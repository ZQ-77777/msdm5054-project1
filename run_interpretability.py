"""
MSDM5054 Project 1 — Direction 2: Interpretability (SHAP) & publication figures
================================================================================
Retrains the production all-tables LightGBM (5-fold CV, seed=42, identical to the
main pipeline) on the real competition data and produces every artifact needed for
the Direction-2 deliverables:

  1. OOF predictions of the all-tables model  -> fig_rq2_calibration (redrawn)
  2. TreeSHAP contributions on a fixed sample -> fig_rq3_top20 (redrawn),
                                                 SHAP dependence plots (top-5),
                                                 group-wise SHAP comparison
  3. Group masks (thin-file vs with bureau)   -> RQ2 / RQ3 cross-analysis

Artifacts written to --out (default ./results):
  oof_alltables.npy          OOF predicted probabilities (n_train,)
  y.npy, thin_mask.npy       labels and thin-file mask (BURO_COUNT is NaN)
  shap_contrib.npy           (n_sample, n_features) TreeSHAP values, fold-0 model
  shap_sample_values.npy     matching feature values (same rows)
  shap_sample_cols.json      feature names
  shap_group_contrib.csv     mean |SHAP| per group (thin / with bureau / all)
  shap_mean_abs_contrib.csv  overall mean |SHAP| (sanity check vs report)
  interpretability_summary.json

Usage
  python run_interpretability.py --data ./data --out ./results
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# reuse the production pipeline unchanged
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import the pipeline from this folder
from home_credit_pipeline import build, cv_lgb, resolve_data, SEED, KEY, TARGET  # noqa: E402

N_SHAP_SAMPLE = 50000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    args.data = resolve_data(args.data)
    log("building features (7 tables) ...")
    tr, _te = build(args.data, use_aux=True)
    # pandas >= 3.0 uses the new 'str' dtype, which the original pipeline's
    # `dtype == "object"` check misses — convert every non-numeric column to
    # category explicitly so LightGBM accepts the frame.
    for c in tr.columns:
        if c not in (KEY, TARGET) and not pd.api.types.is_numeric_dtype(tr[c]):
            tr[c] = tr[c].astype("category")
    y = tr[TARGET].astype(int).values
    feats = [c for c in tr.columns if c not in (KEY, TARGET)]
    thin = tr["BURO_COUNT"].isna().values

    # ---------- 1. all-tables model, same config/seed as the main pipeline
    log("training all-tables LightGBM (5-fold, seed=42) ...")
    p_full, _pred_te, _imp, models = cv_lgb(tr[feats], y, tag="D2 all-tables", seed=SEED)
    auc_full = roc_auc_score(y, p_full)
    log(f"all-tables OOF AUC = {auc_full:.5f}  (report: 0.79261)")
    np.save(os.path.join(args.out, "oof_alltables.npy"), p_full.astype("float64"))
    np.save(os.path.join(args.out, "y.npy"), y)
    np.save(os.path.join(args.out, "thin_mask.npy"), thin)

    # per-group AUC (sanity check vs rq2_thinfile.csv)
    for name, m in [("thin", thin), ("buro", ~thin)]:
        log(f"  AUC[{name}] = {roc_auc_score(y[m], p_full[m]):.5f}")

    # ---------- 2. TreeSHAP on a fixed sample, fold-0 model (same recipe as rq3_contrib)
    log(f"computing TreeSHAP on {N_SHAP_SAMPLE:,} rows (fold-0 model) ...")
    Xs = tr[feats].sample(min(N_SHAP_SAMPLE, len(tr)), random_state=SEED)
    contrib = models[0].booster_.predict(Xs, pred_contrib=True)[:, :-1]  # drop bias column
    np.save(os.path.join(args.out, "shap_contrib.npy"), contrib.astype("float32"))
    # categorical columns -> integer codes (NaN preserved), so everything fits float32
    Xn = Xs.copy()
    for c in Xn.columns:
        if str(Xn[c].dtype) in ("category", "str", "object"):
            codes = Xn[c].astype("category").cat.codes
            Xn[c] = codes.astype("float32").mask(codes < 0, np.nan)
    np.save(os.path.join(args.out, "shap_sample_values.npy"), Xn.to_numpy(dtype="float32"))
    json.dump(list(Xs.columns), open(os.path.join(args.out, "shap_sample_cols.json"), "w"))

    mean_abs = pd.Series(np.abs(contrib).mean(0), index=feats).sort_values(ascending=False)
    mean_abs.to_csv(os.path.join(args.out, "shap_mean_abs_contrib.csv"))

    # ---------- 3. group-wise mean |SHAP| (thin-file vs with bureau)
    thin_pos = tr.index.get_indexer(Xs.index)  # Xs sampled from tr, so all found
    m_thin = thin[thin_pos]
    g = pd.DataFrame({
        "all": np.abs(contrib).mean(0),
        "thin_file": np.abs(contrib[m_thin]).mean(0),
        "with_bureau": np.abs(contrib[~m_thin]).mean(0),
    }, index=feats)
    g["diff_thin_minus_buro"] = g["thin_file"] - g["with_bureau"]
    g.sort_values("all", ascending=False).to_csv(os.path.join(args.out, "shap_group_contrib.csv"))
    log(f"group sample sizes: thin={int(m_thin.sum()):,}, with_bureau={int((~m_thin).sum()):,}")

    # sanity check against the report's rq3_mean_abs_contrib.csv
    ref = "../results/rq3_mean_abs_contrib.csv"
    if os.path.exists(ref):
        old = pd.read_csv(ref, index_col=0).iloc[:, 0]
        cmp = pd.DataFrame({"report": old, "retrained": mean_abs}).dropna()
        corr = np.corrcoef(cmp["report"], cmp["retrained"])[0, 1]
        log(f"sanity: corr(report |SHAP|, retrained |SHAP|) = {corr:.4f} over {len(cmp)} features")

    summary = {
        "auc_all_tables": auc_full,
        "auc_thin": float(roc_auc_score(y[thin], p_full[thin])),
        "auc_buro": float(roc_auc_score(y[~thin], p_full[~thin])),
        "n_shap_sample": int(len(Xs)),
        "n_thin_sample": int(m_thin.sum()),
        "top10_mean_abs_shap": {k: float(v) for k, v in mean_abs.head(10).items()},
    }
    json.dump(summary, open(os.path.join(args.out, "interpretability_summary.json"), "w"), indent=2)
    log(f"SUMMARY {summary}")
    log("DONE")


if __name__ == "__main__":
    main()
