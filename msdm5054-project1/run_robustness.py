"""
MSDM5054 Project 1 — Direction 1: Robustness & hyperparameter neighbourhood check
=================================================================================
Run multi-seed stability checks and a small hyperparameter grid on top of the
already-built feature matrix. Feature engineering is NOT repeated per seed —
only model fitting and fold assignment vary, which is exactly the source of
variance we want to quantify.

Outputs (in --out):
  robustness_multiseed.csv   one row per seed: OOF AUC for app-only / all-tables,
                             RQ2 thin-file & with-bureau gains (mean, lo, hi),
                             RQ1 logit median vs +indicator AUC.
  robustness_tuning.csv       one row per hyperparameter setting (3-fold CV AUC
                             on the all-tables design).

Usage
  python run_robustness.py --data ./data --out ./results
  python run_robustness.py --data ./data --out ./results --fast     # 20% sample, smoke test
  python run_robustness.py --data ./data --out ./results --seeds 42,2026,7,123,2024

Time budget (Kaggle Notebook, 30 GB RAM):
  - feature engineering once:            ~3-5 min
  - all-tables 5-fold CV per seed:      ~15-20 min  x 5 seeds = ~1.5-2 h
  - app-only 5-fold CV per seed:        ~3-5 min   x 5 seeds = ~25 min
  - logit strategies per seed:          ~2 min     x 5 seeds = ~10 min
  - tuning grid (3-fold, 6 settings):  ~10 min    x 6      = ~1 h
  TOTAL ~ 4 h.  Use --fast for a quick sanity run.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from home_credit_pipeline import (
    build, cv_lgb, cv_logit, boot_auc_diff, log, resolve_data, SEED, KEY, TARGET, HAS_LGB,
)


# Seeds used for the multi-seed stability run. 5 seeds give df=4 for the std,
# which is the minimum that makes a reported std interpretable.
DEFAULT_SEEDS = [42, 2026, 7, 123, 2024]

# Hyperparameter neighbourhood grid. 3-fold CV per row to keep it fast.
# "current" reproduces the production setting exactly.
TUNING_GRID = [
    dict(tag="current",        num_leaves=32, min_child_samples=100, colsample_bytree=0.5, reg_lambda=5.0),
    dict(tag="deeper",         num_leaves=63, min_child_samples=100, colsample_bytree=0.5, reg_lambda=5.0),
    dict(tag="more_conserv",   num_leaves=15, min_child_samples=200, colsample_bytree=0.5, reg_lambda=5.0),
    dict(tag="colsample_0.8",  num_leaves=32, min_child_samples=100, colsample_bytree=0.8, reg_lambda=5.0),
    dict(tag="reg_l20",        num_leaves=32, min_child_samples=100, colsample_bytree=0.5, reg_lambda=20.0),
    dict(tag="kaggle_common",  num_leaves=31, min_child_samples=30,  colsample_bytree=0.7, reg_lambda=1.0),
]


def thin_file_mask(tr):
    """Thin-file = no Credit Bureau record at all (BURO_COUNT is NaN after left join)."""
    return tr["BURO_COUNT"].isna().values


def run_multiseed(tr, y, app_cols, feats, seeds, out):
    log(f"=== Multi-seed robustness: {len(seeds)} seeds ===")
    thin = thin_file_mask(tr)
    rows = []
    for seed in seeds:
        log(f"--- seed = {seed} ---")
        # app-only model
        p_app, _, _, _ = cv_lgb(tr[app_cols], y, tag=f"seed{seed} app-only", seed=seed)
        auc_app = roc_auc_score(y, p_app)
        # all-tables model
        p_full, _, _, _ = cv_lgb(tr[feats], y, tag=f"seed{seed} all-tables", seed=seed)
        auc_full = roc_auc_score(y, p_full)
        # RQ2 subgroup gains with bootstrap CI (re-computed per seed because OOF changes)
        g_thin = boot_auc_diff(y, p_app, p_full, thin, seed=seed)
        g_buro = boot_auc_diff(y, p_app, p_full, ~thin, seed=seed)
        # RQ1: logit median vs +indicator on app features
        auc_logit_med = roc_auc_score(y, cv_logit(tr[app_cols], y, add_indicator=False,
                                                   tag=f"seed{seed} logit median", seed=seed))
        auc_logit_ind = roc_auc_score(y, cv_logit(tr[app_cols], y, add_indicator=True,
                                                   tag=f"seed{seed} logit median+ind", seed=seed))
        rows.append(dict(
            seed=seed,
            auc_lgb_app=auc_app,
            auc_lgb_all=auc_full,
            rq2_gain_thin_mean=g_thin[0], rq2_gain_thin_lo=g_thin[1], rq2_gain_thin_hi=g_thin[2],
            rq2_gain_buro_mean=g_buro[0], rq2_gain_buro_lo=g_buro[1], rq2_gain_buro_hi=g_buro[2],
            rq1_logit_median=auc_logit_med,
            rq1_logit_plus_ind=auc_logit_ind,
            rq1_ind_boost=auc_logit_ind - auc_logit_med,
        ))
        log(f"  seed {seed}: app={auc_app:.4f}  all={auc_full:.4f}  "
            f"gain_thin={g_thin[0]:+.4f} [{g_thin[1]:+.4f},{g_thin[2]:+.4f}]  "
            f"gain_buro={g_buro[0]:+.4f} [{g_buro[1]:+.4f},{g_buro[2]:+.4f}]  "
            f"RQ1 boost={auc_logit_ind - auc_logit_med:+.4f}")
    df = pd.DataFrame(rows)
    # append a summary row with mean / std across seeds
    summary = {c: ("mean", "std") for c in df.columns if c != "seed"}
    agg = df.agg(summary)
    log("=== Across-seed summary ===")
    log(agg.round(4).to_string())
    df.to_csv(os.path.join(out, "robustness_multiseed.csv"), index=False)
    agg.to_csv(os.path.join(out, "robustness_multiseed_summary.csv"))
    return df


def run_tuning(tr, y, feats, out, base_seed=SEED):
    log("=== Hyperparameter neighbourhood check (3-fold CV) ===")
    rows = []
    for setting in TUNING_GRID:
        tag = setting.pop("tag")
        log(f"--- {tag}: {setting} ---")
        # 3-fold CV, fixed seed, no test prediction — we only care about OOF AUC here.
        oof, _, _, _ = cv_lgb(tr[feats], y, folds=3, params=setting,
                              tag=f"tune {tag}", seed=base_seed)
        auc = roc_auc_score(y, oof)
        rows.append(dict(tag=tag, **setting, oof_auc_3fold=auc))
        log(f"  {tag}: OOF AUC (3-fold) = {auc:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "robustness_tuning.csv"), index=False)
    log("=== Tuning summary ===")
    log(df.round(4).to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--fast", action="store_true", help="20%% row sample for smoke testing")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                    help="comma-separated list of seeds, e.g. 42,2026,7")
    ap.add_argument("--skip-tuning", action="store_true", help="only run multi-seed, skip hyperparam grid")
    ap.add_argument("--skip-multiseed", action="store_true", help="only run tuning grid")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not HAS_LGB:
        raise SystemExit("lightgbm is required for run_robustness.py")

    args.data = resolve_data(args.data)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Feature engineering runs ONCE — it has no randomness.
    tr, _ = build(args.data, use_aux=True)
    if args.fast:
        tr = tr.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    y = tr[TARGET].astype(int).values
    feats = [c for c in tr.columns if c not in (KEY, TARGET)]
    # Re-derive app_cols the same way the main pipeline does.
    raw_app = pd.read_csv(os.path.join(args.data, "application_train.csv"), nrows=5).columns
    app_cols = [c for c in feats if c in raw_app or c in
                ("DAYS_EMPLOYED_ANOM", "CREDIT_INCOME", "ANNUITY_INCOME", "PAYMENT_RATE", "GOODS_CREDIT",
                 "EMPLOYED_AGE", "INCOME_PER_PERSON", "EXT_MEAN", "EXT_STD", "EXT_PROD", "N_MISSING_APP")]
    log(f"train={tr.shape[0]}, n_features={len(feats)}, n_app_features={len(app_cols)}, seeds={seeds}")

    if not args.skip_multiseed:
        run_multiseed(tr, y, app_cols, feats, seeds, args.out)
    if not args.skip_tuning:
        run_tuning(tr, y, feats, args.out)

    log("DONE. Files written:")
    for f in ["robustness_multiseed.csv", "robustness_multiseed_summary.csv", "robustness_tuning.csv"]:
        p = os.path.join(args.out, f)
        if os.path.exists(p):
            log(f"  {p}")


if __name__ == "__main__":
    main()
