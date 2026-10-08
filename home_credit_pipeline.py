"""
MSDM5054 Project 1 — Home Credit Default Risk: full experimental pipeline
===========================================================================
Research questions
  RQ0  Baselines: logistic regression vs LightGBM (application table only vs all tables)
  RQ1  Is missingness informative (MNAR-like)?  Compare imputation strategies.
  RQ2  Financial inclusion: does the model serve "thin-file" applicants (no Credit Bureau history)
       as well as others, and does alternative data (previous apps / installments / POS / cards) help them more?
  RQ3  Interpretation: SHAP-style contributions (LightGBM pred_contrib, no extra package needed)

Usage
  pip install lightgbm pandas numpy scikit-learn matplotlib
  python home_credit_pipeline.py --data ./data --out ./results            # full run
  python home_credit_pipeline.py --data ./data --out ./results --fast     # 20% sample, quick debug
Recommended: Kaggle Notebook (data already attached, 30GB RAM) or Colab.

Expected files in --data (from the Kaggle competition page):
  application_train.csv application_test.csv bureau.csv bureau_balance.csv previous_application.csv
  POS_CASH_balance.csv installments_payments.csv credit_card_balance.csv
"""
import argparse, gc, json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[warn] lightgbm not installed: LightGBM experiments will be skipped.")

warnings.filterwarnings("ignore")
SEED = 42
KEY, TARGET = "SK_ID_CURR", "TARGET"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reduce_mem(df):
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64" and c not in (KEY, "SK_ID_BUREAU", "SK_ID_PREV"):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def resolve_data(path):
    """Find the folder that actually contains application_train.csv.

    Handles the common Kaggle cases: the competition data mounted under a
    slightly different folder name, nested one level deeper, or not attached at all.
    """
    if os.path.exists(os.path.join(path, "application_train.csv")):
        return path
    roots = [path, "/kaggle/input", os.getcwd(), "./data"]
    seen = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dirpath, _dirs, files in os.walk(r):
            seen.append(dirpath)
            if "application_train.csv" in files:
                log(f"data found at {dirpath} (requested: {path})")
                return dirpath
    listing = "\n  ".join(sorted(set(seen))[:40]) or "(nothing under /kaggle/input)"
    raise SystemExit(
        f"application_train.csv not found. Looked under: {roots}\n"
        f"Directories seen:\n  {listing}\n"
        "Fix: in the notebook editor open the right-hand panel -> Input -> '+ Add Input' -> "
        "Competitions -> 'Home Credit Default Risk' -> Add. You must have clicked "
        "'Join Competition' and accepted the rules first."
    )


def read(data, name, nrows=None):
    return reduce_mem(pd.read_csv(os.path.join(data, name), nrows=nrows))


def num_agg(df, key, prefix, aggs=("mean", "max", "min", "sum")):
    """Collapse a one-row-per-record table to one row per client, aggregating numeric columns."""
    num = df.select_dtypes(include=[np.number]).drop(columns=[c for c in df.columns if c.startswith("SK_ID")], errors="ignore")
    num[key] = df[key]
    g = num.groupby(key).agg(list(aggs))
    g.columns = [f"{prefix}_{a}_{b.upper()}" for a, b in g.columns]
    return g


def cat_share(df, key, prefix, cols):
    """Share of each category level per client (one-hot mean)."""
    if not cols:
        return pd.DataFrame(index=df[key].unique())
    d = pd.get_dummies(df[cols], dummy_na=False).astype("float32")
    d[key] = df[key].values
    g = d.groupby(key).mean()
    g.columns = [f"{prefix}_{c}_SHARE" for c in g.columns]
    return g


# =============================================================== feature engineering
def application(data, nrows=None):
    """Load train+test application tables, repair known anomalies and add domain ratios.

    Train and test are concatenated so that every engineered column exists in both; no
    target information is used here, so this cannot leak.
    """
    tr, te = read(data, "application_train.csv", nrows), read(data, "application_test.csv", nrows)
    df = pd.concat([tr, te], ignore_index=True)
    df["N_MISSING_APP"] = df.drop(columns=[TARGET]).isna().sum(axis=1)   # count of missing raw fields
    # Known data anomalies: 365243 is a placeholder for "never employed" (mostly pensioners),
    # and a handful of rows carry an XNA gender. Both become NaN; the employment case keeps a flag.
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan
    # Ratios that lenders actually look at: leverage, affordability and schedule intensity.
    df["CREDIT_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]            # ~ 1/term
    df["GOODS_CREDIT"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["EMPLOYED_AGE"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_MEAN"] = df[ext].mean(axis=1)
    df["EXT_STD"] = df[ext].std(axis=1)
    df["EXT_PROD"] = df[ext].prod(axis=1, min_count=3)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def bureau_features(data, nrows=None):
    """Credit-bureau records at other institutions, plus their monthly balance history."""
    b, bb = read(data, "bureau.csv", nrows), read(data, "bureau_balance.csv", nrows)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(BB_MONTHS=("MONTHS_BALANCE", "size"))
    bb_bad = bb.assign(BAD=bb["STATUS"].astype(str).isin(["1", "2", "3", "4", "5"]).astype("float32")).groupby("SK_ID_BUREAU")["BAD"].mean()
    b = b.join(bb_agg, on="SK_ID_BUREAU").join(bb_bad.rename("BB_BAD_SHARE"), on="SK_ID_BUREAU")
    b["DEBT_CREDIT"] = b["AMT_CREDIT_SUM_DEBT"] / b["AMT_CREDIT_SUM"]
    g = num_agg(b, KEY, "BURO", ("mean", "max", "sum"))
    g["BURO_COUNT"] = b.groupby(KEY).size()
    g["BURO_ACTIVE_COUNT"] = b[b["CREDIT_ACTIVE"] == "Active"].groupby(KEY).size()
    g = g.join(cat_share(b, KEY, "BURO", ["CREDIT_ACTIVE", "CREDIT_TYPE"]))
    del b, bb; gc.collect()
    return g


def prev_features(data, nrows=None):
    """Previous Home Credit applications (approved, refused or cancelled)."""
    p = read(data, "previous_application.csv", nrows)
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        p.loc[p[c] == 365243, c] = np.nan
    p["APP_CREDIT_RATIO"] = p["AMT_APPLICATION"] / p["AMT_CREDIT"]
    g = num_agg(p, KEY, "PREV", ("mean", "max", "min"))
    g["PREV_COUNT"] = p.groupby(KEY).size()
    g = g.join(cat_share(p, KEY, "PREV", ["NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE"]))
    del p; gc.collect()
    return g


def pos_features(data, nrows=None):
    pos = read(data, "POS_CASH_balance.csv", nrows)
    g = num_agg(pos, KEY, "POS", ("mean", "max"))
    g["POS_COUNT"] = pos.groupby(KEY).size()
    del pos; gc.collect()
    return g


def ins_features(data, nrows=None):
    """Instalment payment history: how much was due, how much was paid, and how late."""
    ins = read(data, "installments_payments.csv", nrows)
    ins["PAY_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["PAY_RATIO"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)   # days past due
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)   # days before due
    ins["LATE"] = (ins["DPD"] > 0).astype("float32")
    g = num_agg(ins[[KEY, "PAY_DIFF", "PAY_RATIO", "DPD", "DBD", "LATE", "AMT_PAYMENT"]], KEY, "INS", ("mean", "max", "sum"))
    g["INS_COUNT"] = ins.groupby(KEY).size()
    del ins; gc.collect()
    return g


def cc_features(data, nrows=None):
    cc = read(data, "credit_card_balance.csv", nrows)
    cc["UTIL"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    g = num_agg(cc, KEY, "CC", ("mean", "max"))
    g["CC_COUNT"] = cc.groupby(KEY).size()
    del cc; gc.collect()
    return g


def build(data, nrows=None, use_aux=True):
    """Assemble the full design matrix and split it back into labelled train and unlabelled test."""
    log("application ...")
    df = application(data, nrows)
    if use_aux:
        for name, fn in [("bureau", bureau_features), ("previous", prev_features), ("pos", pos_features),
                         ("installments", ins_features), ("credit card", cc_features)]:
            log(f"{name} ...")
            df = df.join(fn(data, nrows), on=KEY)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cats = list(df.select_dtypes(exclude=[np.number]).columns)   # version-proof: pandas 3 reads text as StringDtype
    for c in cats:
        df[c] = df[c].astype("category")
    df.columns = [str(c).replace(" ", "_").replace(",", "").replace(":", "").replace('"', "") for c in df.columns]
    tr, te = df[df[TARGET].notna()].reset_index(drop=True), df[df[TARGET].isna()].reset_index(drop=True)
    log(f"features built: train {tr.shape}, test {te.shape}")
    return tr, te


# =============================================================== models
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=32, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                  n_estimators=5000, random_state=SEED, verbose=-1, n_jobs=-1)


CAT_AS_CODES = False   # set True (or pass --cat-codes) for LightGBM builds that reject pandas categoricals


def _encode_cats_for_lgb(df):
    """Integer-code the categorical columns, returning (frame, column names).

    Some LightGBM builds refuse pandas CategoricalDtype. Passing the codes together with
    `categorical_feature` keeps LightGBM's categorical splits; passing them alone would
    make the trees treat an arbitrary code order as ordinal.
    """
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not cats:
        return df, []
    df = df.copy()
    for c in cats:
        df[c] = df[c].astype("category").cat.codes.astype("int32")
    return df, cats


def cv_lgb(X, y, Xte=None, folds=5, params=None, tag="lgb", seed=SEED):
    params = dict(LGB_PARAMS, **(params or {}))
    params["random_state"] = seed
    cat_cols = []
    if CAT_AS_CODES:
        X, cat_cols = _encode_cats_for_lgb(X)
        if Xte is not None:
            Xte, _ = _encode_cats_for_lgb(Xte)
    oof, pred = np.zeros(len(X)), np.zeros(len(Xte)) if Xte is not None else None
    imp, models = pd.Series(0.0, index=X.columns), []
    for k, (a, b) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y)):
        m = lgb.LGBMClassifier(**params)
        fit_kw = {"categorical_feature": cat_cols} if cat_cols else {}
        m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)], **fit_kw)
        oof[b] = m.predict_proba(X.iloc[b], num_iteration=m.best_iteration_)[:, 1]
        if Xte is not None:
            pred += m.predict_proba(Xte, num_iteration=m.best_iteration_)[:, 1] / folds
        imp += pd.Series(m.booster_.feature_importance("gain"), index=X.columns) / folds
        models.append(m)
        log(f"  {tag} fold {k}: AUC={roc_auc_score(y[b], oof[b]):.4f}, iters={m.best_iteration_}")
    log(f"{tag} OOF AUC = {roc_auc_score(y, oof):.4f}")
    return oof, pred, imp.sort_values(ascending=False), models


def logit_design(X, add_indicator):
    """One-hot categoricals; numeric -> median impute (+ optional missing indicators) -> standardize."""
    Xn = pd.get_dummies(X, dummy_na=False).astype("float32")
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=add_indicator),
                         StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    return Xn, pipe


def cv_logit(X, y, add_indicator=False, folds=5, tag="logit", seed=SEED):
    Xn, pipe = logit_design(X, add_indicator)
    oof = np.zeros(len(X))
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(Xn, y):
        pipe.fit(Xn.iloc[a], y[a])
        oof[b] = pipe.predict_proba(Xn.iloc[b])[:, 1]
    log(f"{tag} OOF AUC = {roc_auc_score(y, oof):.4f}")
    return oof


def boot_auc_diff(y, p1, p2, mask=None, B=300, seed=SEED):
    """Bootstrap 95% CI for AUC(p2) - AUC(p1) on subset mask."""
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0] if mask is not None else np.arange(len(y))
    d = []
    for _ in range(B):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].min() == y[s].max():
            continue
        d.append(roc_auc_score(y[s], p2[s]) - roc_auc_score(y[s], p1[s]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# =============================================================== experiments
def rq1_missingness(tr, y, out, app_cols):
    """RQ1: is missingness itself predictive, and does encoding it explicitly help?"""
    """Descriptive: default rate when a column is missing vs observed; predictive: indicators-only model."""
    log("RQ1: missingness analysis")
    X = tr[app_cols]
    rows = []
    for c in X.columns:
        m = X[c].isna().values
        if 0.01 < m.mean() < 0.99:
            rows.append(dict(feature=c, missing_rate=m.mean(), default_if_missing=y[m].mean(),
                             default_if_observed=y[~m].mean(), diff=y[m].mean() - y[~m].mean()))
    tab = pd.DataFrame(rows).sort_values("diff", key=np.abs, ascending=False)
    tab.to_csv(os.path.join(out, "rq1_missing_table.csv"), index=False)
    # indicators-only model: can "which fields are missing" alone predict default?
    M = X.isna().astype("float32")
    M = M.loc[:, (M.mean() > 0.01) & (M.mean() < 0.99)]
    oof = np.zeros(len(M))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(M, y):
        oof[b] = LogisticRegression(max_iter=2000).fit(M.iloc[a], y[a]).predict_proba(M.iloc[b])[:, 1]
    res = {"indicators_only_auc": roc_auc_score(y, oof), "n_indicator_cols": int(M.shape[1])}
    # strategy comparison on application features
    res["logit_median"] = roc_auc_score(y, cv_logit(X, y, False, tag="logit median-impute"))
    res["logit_median_plus_indicator"] = roc_auc_score(y, cv_logit(X, y, True, tag="logit median+indicator"))
    if HAS_LGB:
        res["lgb_native_nan"] = roc_auc_score(y, cv_lgb(X, y, tag="lgb native NaN")[0])
        Xi = X.copy()
        num = Xi.select_dtypes(include=[np.number]).columns
        Xi[num] = Xi[num].fillna(Xi[num].median())
        res["lgb_median_imputed"] = roc_auc_score(y, cv_lgb(Xi, y, tag="lgb median-imputed")[0])
    json.dump(res, open(os.path.join(out, "rq1_results.json"), "w"), indent=2)
    log(f"RQ1 results: {res}")
    return tab, res


def calib_table(y, p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"pred": p, "obs": y, "bin": q}).groupby("bin").agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))


def rq2_thinfile(tr, y, p_app, p_full, out):
    """RQ2: discrimination and calibration for applicants with no credit-bureau record."""
    log("RQ2: thin-file subgroup analysis")
    thin = tr["BURO_COUNT"].isna().values  # no Credit Bureau record at all
    rows = []
    for name, m in [("all", np.ones(len(y), bool)), ("thin-file (no bureau)", thin), ("with bureau history", ~thin)]:
        r = dict(group=name, n=int(m.sum()), share=float(m.mean()), default_rate=float(y[m].mean()),
                 auc_app_only=roc_auc_score(y[m], p_app[m]), auc_all_tables=roc_auc_score(y[m], p_full[m]),
                 brier_app_only=brier_score_loss(y[m], p_app[m]), brier_all_tables=brier_score_loss(y[m], p_full[m]))
        r["gain_mean"], r["gain_lo"], r["gain_hi"] = boot_auc_diff(y, p_app, p_full, m)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "rq2_thinfile.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, m in [("thin-file", thin), ("with bureau", ~thin)]:
        c = calib_table(y[m], p_full[m])
        ax.plot(c["pred"], c["obs"], "o-", label=name)
    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], "k:", lw=.8)
    ax.set_xlabel("mean predicted PD (decile)"); ax.set_ylabel("observed default rate"); ax.legend()
    ax.set_title("Calibration by group (all-tables model, OOF)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_rq2_calibration.png"), dpi=160)
    log("\n" + tab.round(4).to_string(index=False))
    return tab


def rq3_contrib(model, X, out, n=20000):
    """RQ3: exact TreeSHAP contributions, ranked by mean absolute effect on the log-odds."""
    log("RQ3: feature contributions (TreeSHAP via pred_contrib)")
    Xs = X.sample(min(n, len(X)), random_state=SEED)
    contrib = model.booster_.predict(Xs, pred_contrib=True)[:, :-1]      # last column = bias
    s = pd.Series(np.abs(contrib).mean(0), index=X.columns).sort_values(ascending=False)
    s.to_csv(os.path.join(out, "rq3_mean_abs_contrib.csv"))
    top = s.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(top.index, top.values); ax.set_xlabel("mean |contribution| (log-odds)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_rq3_top20.png"), dpi=160)
    return s


# =============================================================== main
def app_feature_cols(data, feats):
    raw_app = pd.read_csv(os.path.join(data, "application_train.csv"), nrows=5).columns
    return [c for c in feats if c in raw_app or c in
            ("DAYS_EMPLOYED_ANOM", "CREDIT_INCOME", "ANNUITY_INCOME", "PAYMENT_RATE", "GOODS_CREDIT",
             "EMPLOYED_AGE", "INCOME_PER_PERSON", "EXT_MEAN", "EXT_STD", "EXT_PROD", "N_MISSING_APP")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--fast", action="store_true", help="20%% row sample of application tables for debugging")
    ap.add_argument("--nrows", type=int, default=None, help="read only first n rows of every csv (smoke test)")
    ap.add_argument("--cat-codes", dest="cat_codes", action="store_true",
                    help="pass categoricals to LightGBM as integer codes (for builds that reject pandas categoricals)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global CAT_AS_CODES
    CAT_AS_CODES = args.cat_codes

    args.data = resolve_data(args.data)
    tr, te = build(args.data, args.nrows, use_aux=True)
    if args.fast:
        tr = tr.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    y = tr[TARGET].astype(int).values
    feats = [c for c in tr.columns if c not in (KEY, TARGET)]
    app_cols = app_feature_cols(args.data, feats)
    summary = {"n_train": int(len(tr)), "default_rate": float(y.mean()), "n_features_all": len(feats), "n_features_app": len(app_cols)}

    # ---------- RQ0 baselines
    p_logit = cv_logit(tr[app_cols], y, add_indicator=True, tag="RQ0 logit (app only)")
    summary["auc_logit_app"] = roc_auc_score(y, p_logit)
    if HAS_LGB:
        p_app, _, imp_app, _ = cv_lgb(tr[app_cols], y, tag="RQ0 lgb (app only)")
        p_full, p_test, imp_full, models = cv_lgb(tr[feats], y, te[feats], tag="RQ0 lgb (all tables)")
        summary["auc_lgb_app"], summary["auc_lgb_all"] = roc_auc_score(y, p_app), roc_auc_score(y, p_full)
        imp_full.to_csv(os.path.join(args.out, "lgb_gain_importance.csv"))
        if not args.fast:
            pd.DataFrame({KEY: te[KEY].astype(int), TARGET: p_test}).to_csv(os.path.join(args.out, "submission.csv"), index=False)
            log("submission.csv written -> upload to Kaggle (Late Submission)")
        # ---------- RQ2, RQ3
        rq2_thinfile(tr, y, p_app, p_full, args.out)
        rq3_contrib(models[0], tr[feats], args.out)
    # ---------- RQ1
    rq1_missingness(tr, y, args.out, app_cols)

    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    log(f"SUMMARY {summary}")


if __name__ == "__main__":
    main()
