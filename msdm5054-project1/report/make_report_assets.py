"""
Turn pipeline outputs (results/) into LaTeX tables + figures used by report_main.tex.
Every number in the report then comes straight from the result files (no manual copying).

Usage:
    python make_report_assets.py --results ../results --kaggle_public 0.xxx --kaggle_private 0.xxx
Outputs (in this folder):
    tables/tab_main.tex  tables/tab_missing.tex  tables/tab_thinfile.tex  tables/macros.tex
    fig_rq1_missing.png  (+ copies of fig_rq2_calibration.png, fig_rq3_top20.png if present)
Missing results are rendered as "TBD" so the LaTeX always compiles.
"""
import argparse, json, os, shutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="../results")
ap.add_argument("--kaggle_public", default=None)
ap.add_argument("--kaggle_private", default=None)
ap.add_argument("--keep_figures", action="store_true", help="do not regenerate/overwrite the polished figures")
a = ap.parse_args()
R = a.results
os.makedirs("tables", exist_ok=True)


def load_json(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def f(x, d=4):
    return "TBD" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


S, R1 = load_json("summary.json"), load_json("rq1_results.json")

# ---------------- robustness (Direction 1)
rp = os.path.join(R, "robustness_summary.csv")
rows_rb, rb_macros = [], {}
if os.path.exists(rp):
    rb = pd.read_csv(rp, index_col=0)
    seeds = pd.read_csv(os.path.join(R, "robustness_seeds.csv"))
    labels = [("auc_app", "OOF AUC, application only"), ("auc_all", "OOF AUC, all tables"),
              ("auc_all_thin", "AUC, thin-file (all tables)"), ("auc_all_thick", "AUC, with bureau (all tables)"),
              ("gain_thin", "Auxiliary-data gain, thin-file"), ("gain_thick", "Auxiliary-data gain, with bureau"),
              ("gain_diff", "Difference of the two gains")]
    for k, lab in labels:
        if k in rb.index:
            rows_rb.append((lab, rb.loc[k, "mean"], rb.loc[k, "std"], rb.loc[k, "min"], rb.loc[k, "max"]))
    rb_macros = {"RobSeeds": str(len(seeds)),
                 "RobAucAll": f(rb.loc["auc_all", "mean"]), "RobAucAllSd": f(rb.loc["auc_all", "std"]),
                 "RobGainThin": f(rb.loc["gain_thin", "mean"]), "RobGainThick": f(rb.loc["gain_thick", "mean"]),
                 "RobGainDiff": f(rb.loc["gain_diff", "mean"]), "RobGainDiffSd": f(rb.loc["gain_diff", "std"]),
                 "RobHolds": str(int((seeds.gain_diff > 0).sum()))}
else:
    rb_macros = {k: "TBD" for k in ["RobSeeds", "RobAucAll", "RobAucAllSd", "RobGainThin",
                                    "RobGainThick", "RobGainDiff", "RobGainDiffSd", "RobHolds"]}
with open("tables/tab_robust.tex", "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\nQuantity & mean & sd & min & max\\\\\n\\midrule\n")
    if rows_rb:
        for lab, m, sd, lo, hi in rows_rb:
            fh.write(f"{lab} & {m:.4f} & {sd:.4f} & {lo:.4f} & {hi:.4f}\\\\\n")
    else:
        for _, lab in [("", "OOF AUC, all tables"), ("", "Auxiliary-data gain, thin-file"),
                       ("", "Auxiliary-data gain, with bureau"), ("", "Difference of the two gains")]:
            fh.write(f"{lab} & TBD & TBD & TBD & TBD\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- macros (numbers quoted in the text)
macros = {
    "NTrain": f"{S['n_train']:,}" if "n_train" in S else "TBD",
    "DefaultRate": f(100 * S["default_rate"], 2) + r"\%" if "default_rate" in S else "TBD",
    "NFeatAll": f(S.get("n_features_all"), 0) if "n_features_all" in S else "TBD",
    "NFeatApp": f(S.get("n_features_app"), 0) if "n_features_app" in S else "TBD",
    "AucLogit": f(S.get("auc_logit_app")), "AucLgbApp": f(S.get("auc_lgb_app")), "AucLgbAll": f(S.get("auc_lgb_all")),
    "KagglePublic": a.kaggle_public or "TBD", "KagglePrivate": a.kaggle_private or "TBD",
    "AucIndOnly": f(R1.get("indicators_only_auc")),
}
thin_path = os.path.join(R, "rq2_thinfile.csv")
if os.path.exists(thin_path):
    t = pd.read_csv(thin_path)
    thin = t[t.group.str.startswith("thin")].iloc[0]
    macros["ThinShare"] = f(100 * thin.share, 1) + r"\%"
else:
    macros["ThinShare"] = "TBD"
macros.update(rb_macros)
with open("tables/macros.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------- Table 1
rows = [("Logistic regression (L2)", "application", S.get("auc_logit_app"), "--", "--"),
        ("LightGBM", "application", S.get("auc_lgb_app"), "--", "--"),
        ("LightGBM", "all tables", S.get("auc_lgb_all"), a.kaggle_public or "TBD", a.kaggle_private or "TBD")]
with open("tables/tab_main.tex", "w") as fh:
    fh.write("\\begin{tabular}{llccc}\n\\toprule\nModel & Features & OOF AUC & Kaggle public & Kaggle private\\\\\n\\midrule\n")
    for r in rows:
        fh.write(f"{r[0]} & {r[1]} & {f(r[2])} & {r[3]} & {r[4]}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- Table 2
rows = [("Missingness indicators only (logistic)", R1.get("indicators_only_auc")),
        ("Logistic, median imputation", R1.get("logit_median")),
        ("Logistic, median imputation + indicators", R1.get("logit_median_plus_indicator")),
        ("LightGBM, native NaN handling", R1.get("lgb_native_nan")),
        ("LightGBM, median-imputed", R1.get("lgb_median_imputed"))]
with open("tables/tab_missing.tex", "w") as fh:
    fh.write("\\begin{tabular}{lc}\n\\toprule\nStrategy (application features) & OOF AUC\\\\\n\\midrule\n")
    for n, v in rows:
        fh.write(f"{n} & {f(v)}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- Table 3
with open("tables/tab_thinfile.tex", "w") as fh:
    fh.write("\\begin{tabular}{lrcccccc}\n\\toprule\nGroup & $n$ & Default & AUC (app) & AUC (all) & Gain [95\\% CI] & Brier (all)\\\\\n\\midrule\n")
    if os.path.exists(thin_path):
        for _, r in t.iterrows():
            fh.write(f"{esc(r.group)} & {int(r.n):,} & {100*r.default_rate:.2f}\\% & {r.auc_app_only:.4f} & {r.auc_all_tables:.4f} & "
                     f"{r.gain_mean:+.4f} [{r.gain_lo:+.4f}, {r.gain_hi:+.4f}] & {r.brier_all_tables:.4f}\\\\\n")
    else:
        for g in ["All", "Thin-file (no bureau)", "With bureau history"]:
            fh.write(f"{g} & TBD & TBD & TBD & TBD & TBD & TBD\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- Figure 1 (RQ1)
mp = os.path.join(R, "rq1_missing_table.csv")
if os.path.exists(mp) and not a.keep_figures:
    m = pd.read_csv(mp)
    # Many columns share an identical missingness pattern (e.g. the six AMT_REQ_CREDIT_BUREAU_*
    # fields, or ENTRANCES_{AVG,MODE,MEDI}); collapse them so the figure shows distinct patterns.
    m["_key"] = m.missing_rate.round(6).astype(str) + "_" + m.default_if_missing.round(6).astype(str)
    grp = m.groupby("_key", sort=False)
    m = grp.first().assign(n_same=grp.size().values).reset_index(drop=True).head(10)[::-1]
    labels = [f"{c} ({100*r:.0f}% miss)" + (f" +{k-1} alike" if k > 1 else "")
              for c, r, k in zip(m.feature, m.missing_rate, m.n_same)]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    y = range(len(m))
    ax.barh([i + 0.2 for i in y], 100 * m.default_if_missing, height=0.4, label="field missing")
    ax.barh([i - 0.2 for i in y], 100 * m.default_if_observed, height=0.4, label="field observed")
    ax.axvline(100 * 0.0807, color="k", ls=":", lw=1, label="overall rate (8.1%)")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("default rate (%)"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig("fig_rq1_missing.png", dpi=180)
for fn in ([] if a.keep_figures else ["fig_rq2_calibration.png", "fig_rq3_top20.png"]):
    if os.path.exists(os.path.join(R, fn)):
        shutil.copy(os.path.join(R, fn), fn)
print("assets written:", sorted(os.listdir("tables")), [x for x in os.listdir(".") if x.endswith(".png")])
