"""
MSDM5054 Project 1 — Direction 2: publication-quality figures
==============================================================
Reads the artifacts from run_interpretability.py (./results) plus the original
rq1_missing_table.csv, and produces all final figures (300 dpi, unified style,
colorblind-safe colors that remain distinguishable in grayscale print):

  fig_rq1_missing.png            RQ1 missing-vs-observed default rates (redrawn)
  fig_rq2_calibration.png        RQ2 calibration by group (redrawn, real OOF)
  fig_rq3_top20.png              RQ3 top-20 mean |SHAP| (redrawn)
  fig_shap_dependence_top5.png   NEW: SHAP dependence for the top-5 features
  fig_shap_group_comparison.png  NEW: thin-file vs with-bureau SHAP profiles

Usage:
  python make_figures.py --artifacts ./results --rq1csv ../results/rq1_missing_table.csv --out ./figures
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--artifacts", default="./results")
ap.add_argument("--rq1csv", default="../results/rq1_missing_table.csv")
ap.add_argument("--out", default="./figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
A = a.artifacts

# ---------------------------------------------------------------- unified style
BLUE, ORANGE, RED, GREY = "#2166ac", "#e08214", "#b2182b", "#8c8c8c"
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def save(fig, name):
    fig.savefig(os.path.join(a.out, name))
    plt.close(fig)
    print("wrote", name)


# ================================================================ Fig 1: RQ1
m = pd.read_csv(a.rq1csv)
# collapse columns sharing an identical missingness pattern, keep top-10 patterns
m["_key"] = m.missing_rate.round(6).astype(str) + "_" + m.default_if_missing.round(6).astype(str)
grp = m.groupby("_key", sort=False)
m = grp.first().assign(n_same=grp.size().values).reset_index(drop=True).head(10)[::-1]
labels = [f"{c} ({100*r:.0f}% miss)" + (f"  +{k-1} alike" if k > 1 else "")
          for c, r, k in zip(m.feature, m.missing_rate, m.n_same)]
overall = 8.07

fig, ax = plt.subplots(figsize=(6.4, 3.9))
ypos = np.arange(len(m))
ax.barh(ypos + 0.19, 100 * m.default_if_missing, height=0.36, color=ORANGE, label="field missing")
ax.barh(ypos - 0.19, 100 * m.default_if_observed, height=0.36, color=BLUE, label="field observed")
ax.axvline(overall, color="k", ls=":", lw=1.0, label="overall rate (8.1%)")
for i, v in enumerate(100 * m.default_if_missing):
    ax.text(v + 0.1, i + 0.19, f"{v:.1f}", va="center", fontsize=6.5, color=ORANGE)
ax.set_yticks(ypos, labels)
ax.set_xlabel("default rate (%)")
ax.set_xlim(0, 11.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False)
save(fig, "fig_rq1_missing.png")

# ================================================================ Fig 2: RQ2 calibration
y = np.load(os.path.join(A, "y.npy"))
p = np.load(os.path.join(A, "oof_alltables.npy"))
thin = np.load(os.path.join(A, "thin_mask.npy"))


def calib_table(pred, obs, bins=10):
    q = pd.qcut(pred, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"pred": pred, "obs": obs, "bin": q}).groupby("bin").agg(
        pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))


fig, ax = plt.subplots(figsize=(4.4, 3.9))
# axis range driven by the decile aggregates, not by extreme individual predictions
curves = [calib_table(p[mk], y[mk]) for mk in (thin, ~thin)]
lim = max(max(c["pred"].max(), c["obs"].max()) for c in curves) * 1.1
ax.plot([0, lim], [0, lim], ls=":", lw=0.9, color="k", zorder=1)
for name, mask, color, c in [("thin-file", thin, ORANGE, curves[0]),
                             ("with bureau", ~thin, BLUE, curves[1])]:
    ax.plot(c["pred"], c["obs"], "o-", ms=4.5, lw=1.4, color=color,
            label=f"{name}  (AUC {roc_auc_score(y[mask], p[mask]):.3f})")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_aspect("equal")
ax.set_xlabel("mean predicted PD (decile)")
ax.set_ylabel("observed default rate")
ax.legend(loc="upper left", frameon=False)
ax.set_title("Calibration by group (all-tables model, OOF)", pad=8)
save(fig, "fig_rq2_calibration.png")

# ================================================================ Fig 3: RQ3 top-20
contrib = np.load(os.path.join(A, "shap_contrib.npy"))
cols = json.load(open(os.path.join(A, "shap_sample_cols.json")))
s = pd.Series(np.abs(contrib).mean(0), index=cols).sort_values(ascending=False)
top = s.head(20)[::-1]
fig, ax = plt.subplots(figsize=(4.6, 4.6))
ax.barh(top.index, top.values, color=BLUE, height=0.68)
for i, v in enumerate(top.values):
    ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=6.5, color="#444444")
ax.set_xlabel("mean |SHAP contribution| (log-odds)")
ax.set_xlim(0, top.values[-1] * 1.14)
ax.tick_params(axis="y", length=0)
save(fig, "fig_rq3_top20.png")

# ================================================================ Fig 4: SHAP dependence top-5
Xs = np.load(os.path.join(A, "shap_sample_values.npy"))
buro_count_idx = cols.index("BURO_COUNT")
thin_s = np.isnan(Xs[:, buro_count_idx])  # thin-file ⇔ no bureau record

top5 = list(s.head(5).index)


def draw_dependence(ax, feat, logx=False):
    j = cols.index(feat)
    xv = Xs[:, j].astype("float64")
    sh = contrib[:, j]
    obs = ~np.isnan(xv)
    nan_share = 1 - obs.mean()
    color = np.where(thin_s, ORANGE, BLUE)
    if nan_share > 0.01:
        xm = xv[obs]
        if logx:
            lo, hi = np.log10(xm[xm > 0].min()), np.log10(xm.max())
        else:
            # clip display range to robust quantiles: a few extreme outliers
            # (e.g. debt >> credit glitches) would otherwise flatten the panel
            qlo, qhi = np.quantile(xm, [0.005, 0.995])
            lo, hi = max(qlo, 0) if (xm >= 0).all() else qlo, qhi
        pad = 0.06 * (hi - lo)
        gap = 8 * pad
        gpos = lo - gap  # x position of the "missing" strip
        jit = (np.random.default_rng(0).random(int((~obs).sum())) - 0.5) * 1.6 * pad
        xo = np.log10(xv[obs]) if logx else np.clip(xv[obs], lo, hi)
        ax.scatter(xo, sh[obs], s=3.5, alpha=0.12, c=color[obs], lw=0, rasterized=True)
        ax.scatter(gpos + jit, sh[~obs], s=4, alpha=0.25, c=GREY, lw=0, marker="D", rasterized=True)
        ax.axvline(lo - pad * 1.4, color=GREY, lw=0.6, ls="--")
        if logx:
            t = np.linspace(np.floor(lo), np.ceil(hi), 4)
            ticks, tl = [gpos] + list(10 ** t), ["missing"] + [f"$10^{{{v:g}}}$" for v in t]
        else:
            t = np.linspace(lo, hi, 4)
            ticks, tl = [gpos] + list(t), ["missing"] + [f"{v:.3g}" for v in t]
        ax.set_xticks(ticks, tl)
        ax.text(gpos, np.nanmax(sh) * 0.92, f"{100*nan_share:.0f}% missing", fontsize=6.5,
                color=GREY, ha="center")
        ax.set_xlim(gpos - 2 * pad, hi + pad)
    else:
        xo = np.log10(xv[obs]) if logx else np.clip(
            xv[obs], *np.quantile(xv[obs], [0.005, 0.995]))
        ax.scatter(xo, sh[obs], s=3.5, alpha=0.12, c=color[obs], lw=0, rasterized=True)
        if logx:
            t = np.linspace(np.floor(xo.min()), np.ceil(xo.max()), 4)
            ax.set_xticks(t, [f"$10^{{{v:g}}}$" for v in t])
    rho = spearmanr(xv[obs], sh[obs]).statistic
    ax.set_xlabel(feat)
    ax.set_ylabel("SHAP value (log-odds)")
    ax.set_title(f"{feat}   (ρ = {rho:+.2f})", fontsize=8)


def draw_gender(ax):
    j = cols.index("CODE_GENDER")
    xv = Xs[:, j]
    sh = contrib[:, j]
    codes, counts = np.unique(xv[~np.isnan(xv)].astype(int), return_counts=True)
    order = codes[np.argsort(-counts)]
    lbl = {0: "F", 1: "M"}  # sorted(["F","M"]) → F=0, M=1
    for jj, code in enumerate(order):
        sel = (xv.astype(int) == code) & ~np.isnan(xv)
        jit = (np.random.default_rng(1).random(int(sel.sum())) - 0.5) * 0.5
        ax.scatter(np.full(int(sel.sum()), jj) + jit, sh[sel], s=3.5, alpha=0.12,
                   c=np.where(thin_s, ORANGE, BLUE)[sel], lw=0, rasterized=True)
    nmiss = np.isnan(xv)
    if nmiss.sum() > 0:
        jit = (np.random.default_rng(2).random(int(nmiss.sum())) - 0.5) * 0.5
        ax.scatter(np.full(int(nmiss.sum()), len(order)) + jit, sh[nmiss], s=4, alpha=0.25,
                   c=GREY, lw=0, marker="D", rasterized=True)
        ax.set_xticks(range(len(order) + 1), [lbl.get(int(c), f"code {c}") for c in order] + ["missing"])
    else:
        ax.set_xticks(range(len(order)), [lbl.get(int(c), f"code {c}") for c in order])
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_ylabel("SHAP value (log-odds)")
    ax.set_title("CODE_GENDER", fontsize=8)


fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.9))
axes = axes.ravel()
i = 0
for feat in top5:
    if feat == "CODE_GENDER":
        draw_gender(axes[i])
    else:
        draw_dependence(axes[i], feat, logx=feat.startswith("AMT_"))
    i += 1
# 6th panel: legend instead of an empty plot
axes[5].axis("off")
handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="with bureau"),
           plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="thin-file (no bureau)"),
           plt.Line2D([], [], marker="D", ls="", color=GREY, label="value missing")]
axes[5].legend(handles=handles, loc="center", frameon=False, title="Point color = applicant group")
fig.tight_layout(h_pad=1.4, w_pad=1.2)
save(fig, "fig_shap_dependence_top5.png")

# ================================================================ Fig 5: group SHAP comparison
g = pd.read_csv(os.path.join(A, "shap_group_contrib.csv"), index_col=0)
g["maxg"] = g[["thin_file", "with_bureau"]].max(axis=1)
sel = g.sort_values("maxg", ascending=False).head(12)[::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
yp = np.arange(len(sel))
ax1.barh(yp + 0.19, sel.thin_file, height=0.36, color=ORANGE, label="thin-file (no bureau)")
ax1.barh(yp - 0.19, sel.with_bureau, height=0.36, color=BLUE, label="with bureau history")
ax1.set_yticks(yp, sel.index)
ax1.set_xlabel("mean |SHAP| (log-odds)")
ax1.legend(loc="lower right", frameon=False)
ax1.tick_params(axis="y", length=0)
ax1.set_title("(a) Feature reliance by group", fontsize=8.5)

d = sel.diff_thin_minus_buro
ax2.barh(yp, d, height=0.62, color=[RED if v > 0 else BLUE for v in d])
ax2.margins(x=0.18)
ax2.set_yticks(yp, [""] * len(sel))
ax2.axvline(0, color="k", lw=0.8)
ax2.set_xlabel("thin-file $-$ with bureau\n(mean |SHAP| difference)")
ax2.set_title("(b) Reliance gap (red: thin-file relies more)", fontsize=8.5)
for i, v in enumerate(d):
    ax2.text(v + (0.004 if v > 0 else -0.004), i, f"{v:+.3f}", va="center",
             ha="left" if v > 0 else "right", fontsize=6.5, color="#444444")
fig.tight_layout(w_pad=1.6)
save(fig, "fig_shap_group_comparison.png")

print("all figures done ->", a.out)
