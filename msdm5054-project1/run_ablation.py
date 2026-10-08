"""
Leave-one-source-out ablation of the auxiliary data sources.

Research question: is the predictive value of alternative data spread evenly across
sources, or does one source dominate? Starting from the full 339-feature design we drop
the aggregates of one auxiliary table at a time and refit with identical feature
engineering, hyperparameters, seed and 5-fold CV, so the change in out-of-fold AUC is
attributable to that source alone.

    AUC drop = AUC(full model) - AUC(model without the source)

Usage:
    python run_ablation.py --data ./data --out ./results

Outputs:
    results/ablation_auc.csv        one row per removed source
    figures/fig_ablation_auc.png    bar chart of the AUC drops
"""
from pathlib import Path
import argparse
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent  # the pipeline sits next to this script
sys.path.insert(0, str(PROJECT_ROOT))

from home_credit_pipeline import build, cv_lgb, KEY, TARGET, SEED


# Every engineered feature carries the prefix of the table it was aggregated from, so a
# source is removed simply by dropping the columns with that prefix.
SOURCE_PREFIXES = {
    "Bureau": ("BURO_",),
    "Previous applications": ("PREV_",),
    "POS/CASH": ("POS_",),
    "Installments": ("INS_",),
    "Credit cards": ("CC_",),
}


def select_features(features, remove_prefixes=()):
    """Return the feature list with every column of the given source(s) removed."""
    return [
        c
        for c in features
        if not any(c.startswith(prefix) for prefix in remove_prefixes)
    ]


def run_model(tr, y, features, tag):
    """Fit the standard 5-fold LightGBM on `features` and return the OOF AUC."""
    oof, _, _, _ = cv_lgb(
        tr[features],
        y,
        Xte=None,
        folds=5,
        tag=tag,
    )
    return roc_auc_score(y, oof)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--out", default="./direction4/results")
    parser.add_argument(
        "--figures",
        default="./direction4/figures",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Read only first n rows from each CSV for debugging.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use 20%% of the training rows for debugging.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    fig_dir = Path(args.figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Direction 4: Alternative-data source ablation")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Seed: {SEED}")

    print("\nBuilding full feature matrix...")
    tr, _ = build(args.data, args.nrows, use_aux=True)

    if args.fast:
        tr = (
            tr.sample(frac=0.2, random_state=SEED)
            .reset_index(drop=True)
        )
        print(f"Fast mode: using {len(tr):,} training rows")

    y = tr[TARGET].astype(int).values
    features = [c for c in tr.columns if c not in (KEY, TARGET)]

    print(f"Training rows: {len(tr):,}")
    print(f"Total features: {len(features)}")

    results = []

    print("\n[1/6] Full model")
    auc_full = run_model(
        tr,
        y,
        features,
        "ablation full",
    )
    results.append(
        {
            "removed_source": "None",
            "n_features": len(features),
            "oof_auc": auc_full,
            "auc_drop_vs_full": 0.0,
        }
    )
    print(f"Full model OOF AUC = {auc_full:.6f}")

    for i, (source, prefixes) in enumerate(SOURCE_PREFIXES.items(), start=2):
        selected = select_features(features, prefixes)

        print(f"\n[{i}/6] Remove: {source}")
        print(f"Remaining features: {len(selected)}")

        auc = run_model(
            tr,
            y,
            selected,
            f"ablation - {source}",
        )

        drop = auc_full - auc

        results.append(
            {
                "removed_source": source,
                "n_features": len(selected),
                "oof_auc": auc,
                "auc_drop_vs_full": drop,
            }
        )

        print(f"AUC = {auc:.6f}")
        print(f"AUC drop = {drop:+.6f}")

    result_df = pd.DataFrame(results)

    csv_path = out_dir / "ablation_auc.csv"
    result_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("Ablation results")
    print("=" * 70)
    print(result_df.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    plot_df = result_df[result_df["removed_source"] != "None"].copy()
    plot_df = plot_df.sort_values("auc_drop_vs_full")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.barh(
        plot_df["removed_source"],
        plot_df["auc_drop_vs_full"],
    )
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel("OOF AUC drop relative to full model")
    ax.set_ylabel("Removed data source")
    ax.set_title("Predictive contribution of alternative data sources")
    ax.grid(axis="x", alpha=0.2)

    for y_pos, value in enumerate(plot_df["auc_drop_vs_full"]):
        ax.text(
            value,
            y_pos,
            f" {value:+.4f}",
            va="center",
        )

    fig.tight_layout()

    fig_path = fig_dir / "fig_ablation_auc.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()