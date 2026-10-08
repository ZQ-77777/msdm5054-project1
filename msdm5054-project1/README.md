# MSDM5054 Project 1 — Home Credit Default Risk

Beyond the Leaderboard: informative missingness and thin-file borrowers in default prediction.

Team: ZHANG Qi (21358199), DU Yunhe (21419400), TANG Yunshu (21275921), WEI Fanjing (21354557).
Kaggle team `msdm5054-zhang-du-tang-wei` — late submission, private AUC **0.79295**, public 0.79713.

## Layout

```
home_credit_pipeline.py     features (7 tables) + baselines + RQ1-RQ3 + Kaggle submission
run_interpretability.py     group-wise TreeSHAP and dependence diagnostics
run_robustness.py           multi-seed stability + hyperparameter neighbourhood check
run_ablation.py             leave-one-source-out ablation of the auxiliary tables
make_mock_data.py           tiny fake data with the real schema, for smoke tests
home_credit_kaggle.ipynb    the pipeline as a ready-to-import Kaggle notebook
results/                    every csv/json/figure the report cites
report/
  report_main.tex           the report; tables and quoted numbers come from results/
  make_report_assets.py     results/ -> tables/*.tex (+ figures, unless --keep_figures)
  make_figures.py           publication-quality redraw of the main figures
  tables/, *.png            generated assets
```

## Reproducing

```bash
pip install lightgbm pandas numpy scikit-learn matplotlib
# data: https://www.kaggle.com/c/home-credit-default-risk/data  -> ./data

python home_credit_pipeline.py --data ./data --out ./results        # ~40-70 min, writes submission.csv
python run_interpretability.py --data ./data --out ./results        # SHAP, group profiles
python run_robustness.py       --data ./data --out ./results        # 5 seeds + 6 settings
python run_ablation.py         --data ./data --out ./results        # 5 leave-one-source-out runs

cd report
python make_report_assets.py --results ../results \
       --kaggle_public 0.79713 --kaggle_private 0.79295 --keep_figures
pdflatex report_main.tex && pdflatex report_main.tex
```

Smoke test without the competition data:
`python make_mock_data.py ./mock && python home_credit_pipeline.py --data ./mock --out ./mock_res`

## Two notes on reproducibility

- **Categorical handling.** By default categorical columns are passed to LightGBM as pandas
  `category` dtype. Some LightGBM builds reject this; `--cat-codes` passes integer codes with
  `categorical_feature` declared instead. The multi-seed table in the report was produced on such
  a build *before* the declaration was added, so its seed-42 column differs from Table 1 by at
  most 0.0014 AUC. Everything else uses the default path.
- **Seeds.** `cv_lgb(..., seed=)` controls both the fold assignment and the LightGBM sampling, so a
  seed sweep gives genuinely independent out-of-fold predictions.

## Research questions

- **RQ1** Is missingness informative, and does encoding it explicitly help? (Informative but redundant.)
- **RQ2** Does the model serve applicants with no credit-bureau history as well as others, and does
  alternative data close the gap? (Narrower, not closed.)
- **RQ3** What drives predictions, and does the picture differ between the two groups?
- Plus a leave-one-source-out ablation of the auxiliary tables and a seed/hyperparameter stability check.

## Use of AI

See the "Use of AI tools" statement in the report.
