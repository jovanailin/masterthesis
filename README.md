# Global Gradient Boosting Models for Intermittent Retail Demand Forecasting

Code for a master's thesis on forecasting daily item-level retail demand with global XGBoost models,
evaluated on the M5 (Walmart) dataset.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory data analysis and demand classes (thesis Sections 6.2, 6.3) |
| `notebooks/02_data_preparation.ipynb` | Inputs of the direct forecasting models and their checks (Section 6.4) |
| `notebooks/03_benchmarks.ipynb` | Classical benchmark methods, accuracy measures, validation scores (Sections 5.2, 5.3, 6.5, 7.1) |
| `notebooks/04_global_xgboost.ipynb` | Global XGBoost: tuning, validation results of the global, partially global and recursive models (Sections 6.6, 7.1–7.4, 7.7) |
| `notebooks/05_models_per_series.ipynb` | Models per series against global models on a sample of 100 series, with a check of the selection bias (Section 7.2) |
| `src/m5_data.py` | Data loading, day indices of the train/validation/test split, the 12 M5 hierarchy levels |
| `src/features.py` | Panel of sales, calendar and price inputs; origins and design matrices of the 7-day models (horizon as an input) |
| `src/benchmarks.py` | Naive, seasonal naive, mean, Croston, SBA, TSB and ADIDA forecasts for all series |
| `src/metrics.py` | RMSSE, WRMSSE over the 12 levels (checked against the official weights), MASE, bias |
| `src/global_xgb.py` | Global XGBoost: variants, Hyperopt tuning on the validation window, refit for the test window; models per store and per demand class; recursive strategy (Sections 6.6, 7.2, 7.7) |
| `src/local_xgb.py` | Models per series and a global model on a sample of 100 series: tuning, refit, selection-bias check (Sections 6.6, 7.2) |
| `src/intermittency.py` | Demand statistics (zero share, ADI, CV²) and the Syntetos–Boylan–Croston classification |
| `src/series_sampling.py` | Stratified sample of series with a largest-remainder (Hamilton) allocation |
| `src/factorial_stats.py` | Wilcoxon–Holm, Friedman–Nemenyi, critical difference diagram |
| `src/plot_style.py` | Shared style for the thesis figures |
| `kaggle_jobs/` | Jobs that run as Kaggle GPU kernels; `upload_src.py` puts `src/` into a private Kaggle dataset for them |
| `kaggle_jobs/speed_test/` | GPU speed test of the global XGBoost (results in `results/kaggle/speed_test/`) |
| `kaggle_jobs/global_xgb/` | Kaggle kernels of the XGBoost models (`push.py <job>`): tuning and refit of the global variants (one kernel per loss), seeds, models per store and per demand class, recursive strategy |
| `figures/` | Figures used in the thesis (PNG and PDF) |
| `results/` | Tables produced by the notebooks (CSV) |
| `archive/replication/` | Abandoned replication of Tsantilis et al. (2025), kept for reference |
| `*.docx`, `Pregled modela.xlsx`, `Primer dataseta.xlsx` | Working documents: EDA summary, summary of Tsantilis et al. (2025), thesis proposal, overview of the models and their validation scores, example rows of the training tables |
| `data/` | Not tracked by git; see below |

## Data

The data are not included in this repository, because the Kaggle competition rules do not allow
redistribution.

1. **Official M5 files (Kaggle).** Accept the rules of the
   [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules)
   competition, create an API token at <https://www.kaggle.com/settings/api>, then run:

   ```bash
   kaggle competitions download -c m5-forecasting-accuracy -p data/kaggle
   unzip data/kaggle/m5-forecasting-accuracy.zip -d data/kaggle
   ```

2. **Test-period ground truth and WRMSSE weights (Nixtla mirror).** The Kaggle release does not contain
   the sales for the final evaluation horizon. They are taken from the Nixtla mirror, whose overlapping
   files were verified to be identical to the Kaggle files:

   ```bash
   curl -L -o data/raw/m5.zip https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip
   unzip data/raw/m5.zip -d data/raw
   ```

| Window | Days | Dates | Use |
|---|---|---|---|
| Training | `d_1` – `d_1913` | 2011-01-29 – 2016-04-24 | model fitting, EDA |
| Validation | `d_1914` – `d_1941` | 2016-04-25 – 2016-05-22 | tuning |
| Test | `d_1942` – `d_1969` | 2016-05-23 – 2016-06-19 | final evaluation only |

Forecasts cover 7 days. Each 28-day window is forecast from 4 weekly origins (rolling evaluation): the models
are not refitted between the origins, but their inputs use the sales observed up to each origin.

Source: Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results,
findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364.

## Running

```bash
pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_data_preparation.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_benchmarks.ipynb
```

The EDA notebook runs in about 1–2 minutes, the data preparation in about 3 and the benchmarks in about 2. The data preparation
saves the panel of inputs to `data/processed/panel/`, which the later notebooks load, so it must run before them.
