# Global Gradient Boosting Models for Intermittent Retail Demand Forecasting

Code for a master's thesis on forecasting daily item-level retail demand with global XGBoost models,
evaluated on the M5 (Walmart) dataset.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory data analysis (thesis Section 6.2) |
| `notebooks/02_data_preparation.ipynb` | Inputs of the direct forecasting models and their checks (Section 6.3) |
| `src/m5_data.py` | Data loading, day indices of the train/validation/test split, the 12 M5 hierarchy levels |
| `src/features.py` | Panel of sales, calendar and price inputs; origins and design matrices of the horizon models |
| `src/intermittency.py` | Demand statistics (zero share, ADI, CV²) and the Syntetos–Boylan–Croston classification |
| `src/series_sampling.py` | Stratified sample of series with a largest-remainder (Hamilton) allocation |
| `src/factorial_stats.py` | Wilcoxon–Holm, Friedman–Nemenyi, critical difference diagram |
| `src/plot_style.py` | Shared style for the thesis figures |
| `figures/` | Figures used in the thesis (PNG and PDF) |
| `results/` | Tables produced by the notebooks (CSV) |
| `archive/replication/` | Abandoned replication of Tsantilis et al. (2025), kept for reference |
| `*.docx` | Working documents: EDA summary, summary of Tsantilis et al. (2025), thesis proposal |
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

Source: Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results,
findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346–1364.

## Running

```bash
pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_data_preparation.ipynb
```

The EDA notebook runs in about 1–2 minutes and the data preparation in about 3 minutes. The data preparation
saves the panel of inputs to `data/processed/panel/`, which the modelling notebooks load.
