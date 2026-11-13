"""Loading helpers for the M5 (Walmart) dataset.

Training data come from the official Kaggle files (data/kaggle). The ground truth for
the final test horizon (d_1942 - d_1969) is not part of the Kaggle release and is taken
from the Nixtla mirror (data/raw), which was verified to be identical to Kaggle on all
overlapping files.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The environment variables point to the data when the code runs elsewhere (Kaggle kernels, kaggle_jobs/).
KAGGLE_DIR = Path(os.environ.get("M5_KAGGLE_DIR", ROOT / "data" / "kaggle"))
NIXTLA_DIR = Path(os.environ.get("M5_NIXTLA_DIR", ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("M5_PROCESSED_DIR", ROOT / "data" / "processed"))

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Day indices are 1-based, as in the M5 column names d_1 ... d_1969.
TRAIN_END = 1913  # last training day
VALID_END = 1941  # last day of the Kaggle release; origin of the test forecasts
TEST_END = 1969  # final test horizon: d_1942 - d_1969
HORIZON = 7  # days forecast from each origin: validation d_1914 - d_1920, test d_1942 - d_1948

# The 12 aggregation levels of the M5 hierarchy (42,840 series in total).
LEVELS = {
    1: [],
    2: ["state_id"],
    3: ["store_id"],
    4: ["cat_id"],
    5: ["dept_id"],
    6: ["state_id", "cat_id"],
    7: ["state_id", "dept_id"],
    8: ["store_id", "cat_id"],
    9: ["store_id", "dept_id"],
    10: ["item_id"],
    11: ["item_id", "state_id"],
    12: ["item_id", "store_id"],
}


def day_cols(start: int = 1, end: int = TRAIN_END) -> list[str]:
    return [f"d_{i}" for i in range(start, end + 1)]


def load_sales(last_day: int = VALID_END) -> pd.DataFrame:
    """Wide sales table: one row per item-store series, one column per day up to `last_day`."""
    days = day_cols(1, last_day)
    dtypes = {c: "category" for c in ID_COLS} | {d: np.int16 for d in days}
    return pd.read_csv(KAGGLE_DIR / "sales_train_evaluation.csv", usecols=["id", *ID_COLS, *days], dtype=dtypes)


def load_test_sales() -> pd.DataFrame:
    """Ground truth for the final test horizon d_1942 - d_1969 (Nixtla mirror)."""
    days = day_cols(VALID_END + 1, TEST_END)
    dtypes = {c: "category" for c in ID_COLS} | {d: np.int16 for d in days}
    df = pd.read_csv(NIXTLA_DIR / "sales_test_evaluation.csv", dtype=dtypes)
    df.insert(0, "id", df["item_id"].astype(str) + "_" + df["store_id"].astype(str) + "_evaluation")
    return df


def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(KAGGLE_DIR / "calendar.csv", parse_dates=["date"])
    cal["d_num"] = cal["d"].str[2:].astype(np.int16)
    for c in ["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        cal[c] = cal[c].astype("category")
    return cal


def load_prices() -> pd.DataFrame:
    dtypes = {"store_id": "category", "item_id": "category", "wm_yr_wk": np.int16, "sell_price": np.float32}
    return pd.read_csv(KAGGLE_DIR / "sell_prices.csv", dtype=dtypes)


def aggregate(sales: pd.DataFrame, level: int, days: list[str]) -> pd.DataFrame:
    """Sum bottom-level sales to one of the 12 hierarchy levels."""
    cols = LEVELS[level]
    if not cols:
        return sales[days].sum().to_frame("Total").T
    return sales.groupby(cols, observed=True)[days].sum()
