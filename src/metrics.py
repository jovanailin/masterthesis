"""Accuracy measures of the 28-day forecasts (thesis Section 5.3).

RMSSE and MASE divide the forecast error of a series by the in-sample error of the one-step naive forecast on
its own history, from its first non-zero sale to the forecast origin (the M5 convention). WRMSSE, the official
M5 measure, averages the RMSSE of all 42,840 series of the 12 hierarchy levels: within a level each series is
weighted by its dollar sales in the 28 days up to the forecast origin, and the 12 levels count equally.

As in the official M5 evaluation, the scales are computed on the raw sales history, with Christmas Day as
recorded (zero sales).
"""
import numpy as np
import pandas as pd
from scipy import sparse

import m5_data as m5

# Key columns of each level, in the order of Agg_Level_1 and Agg_Level_2 in the official weight files.
LEVEL_KEYS = {
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
    11: ["state_id", "item_id"],
    12: ["item_id", "store_id"],
}


def level_matrices(hier: pd.DataFrame) -> dict:
    """Per level: the 0/1 matrix that sums the bottom series into the level's series, and their keys.

    Level 12 keeps the order of the bottom series, so that its rows line up with the panel.
    """
    n = len(hier)
    out = {}
    for lvl, cols in LEVEL_KEYS.items():
        if lvl == 12:
            S = sparse.identity(n, format="csr")
            keys = pd.DataFrame({"agg_1": hier["item_id"].astype(str), "agg_2": hier["store_id"].astype(str)})
        else:
            if cols:
                parts = [hier[c].astype(str) for c in cols]
                key = parts[0] if len(parts) == 1 else parts[0] + "|" + parts[1]
                codes, uniques = pd.factorize(key, sort=True)
                split = [u.split("|") for u in uniques]
                keys = pd.DataFrame({"agg_1": [s[0] for s in split], "agg_2": [s[1] if len(s) > 1 else "X" for s in split]})
            else:
                codes = np.zeros(n, dtype=int)
                keys = pd.DataFrame({"agg_1": ["Total"], "agg_2": ["X"]})
            S = sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(len(keys), n))
        out[lvl] = (S, keys.assign(level=lvl)[["level", "agg_1", "agg_2"]])
    return out


def naive_scale(hist: np.ndarray, power: int) -> np.ndarray:
    """Mean |one-step change|**power of each row of hist, from its first non-zero value onwards."""
    hist = np.asarray(hist, dtype=np.float64)
    started = np.maximum.accumulate(hist > 0, axis=1)[:, :-1]  # change from day t to t+1 counts if t >= first sale
    change = np.abs(np.diff(hist, axis=1)) ** power
    return (change * started).sum(axis=1) / started.sum(axis=1)


def dollar_sales(sales: pd.DataFrame, origin: int, days: int = 28) -> np.ndarray:
    """Dollar sales of each bottom series over the `days` days up to `origin` (the WRMSSE weights)."""
    calendar = m5.load_calendar().set_index("d_num")
    prices = m5.load_prices()
    window = list(range(origin - days + 1, origin + 1))
    week = calendar.loc[window, "wm_yr_wk"].to_numpy()
    price = (prices[prices["wm_yr_wk"].isin(set(week))]
             .pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price", observed=True))
    rows = pd.MultiIndex.from_arrays([sales["store_id"].astype(str), sales["item_id"].astype(str)])
    price.index = pd.MultiIndex.from_arrays([price.index.get_level_values(0).astype(str),
                                             price.index.get_level_values(1).astype(str)])
    p = price.reindex(rows)[week].to_numpy()
    units = sales[m5.day_cols(window[0], window[-1])].to_numpy(np.float64)
    return np.nansum(units * p, axis=1)


class Evaluator:
    """Scales and weights of one forecast origin; scores (n_bottom, H) forecasts of the bottom series."""

    def __init__(self, hier: pd.DataFrame, hist: np.ndarray, revenue: np.ndarray, actual: np.ndarray | None = None):
        self.levels = level_matrices(hier)
        self.scale = {}
        self.weight = {}
        self.actual = {}
        total = revenue.sum()
        for lvl, (S, _) in self.levels.items():
            agg = hist if lvl == 12 else S @ hist
            self.scale[lvl] = naive_scale(agg, 2)
            self.weight[lvl] = (S @ revenue) / total
            if actual is not None:
                self.actual[lvl] = actual if lvl == 12 else S @ actual
        self.mae_scale = naive_scale(hist, 1)

    @classmethod
    def for_origin(cls, origin: int, actual_days: tuple[int, int] | None = None) -> "Evaluator":
        """Evaluator built from the raw Kaggle files; `actual_days` (first, last) must lie within d_1 - d_1941."""
        last = max(origin, actual_days[1] if actual_days else origin)
        assert last <= m5.VALID_END, "the test window d_1942 - d_1969 is read only in the final evaluation"
        sales = m5.load_sales(last_day=last)
        hist = sales[m5.day_cols(1, origin)].to_numpy(np.float64)
        actual = sales[m5.day_cols(*actual_days)].to_numpy(np.float64) if actual_days else None
        hier = sales[m5.ID_COLS].astype(str)
        return cls(hier, hist, dollar_sales(sales, origin), actual)

    def weights_table(self) -> pd.DataFrame:
        return pd.concat([keys.assign(weight=self.weight[lvl]) for lvl, (_, keys) in self.levels.items()], ignore_index=True)

    def rmsse(self, forecast: np.ndarray) -> dict:
        out = {}
        for lvl, (S, _) in self.levels.items():
            f = forecast if lvl == 12 else S @ forecast
            out[lvl] = np.sqrt(np.mean((self.actual[lvl] - f) ** 2, axis=1) / self.scale[lvl])
        return out

    def wrmsse(self, forecast: np.ndarray) -> pd.Series:
        """WRMSSE of each level (weighted RMSSE, weights summing to 1) and their mean, the M5 score."""
        r = self.rmsse(forecast)
        by_level = pd.Series({lvl: float(np.sum(self.weight[lvl] * r[lvl])) for lvl in r}, name="WRMSSE")
        by_level.index.name = "level"
        return pd.concat([pd.Series({"all": by_level.mean()}), by_level])

    def bottom(self, forecast: np.ndarray) -> pd.DataFrame:
        """Per-series measures of the bottom level: RMSSE, MASE, scaled mean error, share above and below."""
        a = self.actual[12]
        err = forecast - a
        return pd.DataFrame(
            {
                "rmsse": np.sqrt(np.mean(err**2, axis=1) / self.scale[12]),
                "mase": np.mean(np.abs(err), axis=1) / self.mae_scale,
                "sme": np.mean(err, axis=1) / self.mae_scale,
                "above": np.mean(forecast > a, axis=1),
                "below": np.mean(forecast < a, axis=1),
            }
        )
