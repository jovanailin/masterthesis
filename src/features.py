"""Inputs of the 7-day forecasting models.

A training example is a triple (series i, forecast origin t, horizon h): the model sees the sales up to
day t and predicts day t + h. One model forecasts all horizons h = 1..7 and gets h as an input, so no
forecast is fed back into the model. Besides h, the inputs form four groups:

- L, lags: sales on days t, t-1, ..., t-55;
- E, exogenous inputs known in advance for the target day t + h: weekday, event and the distance to the
  nearest events, SNAP in the store's state, relative price and discount;
- S, inputs derived from the sales history up to day t: rolling means, zero share, mean non-zero size,
  days since the last sale and the age of the series;
- ID, hierarchy identifiers (item, department, category, store, state), kept apart from S so that the
  effect of the series identity can be separated from that of the sales-derived inputs.

Days before a series' first sale and Christmas Day (stores closed) are missing (NaN) in the sales
history, so that they are not mistaken for zero demand. Rows whose target day is Christmas are dropped.

Day numbers are 1-based, as in the M5 column names; arrays indexed by day have a dummy column 0.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import m5_data as m5

N_LAGS = 56
HORIZON = m5.HORIZON
ROLL_WINDOWS = (7, 28, 56, 182)
DISCOUNT_WEEKS = 52  # regular price = highest price of the last 52 weeks

# First forecast origin (last training day) of each stage: tuning forecasts the validation window with
# models fitted up to d_1913, the final models are refitted up to d_1941 and forecast the test window.
# Each window is forecast 7 days at a time from m5.EVAL_WEEKS weekly origins (forecast_origins).
STAGES = {"valid": m5.TRAIN_END, "test": m5.VALID_END}

LAG_COLS = [f"lag_{j}" for j in range(1, N_LAGS + 1)]
S_COLS = [f"mean_{w}" for w in ROLL_WINDOWS] + ["zero_share_28", "zero_share_182", "nz_mean_56",
                                                  "days_since_sale", "age"]
E_COLS = ["wday", "event", "event_type", "days_to_event", "days_since_event", "snap",
          "rel_price", "price_change", "discount"]
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
GROUPS = {"L": LAG_COLS, "E": E_COLS, "S": S_COLS, "ID": ID_COLS}
H_COL = "h"  # always an input, whatever the groups

FEATURE_INFO = {
    H_COL: ("h", f"horizon: days from the origin t to the target day (1..{HORIZON})"),
    **{c: ("L", f"sales on day t - {j - 1}") for j, c in enumerate(LAG_COLS, 1)},
    "wday": ("E", "weekday of the target day (M5 coding, 1 = Saturday)"),
    "event": ("E", "event on the target day (30 named events, categorical; none = no event)"),
    "event_type": ("E", "event type on the target day: Cultural, National, Religious, Sporting (categorical)"),
    "days_to_event": ("E", "days from the target day to the next event (0 on an event day)"),
    "days_since_event": ("E", "days from the previous event to the target day (0 on an event day)"),
    "snap": ("E", "SNAP day in the store's state on the target day (0/1)"),
    "rel_price": ("E", "price in the target week / the series' median price in the training weeks"),
    "price_change": ("E", "price in the target week / price in the previous week - 1"),
    "discount": ("E", "1 - price in the target week / highest price of the last 52 weeks"),
    **{f"mean_{w}": ("S", f"mean daily sales over the last {w} days") for w in ROLL_WINDOWS},
    "zero_share_28": ("S", "share of days without sales over the last 28 days"),
    "zero_share_182": ("S", "share of days without sales over the last 182 days"),
    "nz_mean_56": ("S", "mean sales on days with sales over the last 56 days"),
    "days_since_sale": ("S", "days since the last day with sales (0 = sales on day t)"),
    "age": ("S", "days since the first sale of the series (1 = first sale on day t)"),
    "item_id": ("ID", "item (3,049, categorical)"),
    "dept_id": ("ID", "department (7, categorical)"),
    "cat_id": ("ID", "category (3, categorical)"),
    "store_id": ("ID", "store (10, categorical)"),
    "state_id": ("ID", "state (3, categorical)"),
}


def feature_table() -> pd.DataFrame:
    return pd.DataFrame([(c, g, d) for c, (g, d) in FEATURE_INFO.items()], columns=["feature", "group", "description"])


# --------------------------------------------------------------------------- panel


@dataclass
class Panel:
    ids: pd.Series  # series id, in the order of the M5 sales file
    hier: pd.DataFrame  # ID_COLS as categoricals
    y: np.ndarray  # (n, VALID_END + 1) float32 sales; NaN before the first sale and on Christmas
    first_sale: np.ndarray  # (n,) day of the first sale in the training window
    cal: pd.DataFrame  # per-day calendar inputs, index 0..TEST_END (row 0 unused)
    snap: np.ndarray  # (3, TEST_END + 1) SNAP flags of CA, TX, WI
    state: np.ndarray  # (n,) row of `snap` for each series
    week: np.ndarray  # (TEST_END + 1,) week index of each day
    price: dict  # name -> (n, n_weeks) float32: rel_price, price_change, discount

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "arrays.npz", y=self.y, first_sale=self.first_sale, snap=self.snap, state=self.state,
                 week=self.week, **{f"price_{k}": v for k, v in self.price.items()})
        pd.concat([self.ids, self.hier], axis=1).to_parquet(path / "series.parquet")
        self.cal.to_parquet(path / "calendar.parquet")

    @classmethod
    def load(cls, path: Path) -> "Panel":
        a = np.load(path / "arrays.npz")
        series = pd.read_parquet(path / "series.parquet")
        return cls(ids=series["id"], hier=series[ID_COLS], y=a["y"], first_sale=a["first_sale"],
                   cal=pd.read_parquet(path / "calendar.parquet"), snap=a["snap"], state=a["state"],
                   week=a["week"], price={k[6:]: a[k] for k in a.files if k.startswith("price_")})


def _calendar_inputs(calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar.set_index("d_num").sort_index()
    days = cal.index.to_numpy()
    event_day = cal["event_name_1"].notna().to_numpy()
    ev_days = days[event_day]
    nxt = np.searchsorted(ev_days, days, side="left")  # first event day >= d
    prv = np.searchsorted(ev_days, days, side="right") - 1  # last event day <= d
    to_event = np.where(nxt < ev_days.size, ev_days[np.minimum(nxt, ev_days.size - 1)] - days, np.nan)
    since_event = np.where(prv >= 0, days - ev_days[np.maximum(prv, 0)], np.nan)
    out = pd.DataFrame(
        {
            "wday": cal["wday"].to_numpy(np.int8),
            "event": cal["event_name_1"].cat.add_categories("none").fillna("none").to_numpy(),
            "event_type": cal["event_type_1"].cat.add_categories("none").fillna("none").to_numpy(),
            "days_to_event": to_event.astype(np.float32),
            "days_since_event": since_event.astype(np.float32),
        },
        index=days,
    )
    for c in ["event", "event_type"]:
        out[c] = pd.Categorical(out[c], categories=["none", *sorted(set(out[c]) - {"none"})])
    out = out.reindex(range(0, m5.TEST_END + 1))
    out["wday"] = out["wday"].astype(np.float32)
    return out


def _price_inputs(sales: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame) -> tuple[np.ndarray, dict]:
    weeks = np.sort(prices["wm_yr_wk"].unique())
    week_of_day = np.zeros(m5.TEST_END + 1, dtype=np.int16)
    week_of_day[calendar["d_num"].to_numpy()] = np.searchsorted(weeks, calendar["wm_yr_wk"].to_numpy())

    row = pd.Series(np.arange(len(sales)), index=pd.MultiIndex.from_arrays(
        [sales["store_id"].astype(str), sales["item_id"].astype(str)]))
    r = row.reindex(pd.MultiIndex.from_arrays([prices["store_id"].astype(str), prices["item_id"].astype(str)])).to_numpy()
    p = np.full((len(sales), weeks.size), np.nan, dtype=np.float32)
    p[r, np.searchsorted(weeks, prices["wm_yr_wk"].to_numpy())] = prices["sell_price"].to_numpy()

    train_weeks = week_of_day[m5.TRAIN_END] + 1  # weeks up to and including the week of d_1913
    median = np.nanmedian(p[:, :train_weeks], axis=1, keepdims=True)
    prev = np.concatenate([np.full((len(sales), 1), np.nan, np.float32), p[:, :-1]], axis=1)
    regular = pd.DataFrame(p.T).rolling(DISCOUNT_WEEKS, min_periods=1).max().to_numpy().T
    price = {
        "rel_price": p / median,
        "price_change": p / prev - 1,
        "discount": 1 - p / regular,
    }
    return week_of_day, {k: v.astype(np.float32) for k, v in price.items()}


def build_panel() -> Panel:
    """All inputs from the raw M5 files: sales up to d_1941 (the test window stays sealed)."""
    sales = m5.load_sales(last_day=m5.VALID_END)
    calendar = m5.load_calendar()
    prices = m5.load_prices()

    y = np.full((len(sales), m5.VALID_END + 1), np.nan, dtype=np.float32)
    y[:, 1:] = sales[m5.day_cols(1, m5.VALID_END)].to_numpy(np.float32)
    train = y[:, 1 : m5.TRAIN_END + 1]
    first_sale = (np.argmax(train > 0, axis=1) + 1).astype(np.int16)
    assert (train > 0).any(axis=1).all(), "every series sells in the training window"
    y[np.arange(y.shape[1])[None, :] < first_sale[:, None]] = np.nan
    dates = calendar.set_index("d_num")["date"]
    christmas = [d for d in range(1, m5.VALID_END + 1) if dates[d].month == 12 and dates[d].day == 25]
    y[:, christmas] = np.nan

    snap = np.zeros((3, m5.TEST_END + 1), dtype=np.int8)
    for k, s in enumerate(["CA", "TX", "WI"]):
        snap[k, calendar["d_num"].to_numpy()] = calendar[f"snap_{s}"].to_numpy()
    state = sales["state_id"].astype(str).map({"CA": 0, "TX": 1, "WI": 2}).to_numpy(np.int8)

    week, price = _price_inputs(sales, prices, calendar)
    hier = pd.DataFrame({c: pd.Categorical(sales[c].astype(str)) for c in ID_COLS})
    return Panel(ids=sales["id"].reset_index(drop=True), hier=hier, y=y, first_sale=first_sale,
                 cal=_calendar_inputs(calendar), snap=snap, state=state, week=week, price=price)


# --------------------------------------------------------------------------- origins


@dataclass
class OriginTable:
    """Inputs known at the forecast origin (groups L and S) for a set of (series, origin) pairs."""

    series: np.ndarray  # (m,) int32 row in the panel
    t: np.ndarray  # (m,) int16 origin day
    lags: np.ndarray  # (m, N_LAGS) float32
    s: pd.DataFrame  # (m, len(S_COLS)) float32

    def __len__(self) -> int:
        return self.t.size


def training_origins(panel: Panel, stage: str, stride: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Origins t = T - stride * k (k >= 1) from each series' first sale, T = the stage's forecast origin.

    With stride 7 every origin falls on the weekday of T (a Sunday in both stages), so that each horizon
    has the same target weekday in training and in the forecast.
    """
    T = STAGES[stage]
    t_all = np.arange(T - stride, 0, -stride)[::-1].astype(np.int16)
    keep = t_all[None, :] >= panel.first_sale[:, None]
    series, k = np.nonzero(keep)
    return series.astype(np.int32), t_all[k]


def forecast_origins(panel: Panel, stage: str) -> tuple[np.ndarray, np.ndarray]:
    """The EVAL_WEEKS weekly origins of the stage's window for every series, origin by origin.

    The later origins need the sales observed up to them: for the test stage the panel must hold the test
    sales, which are loaded only in the final evaluation.
    """
    n = len(panel.ids)
    origins = STAGES[stage] + HORIZON * np.arange(m5.EVAL_WEEKS)
    assert origins[-1] < panel.y.shape[1], f"the panel has no sales up to the origin d_{origins[-1]}"
    return np.tile(np.arange(n, dtype=np.int32), origins.size), np.repeat(origins, n).astype(np.int16)


def origin_table(panel: Panel, series: np.ndarray, t: np.ndarray) -> OriginTable:
    y = panel.y
    lags = np.empty((t.size, N_LAGS), dtype=np.float32)
    for j in range(N_LAGS):
        d = t.astype(np.int64) - j
        lags[:, j] = np.where(d >= 1, y[series, np.maximum(d, 0)], np.nan)

    valid = ~np.isnan(y)
    sale = np.nan_to_num(y) > 0
    cs = np.cumsum(np.nan_to_num(y), axis=1, dtype=np.float64)
    cnt = np.cumsum(valid, axis=1, dtype=np.int32)
    nz = np.cumsum(sale, axis=1, dtype=np.int32)

    def window(c: np.ndarray, w: int) -> np.ndarray:
        return (c[series, t] - c[series, np.maximum(t.astype(np.int64) - w, 0)]).astype(np.float64)

    s = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for w in ROLL_WINDOWS:
            s[f"mean_{w}"] = window(cs, w) / window(cnt, w)
        for w in (28, 182):
            s[f"zero_share_{w}"] = 1 - window(nz, w) / window(cnt, w)
        s["nz_mean_56"] = window(cs, 56) / window(nz, 56)
    day = np.arange(y.shape[1], dtype=np.int16)
    last_sale = np.maximum.accumulate(np.where(sale, day[None, :], 0), axis=1)
    s["days_since_sale"] = t - last_sale[series, t]
    s["age"] = t - panel.first_sale[series] + 1
    return OriginTable(series=series, t=t, lags=lags, s=pd.DataFrame(s).astype(np.float32))


# --------------------------------------------------------------------------- design matrix


def horizon_rows(n: int, per_origin: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (origin row, horizon) for `n` origins: all HORIZON horizons of each origin, or `per_origin`
    of them drawn at random without replacement (to limit the number of training rows)."""
    if per_origin is None or per_origin >= HORIZON:
        return np.repeat(np.arange(n), HORIZON), np.tile(np.arange(1, HORIZON + 1, dtype=np.int16), n)
    rng = np.random.default_rng(seed)
    h = np.argsort(rng.random((n, HORIZON), dtype=np.float32), axis=1)[:, :per_origin] + 1
    return np.repeat(np.arange(n), per_origin), np.sort(h, axis=1).ravel().astype(np.int16)


def design(panel: Panel, table: OriginTable, h, groups=("L", "E", "S", "ID"), target: bool = True,
           stage: str | None = None, rows: np.ndarray | None = None):
    """Inputs (and target) for origin rows `rows` of `table` (default: all) at horizon(s) `h`.

    `h` is one horizon or an array aligned with `rows`, as returned by `horizon_rows`. For training
    (`target=True`) only rows whose target day t + h is observed in the stage (t + h <= T) and is not
    Christmas are kept. Returns X, y (None for a forecast) and the mask of the kept rows.
    """
    if rows is None:
        rows = np.arange(len(table))
    h = np.broadcast_to(np.asarray(h, dtype=np.int64), rows.shape)
    d = table.t[rows].astype(np.int64) + h
    if target:
        T = STAGES[stage]
        keep = d <= T
        y = np.where(keep, panel.y[table.series[rows], np.minimum(d, panel.y.shape[1] - 1)], np.nan)
        keep &= ~np.isnan(y)
    else:
        keep = np.ones(rows.size, dtype=bool)
    rows, d = rows[keep], d[keep]
    i = table.series[rows]

    parts = [pd.DataFrame({H_COL: h[keep].astype(np.float32)})]
    if "L" in groups:
        parts.append(pd.DataFrame(table.lags[rows], columns=LAG_COLS))
    if "E" in groups:
        cal = panel.cal.iloc[d].reset_index(drop=True)
        w = panel.week[d]
        e = cal.assign(snap=panel.snap[panel.state[i], d].astype(np.float32),
                       **{k: v[i, w] for k, v in panel.price.items()})
        parts.append(e[E_COLS])
    if "S" in groups:
        parts.append(table.s.iloc[rows].reset_index(drop=True))
    if "ID" in groups:
        parts.append(panel.hier.iloc[i].reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    return X, (y[keep].astype(np.float32) if target else None), keep
