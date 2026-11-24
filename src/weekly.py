"""Inputs of the weekly forecasting models (additional experiment: the same series summed to weeks).

Weeks run from Monday to Sunday (ISO weeks), so that every forecast origin of the daily models, a Sunday,
closes a week and the 28-day validation and test windows are exactly 4 weeks. Week w covers the days
7w - 4 .. 7w + 2: week 1 is 31.01.-06.02.2011 (the first two days, 29.-30.01.2011, are left out), week 273
ends on 24.04.2016 (last training day), week 277 on 22.05.2016 and week 281 on 19.06.2016.

A training example is a triple (series i, origin week t, horizon h): the model sees the weekly sales up to
week t and predicts the sales of week t + h. As in features.py the model gets h as an input. The main
design forecasts one week ahead (h = 1) from each of the 4 Sundays of a window, the same origins and
information as the daily models, so that the weekly model can be compared with the weekly sums of the
daily forecasts; with `horizon` > 1 the model forecasts weeks 1..horizon from the first origin only.

The model sees only weekly sums of the sales, never the daily sales. Besides h, the inputs form the groups
of features.py:

- L, past sales of the series: sales in weeks t, t-1, ..., t-12, and in the target week one year
  (52 weeks) earlier;
- E, known in advance for the target week: week of the year and month, number of days with an event, the
  first event of the week and its type, days from the end of the week to the next event and from the
  previous event to the start of the week, number of SNAP days in the store's state, relative price,
  price change and discount (price = mean daily price over the 7 days);
- S, from the weekly sales up to week t: mean weekly sales over the last 4, 13, 26 and 52 weeks, share of
  weeks without sales over the last 13 and 52 weeks, mean of the weeks with sales over the last 13 weeks,
  weeks since the last week with sales and the age of the series in weeks;
- ID, hierarchy identifiers (as in features.py);
- H, mean weekly sales of the item in all stores, of the department in the store and of the whole store
  over the last 1 and 4 weeks, and the ratio of the two.

A week is missing (NaN) if any of its days falls before the series' first sale, so that a partial first
week is not mistaken for low demand. Christmas Day (stores closed) adds nothing to its week, whose event is
Christmas.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import features as F
import m5_data as m5
import metrics as M

FIRST_DAY = 3  # Monday 31.01.2011, first day of week 1


def week_of(day: int) -> int:
    """Week of a day (day >= FIRST_DAY)."""
    return (day - FIRST_DAY) // 7 + 1


def week_days(w: int) -> tuple[int, int]:
    """First and last day (Monday, Sunday) of week w."""
    return 7 * w - 4, 7 * w + 2


N_WEEKS = week_of(m5.TEST_END)  # 281
STAGES = {stage: week_of(T) for stage, T in F.STAGES.items()}  # valid: 273, test: 277
EVAL_WEEKS = m5.EVAL_WEEKS


def add_stage(week: int) -> str:
    """Register an earlier 4-week window whose forecast origin is the Sunday of `week` (backtests)."""
    stage = f"w{week}"
    STAGES[stage], F.STAGES[stage] = week, week_days(week)[1]
    return stage
assert all(week_days(w)[1] == F.STAGES[s] for s, w in STAGES.items()), "the origins are Sundays"

N_LAGS = 13
YEAR = 52
ROLL_WINDOWS = (4, 13, 26, 52)
DISCOUNT_WEEKS = F.DISCOUNT_WEEKS

LAG_COLS = [f"lag_{j}" for j in range(1, N_LAGS + 1)] + ["lag_year"]
E_COLS = ["week_of_year", "month", "n_event_days", "event", "event_type", "days_to_event", "days_since_event",
          "snap_days", "rel_price", "price_change", "discount"]
S_COLS = [f"mean_{w}" for w in ROLL_WINDOWS] + ["zero_share_13", "zero_share_52", "nz_mean_13", "weeks_since_sale",
                                                "age"]
ID_COLS = F.ID_COLS
HIER_LEVELS = F.HIER_LEVELS
HIER_WINDOWS = (1, 4)
HIER_COLS = [c for lvl in HIER_LEVELS for c in (*(f"{lvl}_mean_{w}" for w in HIER_WINDOWS), f"{lvl}_trend")]
GROUPS = {"L": LAG_COLS, "E": E_COLS, "S": S_COLS, "ID": ID_COLS, "H": HIER_COLS}
H_COL = F.H_COL

FEATURE_INFO = {
    H_COL: ("h", "horizon: weeks from the origin week t to the target week"),
    **{f"lag_{j}": ("L", f"sales in week t - {j - 1}") for j in range(1, N_LAGS + 1)},
    "lag_year": ("L", f"sales in the target week {YEAR} weeks earlier (week t + h - {YEAR})"),
    "week_of_year": ("E", "ISO week of the year of the target week (1..53)"),
    "month": ("E", "month of the Thursday of the target week"),
    "n_event_days": ("E", "days of the target week with an event (0..7)"),
    "event": ("E", "first event of the target week (30 named events, categorical; none = no event)"),
    "event_type": ("E", "type of that event: Cultural, National, Religious, Sporting (categorical)"),
    "days_to_event": ("E", "days from the Sunday of the target week to the next event after it"),
    "days_since_event": ("E", "days from the last event before the target week to its Monday"),
    "snap_days": ("E", "SNAP days of the target week in the store's state (0..7)"),
    "rel_price": ("E", "price in the target week / the series' median price in the training weeks"),
    "price_change": ("E", "price in the target week / price in the previous week - 1"),
    "discount": ("E", "1 - price in the target week / highest price of the last 52 weeks"),
    **{f"mean_{w}": ("S", f"mean weekly sales over the last {w} weeks") for w in ROLL_WINDOWS},
    "zero_share_13": ("S", "share of weeks without sales over the last 13 weeks"),
    "zero_share_52": ("S", "share of weeks without sales over the last 52 weeks"),
    "nz_mean_13": ("S", "mean sales of the weeks with sales over the last 13 weeks"),
    "weeks_since_sale": ("S", "weeks since the last week with sales (0 = sales in week t)"),
    "age": ("S", "weeks since the first full week of the series (1 = week t)"),
    **{c: F.FEATURE_INFO[c] for c in ID_COLS},
    **{f"{lvl}_mean_{w}": ("H", f"mean weekly sales of {what} over the last {w} week{'s' if w > 1 else ''}")
       for lvl, what in [("item_all", "the item in all stores"), ("dept_store", "the department in the store"),
                         ("store", "the whole store")] for w in HIER_WINDOWS},
    **{f"{lvl}_trend": ("H", f"{lvl}_mean_{HIER_WINDOWS[0]} / {lvl}_mean_{HIER_WINDOWS[1]}") for lvl in HIER_LEVELS},
}
FEATURE_INFO = {c: FEATURE_INFO[c] for c in [H_COL, *(c for cols in GROUPS.values() for c in cols)]}


def columns(groups) -> list[str]:
    return [H_COL, *(c for g, cols in GROUPS.items() if g in groups for c in cols)]


def feature_table() -> pd.DataFrame:
    return pd.DataFrame([(c, g, d) for c, (g, d) in FEATURE_INFO.items()], columns=["feature", "group", "description"])


def to_weeks(a: np.ndarray) -> np.ndarray:
    """(n, days) daily values whose first column is a Monday -> (n, weeks) weekly sums."""
    assert a.shape[1] % 7 == 0
    return a.reshape(a.shape[0], -1, 7).sum(axis=2)


# --------------------------------------------------------------------------- panel


@dataclass
class Panel:
    ids: pd.Series
    hier: pd.DataFrame  # ID_COLS as categoricals
    y: np.ndarray  # (n, last week + 1) float32 weekly sales; NaN before the first full week; column 0 unused
    first_week: np.ndarray  # (n,) first full week after the first sale
    cal: pd.DataFrame  # per-week calendar inputs, index 0..N_WEEKS
    snap: np.ndarray  # (3, N_WEEKS + 1) SNAP days per week in CA, TX, WI
    state: np.ndarray  # (n,) row of `snap`
    price: dict  # name -> (n, N_WEEKS + 1) float32: rel_price, price_change, discount
    _agg: dict | None = field(default=None, init=False, repr=False)

    def subset(self, rows: np.ndarray) -> "Panel":
        """The series in `rows` only; the sales of the higher levels stay those of all the series."""
        sub = Panel(ids=self.ids.iloc[rows].reset_index(drop=True), hier=self.hier.iloc[rows].reset_index(drop=True),
                    y=self.y[rows], first_week=self.first_week[rows], cal=self.cal, snap=self.snap,
                    state=self.state[rows], price={k: v[rows] for k, v in self.price.items()})
        sub._agg = {lvl: (g[rows], cs, cnt) for lvl, (g, cs, cnt) in self.hier_sales().items()}
        return sub

    def hier_sales(self) -> dict:
        """Per level of HIER_LEVELS: group of each series, cumulative weekly sales of each group and the
        cumulative number of weeks in which at least one of its series is observed."""
        if self._agg is None:
            filled, observed = np.nan_to_num(self.y).astype(np.float64), ~np.isnan(self.y)
            self._agg = {}
            for lvl, cols in HIER_LEVELS.items():
                g = self.hier.groupby(cols, observed=True).ngroup().to_numpy()
                total = np.zeros((g.max() + 1, self.y.shape[1]))
                seen = np.zeros(total.shape, dtype=bool)
                np.add.at(total, g, filled)
                np.logical_or.at(seen, g, observed)
                self._agg[lvl] = (g, np.cumsum(total, axis=1), np.cumsum(seen, axis=1, dtype=np.int32))
        return self._agg


def _calendar_inputs(calendar: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cal = calendar.set_index("d_num").sort_index()
    days = cal.index.to_numpy()
    sel = days >= FIRST_DAY
    days = days[sel]
    event_day = cal["event_name_1"].notna().to_numpy()[sel]
    ev_days = days[event_day]
    weeks = np.arange(1, N_WEEKS + 1)
    start, end = 7 * weeks - 4, 7 * weeks + 2
    nxt = np.searchsorted(ev_days, end, side="right")  # first event day after the Sunday
    prv = np.searchsorted(ev_days, start, side="left") - 1  # last event day before the Monday
    to_event = np.where(nxt < ev_days.size, ev_days[np.minimum(nxt, ev_days.size - 1)] - end, np.nan)
    since_event = np.where(prv >= 0, start - ev_days[np.maximum(prv, 0)], np.nan)

    names = cal["event_name_1"].astype(object).to_numpy()[sel].reshape(-1, 7)
    types = cal["event_type_1"].astype(object).to_numpy()[sel].reshape(-1, 7)
    has = pd.notna(names)
    first = np.where(has.any(axis=1), has.argmax(axis=1), 0)
    row = np.arange(len(weeks))
    event = np.where(has.any(axis=1), names[row, first], "none")
    event_type = np.where(has.any(axis=1), types[row, first], "none")
    thursday = cal["date"].to_numpy()[sel].reshape(-1, 7)[:, 3]
    iso = pd.DatetimeIndex(thursday).isocalendar()
    out = pd.DataFrame({
        "week_of_year": iso["week"].to_numpy(np.float32),
        "month": pd.DatetimeIndex(thursday).month.to_numpy().astype(np.float32),
        "n_event_days": has.sum(axis=1).astype(np.float32),
        "event": event,
        "event_type": event_type,
        "days_to_event": to_event.astype(np.float32),
        "days_since_event": since_event.astype(np.float32),
    }, index=weeks)
    for c in ["event", "event_type"]:
        out[c] = pd.Categorical(out[c], categories=["none", *sorted(set(out[c]) - {"none"})])
    snap = np.zeros((3, N_WEEKS + 1), dtype=np.float32)
    for k, s in enumerate(["CA", "TX", "WI"]):
        snap[k, 1:] = cal[f"snap_{s}"].to_numpy()[sel].reshape(-1, 7).sum(axis=1)
    return out.reindex(range(0, N_WEEKS + 1)), snap


def _price_inputs(sales_hier: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    """Mean daily price of each Monday-Sunday week (Walmart's price weeks run from Saturday to Friday)."""
    wm = np.sort(prices["wm_yr_wk"].unique())
    # row of each (store, item) pair, looked up by the category codes of the price file (light on memory)
    stores, items = prices["store_id"].cat.categories, prices["item_id"].cat.categories
    lookup = np.full((stores.size, items.size), -1, dtype=np.int64)
    lookup[stores.get_indexer(sales_hier["store_id"].astype(str)), items.get_indexer(sales_hier["item_id"].astype(str))] =         np.arange(len(sales_hier))
    r = lookup[prices["store_id"].cat.codes.to_numpy(), prices["item_id"].cat.codes.to_numpy()]
    p = np.full((len(sales_hier), wm.size), np.nan, dtype=np.float32)
    p[r[r >= 0], np.searchsorted(wm, prices["wm_yr_wk"].to_numpy()[r >= 0])] = prices["sell_price"].to_numpy()[r >= 0]

    cal = calendar.set_index("d_num").sort_index()
    wm_of_day = np.searchsorted(wm, cal.loc[FIRST_DAY:m5.TEST_END, "wm_yr_wk"].to_numpy())
    pw = np.full((len(sales_hier), N_WEEKS + 1), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # weeks before the item is on sale: all days NaN
        for w in range(1, N_WEEKS + 1):
            idx = wm_of_day[7 * (w - 1): 7 * w]
            pw[:, w] = np.nanmean(p[:, idx], axis=1)
        median = np.nanmedian(pw[:, 1: STAGES["valid"] + 1], axis=1, keepdims=True)
    prev = np.concatenate([np.full((len(pw), 1), np.nan, np.float32), pw[:, :-1]], axis=1)
    regular = pd.DataFrame(pw.T).rolling(DISCOUNT_WEEKS, min_periods=1).max().to_numpy().T
    price = {"rel_price": pw / median, "price_change": pw / prev - 1, "discount": 1 - pw / regular}
    return {k: v.astype(np.float32) for k, v in price.items()}


def build_panel(daily: F.Panel | None = None) -> Panel:
    """Weekly panel from the daily one (features.build_panel; evaluation.test_panel for the test stage)."""
    daily = F.build_panel() if daily is None else daily
    calendar = m5.load_calendar()
    last_week = week_of(daily.y.shape[1] - 1)
    assert week_days(last_week)[1] == daily.y.shape[1] - 1, "the daily sales end on a Sunday"
    d = daily.y[:, FIRST_DAY:]
    first_week = np.ceil((daily.first_sale.astype(np.int64) + 4) / 7).astype(np.int16)
    y = np.full((len(daily.ids), last_week + 1), np.nan, dtype=np.float32)
    y[:, 1:] = to_weeks(np.nan_to_num(d))
    y[np.arange(y.shape[1])[None, :] < first_week[:, None]] = np.nan
    cal, snap = _calendar_inputs(calendar)
    price = _price_inputs(daily.hier, m5.load_prices(), calendar)
    return Panel(ids=daily.ids, hier=daily.hier, y=y, first_week=first_week, cal=cal, snap=snap, state=daily.state,
                 price=price)


# --------------------------------------------------------------------------- origins and design matrix


@dataclass
class OriginTable:
    series: np.ndarray  # (m,) int32
    t: np.ndarray  # (m,) int16 origin week
    lags: np.ndarray  # (m, N_LAGS) float32
    s: pd.DataFrame

    def __len__(self) -> int:
        return self.t.size


def training_origins(panel: Panel, stage: str) -> tuple[np.ndarray, np.ndarray]:
    """Every week t < T from the series' first full week, T = the stage's origin week."""
    T = STAGES[stage]
    t_all = np.arange(1, T, dtype=np.int16)
    keep = t_all[None, :] >= panel.first_week[:, None]
    series, k = np.nonzero(keep)
    return series.astype(np.int32), t_all[k]


def forecast_origins(panel: Panel, stage: str, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """horizon 1: the EVAL_WEEKS origins T, T+1, ... of the window; else only T. Origin by origin."""
    n = len(panel.ids)
    origins = STAGES[stage] + np.arange(EVAL_WEEKS if horizon == 1 else 1)
    assert origins[-1] < panel.y.shape[1], f"the panel has no sales up to week {origins[-1]}"
    return np.tile(np.arange(n, dtype=np.int32), origins.size), np.repeat(origins, n).astype(np.int16)


def origin_table(panel: Panel, series: np.ndarray, t: np.ndarray) -> OriginTable:
    y = panel.y
    lags = np.empty((t.size, N_LAGS), dtype=np.float32)
    for j in range(N_LAGS):
        w = t.astype(np.int64) - j
        lags[:, j] = np.where(w >= 1, y[series, np.maximum(w, 0)], np.nan)

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
        for w in (13, 52):
            s[f"zero_share_{w}"] = 1 - window(nz, w) / window(cnt, w)
        s["nz_mean_13"] = window(cs, 13) / window(nz, 13)
    week = np.arange(y.shape[1], dtype=np.int16)
    last_sale = np.maximum.accumulate(np.where(sale, week[None, :], 0), axis=1)
    s["weeks_since_sale"] = np.where(last_sale[series, t] > 0, t - last_sale[series, t], np.nan)
    s["age"] = t - panel.first_week[series] + 1
    return OriginTable(series=series, t=t, lags=lags, s=pd.DataFrame(s).astype(np.float32))


def horizon_rows(n: int, horizon: int, per_origin: int | None = None, seed: int = 0):
    """Pairs (origin row, horizon) for `n` origins: all horizons 1..horizon, or `per_origin` drawn at random."""
    if per_origin is None or per_origin >= horizon:
        return np.repeat(np.arange(n), horizon), np.tile(np.arange(1, horizon + 1, dtype=np.int16), n)
    rng = np.random.default_rng(seed)
    h = np.argsort(rng.random((n, horizon), dtype=np.float32), axis=1)[:, :per_origin] + 1
    return np.repeat(np.arange(n), per_origin), np.sort(h, axis=1).ravel().astype(np.int16)


def design(panel: Panel, table: OriginTable, h, groups=("L", "E", "S", "ID"), target: bool = True,
           stage: str | None = None, rows: np.ndarray | None = None):
    """Inputs (and target) for origin rows `rows` of `table` at horizon(s) `h`, as features.design.

    For training only rows whose target week t + h lies in the stage (t + h <= T) and is observed are kept.
    """
    if rows is None:
        rows = np.arange(len(table))
    h = np.broadcast_to(np.asarray(h, dtype=np.int64), rows.shape)
    w = table.t[rows].astype(np.int64) + h
    if target:
        keep = w <= STAGES[stage]
        y = np.where(keep, panel.y[table.series[rows], np.minimum(w, panel.y.shape[1] - 1)], np.nan)
        keep &= ~np.isnan(y)
    else:
        keep = np.ones(rows.size, dtype=bool)
    rows, w, hk = rows[keep], w[keep], h[keep]
    i = table.series[rows]

    parts = [pd.DataFrame({H_COL: hk.astype(np.float32)})]
    if "L" in groups:
        ly = w - YEAR
        year = np.where(ly >= 1, panel.y[i, np.clip(ly, 0, panel.y.shape[1] - 1)], np.nan)
        parts.append(pd.DataFrame(np.column_stack([table.lags[rows], year]).astype(np.float32), columns=LAG_COLS))
    if "E" in groups:
        cal = panel.cal.iloc[w].reset_index(drop=True)
        e = cal.assign(snap_days=panel.snap[panel.state[i], w], **{k: v[i, w] for k, v in panel.price.items()})
        parts.append(e[E_COLS])
    if "S" in groups:
        parts.append(table.s.iloc[rows].reset_index(drop=True))
    if "ID" in groups:
        parts.append(panel.hier.iloc[i].reset_index(drop=True))
    if "H" in groups:
        t = table.t[rows].astype(np.int64)
        hs = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for lvl, (g, cs, cnt) in panel.hier_sales().items():
                gi = g[i]
                for k in HIER_WINDOWS:
                    lo = np.maximum(t - k, 0)
                    hs[f"{lvl}_mean_{k}"] = (cs[gi, t] - cs[gi, lo]) / (cnt[gi, t] - cnt[gi, lo])
                hs[f"{lvl}_trend"] = hs[f"{lvl}_mean_{HIER_WINDOWS[0]}"] / hs[f"{lvl}_mean_{HIER_WINDOWS[1]}"]
        parts.append(pd.DataFrame(hs)[HIER_COLS].astype(np.float32))
    X = pd.concat(parts, axis=1)
    return X, (y[keep].astype(np.float32) if target else None), keep


# --------------------------------------------------------------------------- evaluation on weekly sums


def evaluator(stage: str, actual: np.ndarray | None = None, rows: np.ndarray | None = None) -> M.Evaluator:
    """WRMSSE on weekly sums: scales from the weekly sums of the raw sales history up to the stage's origin
    (Christmas as recorded, as in metrics.py), the dollar-sales weights of the daily evaluation, and the
    actual sales of the 4 weeks of the window. `actual` is (n, 28) daily; for the validation stage it is read
    from the sales file, for the test stage it must be given (the test window is opened only in the final
    evaluation)."""
    T = F.STAGES[stage]
    sales = m5.load_sales(last_day=min(T + 28, m5.VALID_END))
    revenue = M.dollar_sales(sales, T)
    if actual is None:
        assert T + 28 <= m5.VALID_END, "give the actual sales of the test window"
        actual = sales[m5.day_cols(T + 1, T + 28)].to_numpy(np.float64)
    if rows is not None:
        sales, revenue, actual = sales.iloc[rows], revenue[rows], actual[rows]
    hist = sales[m5.day_cols(FIRST_DAY, T)].to_numpy(np.float64)
    return M.Evaluator(sales[m5.ID_COLS].astype(str), to_weeks(hist), revenue, to_weeks(np.asarray(actual, np.float64)))
