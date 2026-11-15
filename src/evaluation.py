"""Final evaluation on the test window d_1942 - d_1969 (Sections 7.1 - 7.9).

This is the only place where the test sales are read. Every model was fixed before: the classical methods
have no tuned parameters, and each XGBoost model is the refit of its best validation trial on targets up
to d_1941 (model_test*.ubj of the Kaggle kernels, models_test.pkl of src/local_xgb.py).

As on the validation window, the 28 days are forecast 7 days at a time from 4 weekly origins (d_1941,
d_1948, d_1955, d_1962) without refitting; the later origins see the sales observed up to them. The scales
and the weights of the measures are those of the first origin (history up to d_1941, revenue of its last
28 days), which are the official ones of the M5 evaluation.

Run:  python src/evaluation.py      (CPU, a few minutes; writes data/processed/forecasts/test_forecasts.npz)
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb

import benchmarks as B
import features as F
import global_xgb as G
import local_xgb as L
import m5_data as m5
import metrics as M

RUNS = m5.PROCESSED_DIR / "global_xgb"
OUT = m5.PROCESSED_DIR / "forecasts" / "test_forecasts.npz"
BACKTEST_ORIGINS = (1885, 1857, 1829)  # earlier 28-day windows: d_1886-1913, d_1858-1885, d_1830-1857


def test_panel() -> F.Panel:
    """The panel of inputs with the test sales appended (days d_1942 - d_1969)."""
    panel = F.build_panel()
    test = m5.load_test_sales()
    assert (test["id"].to_numpy() == panel.ids.to_numpy()).all(), "the test file is in the order of the sales file"
    y = np.full((len(panel.ids), m5.TEST_END + 1), np.nan, dtype=np.float32)
    y[:, : m5.VALID_END + 1] = panel.y
    y[:, m5.VALID_END + 1 :] = test[m5.day_cols(m5.VALID_END + 1, m5.TEST_END)].to_numpy(np.float32)
    panel.y = y
    return panel


def test_evaluator(rows: np.ndarray | None = None) -> M.Evaluator:
    """Scales and weights of the origin d_1941, actual sales of d_1942 - d_1969."""
    sales = m5.load_sales(last_day=m5.VALID_END)
    revenue = M.dollar_sales(sales, m5.VALID_END)
    actual = m5.load_test_sales()[m5.day_cols(m5.VALID_END + 1, m5.TEST_END)].to_numpy(np.float64)
    if rows is not None:
        sales, revenue, actual = sales.iloc[rows], revenue[rows], actual[rows]
    return M.Evaluator(sales[m5.ID_COLS].astype(str), sales[m5.day_cols(1, m5.VALID_END)].to_numpy(np.float64), revenue,
                       actual)


def _booster(path) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(path)
    bst.set_param({"device": "cpu"})
    return bst


def _predict(bst: xgb.Booster, X) -> np.ndarray:
    return bst.predict(xgb.DMatrix(X, enable_categorical=True))


def run() -> None:
    start = time.perf_counter()
    panel = test_panel()
    rows = G.ForecastRows(panel, "test")
    out = {}

    for m, f in B.forecast_rolling(panel.y, m5.VALID_END, m5.EVAL_WEEKS).items():
        out[f"benchmark_{m}"] = f.astype(np.float32)
    print("benchmarks", round(time.perf_counter() - start), "s", flush=True)

    for file in sorted(RUNS.glob("*/*/best.json")):
        best = json.loads(file.read_text())
        name, folder, groups = best["variant"], file.parent, G.INPUT_SETS[best["inputs"]]
        if name.endswith("_recursive"):
            out[name] = G.recursive_forecast(panel, _booster(folder / "model_test.ubj"), groups, "test")
        elif "scheme" in best:
            labels = G.series_groups(panel, best["scheme"])
            pred = np.full(len(rows.series), np.nan, dtype=np.float32)
            for g in best["groups"]:
                sel = (labels == g["group"])[rows.series]
                pred[sel] = _predict(_booster(folder / f"model_test_{g['group']}.ubj"), rows.inputs(groups)[sel])
            assert not np.isnan(pred).any()
            out[name] = rows.matrix(pred)
        else:
            X = rows.inputs(groups)
            out[name] = rows.matrix(_predict(_booster(folder / "model_test.ubj"), X))
            for extra in sorted(folder.glob("model_test_seed*.ubj")):
                out[f"{name}_{extra.stem.replace('model_test_', '')}"] = rows.matrix(_predict(_booster(extra), X))
        print(name, round(time.perf_counter() - start), "s", flush=True)

    # models per series and the model of the sample: forecasts of the series of the sample only
    sample = pd.read_csv(L.RESULTS / "sample.csv")["row"].to_numpy()
    small = panel.subset(sample)
    srows = G.ForecastRows(small, "test")
    with open(L.OUT / "models_test.pkl", "rb") as f:
        models = pickle.load(f)
    for name, model in models.items():
        cols = F.columns(G.INPUT_SETS[name.rsplit("_", 1)[1]])
        if isinstance(model, list):  # one model per series
            pred = np.empty(len(srows.series), dtype=np.float32)
            for i, raw in enumerate(model):
                sel = srows.series == i
                pred[sel] = _predict(xgb.Booster(model_file=bytearray(raw)), srows.X.loc[sel, cols])
        else:
            pred = _predict(xgb.Booster(model_file=bytearray(model)), srows.X[cols])
        out[f"sample_{name}"] = srows.matrix(pred)
    # classical methods on the earlier windows of the robustness check (the XGBoost models of these windows
    # are trained by the Kaggle job "backtest")
    for T in BACKTEST_ORIGINS:
        for m, f in B.forecast_rolling(panel.y, T, m5.EVAL_WEEKS).items():
            out[f"backtest_d{T}_benchmark_{m}"] = f.astype(np.float32)
        print("benchmarks of the window after d_%d" % T, round(time.perf_counter() - start), "s", flush=True)
    np.savez(OUT, **out)
    print(len(out), "forecasts saved,", round((time.perf_counter() - start) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    run()
