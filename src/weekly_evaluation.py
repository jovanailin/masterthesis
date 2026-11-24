"""Evaluation of the weekly experiment: the weekly global XGBoost against the classical methods.

Reads the forecasts of weekly_xgb.py (Kaggle jobs in kaggle_jobs/weekly_xgb/), weekly_classical.py and
weekly_checks.py, computes the test forecasts of the models that have only a saved test model, and writes
the tables read by the notebooks in notebooks/weekly/ to results/weekly/.

Five 4-week windows are scored, all on weekly sums with one-week-ahead forecasts from 4 Sunday origins:
three earlier windows (from 01.02., 29.02. and 28.03.2016), the validation window (25.04.-22.05.2016, on
which the hyperparameters were chosen) and the test window (23.05.-19.06.2016). WRMSSE is that of
weekly.evaluator (7 levels). A series without a weekly scale at an origin (it has just started to sell) is
left out of that window.

Run:  python src/weekly_evaluation.py     (CPU, a few minutes)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

import evaluation as E
import intermittency as I
import m5_data as m5
import weekly as W
import weekly_checks as K
import weekly_classical as C
import weekly_xgb as WX

RESULTS = m5.PROCESSED_DIR.parents[1] / "results" / "weekly"
X = m5.PROCESSED_DIR / "weekly_xgb"
MAIN = "XGBoost L+E+S+ID"
WINDOWS = {"w261": "01.02.-28.02.2016", "w265": "29.02.-27.03.2016", "w269": "28.03.-24.04.2016",
           "valid": "25.04.-22.05.2016", "test": "23.05.-19.06.2016"}
CLASSICAL = {"naive": "Naive", "mean": "Historical mean", "ma_4": "Moving average (4 weeks)", "ses": "SES", "ets": "ETS",
             "arima": "ARIMA"}
TUNED = {MAIN: ("main_mse_cpu", "extra_cpu", "LESID_mse"), "XGBoost L": ("l_mse_cpu", "backtest_l_cpu", "L_mse"),
         "XGBoost L+E": ("le_mse_cpu", None, "LE_mse")}  # name: (tuning job, backtest job, variant)
LEVELS = {1: "total", 2: "state", 3: "store", 4: "category", 8: "store x category", 10: "item", 12: "item x store"}
SEEDS = [1, 2, 3, 4]
TUNING_CORE_MINUTES = 1.69 * 60 * 4  # the tuning kernel of the main model: 1.69 hours on the 4 cores of a Kaggle CPU


def test_forecasts(panel: W.Panel) -> None:
    """Test forecasts of the tuned variants and of the extra seeds, saved next to their models."""
    for job, _, variant in TUNED.values():
        file = X / job / variant / "test_forecast.npy"
        if not file.exists():
            np.save(file, WX.test_forecast(file.parent, panel))
    for s in SEEDS:
        file = X / "extra_cpu" / "LESID_mse" / f"test_forecast_seed{s}.npy"
        if not file.exists():
            np.save(file, WX.test_forecast(file.parent, panel, seed=s))


def forecasts(stage: str, classical: dict) -> dict[str, np.ndarray]:
    """Forecasts (n_series, 4 weeks) of every model of a window."""
    back = stage.startswith("w")
    out = {}
    for name, (job, back_job, variant) in TUNED.items():
        if not back:
            out[name] = np.load(X / job / variant / f"{stage}_forecast.npy")
        elif back_job:
            out[name] = np.load(X / back_job / stage / f"{variant}.npy")
    out["XGBoost L+E+S+ID, default hyperparameters"] = np.load(K.OUT / "default" / f"{stage}.npy")
    daily = m5.PROCESSED_DIR / "global_xgb"
    if stage == "test":
        f = np.load(m5.PROCESSED_DIR / "forecasts" / "test_forecasts.npz")["LESID_tweedie"]
    elif stage == "valid":
        f = np.load(daily / "final" / "LESID_tweedie" / "valid_forecast.npy")
    else:
        f = np.load(daily / "backtest" / f"d{W.week_days(W.STAGES[stage])[1]}" / "LESID_tweedie.npy")
    out["Daily main model, summed to weeks"] = W.to_weeks(f)
    return out | {name: classical[f"{stage}_{m}"] for m, name in CLASSICAL.items()}


def evaluator(stage: str, test_daily: np.ndarray, rows=None):
    ev = W.evaluator(stage, actual=test_daily if stage == "test" else None, rows=rows)
    ok = np.isfinite(ev.scale[12]) & (ev.scale[12] > 0)
    ev.weight[12] = np.where(ok, ev.weight[12], 0.0)
    return ev, ok


def series_groups(y: np.ndarray, first_week: np.ndarray, cat: np.ndarray, T: int) -> pd.DataFrame:
    """Groups of the series at the origin week T: demand class of the weekly history (Syntetos-Boylan), quartile of
    the mean weekly sales of the last year, category and length of the history."""
    stats = I.demand_stats(np.nan_to_num(y[:, 1 : T + 1]))
    volume = pd.Series(np.nanmean(y[:, T - 51 : T + 1], axis=1))
    age = T - first_week + 1
    return pd.DataFrame({
        "demand class": np.asarray(I.classify_sbc(stats["adi"], stats["cv2"]).astype(str)),
        "sales volume": pd.qcut(volume.rank(method="first"), 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype(str),
        "category": cat,
        "history": pd.cut(age, [-1e9, 25, 51, 103, 1e9], labels=["1: under 26 weeks", "2: 26-51 weeks", "3: 52-103 weeks",
                                                               "4: 104 weeks or more"]).astype(str)})


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = W.build_panel(E.test_panel())
    test_forecasts(panel)
    y, first_week, cat = panel.y, panel.first_week.astype(int), panel.hier["cat_id"].astype(str).to_numpy()
    del panel
    test_daily = m5.load_test_sales()[m5.day_cols(m5.VALID_END + 1, m5.TEST_END)].to_numpy(np.float64)
    classical = {**np.load(C.OUT / "forecasts.npz"), **np.load(C.OUT / "forecasts_backtest.npz")}
    for week in C.BACKTEST_WEEKS:
        W.add_stage(week)

    wr, bias, frames, seeds = [], [], [], []
    for stage, window in WINDOWS.items():
        T = W.STAGES[stage]
        ev, ok = evaluator(stage, test_daily)
        F, actual = forecasts(stage, classical), ev.actual[12]
        R = {}
        for name, f in F.items():
            w, b = ev.wrmsse(f), ev.bottom(f)
            wr.append({"window": window, "model": name, "WRMSSE": w["all"], **{LEVELS[k]: w[k] for k in LEVELS}})
            R[name] = b["rmsse"].to_numpy()
            bias.append({"window": window, "model": name, "total forecast vs actual, %": 100 * (f.sum() / actual.sum() - 1),
                         "mean scaled error per series": float(np.nanmean(np.where(ok, b["sme"].to_numpy(), np.nan))),
                         "series over-forecast, %": 100 * float(np.mean(f.sum(1)[ok] > actual.sum(1)[ok]))})
        frame = pd.concat([pd.DataFrame(R), series_groups(y, first_week, cat, T)], axis=1).assign(window=window)
        frames.append(frame[ok & (first_week <= T)])
        if stage in ("valid", "test"):
            runs = {0: F[MAIN]} | {s: np.load(X / "extra_cpu" / "LESID_mse" / f"{stage}_forecast_seed{s}.npy") for s in SEEDS}
            for s, f in runs.items():
                w = ev.wrmsse(f)
                seeds.append({"window": window, "seed": s, "WRMSSE": w["all"], **{LEVELS[k]: w[k] for k in LEVELS},
                              "mean RMSSE per series": float(np.nanmean(np.where(ok, ev.bottom(f)["rmsse"], np.nan)))})
    pd.DataFrame(wr).to_csv(RESULTS / "wrmsse.csv", index=False)
    pd.DataFrame(bias).to_csv(RESULTS / "bias.csv", index=False)
    pd.DataFrame(seeds).to_csv(RESULTS / "seeds.csv", index=False)

    A = pd.concat(frames, ignore_index=True)
    groupings = ["demand class", "sales volume", "category", "history"]
    everywhere = [m for m in A.columns if m not in groupings + ["window"] and A[m].notna().all()]  # in all 5 windows
    others = [m for m in everywhere if m != MAIN]

    # per series: share of series on which the main model is better, Wilcoxon signed-rank test, per window
    tests = []
    for window, a in A.groupby("window", sort=False):
        for m in [c for c in a.columns if c not in groupings + ["window", MAIN] and a[c].notna().all()]:
            tests.append({"window": window, "model": m, "main model better, % of series": 100 * np.mean(a[MAIN] < a[m]),
                          "mean RMSSE": a[m].mean(), "mean RMSSE, main model": a[MAIN].mean(),
                          "Wilcoxon p": wilcoxon(a[MAIN], a[m]).pvalue})
    pd.DataFrame(tests).to_csv(RESULTS / "series_tests.csv", index=False)

    core = [MAIN, "XGBoost L", *CLASSICAL.values()]
    ranks = pd.DataFrame(rankdata(A[core].to_numpy(), axis=1), columns=core)
    table = ranks.groupby(A["window"].to_numpy(), sort=False).mean().T
    table["all windows"] = ranks.mean()
    table.to_csv(RESULTS / "ranks.csv")
    friedman = friedmanchisquare(*[A[m].to_numpy() for m in core])

    rows = []
    for g in groupings:
        for level, a in A[A[g] != "nan"].groupby(g):
            rows.append({"grouping": g, "group": level, "series x windows": len(a), **{m: a[m].mean() for m in [MAIN] + others},
                         **{f"main better than {m}, %": 100 * np.mean(a[MAIN] < a[m]) for m in ["SES", "ETS", "ARIMA"]}})
    pd.DataFrame(rows).to_csv(RESULTS / "groups.csv", index=False)

    # the global model on random samples of the series (weekly_checks.subsets), test and validation windows
    z, recs = np.load(K.OUT / "subsets.npz"), []
    full = {s: np.load(X / "main_mse_cpu" / "LESID_mse" / f"{s}_forecast.npy") for s in ("valid", "test")}
    for key in sorted({k.rsplit("_", 1)[0] for k in z.files}):
        sel, size, sample = z[f"{key}_rows"], int(key.split("_")[0][1:]), int(key.split("_s")[1])
        for stage in ("valid", "test"):
            ev, ok = evaluator(stage, test_daily, rows=sel)
            F = {"XGBoost trained on the sample": z[f"{key}_{stage}"], "XGBoost trained on all series": full[stage][sel]}
            F |= {name: classical[f"{stage}_{m}"][sel] for m, name in CLASSICAL.items()}
            r_sample = ev.bottom(F["XGBoost trained on the sample"])["rmsse"].to_numpy()
            for name, f in F.items():
                r = ev.bottom(f)["rmsse"].to_numpy()
                both = ok & np.isfinite(r) & np.isfinite(r_sample)
                recs.append({"series": size, "sample": sample, "window": WINDOWS[stage], "model": name,
                             "WRMSSE": ev.wrmsse(f)["all"], "mean RMSSE per series": float(r[both].mean()),
                             "sample model better, % of series": 100 * float(np.mean(r_sample[both] < r[both]))})
    pd.DataFrame(recs).to_csv(RESULTS / "subsets.csv", index=False)
    pd.DataFrame(json.loads((K.OUT / "subsets.json").read_text())).to_csv(RESULTS / "subsets_tuning.csv", index=False)

    # computing cost per window: minutes of one core (wall time x cores for XGBoost, an upper bound)
    xt = pd.DataFrame(json.loads((K.OUT / "default_times.json").read_text()))
    xt = xt[xt["stage"] == "test"].set_index("model")
    ct = json.loads((K.OUT / "classical_times.json").read_text())
    cores = K.CORES
    cost = [{"model": name, "core-minutes per window": ct[m]["fit_all_core_minutes"],
             "core-minutes for a new week": ct[m]["forward_all_core_minutes"], "one-off tuning, core-minutes": 0.0}
            for m, name in CLASSICAL.items() if m in ct]
    cost += [{"model": CLASSICAL[m], "core-minutes per window": ct["simple_all_series_seconds"] / 60 / 3,
              "core-minutes for a new week": ct["simple_all_series_seconds"] / 60 / 3, "one-off tuning, core-minutes": 0.0}
             for m in ("naive", "mean", "ma_4")]
    for key, name in [("tuned", MAIN), ("default", "XGBoost L+E+S+ID, default hyperparameters")]:
        t = xt.loc[key]
        cost.append({"model": name, "core-minutes per window": cores * (t["inputs_seconds"] + t["matrix_seconds"] + t["train_seconds"]) / 60,
                     "core-minutes for a new week": cores * (t["inputs_seconds"] + t["predict_seconds"]) / 60,
                     "one-off tuning, core-minutes": TUNING_CORE_MINUTES if key == "tuned" else 0.0,
                     "training seconds (wall)": t["train_seconds"], "memory, GB": t["memory_gb"]})
    pd.DataFrame(cost).to_csv(RESULTS / "cost.csv", index=False)
    (RESULTS / "notes.json").write_text(json.dumps({"friedman_p": float(friedman.pvalue), "series_x_windows": len(A),
                                                    "cores": cores}, indent=1))
    print(pd.DataFrame(wr).pivot(index="model", columns="window", values="WRMSSE").round(3).to_string())


if __name__ == "__main__":
    main()
