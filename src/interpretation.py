"""Shapley values of the global XGBoost models on the test-window forecasts (Section 7.8).

The values are computed by XGBoost's own TreeSHAP (`pred_contribs=True`), which is exact for tree ensembles.
The models use the Tweedie loss with a log link, so the contributions are on the log scale: the base value
plus the contributions of a row add up to the logarithm of its forecast, and exp(contribution) is the factor
by which an input multiplies the forecast.

A random sample of series is explained (all their 28 forecast rows), for the best variant (L+E+S+ID) and for
the same model without the identifiers (L+E+S).

Run:  python src/interpretation.py      (CPU, about 10 minutes; writes data/processed/shap/)
"""
import time

import numpy as np
import xgboost as xgb

import evaluation as E
import global_xgb as G
import m5_data as m5

N_SERIES = 4000
SEED = 0
MODELS = {"LESID_tweedie": "final", "LES_tweedie": "tweedie"}  # variant -> folder of its kernel output
OUT = m5.PROCESSED_DIR / "shap"


def run() -> None:
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = E.test_panel()
    rows = G.ForecastRows(panel, "test")
    series = np.sort(np.random.default_rng(SEED).choice(len(panel.ids), N_SERIES, replace=False))
    sel = np.isin(rows.series, series)
    X = rows.X[sel].reset_index(drop=True)
    X.to_parquet(OUT / "inputs.parquet")
    np.save(OUT / "series.npy", rows.series[sel])
    for name, folder in MODELS.items():
        cols = list(rows.inputs(G.INPUT_SETS[name.split("_")[0]]).columns)
        bst = E._booster(G_RUNS / folder / name / "model_test.ubj")
        d = xgb.DMatrix(X[cols], enable_categorical=True)
        contrib = bst.predict(d, pred_contribs=True)
        margin = bst.predict(d, output_margin=True)
        assert np.abs(contrib.sum(axis=1) - margin).max() < 1e-3, "the contributions add up to the log forecast"
        np.savez(OUT / f"{name}.npz", contrib=contrib[:, :-1].astype(np.float32), base=contrib[:, -1].astype(np.float32),
                 columns=np.array(cols))
        print(name, contrib.shape, round((time.perf_counter() - start) / 60, 1), "min", flush=True)


G_RUNS = m5.PROCESSED_DIR / "global_xgb"

if __name__ == "__main__":
    run()
