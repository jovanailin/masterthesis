"""The three forecasting architectures of the 3 x 2 factorial design.

All models map a window of the last LOOKBACK scaled demand values to the next
HORIZON values in one direct step (no recursion):

- LightGBM with the squared-error objective, one booster per horizon position;
- LSTM whose last hidden state feeds a Linear(hidden, HORIZON) layer;
- Echo State Network: a fixed random leaky reservoir is run over the input
  window and a Ridge readout maps the final reservoir state to the 28 outputs.

The Hyperopt search spaces are architecture-specific but shared by the
per-series and the global regime of the same architecture.
"""
import warnings

import lightgbm as lgb
import numpy as np
import torch

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)  # raised by hyperopt 0.2.7
from hyperopt import hp  # noqa: E402
from hyperopt.pyll import scope  # noqa: E402

ARCHITECTURES = ["lgbm", "lstm", "esn"]


def _log_uniform(label: str, low: float, high: float):
    return hp.loguniform(label, np.log(low), np.log(high))


SEARCH_SPACES = {
    "lgbm": {
        "num_leaves": scope.int(hp.quniform("num_leaves", 4, 64, 4)),
        "learning_rate": _log_uniform("learning_rate", 0.01, 0.3),
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 25)),
        "min_data_in_leaf": scope.int(hp.quniform("min_data_in_leaf", 5, 100, 5)),
        "feature_fraction": hp.uniform("feature_fraction", 0.5, 1.0),
        "bagging_fraction": hp.uniform("bagging_fraction", 0.5, 1.0),
        "lambda_l2": _log_uniform("lambda_l2", 1e-3, 10.0),
    },
    "lstm": {
        "hidden_size": hp.choice("hidden_size", [16, 32, 64, 128]),
        "num_layers": hp.choice("num_layers", [1, 2]),
        "dropout": hp.uniform("dropout", 0.0, 0.4),
        "learning_rate": _log_uniform("learning_rate", 1e-4, 1e-2),
        "batch_size": hp.choice("batch_size", [64, 128, 256, 512]),
        "epochs": scope.int(hp.quniform("epochs", 5, 40, 5)),
    },
    "esn": {
        "n_reservoir": scope.int(hp.quniform("n_reservoir", 50, 500, 25)),
        "spectral_radius": hp.uniform("spectral_radius", 0.1, 1.5),
        "input_scaling": _log_uniform("input_scaling", 0.01, 5.0),
        "leak_rate": hp.uniform("leak_rate", 0.05, 1.0),
        "density": hp.uniform("density", 0.01, 0.5),
        "bias_scaling": hp.uniform("bias_scaling", 0.0, 1.0),
        "ridge_alpha": _log_uniform("ridge_alpha", 1e-6, 1e2),
    },
}


class LGBMForecaster:
    """28 independent LightGBM boosters (squared error), one per horizon position."""

    def __init__(self, params: dict, seed: int, n_threads: int):
        params = dict(params)
        self.n_estimators = int(params.pop("n_estimators"))
        self.params = {
            "objective": "regression",
            "bagging_freq": 1,
            "seed": seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": n_threads,
            "verbosity": -1,
            **params,
        }

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LGBMForecaster":
        Y = np.asfortranarray(Y)  # contiguous label columns
        self.boosters = [
            lgb.train(self.params, lgb.Dataset(X, Y[:, h]), num_boost_round=self.n_estimators)
            for h in range(Y.shape[1])
        ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([b.predict(X, num_threads=self.params["num_threads"]) for b in self.boosters])


class _LSTMNet(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float, horizon: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(self.dropout(out[:, -1]))


class LSTMForecaster:
    """LSTM trained with Adam on the MSE; the last hidden state is mapped to all horizons at once."""

    def __init__(self, params: dict, seed: int, n_threads: int):
        self.params, self.seed, self.n_threads = params, seed, n_threads

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LSTMForecaster":
        p = self.params
        torch.set_num_threads(self.n_threads)
        torch.manual_seed(self.seed)
        self.net = _LSTMNet(int(p["hidden_size"]), int(p["num_layers"]), float(p["dropout"]), Y.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=float(p["learning_rate"]))
        Xt, Yt = torch.from_numpy(np.asarray(X, np.float32)), torch.from_numpy(np.asarray(Y, np.float32))
        gen = torch.Generator().manual_seed(self.seed)
        bs = int(p["batch_size"])
        self.net.train()
        for _ in range(int(p["epochs"])):
            for idx in torch.randperm(len(Xt), generator=gen).split(bs):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(Xt[idx]), Yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        torch.set_num_threads(self.n_threads)
        self.net.eval()
        Xt = torch.from_numpy(np.asarray(X, np.float32))
        return torch.cat([self.net(chunk) for chunk in Xt.split(8192)]).numpy().astype(np.float64)


class ESNForecaster:
    """Leaky echo state network with a Ridge readout on the final reservoir state.

    The reservoir starts from the zero state at the beginning of every input window.
    """

    CHUNK = 32768  # windows processed at once, bounds memory for the pooled training set

    def __init__(self, params: dict, seed: int, n_threads: int = 1):
        p = params
        n = int(p["n_reservoir"])
        rng = np.random.default_rng(seed)
        W = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < float(p["density"]))
        rho = np.abs(np.linalg.eigvals(W)).max()
        W *= float(p["spectral_radius"]) / rho if rho > 0 else 0.0
        self.W_T = W.T.astype(np.float32)
        self.w_in = (rng.uniform(-1, 1, n) * float(p["input_scaling"])).astype(np.float32)
        self.bias = (rng.uniform(-1, 1, n) * float(p["bias_scaling"])).astype(np.float32)
        self.leak = np.float32(p["leak_rate"])
        self.alpha = float(p["ridge_alpha"])

    def _states(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        out = np.empty((len(X), len(self.bias)), np.float32)
        for s in range(0, len(X), self.CHUNK):
            u = X[s : s + self.CHUNK]
            h = np.zeros((len(u), len(self.bias)), np.float32)
            for t in range(u.shape[1]):
                h = (1 - self.leak) * h + self.leak * np.tanh(u[:, t, None] * self.w_in + h @ self.W_T + self.bias)
            out[s : s + self.CHUNK] = h
        return out

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ESNForecaster":
        S = self._states(X).astype(np.float64)
        s_mean, y_mean = S.mean(axis=0), Y.mean(axis=0)
        Sc = S - s_mean
        A = Sc.T @ Sc
        A[np.diag_indices_from(A)] += self.alpha
        self.W_out = np.linalg.solve(A, Sc.T @ (Y - y_mean))
        self.intercept = y_mean - s_mean @ self.W_out
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._states(X).astype(np.float64) @ self.W_out + self.intercept


_MODELS = {"lgbm": LGBMForecaster, "lstm": LSTMForecaster, "esn": ESNForecaster}


def make_forecaster(arch: str, params: dict, seed: int, n_threads: int = 1):
    return _MODELS[arch](params, seed, n_threads)
