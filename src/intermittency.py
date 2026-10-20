"""Demand-pattern statistics and the Syntetos-Boylan-Croston (SBC) classification.

All statistics are computed over each series' active period, i.e. from its first
non-zero sale onwards, so that days before a product was introduced do not count
as zero demand.
"""
import numpy as np
import pandas as pd

# Syntetos, Boylan & Croston (2005) cut-off values.
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49
SBC_CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]


def _longest_zero_run(nonzero: np.ndarray) -> int:
    idx = np.flatnonzero(nonzero)
    if idx.size == 0:
        return 0
    gaps = np.diff(idx) - 1
    trailing = nonzero.size - 1 - idx[-1]
    return int(max(gaps.max(initial=0), trailing))


def demand_stats(y: np.ndarray) -> pd.DataFrame:
    """Per-series statistics for a (n_series, n_days) array of non-negative sales.

    ADI  = active days / days with non-zero demand (average inter-demand interval).
    CV2  = squared coefficient of variation of the non-zero demand sizes
           (population variance).
    """
    y = np.asarray(y, dtype=np.float64)
    n_days = y.shape[1]
    nonzero = y > 0
    has_sale = nonzero.any(axis=1)
    first = np.where(has_sale, nonzero.argmax(axis=1), n_days)
    last = np.where(has_sale, n_days - 1 - nonzero[:, ::-1].argmax(axis=1), -1)

    n_active = n_days - first
    n_nonzero = nonzero.sum(axis=1)
    total = y.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_size = total / n_nonzero
        var_size = np.square(y).sum(axis=1) / n_nonzero - mean_size**2
        cv2 = np.clip(var_size, 0, None) / mean_size**2
        adi = n_active / n_nonzero
        zero_share = 1 - n_nonzero / n_active

    return pd.DataFrame(
        {
            "first_sale": np.where(has_sale, first + 1, np.nan),  # 1-based day index
            "n_active": n_active,
            "n_nonzero": n_nonzero,
            "zero_share": zero_share,
            "adi": adi,
            "cv2": cv2,
            "mean_size": mean_size,
            "longest_zero_run": [_longest_zero_run(row) for row in nonzero],
            "trailing_zeros": np.where(has_sale, n_days - 1 - last, n_days),
        }
    )


def classify_sbc(adi, cv2) -> pd.Categorical:
    """Smooth / erratic / intermittent / lumpy, following Syntetos et al. (2005)."""
    adi, cv2 = np.asarray(adi), np.asarray(cv2)
    conditions = [
        (adi < ADI_CUTOFF) & (cv2 < CV2_CUTOFF),
        (adi < ADI_CUTOFF) & (cv2 >= CV2_CUTOFF),
        (adi >= ADI_CUTOFF) & (cv2 < CV2_CUTOFF),
        (adi >= ADI_CUTOFF) & (cv2 >= CV2_CUTOFF),
    ]
    labels = np.select(conditions, SBC_CLASSES, default="")
    return pd.Categorical(np.where(labels == "", None, labels), categories=SBC_CLASSES)
