"""Statistical comparison of the six cells of the factorial design.

- summary table (mean, median, std, IQR, wins) of RMSSE@28;
- two-sided Wilcoxon signed-rank tests with Holm's step-down correction, for the
  six pre-specified contrasts and for all 15 pairs;
- Friedman omnibus test, Nemenyi critical difference and the CD diagram (Demsar, 2006).

The reference values reported by Tsantilis et al. (2025) are included for comparison.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Pre-specified contrasts: pooling effect within each architecture, architecture effect in the global regime.
PLANNED_CONTRASTS = {
    "Pool. LightGBM (PS vs GL)": ("LGBM-MSE-PS", "LGBM-MSE-GL"),
    "Pool. LSTM (PS vs GL)": ("LSTM-PS", "LSTM-GL"),
    "Pool. Reservoir (PS vs GL)": ("RES-PS", "RES-GL"),
    "Arch. in GL (LGBM vs LSTM)": ("LGBM-MSE-GL", "LSTM-GL"),
    "Arch. in GL (LGBM vs RES)": ("LGBM-MSE-GL", "RES-GL"),
    "Arch. in GL (LSTM vs RES)": ("LSTM-GL", "RES-GL"),
}

# Tsantilis et al. (2025), Table I.
PAPER_TABLE1 = pd.DataFrame(
    {
        "Mean": [0.7859, 0.7675, 0.7690, 0.7718, 0.8101, 0.7652],
        "Median": [0.7236, 0.6991, 0.6930, 0.6974, 0.7227, 0.6802],
        "Std": [0.3543, 0.3358, 0.3520, 0.3444, 0.3664, 0.3386],
        "IQR": [0.3319, 0.3534, 0.3640, 0.3838, 0.3778, 0.3699],
        "Wins": [21, 21, 16, 11, 19, 12],
    },
    index=pd.Index(["LGBM-MSE-PS", "LGBM-MSE-GL", "LSTM-PS", "LSTM-GL", "RES-PS", "RES-GL"], name="Method"),
)

# Table II: mean(A - B), raw p and Holm p of the planned contrasts.
PAPER_TABLE2 = pd.DataFrame(
    {
        "mean_diff": [0.0184, -0.0028, 0.0450, -0.0043, 0.0023, 0.0066],
        "p_raw": [0.0012, 0.4682, 0.0001, 0.8770, 0.1576, 0.0023],
        "p_holm": [0.0059, 0.9363, 0.0006, 0.9363, 0.4728, 0.0093],
    },
    index=pd.Index(list(PLANNED_CONTRASTS), name="Contrast"),
)

# Table III: all 15 pairs (A, B) -> mean(A - B), raw p, Holm p.
PAPER_TABLE3 = pd.DataFrame(
    [
        ("LGBM-MSE-PS", "RES-GL", 0.0208, 0.0001, 0.0012),
        ("RES-PS", "RES-GL", 0.0450, 0.0001, 0.0015),
        ("LGBM-MSE-PS", "LSTM-PS", 0.0170, 0.0004, 0.0051),
        ("LSTM-GL", "RES-PS", -0.0383, 0.0008, 0.0100),
        ("LGBM-MSE-PS", "LGBM-MSE-GL", 0.0184, 0.0012, 0.0129),
        ("LGBM-MSE-GL", "RES-PS", -0.0426, 0.0019, 0.0190),
        ("LSTM-GL", "RES-GL", 0.0066, 0.0023, 0.0208),
        ("LGBM-MSE-PS", "LSTM-GL", 0.0141, 0.0027, 0.0215),
        ("LSTM-PS", "RES-GL", 0.0038, 0.0175, 0.1225),
        ("LSTM-PS", "RES-PS", -0.0412, 0.0280, 0.1681),
        ("LGBM-MSE-GL", "RES-GL", 0.0023, 0.1576, 0.7881),
        ("LSTM-PS", "LSTM-GL", -0.0028, 0.4682, 1.0000),
        ("LGBM-MSE-PS", "RES-PS", -0.0242, 0.5497, 1.0000),
        ("LGBM-MSE-GL", "LSTM-GL", -0.0043, 0.8770, 1.0000),
        ("LGBM-MSE-GL", "LSTM-PS", -0.0015, 0.9698, 1.0000),
    ],
    columns=["A", "B", "mean_diff", "p_raw", "p_holm"],
)

PAPER_FRIEDMAN = {"chi2": 20.97, "p": 0.000822, "cd": 0.754}
PAPER_MEAN_RANKS = pd.Series(
    {"RES-GL": 2.94, "LSTM-GL": 3.35, "LSTM-PS": 3.41, "LGBM-MSE-GL": 3.44, "RES-PS": 3.84, "LGBM-MSE-PS": 4.02}
)


def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-cell summary of a (series x cell) score matrix; a win is the lowest score on a series."""
    q75, q25 = scores.quantile(0.75), scores.quantile(0.25)
    wins = scores.idxmin(axis=1).value_counts().reindex(scores.columns, fill_value=0)
    return pd.DataFrame(
        {"Mean": scores.mean(), "Median": scores.median(), "Std": scores.std(), "IQR": q75 - q25, "Wins": wins}
    ).rename_axis("Method")


def wilcoxon_holm(scores: pd.DataFrame, pairs: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank tests, Holm-corrected within the family of `pairs` (default: all pairs)."""
    if pairs is None:
        pairs = {f"{a} vs {b}": (a, b) for a, b in combinations(scores.columns, 2)}
    rows = []
    for name, (a, b) in pairs.items():
        diff = scores[a] - scores[b]
        rows.append({"contrast": name, "A": a, "B": b, "mean_diff": diff.mean(),
                     "p_raw": stats.wilcoxon(scores[a], scores[b]).pvalue})
    out = pd.DataFrame(rows).set_index("contrast")
    out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    return out.sort_values("p_raw")


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference of mean ranks for k methods on n datasets."""
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(k * (k + 1) / (6 * n)))


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman test, mean ranks (1 = lowest error) and the Nemenyi critical difference."""
    chi2, p = stats.friedmanchisquare(*[scores[c] for c in scores.columns])
    ranks = scores.rank(axis=1).mean().sort_values()
    return {"chi2": chi2, "p": p, "mean_ranks": ranks, "cd": nemenyi_cd(scores.shape[1], len(scores), alpha)}


def cliques(mean_ranks: pd.Series, cd: float) -> list[list[str]]:
    """Maximal groups of methods whose mean ranks all lie within one critical difference."""
    r = mean_ranks.sort_values()
    groups = []
    for i in range(len(r)):
        j = i
        while j + 1 < len(r) and r.iloc[j + 1] - r.iloc[i] < cd:
            j += 1
        if j > i and not any(set(r.index[i : j + 1]) <= set(g) for g in groups):
            groups.append(list(r.index[i : j + 1]))
    return groups


def plot_cd_diagram(ax, mean_ranks: pd.Series, cd: float, colors: dict, ink: str = "#0b0b0b",
                    muted: str = "#898781") -> None:
    """Critical difference diagram: rank axis on top, methods hanging below, clique bars in ink."""
    r = mean_ranks.sort_values()
    k = len(r)
    lo, hi = np.floor(r.min() * 2) / 2, np.ceil(r.max() * 2) / 2
    ax.set_xlim(lo - 0.9, hi + 0.9)
    ax.set_ylim(-(k // 2 + 2.2), 1.6)
    ax.axis("off")

    ax.plot([lo, hi], [0, 0], color=muted, lw=0.8)
    for t in np.arange(lo, hi + 1e-9, 0.5):
        ax.plot([t, t], [0, 0.12], color=muted, lw=0.8)
        ax.text(t, 0.25, f"{t:g}", ha="center", va="bottom", fontsize=8.5, color=ink)
    ax.plot([lo, lo + cd], [1.15, 1.15], color=ink, lw=1.2)
    for x in (lo, lo + cd):
        ax.plot([x, x], [1.05, 1.25], color=ink, lw=1.2)
    ax.text(lo + cd / 2, 1.32, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=8.5, color=ink)

    half = (k + 1) // 2
    for i, (name, rank) in enumerate(r.items()):
        left = i < half
        level = -(i + 1) if left else -(k - i)
        x_text = lo - 0.15 if left else hi + 0.15
        ax.plot([rank, rank, x_text], [0, level, level], color=colors.get(name, ink), lw=1.2)
        ax.text(x_text + (-0.05 if left else 0.05), level, f"{name} ({rank:.2f})" if left else f"({rank:.2f}) {name}",
                ha="right" if left else "left", va="center", fontsize=8.5, color=ink)

    for g_i, group in enumerate(cliques(r, cd)):
        y = -0.35 - 0.22 * g_i
        ax.plot([r[group[0]] - 0.03, r[group[-1]] + 0.03], [y, y], color=ink, lw=3, solid_capstyle="round")
