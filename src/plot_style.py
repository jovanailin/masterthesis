"""Shared matplotlib style for the thesis figures (light, print-oriented).

Colours follow a validated categorical palette: the three slots below pass the
colour-vision-deficiency checks for every pair. Categories are always also
identified by a legend or a direct label, never by colour alone.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#c3c2b7"  # de-emphasised series

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
ACCENT = SERIES[0]
NEGATIVE = "#e34948"  # opposite pole of the blue-red diverging pair
CATEGORY_COLORS = {"FOODS": SERIES[0], "HOBBIES": SERIES[1], "HOUSEHOLD": SERIES[2]}

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9.5,
            "text.color": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def label_line_end(ax, x, y, text: str, dx: float = 6) -> None:
    """Direct label placed just right of a line's last point (text in ink, not series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", color=INK_SECONDARY)


def savefig(fig, name: str, subdir: str = "eda") -> None:
    out = FIG_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
