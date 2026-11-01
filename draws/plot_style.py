from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter

C_TEAL = "#2E9D8F"       # primary series: mean recall / Hyperspheres / GIST1M
C_ORANGE = "#D5765C"     # worst case / search-time reduction / SIFT1M
C_BLUE = "#4176A6"       # P25 / FAISS-HNSW / PaperMatch
C_SLATE = "#5C6A76"      # latency (right axis) / safe
C_GREY = "#7C8288"       # FAISS-Flat baseline

# Lifecycle figures need six distinguishable series; keep them in the same
# muted blue family as the reference instead of introducing a new palette.
C_BLUE_DK = "#2A5A80"    # FAISS-HNSW, batched
C_BLUE_MID = "#5B8BBB"   # FAISS-IVFFlat, batched
C_BLUE_LT = "#8FB2D8"    # FAISS-IVFFlat

C_TITLE = "#263340"
C_INK = "#3D4852"        # tick labels / axis labels / annotations
C_AXIS = "#4E5964"       # spines and tick marks
C_AXIS_MUTED = "#B9C0C8" # minor tick marks on log axes
C_GRID = "#E3E5EA"

class Sizes(NamedTuple):
    title: float
    axlabel: float
    tick: float
    legend: float
    annot: float


SIZES_2 = Sizes(title=18.0, axlabel=16.5, tick=14.5, legend=14.5, annot=13.5)
SIZES_4 = Sizes(title=16.0, axlabel=15.5, tick=13.5, legend=12.5, annot=12.0)

FIG_W = 14.32                     # inches; 4296 px at 300 dpi
FIG_H_2 = 5.334                   # two panels: 4296 x 1600 px at 300 dpi
FIG_H_4 = 4.34                    # four panels: 4296 x 1302 px at 300 dpi
LAYOUT_2 = dict(left=0.082, right=0.961, top=0.821, bottom=0.2425, wspace=0.253)


BOLD_CACHE = Path(tempfile.gettempdir()) / "mpl-bold-instances"
VF_CANDIDATES = (
    "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-VariableFont_wght.ttf",
    "/usr/share/fonts/noto/NotoSans[wght].ttf",
)


def _has_bold(family: str) -> bool:
    for entry in fm.fontManager.ttflist:
        if entry.name != family or entry.style != "normal":
            continue
        weight = entry.weight
        if weight == "bold" or (str(weight).isdigit() and int(weight) >= 700):
            return True
    return False


def register_bold(family: str = "Noto Sans") -> bool:
    if _has_bold(family):
        return True
    src = next((Path(p) for p in VF_CANDIDATES if Path(p).exists()), None)
    if src is None:
        return False
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
    except ImportError:
        return False
    BOLD_CACHE.mkdir(parents=True, exist_ok=True)
    out = BOLD_CACHE / f"{src.stem}-Bold.ttf"
    if not out.exists():
        font = TTFont(str(src))
        instantiateVariableFont(font, {"wght": 700}, updateFontNames=True)
        font.save(str(out))
    fm.fontManager.addfont(str(out))
    return _has_bold(family)


def apply_style(sizes: Sizes = SIZES_2) -> None:
    """Install the reference rcParams; call once before building a figure."""
    family = "Noto Sans" if register_bold("Noto Sans") else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "font.size": sizes.tick,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": C_AXIS,
        "axes.labelcolor": C_INK,
        "axes.titlecolor": C_TITLE,
        "axes.titlesize": sizes.title,
        "axes.titleweight": "bold",
        "axes.labelsize": sizes.axlabel,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "xtick.labelsize": sizes.tick,
        "ytick.labelsize": sizes.tick,
        "legend.labelcolor": C_INK,
        "legend.fontsize": sizes.legend,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,          # embed TrueType instead of Type 3
        "ps.fonttype": 42,
    })


def frame(ax, *, grid_axis: str = "y", tick_pad: float = 8,
          minor_ticks: bool = False) -> None:
    """Apply the reference frame: open top/right, one grid direction only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_AXIS)
        ax.spines[side].set_linewidth(1.4)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, which="major", color=C_GRID, linewidth=1.1)
    other = "x" if grid_axis == "y" else "y"
    ax.grid(axis=other, which="major", visible=False)
    ax.grid(which="minor", visible=False)
    ax.tick_params(axis="both", which="major", length=5.5, width=1.4,
                   color=C_AXIS, pad=tick_pad)
    if minor_ticks:
        ax.tick_params(axis="both", which="minor", length=2.6, width=0.9,
                       color=C_AXIS_MUTED)


def percent_y(ax, decimals: int = 0) -> None:
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.{decimals}f}%"))


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_decade_formatter() -> FuncFormatter:
    def _fmt(value, _pos=None):
        exponent = int(round(math.log10(value)))
        return "10" + str(exponent).translate(_SUPERSCRIPT)

    return FuncFormatter(_fmt)


def figure_legend(fig, handles, labels, *, ncol: int = 4, y: float = 0.995,
                  fontsize: float | None = None) -> None:
    """The reference top-centred, frameless legend."""
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, frameon=False, handlelength=2.0, handletextpad=0.7,
               columnspacing=2.4, borderaxespad=0.0,
               fontsize=plt.rcParams["legend.fontsize"] if fontsize is None
               else fontsize)


MEAN = dict(marker="o", ms=10.5, lw=2.7, color=C_TEAL)
P25 = dict(marker="s", ms=9.5, lw=2.2, ls=(0, (5, 3)), color=C_BLUE)
WORST = dict(marker="v", ms=11.5, lw=2.7, color=C_ORANGE)
LATENCY = dict(marker="D", ms=9.0, lw=2.3, ls=(0, (1.2, 2.2)), color=C_SLATE)
SERIES = dict(lw=2.7, ms=10.5)
