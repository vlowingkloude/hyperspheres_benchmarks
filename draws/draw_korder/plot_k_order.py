from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Shared reference style (palette, fonts, axes recipe).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plot_style as ps  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

HERE = Path(__file__).resolve().parent
K_DIR = "k"
ORDER_DIR = "order"
FILE_PREFIX = "hyperspheres-aggressive-1-0"

# Drawing order (also the legend order).
DATASETS = ["gist1m", "sift1m", "mrl_1024"]

LABELS = {
    "gist1m": "GIST1M",
    "sift1m": "SIFT1M",
    "mrl_1024": "PaperMatch",
}

# Reference palette: teal/orange/blue with circle/square/diamond markers.
STYLE = {
    "gist1m": dict(color=ps.C_TEAL, marker="o"),
    "sift1m": dict(color=ps.C_ORANGE, marker="s"),
    "mrl_1024": dict(color=ps.C_BLUE, marker="D"),
}

LINE_WIDTH = 2.7
MARKER_SIZE = 10.5
MARKER_EDGE_WIDTH = 1.2

FIG_SIZE = (ps.FIG_W, ps.FIG_H_2)
LAYOUT = ps.LAYOUT_2

# Panel (a) always starts at 0 (the k=1 point is 0 % on GIST1M/SIFT1M).
Y_LIM_A = (0.0, 103.0)
Y_STEP_A = 10.0

# Panel (b): zoom in so the shuffle-to-shuffle spread is actually visible.
# Set to None to force the same 0-100 % scale as panel (a).
Y_ZOOM_B = (80.0, 100.6)
Y_STEP_B = 5.0

# Write the worst percentage next to each series minimum in panel (b).
LABEL_WORST = False

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def split_name(path: Path, tail_is_number: bool) -> tuple[str, int] | None:
    """Split ``hyperspheres-aggressive-1-0-<a>-<b>.json``.

    Returns (dataset, number), where (a, b) is (k, dataset) in the k files and
    (dataset, shuffle index) in the order files.
    """
    stem = path.stem
    if not stem.startswith(FILE_PREFIX + "-"):
        return None
    head, _, tail = stem[len(FILE_PREFIX) + 1:].rpartition("-")
    if not head or not tail:
        return None
    if tail_is_number:                       # order: "<dataset>-<index>"
        return (head, int(tail)) if tail.isdigit() else None
    return (tail, int(head)) if head.isdigit() else None   # k: "<k>-<dataset>"


def _scan(root: Path, subdir: str, tail_is_number: bool) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for path in sorted((root / subdir).rglob("*.json")):
        parsed = split_name(path, tail_is_number)
        if parsed is None:
            continue
        dataset, number = parsed
        with path.open() as fh:
            series.setdefault(dataset, []).append(
                (number, 100.0 * float(json.load(fh)["recall_min"])))
    return series


def scan_k(root: Path) -> dict[str, list[tuple[int, float]]]:
    return {d: sorted(v) for d, v in _scan(root, K_DIR, tail_is_number=False).items()}


def scan_order(root: Path) -> dict[str, list[tuple[int, float]]]:
    return {d: sorted(v) for d, v in _scan(root, ORDER_DIR, tail_is_number=True).items()}


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #
def style_axes(ax, ylim: tuple[float, float], ystep: float,
               show_ylabels: bool = True) -> None:
    """Reference look: open top/right, horizontal guides, percent ticks."""
    ps.frame(ax, grid_axis="y", tick_pad=9)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ps.percent_y(ax)
    if not show_ylabels:
        ax.tick_params(axis="y", left=False, labelleft=False)


def plot_series(ax, x, y, dataset):
    style = STYLE[dataset]
    (line,) = ax.plot(
        x,
        y,
        label=LABELS[dataset],
        color=style["color"],
        marker=style["marker"],
        markersize=MARKER_SIZE,
        markerfacecolor=style["color"],
        markeredgecolor=style["color"],
        markeredgewidth=MARKER_EDGE_WIDTH,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        clip_on=False,
        zorder=3,
    )
    return line


def annotate_worst(ax, x, y, dataset) -> None:
    idx = min(range(len(y)), key=lambda i: y[i])
    dy = -13 if y[idx] > 97.0 else 7
    ax.annotate(
        f"{y[idx]:.0f}%",
        xy=(x[idx], y[idx]),
        xytext=(0, dy),
        textcoords="offset points",
        ha="center",
        va="bottom" if dy > 0 else "top",
        fontsize=ps.SIZES_2.annot,
        color=ps.C_INK,
        clip_on=False,
        zorder=5,
    )


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build_figure(root: Path, zoom_b: bool = True) -> plt.Figure:
    ps.apply_style(ps.SIZES_2)

    k_data, o_data = scan_k(root), scan_order(root)
    if not k_data:
        raise SystemExit(f"no result files found under {root / K_DIR}")

    datasets = [d for d in DATASETS if d in k_data and d in o_data]
    datasets += sorted(d for d in k_data if d not in datasets and d in o_data)
    unknown = [d for d in datasets if d not in STYLE or d not in LABELS]
    if unknown:
        raise SystemExit(f"datasets missing from STYLE/LABELS: {unknown}")

    zoom_b = bool(zoom_b and Y_ZOOM_B)
    fig, (ax_k, ax_o) = plt.subplots(
        1, 2, figsize=FIG_SIZE, sharey=not zoom_b
    )
    fig.subplots_adjust(**LAYOUT)

    handles = []
    for dataset in datasets:
        (ks, rk), (idxs, ro) = zip(*k_data[dataset]), zip(*o_data[dataset])
        handles.append(plot_series(ax_k, ks, list(rk), dataset))
        plot_series(ax_o, idxs, list(ro), dataset)
        if LABEL_WORST:
            annotate_worst(ax_o, idxs, list(ro), dataset)

    # ---- panel (a): sensitivity to k
    ax_k.set_title("(a) Sensitivity to k", pad=22)
    ax_k.set_xlabel("k", labelpad=12)
    ax_k.set_ylabel("Worst-case recall", labelpad=8)
    all_ks = sorted({k for d in datasets for k, _ in k_data[d]})
    ax_k.set_xticks(all_ks)
    ax_k.set_xlim(min(all_ks) - 4, max(all_ks) + 4)
    style_axes(ax_k, Y_LIM_A, Y_STEP_A)

    # ---- panel (b): sensitivity to vector insertion order
    ax_o.set_title("(b) Sensitivity to vector order", pad=22)
    ax_o.set_xlabel("Random shuffle #", labelpad=12)
    all_idx = sorted({i for d in datasets for i, _ in o_data[d]})
    ax_o.set_xticks(all_idx)
    ax_o.set_xlim(min(all_idx) - 0.35, max(all_idx) + 0.35)
    if zoom_b:
        style_axes(ax_o, Y_ZOOM_B, Y_STEP_B)
    else:
        style_axes(ax_o, Y_LIM_A, Y_STEP_A, show_ylabels=False)

    ps.figure_legend(fig, handles, [LABELS[d] for d in datasets], ncol=3)

    return fig


def save(fig: plt.Figure, out: Path, dpi: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=dpi)
    print(f"saved: {out}")
    print(f"saved: {out.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE, help="result directory")
    parser.add_argument("--out", type=Path, default=Path("k_order_sensitivity.pdf"))
    parser.add_argument("--dpi", type=int, default=300, help="raster dpi for the .png")
    parser.add_argument("--no-shared-variant", action="store_true",
                        help="skip the same-scale (0-100%%) copy")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out = args.out.expanduser()
    if not out.is_absolute():
        out = (root / out).resolve()

    save(build_figure(root, zoom_b=True), out, args.dpi)
    if not args.no_shared_variant:
        save(build_figure(root, zoom_b=False),
             out.with_name(out.stem + "_sharedy" + out.suffix), args.dpi)

    plt.close("all")


if __name__ == "__main__":
    main()
