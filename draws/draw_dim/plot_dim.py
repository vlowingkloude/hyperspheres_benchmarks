#!/usr/bin/env python3
"""Draw the embedding-dimension sweep and MRL geometric-risk figure.

Inputs
    aggressive/, safe/
        per-dimension benchmark JSONs, e.g.
        aggressive/hyperspheres-aggressive-1-0-mrl_512.json
        safe/hyperspheres-safe-1-0-mrl_512.json
    theta.csv
        per-candidate diagnostics from the MRL-256 run:
        r_over_d, cos_theta_prime, in_page_closest, current_global_max,
        has_been_pruned

Figure layout
    (a) effect of dimensionality, dual y-axis
        left  (percentage): safe pruning, aggressive pruning,
              search-time reduction = (safe.avg_query_ms - aggressive.avg_query_ms)
                                      / safe.avg_query_ms
        right (recall@k): mean = recall_at_k, P25 = recall_p25_at_k,
              worst case = recall_min
    (b) geometric risk at MRL 256
        Only candidates with has_been_pruned == 1 are shown.
        x = cos_theta_prime - r_over_d
        y = in_page_closest / current_global_max
        The pruned candidates are drawn as a log-count hexbin; the misprunes
        (y < 1) are overlaid in red and counted.  The x=0 and y=1 guides mark
        the unsafe corner.

Usage
    ./env/bin/python plot_dim.py                    # every dim found, PNG + PDF
    ./env/bin/python plot_dim.py --annotate none    # no value labels in (a)
    ./env/bin/python plot_dim.py --dims 64 128 256 512 768 1024
    ./env/bin/python plot_dim.py --out figs/dim_sweep --formats png,pdf,svg
    ./env/bin/python plot_dim.py --theta theta.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

# Shared reference style (palette, fonts, axes recipe).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plot_style as ps  # noqa: E402

C_TEAL = ps.C_TEAL          # mean recall
C_ORANGE = ps.C_ORANGE      # worst-case recall
C_BLUE = ps.C_BLUE          # aggressive pruning
C_BLUE_DK = ps.C_BLUE_DK    # P25 recall
C_BLUE_LT = ps.C_BLUE_LT    # search-time reduction
C_SLATE = ps.C_SLATE        # safe pruning
C_MISPRUNE = "#C0392B"      # the rare unsafe point
# Very pale quadrant tints; the hexbin must stay the dominant layer.
C_SAFE_BG = "#DCEFD4"       # x < 0, y > 1
C_RISK_BG = "#FFF0BE"       # x > 0, y > 1
C_ERROR_BG = "#F6D0C9"      # x > 0, y < 1
C_IMPOSSIBLE_BG = "#EFEFF0" # x < 0, y < 1
QUADRANT_ALPHA = 0.55
C_TITLE = ps.C_TITLE
C_INK = ps.C_INK
C_AXIS = ps.C_AXIS
C_GRID = ps.C_GRID

HERE = Path(__file__).resolve().parent
THETA_CSV = HERE / "theta.csv"
FIG_W, FIG_H = ps.FIG_W, ps.FIG_H_2
LAYOUT = dict(left=0.09, right=0.905, bottom=0.18, top=0.76, wspace=0.42)
HEX_GRID = 60

FS_TITLE = ps.SIZES_2.title
FS_AXLABEL = ps.SIZES_2.axlabel
FS_TICK = ps.SIZES_2.tick
FS_LEGEND = ps.SIZES_2.legend
FS_ANNOT = ps.SIZES_2.annot


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _dim_of(doc: dict, path: Path) -> int:
    if "dim" in doc:
        return int(doc["dim"])
    return int(path.stem.rsplit("_", 1)[-1])


def load_records(data_dirs: list[str]) -> list[tuple[dict, Path]]:
    """Read every result JSON found in the given dirs (or cwd / this dir)."""
    if data_dirs:
        roots = [Path(d) for d in data_dirs]
    else:   # cwd, and next to this script when it is run from elsewhere
        roots = sorted({Path(".").resolve(), Path(__file__).resolve().parent})
    paths: list[Path] = []
    for root in roots:
        root = Path(root)
        paths.extend(sorted(root.glob("*.json")))
        paths.extend(sorted(root.glob("*/*.json")))       # one level down

    records = []
    for path in sorted(set(paths)):
        if "env" in path.parts:
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if "avg_query_ms" not in doc:
            continue
        records.append((doc, path))
    if not records:
        raise SystemExit(f"no result JSON found under {', '.join(map(str, roots))}")
    return records


def index_records(records):
    """engine -> mode -> dim -> doc"""
    table: dict[str, dict[str, dict[int, dict]]] = {}
    for doc, path in records:
        table.setdefault(doc.get("engine", "unknown"), {}) \
             .setdefault(doc.get("mode", "unknown"), {})[_dim_of(doc, path)] = doc
    return table


def pick_engine(table, requested: str | None) -> str:
    if requested:
        if requested not in table:
            raise SystemExit(f"engine {requested!r} not found; have {sorted(table)}")
        return requested
    return max(table, key=lambda e: sum(len(m) for m in table[e].values()))


def pct(v: float, nd: int = 0) -> str:
    return f"{v:.{nd}f}%"


# --------------------------------------------------------------------------- #
# theta.csv (MRL-1024 geometric diagnostics)
# --------------------------------------------------------------------------- #
def load_theta(path: Path) -> dict:
    """Read the candidate diagnostics and derive the panel (b) coordinates."""
    if not path.is_file():
        raise SystemExit(f"theta CSV not found: {path}")
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    needed = ["r_over_d", "cos_theta_prime", "in_page_closest",
              "current_global_max", "has_been_pruned"]
    try:
        cols = [header.index(name) for name in needed]
    except ValueError as exc:
        raise SystemExit(f"theta CSV is missing a column: {exc}") from exc

    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=cols, ndmin=2)
    r_over_d, cos_tp, in_page, global_max, pruned = arr.T
    x = cos_tp - r_over_d
    y = in_page / global_max
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    in_page, global_max, pruned = in_page[finite], global_max[finite], pruned[finite]
    n_total = int(x.size)

    # Panel (b) is about the pruned candidates only: every point in this
    # subset should be at y >= 1, and any point below 1 is a misprune.
    keep = pruned == 1
    x, y = x[keep], y[keep]
    in_page, global_max = in_page[keep], global_max[keep]
    misprune = in_page < global_max
    return {
        "x": x,
        "y": y,
        "misprune": misprune,
        "n_total": n_total,
        "n_pruned": int(keep.sum()),
        "n_misprune": int(misprune.sum()),
    }


# --------------------------------------------------------------------------- #
# Axes helpers
# --------------------------------------------------------------------------- #
def annotate_worst(ax, xs, ys) -> None:
    """Label worst-case points: above at the ceiling, otherwise under the curve."""
    for x, y in zip(xs, ys):
        if y >= 99.6:                       # flat at the ceiling: label above
            offset, ha, va = (0, 11), "center", "bottom"
        else:                               # beside the marker, under the line
            offset, ha, va = (11, -11), "left", "top"
        ax.annotate(pct(y), (x, y), textcoords="offset points", xytext=offset,
                    ha=ha, va=va, fontsize=FS_ANNOT, color=C_INK, zorder=6,
                    clip_on=False)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def collect(table: dict, engine: str, args):
    """Assemble every plotted series, so figure and table cannot drift apart."""
    modes = table[engine]
    if "aggressive" not in modes:
        raise SystemExit("no 'aggressive' results found")
    safe = modes.get("safe", {})

    dims_a = sorted(set(modes["aggressive"]) & set(safe))
    dims_b = sorted(modes[getattr(args, "quality_mode", "aggressive")])
    if args.dims:
        dims_a = [d for d in dims_a if d in args.dims]
        dims_b = [d for d in dims_b if d in args.dims]
    if not dims_a:
        raise SystemExit("no dimension has both safe and aggressive results")
    if not dims_b:
        raise SystemExit("no dimension left for the recall series")
    if dims_a != dims_b:
        raise SystemExit(
            "panel (a) overlays pruning and recall, so both modes need the "
            f"same dimensions; got {dims_a} vs {dims_b}")

    aggr = modes["aggressive"]
    quality = modes[args.quality_mode]
    series = {
        "dims_a": dims_a,
        "safe_prune": [100.0 * safe[d]["pruning_rate"] for d in dims_a],
        "aggr_prune": [100.0 * aggr[d]["pruning_rate"] for d in dims_a],
        "reduction": [100.0 * (1.0 - aggr[d]["avg_query_ms"] / safe[d]["avg_query_ms"])
                      for d in dims_a],
        "safe_ms": [safe[d]["avg_query_ms"] for d in dims_a],
        "aggr_ms": [aggr[d]["avg_query_ms"] for d in dims_a],
        "dims_b": dims_b,
        "mean": [100.0 * quality[d]["recall_at_k"] for d in dims_b],
        "p25": [100.0 * quality[d]["recall_p25_at_k"] for d in dims_b],
        "worst": [100.0 * quality[d]["recall_min"] for d in dims_b],
        "k": int(quality[dims_b[0]].get("k", 100)),
    }
    return series


def plot_combined(ax, ax2, series: dict, args):
    """Panel (a): pruning / reduction on the left, recall on the right."""
    dims = series["dims_a"]
    x = range(len(dims))

    left = (
        ("Safe pruning", series["safe_prune"], C_SLATE, "s"),
        ("Aggressive pruning", series["aggr_prune"], C_BLUE, "o"),
        ("Search-time reduction", series["reduction"], C_BLUE_LT, "D"),
    )
    for label, y, color, marker in left:
        ax.plot(x, y, marker=marker, ms=9, lw=2.2, ls=(0, (5, 3)),
                color=color, mfc="white", mew=1.6, mec=color,
                label=label, zorder=3)

    right = (
        ("Mean recall", series["mean"], C_TEAL, "o"),
        ("P25 recall", series["p25"], C_BLUE_DK, "s"),
        ("Worst-case recall", series["worst"], C_ORANGE, "v"),
    )
    for label, y, color, marker in right:
        ax2.plot(x, y, marker=marker, ms=10.5, lw=2.7,
                 color=color, mfc=color, mec=color, mew=1.2,
                 label=label, zorder=4)

    ps.frame(ax, grid_axis="y", tick_pad=9)
    ax.set_ylim(-5, 104)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ps.percent_y(ax)
    ax.set_xlim(-0.35, len(dims) - 0.65)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Embedding dimension", labelpad=12)
    ax.set_ylabel("Percentage", labelpad=8)
    ax.set_title("(a) Effect of dimensionality", pad=22)

    # The recall axis starts at 60% so the three recall curves do not sit on
    # top of the 75-95% pruning curves on the left axis.
    ax2.set_ylim(60, 101.5)
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ps.percent_y(ax2)
    ax2.set_ylabel("Recall@100", labelpad=8)
    ax2.tick_params(axis="y", length=5.5, width=1.4, color=C_AXIS,
                    pad=8, labelsize=FS_TICK)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_AXIS)
    ax2.spines["right"].set_linewidth(1.4)
    for side in ("top", "left", "bottom"):
        ax2.spines[side].set_visible(False)

    if getattr(args, "annotate", "none") == "worst":
        annotate_worst(ax2, x, series["worst"])

    return ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()


def shade_quadrants(ax) -> None:
    """Tint the four x=0 / y=1 quadrants behind the hexbin."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rects = (
        (x0, 1.0, 0.0 - x0, y1 - 1.0, C_SAFE_BG),
        (0.0, 1.0, x1 - 0.0, y1 - 1.0, C_RISK_BG),
        (0.0, y0, x1 - 0.0, 1.0 - y0, C_ERROR_BG),
        (x0, y0, 0.0 - x0, 1.0 - y0, C_IMPOSSIBLE_BG),
    )
    for x, y, w, h, color in rects:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                               edgecolor="none", alpha=QUADRANT_ALPHA,
                               zorder=0))


def plot_theta(ax, theta: dict) -> None:
    """Panel (b): log-count hexbin plus the rare mispruned candidates."""
    hb = ax.hexbin(theta["x"], theta["y"], gridsize=HEX_GRID,
                   cmap="Blues", bins="log", mincnt=1, linewidths=0,
                   zorder=2)
    ax.set_xlim(-0.42, 0.10)
    ax.set_ylim(0.84, 4.65)
    shade_quadrants(ax)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ps.frame(ax, grid_axis="y", tick_pad=8)
    ax.set_xlabel(r"$\cos\theta_x - r_i / d$", labelpad=12)
    ax.set_ylabel(r"$\min\,(\|q-x\|/w)^2$", labelpad=8)
    ax.set_title("(b) Geometric risk at MRL 256", pad=22)
    #ax.text(0.02, 0.96,
    #        f"has_been_pruned = 1  ({theta['n_pruned']:,})",
    #        transform=ax.transAxes, ha="left", va="top",
    #        fontsize=FS_ANNOT, color=C_INK, zorder=12,
    #        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
    #                  pad=1.5))

    # Risk boundary: x=0 and y=1.
    ax.axvline(0, color=C_AXIS, lw=1.2, ls=(0, (2, 3)), alpha=0.75,
               zorder=4)
    ax.axhline(1, color=C_AXIS, lw=1.2, ls=(0, (2, 3)), alpha=0.75,
               zorder=4)
    #ax.text(0.012, 4.58, "x = 0", ha="left", va="top",
    #        fontsize=FS_ANNOT, color=C_AXIS, zorder=5)
    #ax.text(-0.41, 1.04, "y = 1", ha="left", va="bottom",
    #        fontsize=FS_ANNOT, color=C_AXIS, zorder=5)

    mask = theta["misprune"]
    if mask.any():
        ax.scatter(theta["x"][mask], theta["y"][mask], s=75,
                   facecolor=C_MISPRUNE, edgecolor="white", linewidth=1.3,
                   zorder=8)
        n = int(mask.sum())
        label = "1 misprune" if n == 1 else f"{n:,} misprunes"
        i = int(np.argmax(mask))
        ax.annotate(label, xy=(theta["x"][i], theta["y"][i]),
                    xytext=(-44, 20), textcoords="offset points",
                    ha="right", va="bottom", fontsize=FS_ANNOT,
                    color=C_MISPRUNE, zorder=9,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.8, pad=1.5),
                    arrowprops=dict(arrowstyle="-", color=C_MISPRUNE,
                                    lw=1.0, shrinkA=0, shrinkB=4))

    cbar = ax.figure.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Count (log scale)", fontsize=FS_AXLABEL - 2,
                   labelpad=8, color=C_INK)
    cbar.ax.tick_params(labelsize=FS_TICK - 2, length=4, width=1.0,
                        color=C_AXIS, labelcolor=C_INK)
    cbar.outline.set_edgecolor(C_AXIS)
    cbar.outline.set_linewidth(1.0)


def plot(series: dict, theta: dict, args) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FIG_W, FIG_H),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    fig.subplots_adjust(**LAYOUT)
    ax_a2 = ax_a.twinx()

    (h1, l1), (h2, l2) = plot_combined(ax_a, ax_a2, series, args)
    plot_theta(ax_b, theta)
    # Interleave left/right handles so the 3-column legend has one row for the
    # percentage series (dashed) and one row for the recall series (solid).
    handles = [h1[0], h2[0], h1[1], h2[1], h1[2], h2[2]]
    labels = [l1[0], l2[0], l1[1], l2[1], l1[2], l2[2]]
    ps.figure_legend(fig, handles, labels, ncol=3, y=0.995)
    return fig


def print_table(series: dict) -> None:
    print(f"{'dim':>5} {'safe_prune%':>11} {'aggr_prune%':>11} {'reduce%':>8} "
          f"{'safe_ms':>9} {'aggr_ms':>9} | {'mean%':>7} {'p25%':>7} {'worst%':>7}")
    for i, d in enumerate(series["dims_a"]):
        print(f"{d:>5} {series['safe_prune'][i]:>11.2f} {series['aggr_prune'][i]:>11.2f} "
              f"{series['reduction'][i]:>8.2f} {series['safe_ms'][i]:>9.3f} "
              f"{series['aggr_ms'][i]:>9.3f} |", end="")
        j = series["dims_b"].index(d) if d in series["dims_b"] else None
        if j is None:
            print()
        else:
            print(f" {series['mean'][j]:>7.2f} {series['p25'][j]:>7.2f} "
                  f"{series['worst'][j]:>7.2f}")


def write_csv(series: dict, path: Path) -> None:
    rows = {}
    for i, d in enumerate(series["dims_a"]):
        rows[d] = {
            "dim": d,
            "safe_pruning_rate": f"{series['safe_prune'][i] / 100:.6f}",
            "aggressive_pruning_rate": f"{series['aggr_prune'][i] / 100:.6f}",
            "search_time_reduction": f"{series['reduction'][i] / 100:.6f}",
            "safe_avg_query_ms": f"{series['safe_ms'][i]:.3f}",
            "aggressive_avg_query_ms": f"{series['aggr_ms'][i]:.3f}",
        }
    for j, d in enumerate(series["dims_b"]):
        row = rows.setdefault(d, {"dim": d})
        row.update(recall_mean=f"{series['mean'][j] / 100:.6f}",
                   recall_p25=f"{series['p25'][j] / 100:.6f}",
                   recall_worst=f"{series['worst'][j] / 100:.6f}")
    fields = ["dim", "safe_pruning_rate", "aggressive_pruning_rate",
              "search_time_reduction", "safe_avg_query_ms", "aggressive_avg_query_ms",
              "recall_mean", "recall_p25", "recall_worst"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for d in sorted(rows):
            writer.writerow({k: rows[d].get(k, "") for k in fields})
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=[], metavar="DIR",
                    help="directories with result JSONs (default: cwd + one level down)")
    ap.add_argument("--engine", default=None, help="engine name if several are present")
    ap.add_argument("--dims", nargs="+", type=int, default=None, help="restrict dimensions")
    ap.add_argument("--quality-mode", default="aggressive",
                    choices=["aggressive", "safe"], help="mode shown in panel (a)")
    ap.add_argument("--annotate", default="worst", choices=["none", "worst"],
                    help="label the worst-case recall values in panel (a)")
    ap.add_argument("--theta", default=str(THETA_CSV), metavar="CSV",
                    help="candidate diagnostics for panel (b) (default: theta.csv)")
    ap.add_argument("--out", default="dim_sweep", help="output path without extension")
    ap.add_argument("--formats", default="png,pdf", help="comma separated: png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    ap.add_argument("--csv", default=None, metavar="PATH", help="also dump the plotted values")
    args = ap.parse_args()

    ps.apply_style(ps.SIZES_2)
    table = index_records(load_records(args.data))
    engine = pick_engine(table, args.engine)
    theta = load_theta(Path(args.theta))

    series = collect(table, engine, args)
    fig = plot(series, theta, args)
    print_table(series)
    n_mis = theta["n_misprune"]
    print(f"theta: {theta['n_pruned']:,} pruned of {theta['n_total']:,} "
          f"candidates, {n_mis} misprune" + ("" if n_mis == 1 else "s"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.csv:
        write_csv(series, Path(args.csv))
    for fmt in [f.strip().lower() for f in args.formats.split(",") if f.strip()]:
        target = out.with_suffix("." + fmt)
        fig.savefig(target, dpi=args.dpi if fmt == "png" else None)
        print(f"wrote {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
