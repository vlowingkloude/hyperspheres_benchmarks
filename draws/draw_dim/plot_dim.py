from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Shared reference style (palette, fonts, axes recipe).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plot_style as ps  # noqa: E402

C_TEAL = ps.C_TEAL      # aggressive / mean
C_ORANGE = ps.C_ORANGE  # search-time reduction / worst case
C_BLUE = ps.C_BLUE      # P25
C_SLATE = ps.C_SLATE    # safe
C_TITLE = ps.C_TITLE
C_INK = ps.C_INK
C_AXIS = ps.C_AXIS
C_GRID = ps.C_GRID

FIG_W, FIG_H = ps.FIG_W, ps.FIG_H_2
LAYOUT = ps.LAYOUT_2

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
# Axes helpers
# --------------------------------------------------------------------------- #
def style_axes(ax, title: str, ylabel: str | None, dims: list[int]) -> None:
    ax.set_title(title, pad=22)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_AXIS)
        ax.spines[side].set_linewidth(1.4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=C_GRID, linewidth=1.1)
    ax.tick_params(axis="both", length=5.5, width=1.4, color=C_AXIS, pad=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v)))
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlim(-0.35, len(dims) - 0.65)
    ax.set_xlabel("Embedding dimension", labelpad=12)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)


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
        raise SystemExit("no dimension left for the quality panel")

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


def plot(series: dict, args) -> plt.Figure:
    dims_a, dims_b = series["dims_a"], series["dims_b"]
    xa, xb = range(len(dims_a)), range(len(dims_b))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(**LAYOUT)

    # ---- (a) pruning and search-time reduction ------------------------------
    ax1.plot(xa, series["safe_prune"], marker="s", ms=9.5, lw=2.6, color=C_SLATE,
             label="Safe", zorder=3)
    ax1.plot(xa, series["aggr_prune"], marker="o", ms=11, lw=2.8, color=C_TEAL,
             label="Aggressive", zorder=4)
    ax1.plot(xa, series["reduction"], marker="D", ms=11, lw=2.8, color=C_ORANGE,
             label="Search-time reduction", zorder=5)

    style_axes(ax1, "(a) Pruning and search-time reduction", "Percentage", dims_a)
    ax1.set_ylim(-5, 104)
    ax1.yaxis.set_major_locator(MultipleLocator(10))

    ax1.legend(loc="center right", bbox_to_anchor=(0.985, 0.30), frameon=False,
               handlelength=1.6, handletextpad=0.9, labelspacing=0.55,
               borderaxespad=0.0)

    # ---- (b) per-query result quality --------------------------------------
    ax2.plot(xb, series["mean"], marker="o", ms=11, lw=2.8, color=C_TEAL,
             label="Mean", zorder=4)
    ax2.plot(xb, series["p25"], marker="s", ms=10, lw=2.3, ls=(0, (5, 3)),
             color=C_BLUE, label="P25", zorder=3)
    ax2.plot(xb, series["worst"], marker="v", ms=12, lw=2.8, color=C_ORANGE,
             label="Worst case", zorder=5)

    style_axes(ax2, "(b) Per-query result quality", f"Recall@{series['k']}", dims_b)
    lo = min(series["worst"] + series["mean"] + series["p25"])
    ax2.set_ylim(max(-1.0, lo - 4.0), 101.6)
    ax2.yaxis.set_major_locator(MultipleLocator(5))

    if args.annotate == "worst":
        annotate_worst(ax2, xb, series["worst"])

    ax2.legend(loc="lower right", bbox_to_anchor=(0.985, 0.03), frameon=False,
               handlelength=1.6, handletextpad=0.9, labelspacing=0.55,
               borderaxespad=0.0)
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
                    choices=["aggressive", "safe"], help="mode shown in panel (b)")
    ap.add_argument("--annotate", default="worst", choices=["none", "worst"],
                    help="label the worst-case recall values in panel (b)")
    ap.add_argument("--out", default="dim_sweep", help="output path without extension")
    ap.add_argument("--formats", default="png,pdf", help="comma separated: png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    ap.add_argument("--csv", default=None, metavar="PATH", help="also dump the plotted values")
    args = ap.parse_args()

    ps.apply_style(ps.SIZES_2)
    table = index_records(load_records(args.data))
    engine = pick_engine(table, args.engine)

    series = collect(table, engine, args)
    fig = plot(series, args)
    print_table(series)

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
