from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (FuncFormatter, LogLocator, MultipleLocator,
                               NullFormatter)

# Shared reference style (palette, fonts, axes recipe).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
#  Dataset / workload assumptions
# --------------------------------------------------------------------------- #
DATASET = "GIST1M"
RECALL_TARGET = 95.0
RECALL_LABEL = f"{RECALL_TARGET:.0f}%"
LIFETIME_MAX = 1_000_000          # [queries] x-range of the left panel
TOTAL_QUERIES = 100_000           # [queries] fixed workload of the middle panel
RIGHT_X_MIN, RIGHT_X_MAX = 1.0, float(TOTAL_QUERIES)
LEFT_LIMITS = (0.1, 1e6)          # y-limits of the left panel
RIGHT_LIMITS = (1e2, 4e5)         # y-limits of the middle panel

# Third panel: latency-recall sweeps, one sub-folder per system.  The mode
# filter drops stray files that do not belong to that engine (e.g. the lone
# HNSW run that sits in lrivf/) and the build-only IVF record.
LATENCY_DIR = "latency_recall"
LATENCY_DIRS = {
    "hyperspheres": ("hyperspheres", "aggressive"),
    "hnsw": ("lrhnsw", "hnsw"),
    "hnsw_batch": ("lrhnswbatch", "hnsw"),
    "ivf": ("lrivf", "ivf"),
    "ivf_batch": ("lrivfbatch", "ivf"),
}
RECALL_X_MIN, RECALL_X_MAX = 70.0, 100.5     # x-range of panel (c)
LATENCY_Y_MIN, LATENCY_Y_MAX = 0.1, 100.0    # log y-range of panel (c)

# --------------------------------------------------------------------------- #
#  Style
# --------------------------------------------------------------------------- #
GREEN = ps.C_TEAL
GRAY = ps.C_GREY
BLUE_MID = ps.C_BLUE
BLUE_DARK = ps.C_BLUE_DK
BLUE_PALE = ps.C_BLUE_LT
BLUE = ps.C_BLUE_MID

# Three panels need a wider canvas than the two-panel reference; keep the
# reference height and type scale so the figures still look like a set.
FIGSIZE = (18.0, ps.FIG_H_2)
DPI = 300
LAYOUT = dict(ps.LAYOUT_2, left=0.085, right=0.965, top=0.76, wspace=0.28)

FS_TITLE = ps.SIZES_2.title
FS_AXLABEL = ps.SIZES_2.axlabel
FS_TICK = ps.SIZES_2.tick
FS_LEGEND = ps.SIZES_2.legend


@dataclass(frozen=True)
class Engine:
    key: str
    label: str
    color: str
    linestyle: object
    linewidth: float
    marker: str | None
    markersize: float
    run_file: str
    build_file: str | None = None
    baseline: bool = False


# Legend order is column-major: (Hyperspheres, Flat) | (HNSW, HNSW-B) | (IVF, IVF-B)
ENGINES: tuple[Engine, ...] = (
    Engine("hyperspheres", "Hyperspheres-Aggressive", GREEN, "-", 2.8, None, 0,
           "hyperspheres-aggressive-0.9-170-gist1m.json"),
    Engine("flat", "FAISS-Flat", GRAY, (0, (1.2, 1.7)), 2.2, None, 0,
           "faiss-flat-gist1m.json", baseline=True),
    Engine("hnsw", "FAISS-HNSW", BLUE_MID, "-", 2.2, None, 0,
           "faiss-hnsw-1000-gist1m.json"),
    Engine("hnsw_batch", "FAISS-HNSW-Batch", BLUE_DARK, (0, (5.2, 2.0)), 2.5,
           "s", 9.0, "faiss-hnsw-1000-gist1m-batched.json"),
    Engine("ivf", "FAISS-IVFFlat", BLUE_PALE, "-", 2.2, None, 0,
           "faiss-ivf-180-gist1m.json", build_file="faiss-ivf-build.json"),
    Engine("ivf_batch", "FAISS-IVFFlat-Batch", BLUE, (0, (5.2, 2.0)), 2.5,
           "^", 9.5, "faiss-ivf-180-gist1m-batched.json",
           build_file="faiss-ivf-build.json"),
)


@dataclass(frozen=True)
class Cost:
    """All costs in seconds."""

    ingest: float          # I_e  one-time ingestion + training + build
    save: float            # one-time index write (storage)
    load: float            # L_e  per-session load
    query: float           # S_e  per query


def read_json(name: str) -> dict:
    with (HERE / name).open(encoding="utf-8") as fh:
        return json.load(fh)


def costs_of(engine: Engine) -> Cost:
    """I_e / save / load / query for one engine.

    The IVF runs reuse an index built earlier, so their own JSON reports
    train_ms = build_ms = save_ms = 0.  The one-time cost is taken from the
    companion faiss-ivf-build.json in that case.  Load time is always taken
    from the run itself (cold load in a fresh process).
    """
    run = read_json(engine.run_file)
    build = read_json(engine.build_file) if engine.build_file else None

    def sec(rec: dict, *keys: str) -> float:
        return sum(float(rec.get(k, 0.0)) for k in keys) / 1000.0

    if engine.baseline:
        return Cost(0.0, 0.0, 0.0, float(run["avg_query_ms"]) / 1000.0)

    source = build if build is not None else run
    ingest = sec(source, "train_ms", "build_ms", "read_ms")
    save = sec(source, "save_ms")
    load = sec(run, "load_ms") or sec(source, "load_ms")
    query = float(run["avg_query_ms"]) / 1000.0
    return Cost(ingest, save, load, query)


def lifetime_cost(cost: Cost, n: np.ndarray) -> np.ndarray:
    """K = 1: I_e + N * S_e (the left panel)."""
    return cost.ingest + n * cost.query


def session_cost(cost: Cost, per_session: np.ndarray, total: float) -> np.ndarray:
    """Fixed total workload: I_e + Save_e + K * L_e + total * S_e."""
    sessions = total / per_session
    return cost.ingest + cost.save + sessions * cost.load + total * cost.query


# --------------------------------------------------------------------------- #
#  Latency-recall data
# --------------------------------------------------------------------------- #
def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left envelope of a recall/latency cloud.

    Hyperspheres has one point per ``(beta, pages_to_probe)`` pair, so several
    configurations can share a recall.  Walking from high recall to low and
    keeping a point only when it is faster than everything already seen leaves
    the best latency achievable at each recall; by construction the result is
    monotone and can be drawn as one curve.
    """
    best = float("inf")
    front: list[tuple[float, float]] = []
    for recall, latency in sorted(points, key=lambda p: (-p[0], p[1])):
        if latency < best - 1e-12:
            front.append((recall, latency))
            best = latency
    return sorted(front)


def hyperspheres_betas() -> list[float]:
    """Distinct Hyperspheres beta values that carry a query result."""
    folder, mode = LATENCY_DIRS["hyperspheres"]
    betas: set[float] = set()
    for path in sorted((HERE / LATENCY_DIR / folder).glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("mode") != mode or "beta" not in doc:
            continue
        if "avg_query_ms" not in doc or "recall_at_k" not in doc:
            continue
        betas.add(round(float(doc["beta"]), 6))
    return sorted(betas)


def load_latency_curve(engine: Engine,
                       beta: float | None = None) -> list[tuple[float, float]]:
    """Read one system's sweep and return (recall %, avg latency ms) points.

    With ``beta`` set, Hyperspheres uses only that fixed beta (one smooth
    pages_to_probe sweep); otherwise it uses the Pareto lower envelope over the
    whole beta x pages_to_probe grid.  Other systems ignore ``beta``.
    """
    folder, mode = LATENCY_DIRS[engine.key]
    data_dir = HERE / LATENCY_DIR / folder
    points: list[tuple[float, float]] = []
    for path in sorted(data_dir.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("mode") != mode:          # stray file from another engine
            continue
        if "avg_query_ms" not in doc or "recall_at_k" not in doc:
            continue                          # build-only record
        if engine.key == "hyperspheres" and beta is not None:
            if abs(float(doc.get("beta", -1e9)) - beta) > 1e-6:
                continue
        latency = float(doc["avg_query_ms"])
        recall = 100.0 * float(doc["recall_at_k"])
        if latency > 0 and recall >= RECALL_X_MIN:
            points.append((recall, latency))
    if not points:
        print(f"warning: no latency-recall points for {engine.label} "
              f"under {data_dir}")
        return []
    return pareto_front(points)


# --------------------------------------------------------------------------- #
#  Plot helpers
# --------------------------------------------------------------------------- #
def style_axis(ax, xlim, ylim, xlabel, ylabel, xticks, yticks) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel, labelpad=12)
    ax.set_ylabel(ylabel, labelpad=8)

    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=xticks))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=yticks))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    fmt = ps.log_decade_formatter()
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

    ps.frame(ax, grid_axis="y", tick_pad=8, minor_ticks=True)
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK, pad=8)
    ax.grid(True, which="major", axis="y", color=ps.C_GRID, linewidth=1.1)


def plot_latency_panel(ax, curves: dict[str, list[tuple[float, float]]]) -> None:
    """Panel (c): average query latency against mean recall."""
    for eng in ENGINES:
        points = curves.get(eng.key)
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        marker = eng.marker
        ax.plot(
            x, y,
            color=eng.color,
            linestyle=eng.linestyle,
            linewidth=eng.linewidth,
            marker=marker,
            markersize=7.0 if marker else 0,
            markevery=(max(1, len(x) // 6) if marker else None),
            markerfacecolor=eng.color,
            markeredgecolor=eng.color,
            markeredgewidth=1.2,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=3,
        )

    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlim(RECALL_X_MIN, RECALL_X_MAX)
    ax.set_ylim(LATENCY_Y_MIN, LATENCY_Y_MAX)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1,
                                          numticks=100))
    ax.yaxis.set_major_formatter(ps.log_decade_formatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ps.frame(ax, grid_axis="y", tick_pad=8, minor_ticks=True)
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK, pad=8)
    ax.set_xlabel("Recall@100", labelpad=12)
    ax.set_ylabel("Avg. query latency (ms)", labelpad=8)
    ax.set_title("(c) Latency vs. recall", pad=22)

    # Tie panel (c) to the recall target used by panels (a) and (b).
    ax.axvline(RECALL_TARGET, color=ps.C_AXIS, lw=1.2, ls=(0, (2, 3)),
               alpha=0.5, zorder=1)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default=str(HERE / "lifecycle_gist1m_recall95.png"))
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--beta", type=float, default=None,
                        help="fixed Hyperspheres beta (default: the only beta "
                             "present in the data, else Pareto)")
    parser.add_argument("--pareto", action="store_true",
                        help="use the Pareto envelope over the beta grid "
                             "instead of one fixed beta")
    args = parser.parse_args()

    ps.apply_style(ps.SIZES_2)
    costing = {eng.key: costs_of(eng) for eng in ENGINES}
    betas = hyperspheres_betas()
    if args.pareto:
        beta = None
    elif args.beta is not None:
        beta = args.beta
        if beta not in betas:
            available = ", ".join(f"{b:g}" for b in betas) or "none"
            print(f"warning: requested beta={beta:g} not found; "
                  f"available: {available}")
            beta = betas[0] if len(betas) == 1 else None
    else:
        beta = betas[0] if len(betas) == 1 else None
        if beta is not None:
            print(f"using the only Hyperspheres beta in the data: {beta:g}")

    curves = {eng.key: load_latency_curve(eng, beta) for eng in ENGINES
              if eng.key in LATENCY_DIRS}

    fig, (ax_l, ax_r, ax_c) = plt.subplots(1, 3, figsize=FIGSIZE, dpi=args.dpi)
    fig.subplots_adjust(**LAYOUT)

    left_x = np.logspace(0, np.log10(LIFETIME_MAX), 400)
    right_x = np.logspace(np.log10(RIGHT_X_MIN), np.log10(RIGHT_X_MAX), 500)
    left_marks = np.logspace(0, np.log10(LIFETIME_MAX), 7)
    right_marks = np.logspace(np.log10(RIGHT_X_MIN), np.log10(RIGHT_X_MAX), 6)

    for eng in ENGINES:
        c = costing[eng.key]
        left_y = lifetime_cost(c, left_x)
        right_y = session_cost(c, right_x, TOTAL_QUERIES)

        for ax, x, y, marks in ((ax_l, left_x, left_y, left_marks),
                                (ax_r, right_x, right_y, right_marks)):
            ax.plot(x, y, color=eng.color, linestyle=eng.linestyle,
                    linewidth=eng.linewidth, solid_capstyle="round",
                    dash_capstyle="round", zorder=3 if not eng.baseline else 2)
            if eng.marker:
                ax.plot(marks, np.interp(np.log10(marks), np.log10(x), y),
                        linestyle="none", marker=eng.marker,
                        markersize=eng.markersize,
                        markerfacecolor=eng.color, markeredgecolor=eng.color,
                        markeredgewidth=1.2, zorder=4)

    style_axis(ax_l, (1, LIFETIME_MAX), LEFT_LIMITS,
               "Lifetime queries", "Lifecycle time (s)",
               xticks=4, yticks=4)
    style_axis(ax_r, (RIGHT_X_MIN, RIGHT_X_MAX), RIGHT_LIMITS,
               "Queries / session", "Total time (s)",
               xticks=4, yticks=5)
    ax_l.set_title(f"(a) Lifecycle cost at {RECALL_LABEL} recall", pad=22)
    ax_r.set_title(f"(b) Recurring sessions at {RECALL_LABEL} recall", pad=22)
    plot_latency_panel(ax_c, curves)

    # Fig-level legend: 2 rows x 3 columns, column-major fill.
    handles = []
    for eng in ENGINES:
        handles.append(Line2D([], [], color=eng.color, linestyle=eng.linestyle,
                              linewidth=eng.linewidth, marker=eng.marker,
                              markersize=eng.markersize, markerfacecolor=eng.color,
                              markeredgecolor=eng.color, markeredgewidth=1.2,
                              solid_capstyle="round", dash_capstyle="round"))
    ps.figure_legend(fig, handles, [eng.label for eng in ENGINES], ncol=3, y=0.99)

    fig.text(0.018, 0.5, DATASET, rotation=90, ha="center", va="center",
             fontsize=FS_TITLE, fontweight="bold", color=ps.C_TITLE)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    print(f"wrote {args.out}\nwrote {pdf}")

    # ------------------------------------------------------------------ #
    #  Text summary
    # ------------------------------------------------------------------ #
    flat = costing["flat"]
    meta = read_json(ENGINES[0].run_file)
    print(f"\n{DATASET}: {meta['count']:,} vectors, dim={meta['dim']}, k={meta['k']}, "
          f"{TOTAL_QUERIES:,} queries / lifetime, left panel K=1")
    head = (f"{'engine':24s} {'recall':>7s} {'ingest':>9s} {'save':>7s} {'load':>7s} "
            f"{'query':>8s} {'break-even':>11s} {'right@1':>11s} {'right@1e4':>11s}")
    print(head)
    print("-" * len(head))
    for eng in ENGINES:
        c = costing[eng.key]
        recall = read_json(eng.run_file).get("recall_at_k", float("nan"))
        be = "-" if eng.baseline else f"{c.ingest / (flat.query - c.query):.0f}"
        r1 = session_cost(c, np.array([1.0]), TOTAL_QUERIES)[0]
        r4 = session_cost(c, np.array([1e4]), TOTAL_QUERIES)[0]
        print(f"{eng.label:24s} {recall:7.4f} {c.ingest:8.1f}s {c.save:6.2f}s "
              f"{c.load:6.2f}s {c.query * 1e3:7.2f}ms {be:>11s} "
              f"{r1:10.0f}s {r4:10.0f}s")
    print("break-even = lifetime queries where an index beats FAISS-Flat; "
          "right@x = total time at x queries/session")

    mode = "Pareto envelope" if beta is None else f"fixed beta={beta:g}"
    print(f"\nLatency-recall curves ({mode}, recall >= {RECALL_X_MIN:.0f}%):")
    for eng in ENGINES:
        points = curves.get(eng.key)
        if not points:
            continue
        recalls = [p[0] for p in points]
        latencies = [p[1] for p in points]
        print(f"  {eng.label:<24} {len(points):3d} points  "
              f"recall {recalls[0]:5.1f}..{recalls[-1]:5.1f}%  "
              f"latency {min(latencies):6.2f}..{max(latencies):6.2f} ms")


if __name__ == "__main__":
    main()
