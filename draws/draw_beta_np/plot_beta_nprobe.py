from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Shared reference style (palette, fonts, axes recipe).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plot_style as ps  # noqa: E402

C_TEAL = ps.C_TEAL      # mean recall / primary series
C_ORANGE = ps.C_ORANGE  # worst case / secondary highlight
C_BLUE = ps.C_BLUE      # P25
C_SLATE = ps.C_SLATE    # latency, secondary axis
C_TITLE = ps.C_TITLE
C_INK = ps.C_INK
C_AXIS = ps.C_AXIS
C_GRID = ps.C_GRID

FIG_W, FIG_H = ps.FIG_W, ps.FIG_H_4
LEFT, RIGHT = 0.076, 0.930     # figure fractions used by the axes box
TOP, BOTTOM = 0.800, 0.282
# Gap after (a), (b) and (c), as a fraction of one panel width.  The last gap is
# wider so (d)'s y-axis label and its tick labels clear panel (c)'s curves.
GAPS = (0.46, 0.46, 0.60)

FS_TITLE = ps.SIZES_4.title
FS_AXLABEL = ps.SIZES_4.axlabel
FS_TICK = ps.SIZES_4.tick
FS_LEGEND = ps.SIZES_4.legend
FS_ANNOT = ps.SIZES_4.annot


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
NAME_RE = re.compile(r"^(?P<engine>[a-z]+)-(?P<mode>[a-z]+)-"
                     r"(?P<beta>-?\d+(?:\.\d+)?)-(?P<np>\d+(?:\.\d+)?)"
                     r"(?:-mrl_(?P<dim>\d+))?\.json$")


def read_dir(path: Path) -> list[dict]:
    """Every result JSON in a folder, with the file name as a fallback source."""
    if not path.is_dir():
        return []
    docs = []
    for file in sorted(path.glob("*.json")):
        with open(file, encoding="utf-8") as fh:
            doc = json.load(fh)
        if "avg_query_ms" not in doc:
            continue
        hit = NAME_RE.match(file.name)
        if hit:                        # name only fills gaps: JSON wins
            doc.setdefault("engine", hit["engine"])
            doc.setdefault("mode", hit["mode"])
            doc.setdefault("beta", float(hit["beta"]))
            doc.setdefault("pages_to_probe", int(float(hit["np"])))
            doc.setdefault("dim", int(hit["dim"]) if hit["dim"] else None)
        doc["_source"] = str(file)
        docs.append(doc)
    return docs


def load_sweeps(beta_dir: Path, nprobe_dir: Path) -> tuple[list[dict], list[dict]]:
    beta = [d for d in read_dir(beta_dir)
            if int(d.get("pages_to_probe", 0)) == 0]          # full search, beta varies
    nprobe = [d for d in read_dir(nprobe_dir)
              if int(d.get("pages_to_probe", 0)) > 0]         # beta fixed, nprobe varies

    def dedupe(docs, key):
        """One record per x value; keep the first file name in sorted order."""
        seen: dict = {}
        for doc in docs:
            seen.setdefault(key(doc), doc)
        return list(seen.values())

    beta = sorted(dedupe(beta, lambda d: round(float(d["beta"]), 4)),
                  key=lambda d: float(d["beta"]))
    nprobe = sorted(dedupe(nprobe, lambda d: int(d["pages_to_probe"])),
                    key=lambda d: int(d["pages_to_probe"]))
    if not beta and not nprobe:
        raise SystemExit(f"no result JSON found under {beta_dir} or {nprobe_dir}")
    return beta, nprobe


def check_compatible(docs: list[dict], what: str) -> None:
    """Warn when a sweep mixes runs that cannot share one axis."""
    for field in ("engine", "mode", "dim", "k", "count", "query_count"):
        values = {d.get(field) for d in docs}
        if len(values) > 1:
            print(f"warning: {what} mixes {field}={sorted(map(str, values))}; "
                  "keeping all points anyway")


# --------------------------------------------------------------------------- #
# Series
# --------------------------------------------------------------------------- #
def recall_series(docs: list[dict]) -> dict:
    """Recall triplet in percent, plus the latency used on the second axis."""
    return {
        "mean": [100.0 * d["recall_at_k"] for d in docs],
        "p25": [100.0 * d["recall_p25_at_k"] for d in docs],
        "worst": [100.0 * d["recall_min"] for d in docs],
        "prune": [100.0 * d["pruning_rate"] for d in docs],
        "ms": [d["avg_query_ms"] for d in docs],
    }


def collect(beta: list[dict], nprobe: list[dict], args) -> dict:
    if args.beta_range:
        lo, hi = args.beta_range
        beta = [d for d in beta if lo <= float(d["beta"]) <= hi]
    if args.nprobe_max:
        nprobe = [d for d in nprobe if int(d["pages_to_probe"]) <= args.nprobe_max]
    if not beta:
        raise SystemExit("no beta results left after filtering")
    if not nprobe:
        raise SystemExit("no nprobe results left after filtering")

    check_compatible(beta, "the beta sweep")
    check_compatible(nprobe, "the nprobe sweep")

    k = int(beta[0].get("k", 100))
    n_pages = int(beta[0].get("n_database_pages") or nprobe[0].get("n_database_pages") or 0)
    return {
        "beta": beta,
        "nprobe": nprobe,
        "beta_x": [float(d["beta"]) for d in beta],
        "nprobe_x": [int(d["pages_to_probe"]) for d in nprobe],
        "beta_s": recall_series(beta),
        "nprobe_s": recall_series(nprobe),
        "k": k,
        "n_pages": n_pages,
    }


def beta_label(b: float) -> str:
    text = f"{b:g}"
    return text.replace("-", "\u2212")      # real minus sign


def nprobe_label(np_: int, n_pages: int) -> str:
    if n_pages and np_ >= n_pages:
        return f"{np_}"
    return str(np_)


# --------------------------------------------------------------------------- #
# Axes helpers
# --------------------------------------------------------------------------- #
def style_axes(ax, title: str, ylabel: str | None, xlabel: str,
               ticks: list[str] | None = None, pct_y: bool = True) -> None:
    ax.set_title(title, pad=20)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_AXIS)
        ax.spines[side].set_linewidth(1.4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=C_GRID, linewidth=1.1)
    ax.tick_params(axis="both", length=5.5, width=1.4, color=C_AXIS, pad=8)
    if pct_y:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if ticks is not None:
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_xlim(-0.42, len(ticks) - 0.58)
    ax.set_xlabel(xlabel, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=3)


def annotate_plateaus(ax, xs, ys, min_delta: float = 5.0, which=None) -> None:
    """Label a series where it moves, keeping clear of the curves above it."""
    top = ax.get_ylim()[1]
    thr = 0.5 * top
    prev = None
    for i, (x, y) in enumerate(zip(xs, ys)):
        if which is not None and i not in which:
            continue
        if prev is not None and abs(y - prev) < min_delta:
            continue
        prev = y
        if y < thr:                       # low values: label above the marker
            offset, va = (7, 7), "bottom"
        else:                             # high values: label underneath
            offset, va = (7, -7), "top"
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=offset,
                    ha="left", va=va, fontsize=FS_ANNOT, color=C_INK, zorder=7,
                    clip_on=False)


def tag(ax, x, y, text, dx, dy, ha="left", va="bottom", box: bool = False) -> None:
    """One small data label with an explicit offset in points."""
    kw = {}
    if box:                      # knock out the line running under the label
        kw["bbox"] = dict(boxstyle="square,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.85)
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va=va, fontsize=FS_ANNOT, color=C_INK, zorder=8,
                clip_on=False, **kw)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot(series: dict, args) -> plt.Figure:
    beta, nprobe = series["beta"], series["nprobe"]
    sb, sn = series["beta_s"], series["nprobe_s"]
    xb = list(range(len(beta)))
    xn = list(range(len(nprobe)))
    k = series["k"]
    bx = [beta_label(b) for b in series["beta_x"]]
    nx = [nprobe_label(n, series["n_pages"]) for n in series["nprobe_x"]]

    panel_w = (RIGHT - LEFT) / (4.0 + sum(GAPS))
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes, x0 = [], LEFT
    for i in range(4):
        axes.append(fig.add_axes([x0, BOTTOM, panel_w, TOP - BOTTOM]))
        x0 += panel_w * (1.0 + GAPS[min(i, 2)])
    ax1, ax2, ax3, ax4 = axes

    def guide(ax):
        if args.mark_beta is None:
            return
        i = min(xb, key=lambda i: abs(series["beta_x"][i] - args.mark_beta))
        ax.axvline(i, color=C_AXIS, lw=1.2, ls=(0, (2, 3)), alpha=0.5, zorder=1)

    # ---- (a) recall distribution vs. beta ----------------------------------
    ax1.plot(xb, sb["mean"], marker="o", ms=10.5, lw=2.7, color=C_TEAL,
             label="Mean", zorder=4)
    ax1.plot(xb, sb["p25"], marker="s", ms=9.5, lw=2.2, ls=(0, (5, 3)),
             color=C_BLUE, label="P25", zorder=3)
    ax1.plot(xb, sb["worst"], marker="v", ms=11.5, lw=2.7, color=C_ORANGE,
             label="Worst case", zorder=5)
    style_axes(ax1, f"(a) Recall@{k} vs. β", f"Recall@{k}", None, bx)
    ax1.set_ylim(0, 110)
    ax1.yaxis.set_major_locator(MultipleLocator(20))
    guide(ax1)
    if args.annotate == "all":
        annotate_plateaus(ax1, xb, sb["worst"], min_delta=5.0)
    elif args.annotate == "key":
        i_knee = min(xb, key=lambda i: abs(series["beta_x"][i] - 0.5))
        tag(ax1, xb[0], sb["worst"][0], f'{sb["worst"][0]:.0f}%', 8, 5)
        tag(ax1, xb[i_knee], sb["worst"][i_knee], f'{sb["worst"][i_knee]:.0f}%',
            7, -17, va="top")

    # ---- (b) pruning rate vs. beta -----------------------------------------
    ax2.plot(xb, sb["prune"], marker="o", ms=10.5, lw=2.7, color=C_TEAL, zorder=4)
    ax2.fill_between(xb, 0, sb["prune"], color=C_TEAL, alpha=0.12, zorder=2)
    style_axes(ax2, "(b) Pruning rate vs. β", "Pruning rate",
               "β", bx)
    ax2.set_ylim(0, 112)
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    guide(ax2)
    if args.annotate != "none":
        tag(ax2, xb[0], sb["prune"][0], f'{sb["prune"][0]:.0f}%', 7, 6)
        tag(ax2, xb[-1], sb["prune"][-1], f'{sb["prune"][-1]:.0f}%',
            -8, 2, ha="right")

    # ---- (c) search latency vs. beta ---------------------------------------
    ax3.plot(xb, sb["ms"], marker="o", ms=10.5, lw=2.7, color=C_TEAL, zorder=4)
    ax3.fill_between(xb, 0, sb["ms"], color=C_TEAL, alpha=0.12, zorder=2)
    style_axes(ax3, "(c) Query latency vs. β", "Avg. query latency (ms)",
               None, bx, pct_y=False)
    ax3.set_ylim(0, max(sb["ms"]) * 1.16)
    ax3.yaxis.set_major_locator(MultipleLocator(10))
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    guide(ax3)
    if args.annotate != "none":
        i = len(sb["ms"]) - 1
        tag(ax3, i, sb["ms"][i], f"{sb['ms'][i]:.1f} ms", -8, 8, ha="right")

    # ---- (d) recall and latency vs. nprobe ---------------------------------
    ax4.plot(xn, sn["mean"], marker="o", ms=10.5, lw=2.7, color=C_TEAL,
             label="Mean", zorder=4)
    ax4.plot(xn, sn["p25"], marker="s", ms=9.5, lw=2.2, ls=(0, (5, 3)),
             color=C_BLUE, label="P25", zorder=3)
    ax4.plot(xn, sn["worst"], marker="v", ms=11.5, lw=2.7, color=C_ORANGE,
             label="Worst case", zorder=5)
    xlabel = "Pages to probe"
    style_axes(ax4, f"(d) Recall@{k} vs. nprobe", f"Recall@{k}", xlabel, nx)
    ax4.set_ylim(0, 110)
    ax4.yaxis.set_major_locator(MultipleLocator(20))

    ax5 = ax4.twinx()
    ax5.plot(xn, sn["ms"], marker="D", ms=9, lw=2.3, ls=(0, (1.2, 2.2)),
             color=C_SLATE, label="Latency", zorder=6)
    ax5.set_ylabel("Avg. query latency (ms)", labelpad=6)
    ax5.set_ylim(0, max(sn["ms"]) * 1.30)
    ax5.yaxis.set_major_locator(MultipleLocator(5))
    ax5.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax5.tick_params(axis="y", length=5.5, width=1.4, color=C_AXIS, pad=5)
    ax5.spines["top"].set_visible(False)
    ax5.tick_params(axis="x", top=False)

    if args.annotate == "all":
        annotate_plateaus(ax4, xn, sn["worst"], min_delta=5.0,
                          which={0, len(xn) - 1})

    handles = ax1.get_legend_handles_labels()[0] + ax5.get_legend_handles_labels()[0]
    labels = ["Mean", "P25", "Worst case", "Latency (right axis)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4, frameon=False, handlelength=2.0, handletextpad=0.7,
               columnspacing=2.4, borderaxespad=0.0)

    return fig


# --------------------------------------------------------------------------- #
# Table / CSV
# --------------------------------------------------------------------------- #
def print_table(series: dict) -> None:
    sb, sn = series["beta_s"], series["nprobe_s"]
    print(f"{'beta':>7} {'mean%':>7} {'p25%':>7} {'worst%':>7} {'prune%':>7} {'ms':>8}")
    for i, b in enumerate(series["beta_x"]):
        print(f"{b:>7g} {sb['mean'][i]:>7.2f} {sb['p25'][i]:>7.2f} {sb['worst'][i]:>7.2f} "
              f"{sb['prune'][i]:>7.2f} {sb['ms'][i]:>8.3f}")
    print(f"\n{'nprobe':>7} {'mean%':>7} {'p25%':>7} {'worst%':>7} {'prune%':>7} {'ms':>8}")
    for i, n in enumerate(series["nprobe_x"]):
        print(f"{n:>7d} {sn['mean'][i]:>7.2f} {sn['p25'][i]:>7.2f} {sn['worst'][i]:>7.2f} "
              f"{sn['prune'][i]:>7.2f} {sn['ms'][i]:>8.3f}")


def write_csv(series: dict, path: Path) -> None:
    fields = ["sweep", "beta", "pages_to_probe", "recall_mean", "recall_p25",
              "recall_worst", "pruning_rate", "avg_query_ms"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, doc in enumerate(series["beta"]):
            s = series["beta_s"]
            writer.writerow({
                "sweep": "beta", "beta": f"{series['beta_x'][i]:g}",
                "pages_to_probe": doc.get("pages_to_probe", 0),
                "recall_mean": f"{s['mean'][i] / 100:.6f}",
                "recall_p25": f"{s['p25'][i] / 100:.6f}",
                "recall_worst": f"{s['worst'][i] / 100:.6f}",
                "pruning_rate": f"{s['prune'][i] / 100:.6f}",
                "avg_query_ms": f"{s['ms'][i]:.4f}",
            })
        for i, doc in enumerate(series["nprobe"]):
            s = series["nprobe_s"]
            writer.writerow({
                "sweep": "nprobe", "beta": f"{float(doc['beta']):g}",
                "pages_to_probe": series["nprobe_x"][i],
                "recall_mean": f"{s['mean'][i] / 100:.6f}",
                "recall_p25": f"{s['p25'][i] / 100:.6f}",
                "recall_worst": f"{s['worst'][i] / 100:.6f}",
                "pruning_rate": f"{s['prune'][i] / 100:.6f}",
                "avg_query_ms": f"{s['ms'][i]:.4f}",
            })
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beta-dir", default="beta", metavar="DIR",
                    help="beta sweep JSONs (default: ./beta)")
    ap.add_argument("--nprobe-dir", default="nprobe", metavar="DIR",
                    help="nprobe sweep JSONs (default: ./nprobe)")
    ap.add_argument("--beta-range", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"), help="keep betas within [LO, HI]")
    ap.add_argument("--nprobe-max", type=int, default=None,
                    help="drop nprobe points above this value")
    ap.add_argument("--mark-beta", default="1.0",
                    help="dashed guide at this beta in panels (a)-(c); 'none' disables")
    ap.add_argument("--annotate", default="key", choices=["none", "key", "all"],
                    help="print data labels on the markers")
    ap.add_argument("--out", default="beta_nprobe", help="output path without extension")
    ap.add_argument("--formats", default="png,pdf", help="comma separated: png,pdf,svg")
    ap.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    ap.add_argument("--csv", default=None, metavar="PATH", help="also dump the plotted values")
    args = ap.parse_args()
    if isinstance(args.mark_beta, str):
        args.mark_beta = None if args.mark_beta.lower() in ("none", "off", "") \
            else float(args.mark_beta)

    ps.apply_style(ps.SIZES_4)
    beta, nprobe = load_sweeps(Path(args.beta_dir), Path(args.nprobe_dir))
    series = collect(beta, nprobe, args)

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
