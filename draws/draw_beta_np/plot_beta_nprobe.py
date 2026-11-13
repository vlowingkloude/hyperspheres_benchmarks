#!/usr/bin/env python3
"""Draw the beta sweeps and the MRL cosine statistics as one figure.

Inputs are the benchmark JSONs in two folders:

    beta/    hyperspheres-aggressive--0.25-0-mrl_1024.json   (beta sweep, nprobe=0)
    nprobe/  hyperspheres-aggressive-1-125-mrl_1024.json     (nprobe sweep, beta=1)

The file name is ambiguous for negative betas ("--0.25-0" is beta=-0.25, nprobe=0),
so every value is read from the JSON fields, not from the name.

Panels
    (a) recall distribution vs. beta  -- mean / P25 / worst case
        recall_at_k, recall_p25_at_k, recall_min
    (b) pruning rate and latency vs. beta, dual y-axis
        pruning_rate on the left, avg_query_ms on the right
    (c) cosine statistics vs. M / D, two stacked rows
        top:    mean cosine for real queries and random directions,
                with P25/median/P75 spread
        bottom: (D-M) * Var(cos), linear; random directions converge to 1,
                real queries start above 1 and decay toward it.  Other
                options: --hs-bottom variance (raw variance, x1e-3) or
                --hs-bottom scaled ((D-M) * mean(cos^2), log y).
    (d) recall distribution and latency vs. nprobe
        pages_to_probe, 0 meaning "probe every one of n_database_pages pages"

The panel (c) numbers are computed by hs_geometry.py from read_hs_file.py's
inputs.  The first run computes and caches hs_geometry.npz; later runs load it.
Use --recompute-hs to rebuild the cache.

Usage
    ./env/bin/python plot_beta_nprobe.py                  # PNG + PDF, from ./beta ./nprobe
    ./env/bin/python plot_beta_nprobe.py --out figs/beta_nprobe --formats png,pdf,svg
    ./env/bin/python plot_beta_nprobe.py --annotate none  # no value labels
    ./env/bin/python plot_beta_nprobe.py --mark-beta 1.0  # vertical guide in (a)-(b)
    ./env/bin/python plot_beta_nprobe.py --recompute-hs   # rebuild panel (c) cache
"""

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
import numpy as np
from matplotlib.ticker import (FuncFormatter, LogLocator, MultipleLocator,
                               NullFormatter)

# Shared reference style (palette, fonts, axes recipe) and panel (c) cache.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(HERE))
import plot_style as ps  # noqa: E402
import hs_geometry as hsg  # noqa: E402

C_TEAL = ps.C_TEAL      # mean recall / primary series
C_ORANGE = ps.C_ORANGE  # worst case / secondary highlight
C_BLUE = ps.C_BLUE      # P25
C_SLATE = ps.C_SLATE    # latency, secondary axis
C_TITLE = ps.C_TITLE
C_INK = ps.C_INK
C_AXIS = ps.C_AXIS
C_GRID = ps.C_GRID

FIG_W, FIG_H = 18.0, ps.FIG_H_2
LEFT, RIGHT = 0.070, 0.930       # figure fractions used by the axes box
TOP, BOTTOM = 0.800, 0.200
# Three columns: (a), (b), and the stacked (c).  The third is wider because it
# carries two rows; gaps are fractions of the base panel width.
WIDTHS = (1.0, 1.0, 1.28, 1.0)
GAPS = (0.45, 0.78, 0.50)
C_HS_REAL = C_TEAL
C_HS_RANDOM = C_ORANGE

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
def plot_hs_panel(ax_top, ax_bot, hs: dict, bottom: str = "scaled_variance") -> None:
    """Panel (c): cosine mean/spread and scaled second moment vs. M/D."""
    x = hs["x"]
    groups = (
        (hs["real"], C_HS_REAL, "Real query", "o"),
        (hs["random"], C_HS_RANDOM, "Random", "s"),
    )

    bottom_curves = []
    for data, color, label, marker in groups:
        mean = data[:, :, 0].mean(axis=1)
        p25, median, p75 = np.percentile(data[:, :, 0], [25, 50, 75], axis=1)
        lower = np.maximum(mean - p25, 0.0)
        upper = np.maximum(p75 - mean, 0.0)
        ax_top.errorbar(x, mean, yerr=[lower, upper], fmt=marker + "-",
                        color=color, ms=4.0, lw=2.0,
                        ecolor=color, elinewidth=1.0, capsize=2.4,
                        capthick=1.0, alpha=0.95, label=label, zorder=4)
        ax_top.plot(x, median, color=color, lw=1.1, ls=(0, (4, 2)),
                    alpha=0.9, zorder=3)
        if bottom == "variance":
            y_bot = data[:, :, 1].mean(axis=1) * 1e3
        elif bottom == "scaled_variance":
            y_bot = data[:, :, 1].mean(axis=1) * (1.0 - x) * hs["dim"]
        else:
            y_bot = data[:, :, 2].mean(axis=1)
        bottom_curves.append(y_bot)
        ax_bot.plot(x, y_bot, marker=marker, ms=4.0, lw=2.0,
                    color=color, zorder=3)

    ax_top.axhline(0, color=C_AXIS, lw=1.0, ls=(0, (2, 3)),
                   alpha=0.7, zorder=1)
    ax_top.set_xlim(-0.02, 1.0)
    ax_top.set_ylim(-0.02, 0.52)
    ax_top.xaxis.set_major_locator(MultipleLocator(0.25))
    ax_top.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.set_ylabel("Mean cosine", fontsize=FS_AXLABEL - 3.0, labelpad=4)
    ax_top.set_title("(c) Cosine statistics vs. M / D", pad=16)
    ax_top.tick_params(axis="both", labelsize=FS_TICK - 1.0)
    ax_top.legend(loc="upper right", frameon=False,
                  fontsize=FS_LEGEND - 2.5, handlelength=1.8,
                  borderaxespad=0.2)
    ax_top.text(0.98, 0.06, "bars: P25-P75   dashed: median",
                transform=ax_top.transAxes, ha="right", va="bottom",
                fontsize=FS_ANNOT - 3.5, color=C_AXIS)
    ps.frame(ax_top, grid_axis="y", tick_pad=6)

    ax_bot.set_xlim(-0.02, 1.0)
    ax_bot.xaxis.set_major_locator(MultipleLocator(0.25))
    ax_bot.set_xlabel("M / D", fontsize=FS_AXLABEL - 2.5, labelpad=8)
    if bottom == "variance":
        y_max = max(max(curve) for curve in bottom_curves)
        ax_bot.set_ylim(0, y_max * 1.12)
        ax_bot.yaxis.set_major_locator(MultipleLocator(2))
        ax_bot.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax_bot.tick_params(axis="both", labelsize=FS_TICK - 1.0)
        ps.frame(ax_bot, grid_axis="y", tick_pad=6)
        ax_bot.set_ylabel("Variance (×10⁻³)", fontsize=FS_AXLABEL - 3.0,
                          labelpad=4)
    elif bottom == "scaled_variance":
        y_max = max(max(curve) for curve in bottom_curves)
        ax_bot.set_ylim(0, y_max * 1.12)
        ax_bot.yaxis.set_major_locator(MultipleLocator(1))
        ax_bot.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
        ax_bot.axhline(1, color=C_AXIS, lw=1.0, ls=(0, (2, 3)),
                       alpha=0.7, zorder=1)
        ax_bot.tick_params(axis="both", labelsize=FS_TICK - 1.0)
        ps.frame(ax_bot, grid_axis="y", tick_pad=6)
        ax_bot.set_ylabel("Scaled variance", fontsize=FS_AXLABEL - 3.0,
                          labelpad=4)
        ax_bot.text(0.98, 0.95, r"$(D-M)\cdot\mathrm{Var}(\cos)$",
                    transform=ax_bot.transAxes, ha="right", va="top",
                    fontsize=FS_ANNOT - 3.5, color=C_INK,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=1.2))
    else:
        ax_bot.set_yscale("log")
        ax_bot.set_ylim(0.8, 200)
        ax_bot.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax_bot.yaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax_bot.yaxis.set_major_formatter(ps.log_decade_formatter())
        ax_bot.yaxis.set_minor_formatter(NullFormatter())
        ax_bot.axhline(1, color=C_AXIS, lw=1.0, ls=(0, (2, 3)),
                       alpha=0.7, zorder=1)
        ax_bot.tick_params(axis="both", labelsize=FS_TICK - 1.0)
        ps.frame(ax_bot, grid_axis="y", tick_pad=6, minor_ticks=True)
        ax_bot.set_ylabel("Scaled 2nd moment", fontsize=FS_AXLABEL - 3.0,
                          labelpad=4)
        ax_bot.text(0.98, 0.95, r"$(D-M)\cdot\mathrm{mean}(\cos^2)$",
                    transform=ax_bot.transAxes, ha="right", va="top",
                    fontsize=FS_ANNOT - 3.5, color=C_INK,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=1.2))


def plot(series: dict, hs: dict, args) -> plt.Figure:
    beta, nprobe = series["beta"], series["nprobe"]
    sb, sn = series["beta_s"], series["nprobe_s"]
    xb = list(range(len(beta)))
    xn = list(range(len(nprobe)))
    k = series["k"]
    bx = [beta_label(b) for b in series["beta_x"]]
    nx = [nprobe_label(n, series["n_pages"]) for n in series["nprobe_x"]]

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    base = (RIGHT - LEFT) / (sum(WIDTHS) + sum(GAPS))
    x0 = LEFT
    ax1 = fig.add_axes([x0, BOTTOM, base, TOP - BOTTOM])
    x0 += base * (1.0 + GAPS[0])
    ax2 = fig.add_axes([x0, BOTTOM, base, TOP - BOTTOM])
    x0 += base * (1.0 + GAPS[1])
    cw = base * WIDTHS[2]
    vgap = 0.04
    ch = (TOP - BOTTOM - vgap) / 2.0
    axc_t = fig.add_axes([x0, BOTTOM + ch + vgap, cw, ch])
    axc_b = fig.add_axes([x0, BOTTOM, cw, ch], sharex=axc_t)
    x0 += cw + base * GAPS[2]
    axd = fig.add_axes([x0, BOTTOM, base, TOP - BOTTOM])

    def guide(ax):
        if args.mark_beta is None:
            return
        i = min(xb, key=lambda i: abs(series["beta_x"][i] - args.mark_beta))
        ax.axvline(i, color=C_AXIS, lw=1.2, ls=(0, (2, 3)),
                   alpha=0.5, zorder=1)

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
        tag(ax1, xb[i_knee], sb["worst"][i_knee],
            f'{sb["worst"][i_knee]:.0f}%', 7, -17, va="top")

    # ---- (b) pruning rate and latency vs. beta ------------------------------
    ax2.plot(xb, sb["prune"], marker="o", ms=10.5, lw=2.7, color=C_TEAL,
             label="Pruning rate", zorder=4)
    ax2.fill_between(xb, 0, sb["prune"], color=C_TEAL, alpha=0.12, zorder=2)
    style_axes(ax2, "(b) Pruning & latency vs. β", "Pruning rate", "β", bx)
    ax2.set_ylim(0, 112)
    ax2.yaxis.set_major_locator(MultipleLocator(20))
    guide(ax2)

    ax2r = ax2.twinx()
    ax2r.plot(xb, sb["ms"], marker="D", ms=9, lw=2.3, ls=(0, (1.2, 2.2)),
              color=C_SLATE, label="Latency", zorder=6)
    ax2r.set_ylabel("Avg. query latency (ms)", labelpad=6)
    ax2r.set_ylim(0, max(sb["ms"]) * 1.16)
    ax2r.yaxis.set_major_locator(MultipleLocator(10))
    ax2r.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax2r.tick_params(axis="y", length=5.5, width=1.4, color=C_AXIS, pad=5)
    ax2r.spines["right"].set_visible(True)
    ax2r.spines["right"].set_color(C_AXIS)
    ax2r.spines["right"].set_linewidth(1.4)
    for side in ("top", "left", "bottom"):
        ax2r.spines[side].set_visible(False)

    if args.annotate != "none":
        tag(ax2, xb[0], sb["prune"][0], f'{sb["prune"][0]:.0f}%', 7, 6)
        tag(ax2, xb[-1], sb["prune"][-1], f'{sb["prune"][-1]:.0f}%',
            -8, 2, ha="right")
        tag(ax2r, xb[-1], sb["ms"][-1], f"{sb['ms'][-1]:.1f} ms",
            -8, 8, ha="right")

    # ---- (c) cosine statistics vs. M/D --------------------------------------
    plot_hs_panel(axc_t, axc_b, hs, getattr(args, "hs_bottom", "scaled_variance"))

    # ---- (d) recall and latency vs. nprobe ---------------------------------
    axd.plot(xn, sn["mean"], marker="o", ms=10.5, lw=2.7, color=C_TEAL,
             label="Mean", zorder=4)
    axd.plot(xn, sn["p25"], marker="s", ms=9.5, lw=2.2, ls=(0, (5, 3)),
             color=C_BLUE, label="P25", zorder=3)
    axd.plot(xn, sn["worst"], marker="v", ms=11.5, lw=2.7, color=C_ORANGE,
             label="Worst case", zorder=5)
    style_axes(axd, f"(d) Recall@{k} vs. nprobe", f"Recall@{k}",
               "Pages to probe", nx)
    axd.set_ylim(0, 110)
    axd.yaxis.set_major_locator(MultipleLocator(20))
    axd_r = axd.twinx()
    axd_r.plot(xn, sn["ms"], marker="D", ms=9, lw=2.3, ls=(0, (1.2, 2.2)),
               color=C_SLATE, label="Latency", zorder=6)
    axd_r.set_ylabel("Avg. query latency (ms)", labelpad=6)
    axd_r.set_ylim(0, max(sn["ms"]) * 1.30)
    axd_r.yaxis.set_major_locator(MultipleLocator(5))
    axd_r.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    axd_r.tick_params(axis="y", length=5.5, width=1.4, color=C_AXIS, pad=5)
    axd_r.spines["right"].set_visible(True)
    axd_r.spines["right"].set_color(C_AXIS)
    axd_r.spines["right"].set_linewidth(1.4)
    for side in ("top", "left", "bottom"):
        axd_r.spines[side].set_visible(False)
    if args.annotate == "all":
        annotate_plateaus(axd, xn, sn["worst"], min_delta=5.0,
                          which={0, len(xn) - 1})

    handles = (ax1.get_legend_handles_labels()[0]
               + ax2.get_legend_handles_labels()[0]
               + ax2r.get_legend_handles_labels()[0])
    labels = ["Mean", "P25", "Worst case",
              "Pruning rate (left axis)", "Latency (right axis)"]
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False,
               handlelength=2.0, handletextpad=0.7, columnspacing=2.4,
               borderaxespad=0.0, fontsize=FS_LEGEND)
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
                    help="dashed guide at this beta in panels (a)-(b); 'none' disables")
    ap.add_argument("--annotate", default="key", choices=["none", "key", "all"],
                    help="print data labels on the markers")
    ap.add_argument("--hs-cache", default=str(HERE / "hs_geometry.npz"),
                    metavar="PATH", help="cache for panel (c) statistics")
    ap.add_argument("--recompute-hs", action="store_true",
                    help="recompute the panel (c) cache from read_hs_file.py inputs")
    ap.add_argument("--hs-bottom", default="scaled_variance",
                    choices=["scaled_variance", "variance", "scaled"],
                    help="panel (c) bottom row: scaled variance (default), "
                         "raw variance, or scaled 2nd moment")
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
    hs = hsg.load(Path(args.hs_cache), args.recompute_hs)

    fig = plot(series, hs, args)
    print_table(series)
    print(f"hs: {len(hs['x'])} M/D points, "
          f"{hs['real'].shape[1]} queries per group")

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
