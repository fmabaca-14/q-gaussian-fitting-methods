"""Figures 1–11 and A.1–B.4 of the manuscript."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
import numpy as np

from .fits import METHODS, cumulative_mean, density


NAMES = {"solar": "Solar wind", "bitcoin": "Bitcoin", "discharge": "River discharge"}
X_LABELS = {"solar": r"Fluctuation $dN_p$", "bitcoin": r"Log-return $r_P$",
            "discharge": r"Fluctuation $dQ$"}
LABELS = {"pdf": "Direct PDF", "qlog": "q-log", "mle": "MLE", "cdf": "CDF"}
COLORS = {"pdf": "#176e9c", "qlog": "#c16620", "mle": "#118578", "cdf": "#9c5480"}
STYLES = {"pdf": "-", "qlog": "-", "mle": "--", "cdf": ":"}
STYLE = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
         "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 8, "xtick.labelsize": 8,
         "ytick.labelsize": 8, "legend.fontsize": 7, "axes.linewidth": .65,
         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
         "ytick.right": True, "grid.linewidth": .4, "grid.alpha": .25, "pdf.fonttype": 42}


def save(fig, path):
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None}, dpi=300)
    plt.close(fig)


def _plot(ax, x, y, method, **kwargs):
    return ax.plot(x, y, color=COLORS[method], ls=STYLES[method], lw=.95,
                   label=LABELS[method], **kwargs)[0]


def time_series(series, path):
    fig, ax = plt.subplots(figsize=(170/25.4, 65/25.4), layout="constrained")
    ax.plot(series.dates, series.increments, color=COLORS["pdf"], lw=.45, alpha=.75,
            rasterized=series.name == "solar")
    ax.set(xlabel="Year", ylabel=X_LABELS[series.name], xlim=(series.dates[0], series.dates[-1]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid()
    save(fig, path)


def density_comparison(name, hist, fits, path):
    fig, ax = plt.subplots(figsize=(85/25.4, 65/25.4), layout="constrained")
    occupied = hist.counts > 0
    ax.plot(hist.centers[occupied], hist.density[occupied], ".", color=".44", ms=2.6,
            label="Empirical PDF", zorder=3)
    grid = np.linspace(hist.edges[0], hist.edges[-1], 2400)
    top = max(float(hist.density.max()), *(float(density(0., f.b, f.q)) for f in fits.values()))
    for method in METHODS:
        fit = fits[method]
        _plot(ax, grid, density(grid, fit.b, fit.q), method)
    ax.set(xlabel=X_LABELS[name], ylabel="Probability density", yscale="log",
           xlim=(hist.edges[0], hist.edges[-1]))
    ax.set_ylim(top=10*top)
    ax.grid()
    ax.legend(loc="upper center", ncol=2, frameon=False, columnspacing=.8, handlelength=1.6)
    save(fig, path)


def sensitivity(sweeps, path):
    fig, axes = plt.subplots(3, 2, figsize=(170/25.4, 180/25.4), layout="constrained")
    for row, (name, by_bins) in enumerate(sweeps.items()):
        bins = sorted(by_bins)
        for col, param in enumerate(("q", "b")):
            ax = axes[row, col]
            for method in ("pdf", "qlog"):
                _plot(ax, bins, [getattr(by_bins[k][method], param) for k in bins], method,
                      marker="o", ms=1.8, markevery=2)
            for method in ("mle", "cdf"):
                ax.axhline(getattr(by_bins[bins[0]][method], param), color=COLORS[method],
                           ls=STYLES[method], lw=.85, label=LABELS[method])
            if param == "b":
                ax.set_yscale("log")
            ax.set(title=f"({chr(97+2*row+col)}) {NAMES[name]} | ${param}$", ylabel=f"${param}$",
                   xlabel="Number of bins", xlim=(bins[0], bins[-1]), xticks=range(10, 111, 20))
            ax.grid()
            ax.legend(loc="best", ncol=2, frameon=False, columnspacing=.8, handlelength=1.5)
    save(fig, path)


def absolute_errors(name, hist, errors, path):
    bins = len(hist.counts)
    k, shown = int(np.ceil(.10*bins)), {20: 5, 50: 10, 100: 30}[bins]
    edges, centers = hist.edges, hist.centers
    fig, ax = plt.subplots(figsize=(170/25.4, 87/25.4), layout="constrained")
    for left, right in ((edges[0], edges[k]), (edges[-k-1], edges[-1])):
        ax.axvspan(left, right, color=".90", zorder=0)
    for boundary in (edges[k], edges[-k-1]):
        ax.axvline(boundary, color=".4", lw=.55, ls=":")
    for method in METHODS:
        _plot(ax, centers, errors[method], method, marker=".", ms=2.2)
    ax.set(xlabel=X_LABELS[name], ylabel="Absolute density error", xlim=(edges[0], edges[-1]),
           ylim=(0, 1.24*max(np.max(e) for e in errors.values())))
    ax.grid()
    ax.legend(loc="upper center", ncol=4, frameon=False)
    for side, position, indices, interval in (
        ("Left", [.06, .49, .33, .33], slice(0, shown), (edges[0], edges[k])),
        ("Right", [.61, .49, .33, .33], slice(-shown, None), (edges[-k-1], edges[-1])),
    ):
        iax = ax.inset_axes(position)
        iax.axvspan(*interval, color=".91", zorder=0)
        for method in METHODS:
            err = errors[method][indices]
            _plot(iax, centers[indices], np.where(err > 0, err, np.nan), method, marker=".", ms=1.8)
        span = f"1–{shown}" if side == "Left" else f"{bins-shown+1}–{bins}"
        iax.set(title=f"{side} tail (bins {span})", yscale="log")
        positive_errors = np.concatenate([v[indices][v[indices] > 0] for v in errors.values()])
        if len(positive_errors):
            lo, hi = np.log10([positive_errors.min(), positive_errors.max()])
            iax.set_ylim(10**np.floor(lo), 10**max(np.ceil(hi), np.floor(lo)+1))
        iax.xaxis.set_major_locator(MaxNLocator(3))
        iax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        iax.yaxis.set_minor_formatter(NullFormatter())
        iax.tick_params(labelsize=6, pad=1.5)
        iax.grid()
    save(fig, path)


def cumulative_errors(errors_by_name, bins, path):
    fig, axes = plt.subplots(3, 3, figsize=(170/25.4, 180/25.4))
    fig.subplots_adjust(left=.10, right=.985, bottom=.08, top=.90, wspace=.40, hspace=.49)
    k_tail = int(np.ceil(.10*bins))
    for row, (name, errors) in enumerate(errors_by_name.items()):
        for col, region in enumerate(("All", "Left", "Right")):
            ax = axes[row, col]
            length = bins if region == "All" else bins//2
            if region != "All":
                ax.axvspan(1, k_tail, color=".90", zorder=0)
                ax.axvline(k_tail, color=".5", lw=.55, ls=":")
            for method in METHODS:
                e = errors[method][::-1] if region == "Right" else errors[method]
                _plot(ax, np.arange(1, length+1), cumulative_mean(e[:length]), method)
            ticks = ([1, bins//4, bins//2, 3*bins//4, bins] if region == "All"
                     else [1, k_tail, length//2, length])
            ax.set(title=f"({chr(97+3*row+col)}) {NAMES[name]} / {region}",
                   xlim=(1, length), ylim=(0, None), xticks=ticks)
            if col == 0:
                ax.set_ylabel("Cumulative MAE")
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5,.985))
    fig.supxlabel("Number of accumulated bins, $k$", y=.025, fontsize=9)
    save(fig, path)
