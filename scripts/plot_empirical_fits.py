"""Make per-dataset body PDF and two-sided tail figures for four centered fits.

Run from the repository root with the original three --solar/--bitcoin/--discharge
files. PDF and q-log use --bins (100 by default); MLE/CDF use all observations.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from src.distributions import q_gaussian
from src.empirical import fit_histogram, fit_unbinned, histogram, load_increments


METHODS = ("pdf", "qlog", "mle", "cdf")
STYLE = {
    "pdf": ("PDF directo", "#d34a3e", "-"),
    "qlog": ("q-log", "#8058a9", "--"),
    "mle": ("MLE", "#2b72ac", "-."),
    "cdf": ("CDF", "#309973", ":"),
}
TITLES = {"solar": "Viento solar", "bitcoin": "Bitcoin", "discharge": "Caudal"}


def _tail_probability(cutoff, fit, side):
    nu = (3 - fit.q) / (fit.q - 1)
    scale = 1 / np.sqrt(fit.b * (3 - fit.q))
    z = (cutoff - fit.mu) / scale
    return t.cdf(z, nu) if side == "left" else t.sf(z, nu)


def _fits(x, bins):
    fitted = {method: (fit_histogram(x, bins, method) if method in ("pdf", "qlog")
                       else fit_unbinned(x, method)) for method in METHODS}
    failed = {method: result.message for method, result in fitted.items() if not result.success}
    if failed:
        raise RuntimeError(f"Fits failed: {failed}")
    return fitted


def plot_body(name, x, fits, bins, output):
    """Linear density view of the central 1st–99th percentile interval."""
    _, edges, centers, heights, errors = histogram(x, bins)
    xmin, xmax = np.quantile(x, [.01, .99])
    grid = np.linspace(xmin, xmax, 1600)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.errorbar(centers, heights, yerr=errors, fmt="o", ms=3.5, color="#222222",
                ecolor="#777777", elinewidth=.7, capsize=1.4,
                label=f"Densidad empírica ({bins} bins)", zorder=4)
    for method, fit in fits.items():
        label, color, ls = STYLE[method]
        ax.plot(grid, q_gaussian(grid, fit.b, fit.q, fit.mu), color=color, ls=ls,
                lw=2.1, label=f"{label}  ·  q={fit.q:.3f}")
    ax.set(xlim=(xmin, xmax), ylim=(0, None), xlabel="Incremento (escala original)",
           ylabel="Densidad", title=f"{TITLES[name]} · cuerpo de la distribución")
    ax.grid(alpha=.18)
    ax.legend(fontsize=9, ncol=2)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_tails(name, x, fits, bins, output):
    """Log-density and exceedance views of each tail from its 10% threshold."""
    _, edges, centers, heights, errors = histogram(x, bins)
    median = float(np.median(x))
    sorted_x = np.sort(x)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for column, side in enumerate(("left", "right")):
        magnitude = median - x if side == "left" else x - median
        threshold = float(np.quantile(magnitude, .9))
        max_u = float(magnitude.max())
        grid = np.linspace(threshold, max_u, 1200)
        ax_pdf, ax_exc = axes[:, column]

        # The histogram has equal-width bins over the *whole* sample; no tail
        # renormalization or additional fit is performed here.
        distances = median - centers if side == "left" else centers - median
        mask = distances >= threshold
        ax_pdf.errorbar(distances[mask], heights[mask], yerr=errors[mask],
                        fmt="o", ms=3.5, color="#222222", ecolor="#777777",
                        elinewidth=.7, capsize=1.4, label=f"PDF empírica ({bins} bins)")

        # ECDF step: P(X<=median-u) or P(X>=median+u), divided by full N.
        sorted_u = np.sort(magnitude[magnitude >= threshold])
        observed = (len(sorted_u) - np.arange(len(sorted_u))) / len(x)
        ax_exc.step(sorted_u, observed, where="post", color="#222222",
                    lw=1.35, label="Excedencia empírica")
        for method, fit in fits.items():
            label, color, ls = STYLE[method]
            points = median - grid if side == "left" else median + grid
            ax_pdf.plot(grid, q_gaussian(points, fit.b, fit.q, fit.mu),
                        color=color, ls=ls, lw=1.9, label=label)
            ax_exc.plot(grid, _tail_probability(points, fit, side),
                        color=color, ls=ls, lw=1.9, label=label)
        for ax in (ax_pdf, ax_exc):
            ax.set(xlim=(threshold, max_u), yscale="log",
                   xlabel="Distancia a la mediana empírica")
            ax.grid(alpha=.18)
        ax_pdf.set_title("Cola izquierda" if side == "left" else "Cola derecha")
        ax_pdf.set_ylabel("Densidad")
        ax_exc.set_ylabel("Probabilidad de excedencia")
        top = max(.12, max(float(_tail_probability(
            median - threshold if side == "left" else median + threshold,
            fit, side)) for fit in fits.values()) * 1.2)
        ax_exc.set_ylim(bottom=.5 / len(x), top=top)
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"{TITLES[name]} · colas desde el umbral empírico del 10 %", fontsize=15)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot empirical q-Gaussian body and tails")
    for name in ("solar", "bitcoin", "discharge"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_empirical_figures"))
    args = parser.parse_args(argv)
    if args.bins < 10:
        parser.error("--bins must be at least 10")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in load_increments(args.solar, args.bitcoin, args.discharge).items():
        x = data["x"]
        fits = _fits(x, args.bins)
        body = args.output_dir / f"{name}_body.png"
        tails = args.output_dir / f"{name}_tails.png"
        plot_body(name, x, fits, args.bins, body)
        plot_tails(name, x, fits, args.bins, tails)
        print(f"{name}: n={len(x)}, body={body}, tails={tails}", flush=True)


if __name__ == "__main__":
    main()
