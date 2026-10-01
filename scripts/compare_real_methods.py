"""Run: python -m scripts.compare_real_methods --solar ... --bitcoin ... --discharge ..."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.distributions import q_gaussian
from src.empirical import METHODS, diagnostics, fit_histogram, fit_unbinned, histogram, load_increments


def main(argv=None):
    parser = argparse.ArgumentParser(description="Four centered q-Gaussian fits on real increments")
    for name in ("solar", "bitcoin", "discharge"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_empirical_50"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_increments(args.solar, args.bitcoin, args.discharge)
    rows = []
    fig, axes = plt.subplots(3, 1, figsize=(10, 13), layout="constrained")
    colors = {"pdf": "#d54b40", "mle": "#2c72aa", "cdf": "#3e9a75", "qlog": "#7854ab"}
    for ax, (name, data) in zip(axes, datasets.items()):
        x = data["x"]
        counts, edges, centers, heights, sigma = histogram(x, args.bins)
        ax.errorbar(centers, heights, yerr=sigma, fmt=".", color="#333333", ms=3,
                    label=f"Data ({args.bins} bins)")
        grid = np.linspace(np.quantile(x, .001), np.quantile(x, .999), 1200)
        for method in METHODS:
            result = (fit_histogram(x, args.bins, method) if method in ("pdf", "qlog")
                      else fit_unbinned(x, method))
            rows.append({"dataset": name, "n_raw": data["raw_n"], "n_increments": len(x),
                         "n_gaps": data["n_gaps"], "first": data["first"], "last": data["last"],
                         "transform": data["transform"], "mean_q": data["mean_q"],
                         "bins": args.bins, "occupied_bins": int((counts > 0).sum()),
                         **asdict(result), **diagnostics(x, result, args.bins)})
            if result.success:
                ax.plot(grid, q_gaussian(grid, result.b, result.q, result.mu),
                        lw=1.8, label=method.upper(), color=colors[method])
        ax.set(title=name, xlim=(grid[0], grid[-1]), ylabel="Density")
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(heights.min() / 3, 1e-5))
        ax.grid(alpha=.2)
    axes[0].legend(ncol=5, fontsize=9)
    axes[-1].set_xlabel("Increment (original scale)")
    fig.savefig(args.output_dir / "fits.png", dpi=180)
    plt.close(fig)
    with (args.output_dir / "fits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['dataset']:9} {row['method']:4} q={row['q']:.5f} "
              f"b={row['b']:.5g} mu={row['mu']:.6g} success={row['success']}")


if __name__ == "__main__":
    main()
