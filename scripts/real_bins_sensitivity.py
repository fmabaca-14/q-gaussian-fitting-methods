"""Run: python -m scripts.real_bins_sensitivity --solar ... --bitcoin ... --discharge ..."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.empirical import fit_histogram, fit_unbinned, histogram, load_increments


def main(argv=None):
    parser = argparse.ArgumentParser(description="Centered empirical PDF and q-log bin sensitivity")
    for name in ("solar", "bitcoin", "discharge"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--bins-start", type=int, default=10)
    parser.add_argument("--bins-stop", type=int, default=200)
    parser.add_argument("--bins-step", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_empirical_bins"))
    args = parser.parse_args(argv)
    if args.bins_start < 4 or args.bins_step <= 0 or args.bins_stop < args.bins_start:
        parser.error("Invalid bin range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_increments(args.solar, args.bitcoin, args.discharge)
    rows = []
    refs = []
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), layout="constrained")
    for ax, (name, data) in zip(axes, datasets.items()):
        x = data["x"]
        references = {method: fit_unbinned(x, method) for method in ("mle", "cdf")}
        for method, result in references.items():
            refs.append({"dataset": name, "n_increments": len(x), "transform": data["transform"],
                         "mean_q": data["mean_q"], **asdict(result)})
        for bins in range(args.bins_start, args.bins_stop + 1, args.bins_step):
            counts, *_ = histogram(x, bins)
            for method in ("pdf", "qlog"):
                result = fit_histogram(x, bins, method)
                rows.append({"dataset": name, "n_increments": len(x),
                             "n_gaps": data["n_gaps"], "transform": data["transform"],
                             "mean_q": data["mean_q"], "bins": bins,
                             "occupied_bins": int(np.count_nonzero(counts)),
                             "empty_bins": int(np.count_nonzero(counts == 0)), **asdict(result)})
        for method, color in (("pdf", "#d54b40"), ("qlog", "#7854ab")):
            selected = [row for row in rows if row["dataset"] == name and row["method"] == method]
            ax.plot([r["bins"] for r in selected], [r["q"] for r in selected],
                    "o-", ms=3, lw=1.3, color=color, label=method.upper())
        for method, color, style in (("mle", "#2c72aa", "--"), ("cdf", "#3e9a75", "-.")):
            if references[method].success:
                ax.axhline(references[method].q, color=color, ls=style, lw=1.7,
                           label=f"{method.upper()}: {references[method].q:.3f}")
        ax.set(title=name, xlim=(args.bins_start, args.bins_stop), ylabel="q")
        ax.grid(alpha=.2)
        ax.legend(ncol=4, fontsize=9)
    axes[-1].set_xlabel("Number of bins")
    fig.savefig(args.output_dir / "q_vs_bins.png", dpi=180)
    plt.close(fig)
    for filename, data in (("detail.csv", rows), ("references.csv", refs)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    summary = []
    for name in datasets:
        for method in ("pdf", "qlog"):
            selected = [r for r in rows if r["dataset"] == name and r["method"] == method]
            good = [r for r in selected if r["success"]]
            q = np.array([r["q"] for r in good])
            summary.append({"dataset": name, "method": method, "n_success": len(good),
                            "n_total": len(selected), "q_min": float(q.min()) if len(q) else np.nan,
                            "q_max": float(q.max()) if len(q) else np.nan,
                            "q_sd_across_bins": float(q.std(ddof=1)) if len(q) > 1 else np.nan,
                            "b_min": min((r["b"] for r in good), default=np.nan),
                            "b_max": max((r["b"] for r in good), default=np.nan),
                            "mu_min": min((r["mu"] for r in good), default=np.nan),
                            "mu_max": max((r["mu"] for r in good), default=np.nan)})
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['dataset']:9} {row['method']:4} {row['n_success']}/{row['n_total']} "
              f"q range={row['q_min']:.4f}..{row['q_max']:.4f}")


if __name__ == "__main__":
    main()
