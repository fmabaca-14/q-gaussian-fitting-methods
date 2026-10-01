"""Paper-size Gaussian versus q-Gaussian CDF fits on the same increments."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import norm

from src.distributions import q_gaussian
from src.empirical import load_increments, fit_unbinned
from src.gaussian import fit_gaussian_cdf, histogram_r2

LABELS = {"solar": r"Fluctuation $\delta n_p$", "bitcoin": r"Log-return $r_P$",
          "discharge": r"Fluctuation $\delta Q$"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-column Gaussian/q-Gaussian CDF comparison")
    for name in ("solar", "bitcoin", "discharge"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--discharge-transform", choices=("symmetric", "difference", "mean-scaled"),
                        default="mean-scaled")
    parser.add_argument("--free-mu", action="store_true", help="Historical control; paper fixes mu=0")
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper_gaussian_comparison"))
    args = parser.parse_args(argv)
    if args.bins < 4:
        parser.error("bins must be at least 4")
    data = load_increments(args.solar, args.bitcoin, args.discharge, args.discharge_transform)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "xtick.direction": "in", "ytick.direction": "in"})
    rows, metadata, captions = [], [], []
    for name, info in data.items():
        x = info["x"]
        g = fit_gaussian_cdf(x, centered=not args.free_mu)
        qg = fit_unbinned(x, "cdf", centered=not args.free_mu)
        if not (g.success and qg.success):
            raise RuntimeError(f"{name}: Gaussian {g.message}; q-Gaussian {qg.message}")
        gp = lambda v: norm.pdf(v, loc=g.mu, scale=g.sigma)
        qp = lambda v: q_gaussian(v, qg.b, qg.q, qg.mu)
        rg, rq = histogram_r2(x, gp, args.bins), histogram_r2(x, qp, args.bins)
        density, edges = np.histogram(x, bins=args.bins, density=True)
        centers = (edges[:-1]+edges[1:])/2
        grid = np.unique(np.r_[np.linspace(x.min(), x.max(), 4000),
                              np.linspace(np.quantile(x, .001), np.quantile(x, .999), 3000)])
        fig = plt.figure(figsize=(85/25.4, 70/25.4))
        ax = fig.add_axes([.19, .19, .77, .77])
        ax.plot(centers[density > 0], density[density > 0], linestyle="none", marker="o",
                markersize=2.4, markerfacecolor="none", markeredgecolor=".22",
                markeredgewidth=.55, label="Empirical PDF", zorder=3)
        ax.plot(grid, qp(grid), color="#0072B2", linewidth=1.3,
                label="$q$-Gaussian\n"+fr"$q={qg.q:.3f}$"+"\n"+fr"$R^2={rq:.4f}$")
        ax.plot(grid, gp(grid), color="#D55E00", linestyle="--", linewidth=1.3,
                label="Gaussian\n"+fr"$R^2={rg:.4f}$")
        ax.set(xlabel=LABELS[name], ylabel="Probability density", yscale="log",
               xlim=(x.min(), x.max()))
        ax.set_ylim(density[density > 0].min()*.35,
                    max(density.max(), qp(grid).max(), gp(grid).max())*1.65)
        ax.tick_params(which="both", top=True, right=True, width=.65)
        ax.grid(axis="y", color=".88", linewidth=.45)
        legend = ax.legend(loc="upper right" if name == "discharge" else "upper left",
                framealpha=.94, facecolor="white", edgecolor=".8", handlelength=1.6,
                borderpad=.4, labelspacing=.35)
        fig.canvas.draw()
        a = ax.get_window_extent()
        b = legend.get_window_extent(fig.canvas.get_renderer())
        if not (a.x0 <= b.x0 and a.x1 >= b.x1 and a.y0 <= b.y0 and a.y1 >= b.y1):
            raise RuntimeError(f"{name}: legend outside axes")
        stem = f"{name}_gaussian_comparison"
        fig.savefig(args.output_dir/f"{stem}.pdf", metadata={"Title": stem,
                    "Subject": "CDF fits; 85 x 70 mm; fixed mu=0" if not args.free_mu else "CDF fits; free mu"})
        fig.savefig(args.output_dir/f"{stem}.png", dpi=600)
        plt.close(fig)
        rows.append({"dataset": name, "n": len(x), "bins": args.bins,
            "transform": info["transform"], "mean_q": info["mean_q"],
            "gaussian_mu": g.mu, "gaussian_sigma": g.sigma, "gaussian_r2": rg,
            "gaussian_cdf_mse": g.objective, "qgaussian_mu": qg.mu,
            "qgaussian_b": qg.b, "qgaussian_q": qg.q, "qgaussian_r2": rq,
            "qgaussian_cdf_mse": qg.objective})
        metadata.append({"dataset": name, **{k:v for k,v in info.items() if k != "x"},
            "n": len(x), "sample_mean": float(x.mean()), "sample_median": float(np.median(x)),
            "min": float(x.min()), "max": float(x.max()), "legend_inside_axes": True})
        definition = {"solar": r"$\delta n_p=2(n_{p,t+1}-n_{p,t})/(n_{p,t+1}+n_{p,t})$",
            "bitcoin": r"$r_P=\ln(P_{t+1}/P_t)$", "discharge":
            r"$\delta Q=(Q_{t+1}-Q_t)/\langle Q\rangle$" if args.discharge_transform == "mean-scaled"
            else args.discharge_transform}[name]
        location = r"Both distributions have $\mu=0$; no sample center is subtracted." if not args.free_mu else r"Both location parameters are fitted freely."
        captions.append("% "+stem+".pdf\n"+r"\caption{Empirical PDF of "+name+
            " increments ("+str(args.bins)+" equal-width bins), "+definition+
            r", and normalized Gaussian and $q$-Gaussian models fitted to the unbinned empirical CDF. "+
            location+r" The vertical axis is logarithmic. $R^2$ is evaluated at all centers of the same histogram, including empty bins.}"+"\n")
        print(f"{name}: n={len(x)}, q={qg.q:.6f}, b={qg.b:.6g}, R2_q={rq:.5f}, R2_g={rg:.5f}", flush=True)
    with (args.output_dir/"fits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"mu_fixed": None if args.free_mu else 0, "sample_centering": False,
        "method_both_models": "midpoint ECDF least squares", "histogram_bins": args.bins,
        "figure_size_mm": [85, 70], "font_sizes_pt": {"axes":9, "ticks":8, "legend":8},
        "png_dpi": 600, "pdf_vector": True, "datasets": metadata,
        "inputs_sha256": {name: hashlib.sha256(Path(getattr(args,name)).read_bytes()).hexdigest()
                          for name in ("solar", "bitcoin", "discharge")},
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])):
                          hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__).resolve(),
                                       Path(__file__).resolve().parents[1]/"src/empirical.py",
                                       Path(__file__).resolve().parents[1]/"src/gaussian.py",
                                       Path(__file__).resolve().parents[1]/"src/distributions.py",
                                       Path(__file__).resolve().parents[1]/"src/real_data.py"]}}
    (args.output_dir/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (args.output_dir/"captions.tex").write_text("\n".join(captions), encoding="utf-8")


if __name__ == "__main__":
    main()
