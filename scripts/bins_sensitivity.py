"""Sensibilidad de PDF y q-log a los bins, con muestras emparejadas.

Ejecutar desde la raiz: python -m scripts.bins_sensitivity
Para verificar rapido: python -m scripts.bins_sensitivity --repetitions 2
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from src.estimators import DEFAULT_Q_GRID, fit, histogram
from scripts.run_pilot import generate, parameter_metrics

QS = (1.5, 1.9)
BS = (2.0, 5.0)
NS = (1000, 5000)
BINS = (20, 30, 40, 50, 70, 100, 150, 200)
METHODS = ("pdf", "qlog")
DETAIL_FIELDS = ("q_true", "b_true", "n", "rep", "seed", "bins", "method", "q_hat",
                 "b_hat", "success", "at_bound", "occupied_bins", "max_abs_x", "seconds", "message")
SUMMARY_FIELDS = ("q_true", "b_true", "n", "bins", "method", "repetitions", "valid",
                  "failures", "failure_rate", "at_bound", "mean_occupied_bins",
                  "mean_q_hat", "median_q_hat", "bias_q", "mcse_bias_q", "sd_q", "rmse_q",
                  "mcse_rmse_q", "mean_b_hat", "bias_b", "mcse_bias_b", "sd_b", "rmse_b",
                  "mcse_rmse_b", "mean_seconds")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["q_true"], row["b_true"], row["n"], row["bins"], row["method"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (q, b, n, bins, method), group in sorted(groups.items()):
        valid = [r for r in group if r["success"] and np.isfinite(r["q_hat"])
                 and np.isfinite(r["b_hat"])]
        qm = parameter_metrics(valid, "q_hat", q) if valid else (np.nan,) * 5
        bm = parameter_metrics(valid, "b_hat", b) if valid else (np.nan,) * 5
        summary.append(dict(q_true=q, b_true=b, n=n, bins=bins, method=method,
                            repetitions=len(group), valid=len(valid), failures=len(group)-len(valid),
                            failure_rate=(len(group)-len(valid))/len(group),
                            at_bound=sum(r["at_bound"] for r in group),
                            mean_occupied_bins=np.mean([r["occupied_bins"] for r in group]),
                            mean_q_hat=np.mean([r["q_hat"] for r in valid]) if valid else np.nan,
                            median_q_hat=np.median([r["q_hat"] for r in valid]) if valid else np.nan,
                            **dict(zip(("bias_q", "mcse_bias_q", "sd_q", "rmse_q", "mcse_rmse_q"), qm)),
                            mean_b_hat=np.mean([r["b_hat"] for r in valid]) if valid else np.nan,
                            **dict(zip(("bias_b", "mcse_bias_b", "sd_b", "rmse_b", "mcse_rmse_b"), bm)),
                            mean_seconds=np.mean([r["seconds"] for r in group])))
    return summary


def plot_summary(summary, path, repetitions):
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True, sharey=True)
    color = {"pdf": "#126da8", "qlog": "#d97721"}
    for row_idx, q in enumerate(QS):
        for col_idx, (b, n) in enumerate((b, n) for b in BS for n in NS):
            ax = axs[row_idx, col_idx]
            subset = [s for s in summary if s["q_true"] == q and s["b_true"] == b and s["n"] == n]
            ax.axhline(q, color="black", lw=1.8, ls="--", label=f"q verdadero = {q}")
            for method in METHODS:
                ordered = sorted((s for s in subset if s["method"] == method), key=lambda s: s["bins"])
                bins = np.array([s["bins"] for s in ordered])
                means = np.array([s["mean_q_hat"] for s in ordered])
                sds = np.array([s["sd_q"] for s in ordered])
                ax.plot(bins, means, "o-", color=color[method], lw=2, ms=4,
                        label=f"{method.upper()} media")
                ax.fill_between(bins, means-sds, means+sds, color=color[method], alpha=.13)
                missing = np.array([s["failures"] for s in ordered])
                for x, y, f in zip(bins, means, missing):
                    if f and np.isfinite(y):
                        ax.annotate(str(f), (x, y), xytext=(0, 5), textcoords="offset points",
                                    ha="center", color=color[method], fontsize=7)
            ax.set_title(f"q={q}; b={b}; N={n}")
            ax.grid(alpha=.25)
            if row_idx == 1:
                ax.set_xlabel("Número de bins")
            if col_idx == 0:
                ax.set_ylabel("Estimación de q (media ± DE)")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    # Una leyenda por figura; los valores q verdaderos varian por fila.
    fig.legend(handles[:3], ["q verdadero", "PDF: media ± DE", "q-log: media ± DE"],
               loc="upper center", bbox_to_anchor=(.5, .955), ncol=3, frameon=False)
    fig.suptitle(f"Sensibilidad a los bins · {repetitions} muestras compartidas por escenario", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output-dir", type=Path, default=Path("results/bins_sensitivity"))
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("repetitions debe ser al menos 2 para calcular dispersion")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(q_values=QS, b_values=BS, sample_sizes=NS, bins=BINS,
                  repetitions=args.repetitions, seed=args.seed, mu=0.0,
                  methods=METHODS, qlog_grid=list(DEFAULT_Q_GRID),
                  python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    for iq, q in enumerate(QS):
        for ib, b in enumerate(BS):
            for ins, n in enumerate(NS):
                for rep in range(args.repetitions):
                    # La semilla no depende de bins ni del metodo.
                    seed = [args.seed, iq, ib, ins, rep]
                    rng = np.random.default_rng(np.random.SeedSequence(seed))
                    data = generate(rng, n, q, b)
                    max_abs_x = float(np.max(np.abs(data)))
                    for bins in BINS:
                        hist = histogram(data, bins)
                        assert hist[1].sum() == n
                        for method in METHODS:
                            start = time.perf_counter()
                            result = fit(data, method, bins=bins, hist=hist)
                            rows.append(dict(q_true=q, b_true=b, n=n, rep=rep,
                                             seed="-".join(map(str, seed)), bins=bins,
                                             method=method, q_hat=result.q, b_hat=result.b,
                                             success=result.success, at_bound=result.at_bound,
                                             occupied_bins=result.occupied_bins,
                                             max_abs_x=max_abs_x,
                                             seconds=time.perf_counter()-start,
                                             message=result.message))
                # Guardado parcial tras cada escenario.
                write_csv(args.output_dir / "detail.csv", DETAIL_FIELDS, rows)
                summary = summarize(rows)
                write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summary)
                print(f"Completado q={q}, b={b}, N={n}", flush=True)
    plot_summary(summary, args.output_dir / "q_vs_bins.png", args.repetitions)


if __name__ == "__main__":
    main()
