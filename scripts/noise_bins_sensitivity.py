"""Bins sensitivity under paired clean and 10% noisy q-Gaussian samples.

Run: python -m scripts.noise_bins_sensitivity
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.noise_sensitivity import QS, BS, write_csv, summarise
from scripts.run_pilot import generate
from src.estimators import fit, histogram

METHODS = ("pdf", "qlog")
REFERENCES = ("mle", "cdf")
DETAIL = ("q_true", "b_true", "n", "rep", "condition", "bins", "method",
          "q_hat", "b_hat", "success", "at_bound", "occupied_bins", "message", "seconds")
CHANGE = ("q_true", "b_true", "n", "bins", "method", "paired_valid",
          "mean_delta_q", "sd_delta_q", "mean_delta_b", "sd_delta_b")
ANALYSIS = ("q_true", "b_true", "method", "condition", "valid_at_all_bins",
            "mean_q_10", "mean_q_50", "mean_q_100", "mean_q_200",
            "span_mean_q", "mean_b_10", "mean_b_50", "mean_b_100",
            "mean_b_200", "span_mean_b", "max_failure_rate")


def plot(summary, refs, path, parameter):
    field = "mean_q_hat" if parameter == "q" else "mean_b_hat"
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    scenarios = [(q, b) for q in QS for b in BS]
    for col, (q, b) in enumerate(scenarios):
        for row, method in enumerate(METHODS):
            ax = axes[row, col]
            target = q if parameter == "q" else b
            ax.axhline(target, color="black", ls="--", lw=1.6, label="Verdadero")
            for ref, style, color in (("mle", ":", "#276a42"),
                                       ("cdf", "-.", "#67418c")):
                value = next(r[field] for r in refs if r["q_true"] == q and
                             r["b_true"] == b and r["condition"] == "noisy" and
                             r["method"] == ref)
                ax.axhline(value, color=color, ls=style, lw=1.4,
                           label=ref.upper() + " con ruido")
            for cond, color, style in (("clean", "#2584ae", "-"),
                                       ("noisy", "#df7828", "-")):
                group = sorted((s for s in summary if s["q_true"] == q and
                                s["b_true"] == b and s["method"] == method and
                                s["condition"] == cond), key=lambda s: s["bins"])
                ax.plot([s["bins"] for s in group], [s[field] for s in group],
                        style, color=color, marker="o", ms=2.5,
                        lw=1.5, label="Sin ruido" if cond == "clean" else "Ruido 10 %")
            ax.set_title(f"{method.upper()} · q={q:g}, b={b:g}")
            ax.grid(alpha=.2)
            if row == 1:
                ax.set_xlabel("Número de bins")
            if col == 0:
                ax.set_ylabel("Media de " + ("q estimado" if parameter == "q" else "b estimado"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .99), ncol=5,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--noise-fraction", type=float, default=.10)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--bins-start", type=int, default=10)
    parser.add_argument("--bins-stop", type=int, default=200)
    parser.add_argument("--bins-step", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/noise_bins_10pct"))
    args = parser.parse_args()
    bins_grid = tuple(range(args.bins_start, args.bins_stop + 1, args.bins_step))
    if (args.repetitions < 2 or args.n < 3 or args.noise_fraction < 0 or
        args.bins_step <= 0 or not bins_grid or bins_grid[-1] != args.bins_stop or
        any(v < 10 or v % 10 for v in bins_grid)):
        parser.error("repetitions >= 2, n >= 3, fraction >= 0; bins múltiplos de 10 >= 10")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(q_values=QS, b_values=BS, n=args.n, mu=0, repetitions=args.repetitions,
                  bins=bins_grid, noise_fraction=args.noise_fraction,
                  noise_definition="iid Gaussian SD = fraction / sqrt(b*(3-q))",
                  seed=args.seed, methods=METHODS, references=REFERENCES,
                  reference_bins=50, paired_with="scripts.noise_sensitivity")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, reference_rows = [], []
    for iq, q in enumerate(QS):
        for ib, b in enumerate(BS):
            sigma = args.noise_fraction / np.sqrt(b * (3 - q))
            for rep in range(args.repetitions):
                clean = generate(np.random.default_rng(
                    np.random.SeedSequence([args.seed, iq, ib, rep, 0])), args.n, q, b)
                noise = np.random.default_rng(
                    np.random.SeedSequence([args.seed, iq, ib, rep, 1])).normal(0, sigma, args.n)
                for condition, data in (("clean", clean), ("noisy", clean + noise)):
                    for method in REFERENCES:
                        result = fit(data, method, bins=50)
                        reference_rows.append(dict(q_true=q, b_true=b, n=args.n, rep=rep,
                                                   condition=condition, bins=50, method=method,
                                                   q_hat=result.q, b_hat=result.b,
                                                   success=result.success, at_bound=result.at_bound,
                                                   occupied_bins=result.occupied_bins,
                                                   message=result.message, seconds=0))
                    for bins in bins_grid:
                        hist = histogram(data, bins)
                        for method in METHODS:
                            result = fit(data, method, bins=bins, hist=hist)
                            rows.append(dict(q_true=q, b_true=b, n=args.n, rep=rep,
                                             condition=condition, bins=bins, method=method,
                                             q_hat=result.q, b_hat=result.b,
                                             success=result.success, at_bound=result.at_bound,
                                             occupied_bins=result.occupied_bins,
                                             message=result.message, seconds=0))
            print(f"Completado q={q:g}, b={b:g}", flush=True)
            write_csv(args.output_dir / "detail.csv", DETAIL, rows)
    write_csv(args.output_dir / "references_detail.csv", DETAIL, reference_rows)
    summary = []
    for bins in bins_grid:
        for s in summarise([r for r in rows if r["bins"] == bins]):
            summary.append({"bins": bins, **s})
    write_csv(args.output_dir / "summary.csv", ("bins", *tuple(summarise(rows[:1])[0])), summary)
    references = summarise(reference_rows)
    write_csv(args.output_dir / "references_summary.csv", tuple(references[0]), references)
    index = {(r["q_true"], r["b_true"], r["rep"], r["bins"], r["method"], r["condition"]): r
             for r in rows}
    changes = []
    for q in QS:
        for b in BS:
            for bins in bins_grid:
                for method in METHODS:
                    pairs = [(index[q,b,rep,bins,method,"clean"],
                              index[q,b,rep,bins,method,"noisy"])
                             for rep in range(args.repetitions)]
                    pairs = [(a,z) for a,z in pairs if a["success"] and z["success"]]
                    dq = [z["q_hat"]-a["q_hat"] for a,z in pairs]
                    db = [z["b_hat"]-a["b_hat"] for a,z in pairs]
                    changes.append(dict(q_true=q, b_true=b, n=args.n, bins=bins, method=method,
                                        paired_valid=len(pairs),
                                        mean_delta_q=np.mean(dq) if dq else np.nan,
                                        sd_delta_q=np.std(dq,ddof=1) if len(dq)>1 else np.nan,
                                        mean_delta_b=np.mean(db) if db else np.nan,
                                        sd_delta_b=np.std(db,ddof=1) if len(db)>1 else np.nan))
    write_csv(args.output_dir / "paired_changes.csv", CHANGE, changes)
    analysis = []
    for q in QS:
        for b in BS:
            for method in METHODS:
                for condition in ("clean", "noisy"):
                    sub = sorted((s for s in summary if s["q_true"] == q and
                                  s["b_true"] == b and s["method"] == method and
                                  s["condition"] == condition), key=lambda s: s["bins"])
                    keyed = {s["bins"]: s for s in sub}
                    def at(k, field):
                        return keyed[k][field] if k in keyed else np.nan
                    analysis.append(dict(q_true=q, b_true=b, method=method, condition=condition,
                                         valid_at_all_bins=all(s["valid"]==args.repetitions for s in sub),
                                         **{f"mean_{p}_{k}": at(k, f"mean_{p}_hat")
                                            for p in ("q","b") for k in (10,50,100,200)},
                                         span_mean_q=np.ptp([s["mean_q_hat"] for s in sub]),
                                         span_mean_b=np.ptp([s["mean_b_hat"] for s in sub]),
                                         max_failure_rate=max(s["failures"]/s["repetitions"] for s in sub)))
    write_csv(args.output_dir / "analysis.csv", ANALYSIS, analysis)
    for parameter in ("q", "b"):
        plot(summary, references, args.output_dir / f"{parameter}_vs_bins.png", parameter)


if __name__ == "__main__":
    main()
