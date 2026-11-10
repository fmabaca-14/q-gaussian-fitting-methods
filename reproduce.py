"""Rebuild every empirical figure and table of the manuscript."""

import argparse
import csv
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
import pandas as pd
import scipy

from src import figures
from src.data import load_inputs
from src.fits import (Histogram, METHODS, Q_GRID, SWEEP_BINS, TABLE_BINS,
                      cumulative_mean, diagnostics, fit_direct, fit_qlog, fit_unbinned)


ROOT = Path(__file__).resolve().parent
TABLE_FILES = {20: "table_A1_20bins", 50: "table_B1_50bins", 100: "table_1_100bins"}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(path, rows, bins):
    """Write an Overleaf-ready tabular; the manuscript supplies captions."""
    scale = 100 if bins == 100 else 1
    mae_unit = r" ($10^{-2}$)" if scale == 100 else ""
    header = (r"Dataset & Method & $\hat q$ & $\hat b$ & $d_{\mathrm{CvM}}$ ($10^{-4}$) & $R^2$ & "
              + r"$\mathrm{MAE}_L$" + mae_unit + r" & $\mathrm{MAE}_R$" + mae_unit
              + r" & $\mathrm{MAE}$" + mae_unit)
    lines = [r"% Requires \usepackage{booktabs}", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             header + r" \\", r"\midrule"]
    for row in rows:
        name = figures.NAMES[row["dataset"]] if row["method"] == "pdf" else ""
        method = figures.LABELS[row["method"]]
        values = (f"{row['q']:.3f}", f"{row['b']:.2f}", f"{1e4*row['cvm']:.3f}",
                  f"{row['r2']:.4f}", *(f"{scale*row[key]:.6f}" for key in ("mae_left", "mae_right", "mae")))
        lines.append(" & ".join((name, method, *values)) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(data_dir, output_dir):
    series = load_inputs(data_dir)
    output_dir = Path(output_dir)
    plots, tables = output_dir / "figures", output_dir / "tables"
    for path in (plots, tables):
        path.mkdir(parents=True, exist_ok=True)
    sweeps, bin_rows, param_rows = {}, [], []
    summaries = {k: [] for k in TABLE_BINS}
    errors_by_bins = {k: {} for k in TABLE_BINS}

    with matplotlib.rc_context(figures.STYLE):
        for panel, (name, sample) in enumerate(series.items(), start=1):
            x = sample.increments
            sorted_x = np.sort(x)
            spread = np.std(x, ddof=1)
            print(f"{figures.NAMES[name]}: {len(x):,} increments; fitting MLE and CDF", flush=True)
            references = {method: fit_unbinned(x, method) for method in ("mle", "cdf")}
            sweeps[name] = {}
            figures.time_series(sample, plots / f"figure_{panel:02d}_{name}_fluctuations.pdf")

            print(f"{figures.NAMES[name]}: sweeping {SWEEP_BINS[0]}–{SWEEP_BINS[-1]} bins", flush=True)
            for bins in SWEEP_BINS:
                hist = Histogram.from_sample(x, bins)
                fits = {"pdf": fit_direct(hist, spread), "qlog": fit_qlog(hist), **references}
                sweeps[name][bins] = fits
                param_rows.extend({"dataset": name, "bins": bins, "method": m, "q": f.q, "b": f.b,
                                   "objective": f.objective} for m, f in fits.items())
                if bins not in TABLE_BINS:
                    continue
                errors = errors_by_bins[bins][name] = {}
                for method in METHODS:
                    score, model, err = diagnostics(sorted_x, hist, fits[method])
                    summaries[bins].append({"dataset": name, "bins": bins, "method": method, **score})
                    errors[method] = err
                    for i, (pred, e, left, right) in enumerate(zip(
                            model, err, cumulative_mean(err), cumulative_mean(err[::-1]))):
                        bin_rows.append({"dataset": name, "bins": bins, "method": method, "bin": i+1,
                                         "lower_edge": hist.edges[i], "upper_edge": hist.edges[i+1],
                                         "center": hist.centers[i], "count": hist.counts[i],
                                         "empirical_density": hist.density[i], "fitted_density": pred,
                                         "absolute_error": e, "cumulative_mae_left": left,
                                         "cumulative_mae_right": right})

                if bins == 100:
                    figures.density_comparison(name, hist, fits, plots / f"figure_{panel+3:02d}_{name}_pdf.pdf")
                    error_path = plots / f"figure_{panel+7:02d}_{name}_absolute_error.pdf"
                else:
                    appendix = "A" if bins == 20 else "B"
                    error_path = plots / f"figure_{appendix}{panel}_{name}_absolute_error_{bins}bins.pdf"
                figures.absolute_errors(name, hist, errors, error_path)

        figures.sensitivity(sweeps, plots / "figure_07_bin_sensitivity.pdf")
        for bins, filename in ((20, "figure_A4_cumulative_mae_20bins.pdf"),
                               (50, "figure_B4_cumulative_mae_50bins.pdf"),
                               (100, "figure_11_cumulative_mae_100bins.pdf")):
            figures.cumulative_errors(errors_by_bins[bins], bins, plots / filename)

    for bins, rows in summaries.items():
        write_csv(tables / (TABLE_FILES[bins] + ".csv"), rows)
        write_table(tables / (TABLE_FILES[bins] + ".tex"), rows, bins)
    write_csv(output_dir / "bin_sensitivity.csv", param_rows)
    write_csv(output_dir / "binwise_errors.csv", bin_rows)
    record = {"python": platform.python_version(), "dependencies": {
                  mod.__name__: mod.__version__ for mod in (np, scipy, pd, matplotlib)},
              "location": 0, "recenter_samples": False,
              "increments": {"solar": "2*(next-current)/(next+current)", "bitcoin": "log(next/current)",
                             "discharge": "2*(next-current)/(next+current)"},
              "qlog_selection": "Pearson discrepancy of integrated expected bin counts",
              "qlog_q_grid": Q_GRID.tolist(), "bin_sweep": list(SWEEP_BINS),
              "evaluation_bins": list(TABLE_BINS), "datasets": {k: v.metadata for k, v in series.items()}}
    (output_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote 19 figure PDFs and 3 tables (CSV and LaTeX) to {output_dir.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    try:
        run(args.data_dir, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
