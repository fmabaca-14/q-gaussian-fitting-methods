"""Evaluate river discharge increments (Q[t+1]-Q[t])/mean(Q).

Run from repository root:
python -m scripts.analyze_discharge_mean_scaled --discharge "data/raw/daily-data-mean.xls" --bins 100

The source is sorted chronologically; pairs crossing missing days are excluded.
The denominator is the mean of all valid, positive daily flows BEFORE pairing.
Fits are in-sample and do not assume temporal independence for point estimates.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.empirical import fit_histogram, fit_unbinned
from src.empirical_scores import body_scores, tail_scores, tail_thresholds
from src.real_data import load_discharge
from scripts.plot_empirical_fits import plot_body, plot_tails


METHODS = ("pdf", "qlog", "mle", "cdf")


def increments(path):
    frame = load_discharge(path)
    if frame.index.has_duplicates:
        raise ValueError("Duplicate dates: select one observation per day first")
    values = frame["value"].to_numpy(dtype=float)
    if len(values) < 4 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("At least four finite, positive daily flows are required")
    mean_q = float(np.mean(values))
    consecutive = np.diff(frame.index.values) == np.timedelta64(1, "D")
    x = ((values[1:] - values[:-1]) / mean_q)[consecutive]
    if len(x) < 4:
        raise ValueError("Too few consecutive-day pairs")
    metadata = {
        "definition": "(Q[t+1]-Q[t])/mean(Q)",
        "mean_q": mean_q,
        "mean_q_units": str(frame["unit_of_measure"].iloc[0]) if "unit_of_measure" in frame else "source units",
        "raw_n": int(len(values)),
        "n_increments": int(len(x)),
        "n_missing_day_pairs": int((~consecutive).sum()),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
        "increment_mean": float(np.mean(x)),
        "increment_median": float(np.median(x)),
        "increment_std": float(np.std(x, ddof=1)),
    }
    for lag in (1, 5, 10, 30):
        metadata[f"acf_lag_{lag}"] = (
            float(np.corrcoef(x[:-lag], x[lag:])[0, 1]) if len(x) > lag + 2
            else None
        )
        metadata[f"acf_abs_lag_{lag}"] = (
            float(np.corrcoef(np.abs(x[:-lag]), np.abs(x[lag:]))[0, 1])
            if len(x) > lag + 2 else None
        )
    return x, metadata


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(x, bins):
    fits = {}
    global_rows = []
    threshold_rows = []
    fixed_thresholds = tail_thresholds(x)
    for method in METHODS:
        fit = (fit_histogram(x, bins, method) if method in ("pdf", "qlog")
               else fit_unbinned(x, method))
        if not fit.success:
            raise RuntimeError(f"{method} fit failed: {fit.message}")
        fits[method] = fit
        body = body_scores(x, fit, bins=bins)
        tails, tail_mae = tail_scores(x, fit, fixed_thresholds)
        global_rows.append({
            "method": method, "bins_fit": bins if method in ("pdf", "qlog") else "",
            "n": len(x), "b": fit.b, "q": fit.q, "mu": fit.mu,
            **body, "tail_mae_pp": tail_mae,
        })
        threshold_rows.extend({
            "method": method, "bins_fit": bins if method in ("pdf", "qlog") else "",
            **row,
        } for row in tails)
    return fits, global_rows, threshold_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discharge", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/discharge_mean_scaled"))
    args = parser.parse_args(argv)
    if args.bins < 10:
        parser.error("--bins must be at least 10")
    x, meta = increments(args.discharge)
    fits, global_rows, tail_rows = analyze(x, args.bins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta["bins"] = args.bins
    (args.output_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "body_and_fit_summary.csv", global_rows)
    write_csv(args.output_dir / "tail_threshold_errors.csv", tail_rows)
    plot_body("discharge", x, fits, args.bins,
              args.output_dir / "discharge_body.png")
    plot_tails("discharge", x, fits, args.bins,
               args.output_dir / "discharge_tails.png")
    for row in global_rows:
        print(f"{row['method']:5} b={row['b']:.6g} q={row['q']:.6f} "
              f"mu={row['mu']:.6g} KS={row['ks']:.6g} "
              f"tail_MAE={row['tail_mae_pp']:.4g} pp")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
