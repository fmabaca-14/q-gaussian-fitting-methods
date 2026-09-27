"""Evaluate empirical q-Gaussian fits on common body and tail scores."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.empirical import fit_histogram, fit_unbinned, load_increments
from src.empirical_scores import body_scores, exceedance_curve, tail_scores, tail_thresholds


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_dataset(name, data, bins=(50, 100)):
    """Return body, tail detail, tail summary, and exceedance curve rows."""
    x = data["x"]
    thresholds = tail_thresholds(x)
    configurations = [(method, count) for method in ("pdf", "qlog") for count in bins]
    configurations += [("mle", None), ("cdf", None)]
    body_rows, detail_rows, summary_rows, curve_rows = [], [], [], []
    for method, count in configurations:
        fit = fit_histogram(x, count, method) if count is not None else fit_unbinned(x, method)
        base = {"dataset": name, "method": method, "fit_bins": count or "",
                "n_increments": len(x), "n_raw": data["raw_n"], "n_gaps": data["n_gaps"],
                **asdict(fit)}
        if not fit.success:
            body_rows.append({**base, **{key: np.nan for key in
                              ("dic_dfn", "ks", "r2_hist_100", "chi2_hist_100", "mean_loglik")}})
            summary_rows.append({**base, "tail_mae_pp": np.nan})
            continue
        body_rows.append({**base, **body_scores(x, fit)})
        details, mae = tail_scores(x, fit, thresholds)
        summary_rows.append({**base, "tail_mae_pp": mae})
        detail_rows.extend({"dataset": name, "method": method, "fit_bins": count or "",
                            **row} for row in details)
        threshold_grid, observed, predicted = exceedance_curve(x, fit)
        curve_rows.extend({"dataset": name, "method": method, "fit_bins": count or "",
                           "threshold": float(u), "observed_exceedance": float(obs),
                           "predicted_exceedance": float(pred)}
                          for u, obs, pred in zip(threshold_grid, observed, predicted))
    return body_rows, detail_rows, summary_rows, curve_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare in-sample body and tail scores")
    for name in ("solar", "bitcoin", "discharge"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/empirical_scores"))
    args = parser.parse_args(argv)
    datasets = load_increments(args.solar, args.bitcoin, args.discharge)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [[], [], [], []]
    for name, data in datasets.items():
        rows = evaluate_dataset(name, data)
        for target, part in zip(all_rows, rows):
            target.extend(part)
        print(f"Evaluated {name}: {len(data['x'])} increments", flush=True)
    for filename, rows in zip(("global_evaluation.csv", "tail_threshold_errors.csv",
                               "tail_summary.csv", "exceedance_curves.csv"), all_rows):
        _write_csv(args.output_dir / filename, rows)
    config = {"body_dic": "mean[(F_model(x_i)-F_n(x_i))^2], right-continuous F_n",
              "ks": "max of left- and right-limit CDF differences at sample values",
              "r2_bins": 100, "threshold_percentiles": [1, 5, 10, 90, 95, 99],
              "tail_error": "100*(predicted-observed), left <=u, right >u",
              "fit_bins": [50, 100], "same_sample_evaluation": True}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), layout="constrained")
    for ax, name in zip(axes, datasets):
        curves = [row for row in all_rows[3] if row["dataset"] == name]
        first = next((row for row in curves), None)
        if first is None:
            continue
        observed = [r for r in curves if r["method"] == first["method"] and r["fit_bins"] == first["fit_bins"]]
        ax.plot([r["threshold"] for r in observed], [r["observed_exceedance"] for r in observed],
                "k.", markersize=2, label="Empirical")
        for method in ("pdf", "qlog", "mle", "cdf"):
            for count in ((50, 100) if method in ("pdf", "qlog") else ("",)):
                selected = [r for r in curves if r["method"] == method and r["fit_bins"] == count]
                if selected:
                    label = f"{method.upper()} {count}".strip()
                    ax.plot([r["threshold"] for r in selected],
                            [r["predicted_exceedance"] for r in selected], label=label, lw=1)
        ax.set(title=name, ylabel="P(X > threshold)", yscale="log")
        ax.grid(alpha=.2)
        ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Threshold (original increment units)")
    fig.savefig(args.output_dir / "exceedance_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
