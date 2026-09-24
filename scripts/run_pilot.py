"""Piloto reproducible: python -m scripts.run_pilot --repetitions 100.

Produce `pilot_v2_detail.csv` (una fila por ajuste), `pilot_v2_summary.csv`
(una fila por escenario/metodo) y `pilot_v2.json` (configuracion y versiones).
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

from src.distributions import q_gaussian, q_gaussian_cdf
from src.estimators import DEFAULT_Q_GRID, fit, histogram

QS = (1.4, 1.9, 2.5)
NS = (100, 2000)
METHODS = ("pdf", "mle", "qlog", "cdf")
DETAIL_COLUMNS = ("q_true", "b_true", "mu_true", "n", "rep", "method", "q_hat", "b_hat",
                  "success", "at_bound", "occupied_bins", "max_abs_x", "objective", "nfev",
                  "seconds", "message")
SUMMARY_COLUMNS = ("q_true", "b_true", "mu_true", "n", "method", "repetitions", "valid",
                   "failures", "at_bound", "failure_rate", "mean_occupied_bins", "mean_seconds",
                   "total_seconds", "bias_q", "mcse_bias_q", "sd_q", "rmse_q", "mcse_rmse_q",
                   "bias_b", "mcse_bias_b", "sd_b", "rmse_b", "mcse_rmse_b")


def generate(rng, n, q, b=2.0):
    """Student-t exactamente equivalente a la q-Gaussiana normalizada."""
    nu = (3-q)/(q-1)
    scale = 1/np.sqrt(b*(3-q))
    return t.rvs(df=nu, scale=scale, size=n, random_state=rng)


def check_distributions():
    grid = np.linspace(-20, 20, 1001)
    for q in QS:
        nu = (3-q)/(q-1)
        scale = 1/np.sqrt(2*(3-q))
        assert np.allclose(q_gaussian(grid, 2, q), t.pdf(grid, nu, scale=scale), rtol=1e-12, atol=1e-14)
        assert np.allclose(q_gaussian_cdf(grid, 2, q), t.cdf(grid, nu, scale=scale), rtol=1e-12, atol=1e-14)


def parameter_metrics(rows, field, truth):
    errors = np.array([float(row[field])-truth for row in rows])
    n = len(errors)
    bias = float(np.mean(errors))
    sd = float(np.std(errors, ddof=1)) if n > 1 else np.nan
    rmse = float(np.sqrt(np.mean(errors**2)))
    mcse_bias = sd/np.sqrt(n) if n > 1 else np.nan
    sq = errors**2
    mcse_rmse = (float(np.std(sq, ddof=1)) / (2*np.sqrt(n)*rmse)
                 if n > 1 and rmse > 0 else np.nan)
    return bias, mcse_bias, sd, rmse, mcse_rmse


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["q_true"], row["n"], row["method"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (q, n, method), group in sorted(grouped.items()):
        # Los parametros sobre limites se reportan, pero no se excluyen del
        # calculo del error: son resultados reales del procedimiento.
        valid = [r for r in group if r["success"] and np.isfinite(r["q_hat"]) and np.isfinite(r["b_hat"])]
        metrics_q = parameter_metrics(valid, "q_hat", q) if valid else (np.nan,)*5
        metrics_b = parameter_metrics(valid, "b_hat", 2.0) if valid else (np.nan,)*5
        output.append(dict(q_true=q, b_true=2.0, mu_true=0.0, n=n, method=method,
                           repetitions=len(group), valid=len(valid), failures=len(group)-len(valid),
                           at_bound=sum(r["at_bound"] for r in group),
                           failure_rate=(len(group)-len(valid))/len(group),
                           mean_occupied_bins=np.mean([r["occupied_bins"] for r in group]),
                           mean_seconds=np.mean([r["seconds"] for r in group]),
                           total_seconds=sum(r["seconds"] for r in group),
                           **dict(zip(("bias_q", "mcse_bias_q", "sd_q", "rmse_q", "mcse_rmse_q"), metrics_q)),
                           **dict(zip(("bias_b", "mcse_bias_b", "sd_b", "rmse_b", "mcse_rmse_b"), metrics_b))))
    return output


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/pilot_v2_detail.csv"))
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions debe ser positivo")
    check_distributions()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    summary_stem = (stem.name.replace("_detail", "_summary") if "_detail" in stem.name
                    else stem.name + "_summary")
    config = dict(q_values=QS, sample_sizes=NS, b=2.0, mu=0.0,
                  repetitions=args.repetitions, seed=args.seed, bins=args.bins,
                  q_grid=list(DEFAULT_Q_GRID), methods=METHODS,
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    config_stem = stem.with_name(stem.name.removesuffix("_detail"))
    config_stem.with_suffix(".json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows = []
    for scenario, (q, n) in enumerate((q, n) for q in QS for n in NS):
        for rep in range(args.repetitions):
            # Muestra comun para los cuatro metodos; corrientes independientes.
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, scenario, rep]))
            data = generate(rng, n, q)
            hist = histogram(data, args.bins)
            for method in METHODS:
                start = time.perf_counter()
                result = fit(data, method, bins=args.bins, hist=hist)
                seconds = time.perf_counter() - start
                rows.append(dict(q_true=q, b_true=2.0, mu_true=0.0, n=n, rep=rep,
                                 method=method, q_hat=result.q, b_hat=result.b,
                                 success=result.success, at_bound=result.at_bound,
                                 occupied_bins=result.occupied_bins, max_abs_x=np.max(np.abs(data)),
                                 objective=result.objective, nfev=result.nfev,
                                 seconds=seconds, message=result.message))
        # Checkpoint en cada escenario, util si se interrumpe la ejecucion.
        write_csv(args.output, DETAIL_COLUMNS, rows)
        write_csv(stem.with_name(summary_stem).with_suffix(".csv"),
                  SUMMARY_COLUMNS, summarize(rows))
        print(f"Completado: q={q}, N={n}, muestras={args.repetitions}", flush=True)


if __name__ == "__main__":
    main()
