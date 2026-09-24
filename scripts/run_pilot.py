"""Run: python -m scripts.run_pilot --repetitions 100 --output results/pilot.csv"""
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
from src.estimators import BOUNDS, START, fit

QS = (1.4, 1.9, 2.5)
NS = (100, 2000)
METHODS = ("pdf", "mle", "qlog", "cdf")


def generate(rng, n, q, b=2.0):
    nu = (3 - q) / (q - 1)
    scale = 1 / np.sqrt(b * (3 - q))
    return t.rvs(df=nu, scale=scale, size=n, random_state=rng)


def check_distributions():
    grid = np.linspace(-20, 20, 1001)
    for q in QS:
        nu = (3 - q) / (q - 1)
        scale = 1 / np.sqrt(2 * (3 - q))
        assert np.allclose(q_gaussian(grid, 2, q), t.pdf(grid, nu, scale=scale), rtol=1e-12, atol=1e-14)
        assert np.allclose(q_gaussian_cdf(grid, 2, q), t.cdf(grid, nu, scale=scale), rtol=1e-12, atol=1e-14)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/pilot.csv"))
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    check_distributions()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {"q_values": QS, "sample_sizes": NS, "b": 2.0, "mu": 0.0,
              "repetitions": args.repetitions, "seed": args.seed,
              "methods": METHODS, "bins": 50, "bounds": BOUNDS,
              "python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__}
    args.output.with_suffix(".json").write_text(json.dumps(config, indent=2))
    fields = ["q_true", "b_true", "mu_true", "n", "rep", "method", "q_hat", "b_hat",
              "success", "at_bound", "at_start", "occupied_bins", "objective", "nfev", "seconds", "message"]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for scenario, (q, n) in enumerate((q, n) for q in QS for n in NS):
            for rep in range(args.repetitions):
                # Index-specific seed: independent of scenario loop order.
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, scenario, rep]))
                data = generate(rng, n, q)
                occupied_bins = np.count_nonzero(np.histogram(data, bins=50)[0])
                for method in METHODS:
                    start = time.perf_counter()
                    result = fit(data, method)
                    elapsed = time.perf_counter() - start
                    bound = any(abs(v - endpoint) < 1e-4 for v, pair in
                                zip((result.q, result.b), BOUNDS) for endpoint in pair)
                    writer.writerow(dict(q_true=q, b_true=2.0, mu_true=0.0, n=n,
                                         rep=rep, method=method, q_hat=result.q,
                                         b_hat=result.b, success=result.success,
                                         at_bound=bound, at_start=np.allclose((result.q, result.b), START, rtol=0, atol=1e-8),
                                         occupied_bins=occupied_bins, objective=result.objective,
                                         nfev=result.nfev, seconds=elapsed,
                                         message=result.message))
            stream.flush()
            print(f"Completed q={q}, N={n}", flush=True)


if __name__ == "__main__":
    main()
