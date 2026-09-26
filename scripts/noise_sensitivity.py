"""Paired clean/noisy q-Gaussian Monte Carlo experiment.

Run from repository root: python -m scripts.noise_sensitivity
Noise is additive N(0, (fraction * Student-t scale)**2), independently
for each observation. Both conditions share the exact same latent sample.
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_pilot import generate, parameter_metrics
from src.estimators import fit, histogram

QS = (1.5, 1.9)
BS = (2.0, 5.0)
METHODS = ("pdf", "mle", "qlog", "cdf")
DETAIL = ("q_true", "b_true", "n", "rep", "condition", "noise_fraction", "noise_sigma",
          "method", "q_hat", "b_hat", "success", "at_bound", "occupied_bins",
          "max_abs_x", "objective", "seconds", "message")
SUMMARY = ("q_true", "b_true", "n", "condition", "method", "repetitions", "valid",
           "failures", "at_bound", "mean_q_hat", "sd_q", "bias_q", "mcse_bias_q",
           "rmse_q", "mcse_rmse_q", "mean_b_hat", "sd_b", "bias_b", "mcse_bias_b",
           "rmse_b", "mcse_rmse_b", "mean_seconds")
PAIRED = ("q_true", "b_true", "n", "method", "paired_valid", "mean_delta_q",
          "sd_delta_q", "mcse_delta_q", "mean_abs_delta_q", "mean_delta_b",
          "sd_delta_b", "mcse_delta_b", "mean_abs_delta_b")
DISAGREEMENT = ("q_true", "b_true", "n", "condition", "complete_repetitions",
                "mean_range_q", "sd_range_q", "mean_range_b", "sd_range_b",
                "mean_relative_range_b")
DISAGREEMENT_CHANGE = ("q_true", "b_true", "n", "paired_valid", "mean_delta_range_q",
                       "mcse_delta_range_q", "ci95_low_q", "ci95_high_q",
                       "mean_delta_range_b", "mcse_delta_range_b", "ci95_low_b", "ci95_high_b")


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    grouped = {}
    for row in rows:
        key = (row["q_true"], row["b_true"], row["condition"], row["method"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (q, b, condition, method), group in sorted(grouped.items()):
        good = [r for r in group if r["success"] and np.isfinite(r["q_hat"])
                and np.isfinite(r["b_hat"])]
        qm = parameter_metrics(good, "q_hat", q) if good else (np.nan,) * 5
        bm = parameter_metrics(good, "b_hat", b) if good else (np.nan,) * 5
        output.append(dict(q_true=q, b_true=b, n=group[0]["n"], condition=condition,
                           method=method, repetitions=len(group), valid=len(good),
                           failures=len(group)-len(good), at_bound=sum(r["at_bound"] for r in group),
                           mean_q_hat=np.mean([r["q_hat"] for r in good]) if good else np.nan,
                           sd_q=qm[2], bias_q=qm[0], mcse_bias_q=qm[1], rmse_q=qm[3],
                           mcse_rmse_q=qm[4],
                           mean_b_hat=np.mean([r["b_hat"] for r in good]) if good else np.nan,
                           sd_b=bm[2], bias_b=bm[0], mcse_bias_b=bm[1], rmse_b=bm[3],
                           mcse_rmse_b=bm[4], mean_seconds=np.mean([r["seconds"] for r in group])))
    return output


def comparisons(rows):
    index = {(r["q_true"], r["b_true"], r["rep"], r["condition"], r["method"]): r
             for r in rows}
    paired = []
    disagreements = []
    changes = []
    for q in QS:
        for b in BS:
            reps = sorted({r["rep"] for r in rows if r["q_true"] == q and r["b_true"] == b})
            for method in METHODS:
                pairs = [(index[q,b,rep,"clean",method], index[q,b,rep,"noisy",method])
                         for rep in reps]
                pairs = [(clean, noisy) for clean, noisy in pairs
                         if clean["success"] and noisy["success"]]
                dq = np.array([noisy["q_hat"]-clean["q_hat"] for clean,noisy in pairs])
                db = np.array([noisy["b_hat"]-clean["b_hat"] for clean,noisy in pairs])
                def stats(d):
                    if not len(d): return (np.nan,) * 4
                    sd = float(np.std(d,ddof=1)) if len(d)>1 else np.nan
                    return (float(np.mean(d)),sd,sd/np.sqrt(len(d)),float(np.mean(np.abs(d))))
                qstats, bstats = stats(dq), stats(db)
                paired.append(dict(q_true=q,b_true=b,n=rows[0]["n"],method=method,
                                   paired_valid=len(pairs),mean_delta_q=qstats[0],
                                   sd_delta_q=qstats[1],mcse_delta_q=qstats[2],
                                   mean_abs_delta_q=qstats[3],mean_delta_b=bstats[0],
                                   sd_delta_b=bstats[1],mcse_delta_b=bstats[2],
                                   mean_abs_delta_b=bstats[3]))
            for condition in ("clean","noisy"):
                ranges_q, ranges_b = [], []
                for rep in reps:
                    fits = [index[q,b,rep,condition,method] for method in METHODS]
                    if not all(f["success"] for f in fits):
                        continue
                    ranges_q.append(np.ptp([f["q_hat"] for f in fits]))
                    ranges_b.append(np.ptp([f["b_hat"] for f in fits]))
                disagreements.append(dict(q_true=q,b_true=b,n=rows[0]["n"],condition=condition,
                                          complete_repetitions=len(ranges_q),
                                          mean_range_q=np.mean(ranges_q) if ranges_q else np.nan,
                                          sd_range_q=np.std(ranges_q,ddof=1) if len(ranges_q)>1 else np.nan,
                                          mean_range_b=np.mean(ranges_b) if ranges_b else np.nan,
                                          sd_range_b=np.std(ranges_b,ddof=1) if len(ranges_b)>1 else np.nan,
                                          mean_relative_range_b=np.mean(ranges_b)/b if ranges_b else np.nan))
            delta_range_q, delta_range_b = [], []
            for rep in reps:
                clean = [index[q,b,rep,"clean",method] for method in METHODS]
                noisy = [index[q,b,rep,"noisy",method] for method in METHODS]
                if not all(r["success"] for r in clean + noisy):
                    continue
                delta_range_q.append(np.ptp([r["q_hat"] for r in noisy])
                                     - np.ptp([r["q_hat"] for r in clean]))
                delta_range_b.append(np.ptp([r["b_hat"] for r in noisy])
                                     - np.ptp([r["b_hat"] for r in clean]))
            def paired_stats(values):
                if len(values) < 2: return (np.nan,)*4
                mean = float(np.mean(values))
                se = float(np.std(values,ddof=1)/np.sqrt(len(values)))
                critical = float(student_t.ppf(.975,len(values)-1))
                return mean,se,mean-critical*se,mean+critical*se
            qc,bc = paired_stats(delta_range_q),paired_stats(delta_range_b)
            changes.append(dict(q_true=q,b_true=b,n=rows[0]["n"],paired_valid=len(delta_range_q),
                                mean_delta_range_q=qc[0],mcse_delta_range_q=qc[1],
                                ci95_low_q=qc[2],ci95_high_q=qc[3],
                                mean_delta_range_b=bc[0],mcse_delta_range_b=bc[1],
                                ci95_low_b=bc[2],ci95_high_b=bc[3]))
    return paired, disagreements, changes


def plot_disagreement(disagreement, path, repetitions):
    fig, axes = plt.subplots(1,2,figsize=(11,4.6))
    scenarios = [(q,b) for q in QS for b in BS]
    labels = [f"q={q}, b={b:g}" for q,b in scenarios]
    x = np.arange(len(scenarios))
    for ax,field,title in ((axes[0],"mean_range_q","Rango entre métodos de q"),
                           (axes[1],"mean_relative_range_b","Rango de b / b verdadero")):
        for offset,condition,color in ((-.19,"clean","#277da1"),(.19,"noisy","#e07a29")):
            values = [next(r[field] for r in disagreement if r["q_true"]==q and
                           r["b_true"]==b and r["condition"]==condition) for q,b in scenarios]
            ax.bar(x+offset,values,width=.36,label="Limpia" if condition=="clean" else "Ruido 10%",
                   color=color)
        ax.set_xticks(x,labels,rotation=25,ha="right")
        ax.set_title(title)
        ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Separación media de los cuatro métodos · {repetitions} muestras pareadas")
    fig.tight_layout()
    fig.savefig(path,dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions",type=int,default=10)
    parser.add_argument("--n",type=int,default=2000)
    parser.add_argument("--bins",type=int,default=50)
    parser.add_argument("--noise-fraction",type=float,default=.10)
    parser.add_argument("--seed",type=int,default=20260926)
    parser.add_argument("--output-dir",type=Path,default=Path("results/noise_10pct"))
    args = parser.parse_args()
    if args.repetitions < 2 or args.n < 3 or args.noise_fraction < 0:
        parser.error("repetitions >=2, n >=3 y noise-fraction >=0")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config = dict(q_values=QS,b_values=BS,n=args.n,mu=0.0,bins=args.bins,
                  repetitions=args.repetitions,noise_fraction=args.noise_fraction,
                  noise_definition="additive iid Gaussian; SD = fraction / sqrt(b*(3-q))",
                  seed=args.seed,methods=METHODS,python=platform.python_version(),
                  numpy=np.__version__,scipy=scipy.__version__)
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    rows = []
    for iq,q in enumerate(QS):
        for ib,b in enumerate(BS):
            scale = 1/np.sqrt(b*(3-q))
            sigma = args.noise_fraction*scale
            for rep in range(args.repetitions):
                clean = generate(np.random.default_rng(np.random.SeedSequence([args.seed,iq,ib,rep,0])),args.n,q,b)
                epsilon = np.random.default_rng(np.random.SeedSequence([args.seed,iq,ib,rep,1])).normal(0,sigma,args.n)
                for condition,data in (("clean",clean),("noisy",clean+epsilon)):
                    hist = histogram(data,args.bins)
                    for method in METHODS:
                        start = time.perf_counter()
                        result = fit(data,method,bins=args.bins,hist=hist)
                        rows.append(dict(q_true=q,b_true=b,n=args.n,rep=rep,condition=condition,
                                         noise_fraction=args.noise_fraction,noise_sigma=sigma,
                                         method=method,q_hat=result.q,b_hat=result.b,
                                         success=result.success,at_bound=result.at_bound,
                                         occupied_bins=result.occupied_bins,max_abs_x=np.max(np.abs(data)),
                                         objective=result.objective,seconds=time.perf_counter()-start,
                                         message=result.message))
            print(f"Completado q={q}, b={b}",flush=True)
    write_csv(args.output_dir/"detail.csv",DETAIL,rows)
    write_csv(args.output_dir/"summary.csv",SUMMARY,summarise(rows))
    paired,disagreement,changes = comparisons(rows)
    write_csv(args.output_dir/"paired_changes.csv",PAIRED,paired)
    write_csv(args.output_dir/"disagreement.csv",DISAGREEMENT,disagreement)
    write_csv(args.output_dir/"disagreement_change.csv",DISAGREEMENT_CHANGE,changes)
    plot_disagreement(disagreement,args.output_dir/"disagreement.png",args.repetitions)


if __name__ == "__main__":
    main()
