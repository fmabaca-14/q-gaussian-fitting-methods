"""Compare Gaussian and q-Gaussian CDF fits on three empirical datasets."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src.distributions import q_gaussian
from src.empirical import load_increments, fit_unbinned
from src.gaussian import fit_gaussian_cdf, histogram_r2
from src.real_data import load_discharge


def main(argv=None):
    parser=argparse.ArgumentParser(description="Gaussian versus q-Gaussian, both fitted to CDF")
    for name in ("solar","bitcoin","discharge"):
        parser.add_argument(f"--{name}",required=True)
    parser.add_argument("--bins",type=int,default=100)
    parser.add_argument("--discharge-transform",choices=("symmetric","difference","mean-scaled"),default="symmetric")
    parser.add_argument("--output-dir",type=Path,default=Path("results/gaussian_comparison"))
    args=parser.parse_args(argv)
    if args.bins < 2:
        parser.error("bins must be at least 2")
    data=load_increments(args.solar,args.bitcoin,args.discharge)
    if args.discharge_transform != "symmetric":
        frame=load_discharge(args.discharge)
        q=frame["value"].to_numpy(dtype=float)
        consecutive=np.diff(frame.index.values)==np.timedelta64(1,"D")
        x=np.diff(q)[consecutive]
        if args.discharge_transform == "mean-scaled":
            x=x/q.mean()
        data["discharge"]["x"]=x
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(11,12),layout="constrained")
    rows=[]
    for ax,(name,info) in zip(axes,data.items()):
        x=info["x"]
        g=fit_gaussian_cdf(x)
        qg=fit_unbinned(x,"cdf")
        if not (g.success and qg.success):
            raise RuntimeError(f"{name}: fit failed: Gaussian {g.message}; q-Gaussian {qg.message}")
        g_pdf=lambda v:norm.pdf(v,loc=g.mu,scale=g.sigma)
        q_pdf=lambda v:q_gaussian(v,qg.b,qg.q,qg.mu)
        r_g=histogram_r2(x,g_pdf,args.bins)
        r_q=histogram_r2(x,q_pdf,args.bins)
        density,edges=np.histogram(x,bins=args.bins,density=True)
        centers=(edges[:-1]+edges[1:])/2
        grid=np.unique(np.r_[np.linspace(x.min(),x.max(),2500),
                             np.linspace(np.quantile(x,.01),np.quantile(x,.99),1500)])
        ax.plot(centers[density>0],density[density>0],"k.",label="Datos")
        ax.plot(grid,g_pdf(grid),color="tab:orange",label=fr"Gaussiana: $R^2={r_g:.4f}$")
        ax.plot(grid,q_pdf(grid),color="tab:blue",label=fr"q-Gaussiana: $R^2={r_q:.4f}$, $q={qg.q:.3f}$")
        ax.set(title={"solar":"Viento solar","bitcoin":"Bitcoin","discharge":"Caudal"}[name],
               ylabel="Densidad",yscale="log",xlim=(x.min(),x.max()))
        ax.set_ylim(bottom=max(float(density[density>0].min())/10,1e-12),
                    top=3*max(float(density.max()),float(g_pdf(grid).max()),float(q_pdf(grid).max())))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        rows.append({"dataset":name,"n":len(x),"bins":args.bins,
                     "discharge_transform":args.discharge_transform if name=="discharge" else "",
                     "gaussian_mu":g.mu,"gaussian_sigma":g.sigma,"gaussian_r2":r_g,
                     "qgaussian_mu":qg.mu,"qgaussian_b":qg.b,"qgaussian_q":qg.q,"qgaussian_r2":r_q})
        print(f"{name}: Gaussian R²={r_g:.4f}; q-Gaussian R²={r_q:.4f}, q={qg.q:.4f}",flush=True)
    axes[-1].set_xlabel("Incremento")
    fig.savefig(args.output_dir/"gaussian_vs_qgaussian.png",dpi=180)
    plt.close(fig)
    with (args.output_dir/"fits.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()
