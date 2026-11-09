"""PSO-FMTL (M=35) vs Online FMTL, with and without Nesterov smoothing theta of the task-relation matrix.
K = 10 tasks, D = 200, mu = MU (default 0.05), 1000 rounds, held-out test MSE (dB), mean over SEEDS seeds.
Curves: Online FMTL (full sharing, 400 params), PSO-Fed M=35 (baseline of PSO-FMTL), PSO-FMTL M=35 with theta in THETAS.
Outputs: fig_multitask_coordinated.png, fig_multitask_uncoordinated.png, multitask_theta_results.json
Needs fed_simulation.py and run_all_paper_experiments.py in the same folder."""
import json, os, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import run_all_paper_experiments as ra
from fed_simulation import generate_tasks, RFFMap, run_mocha, make_client_probe
MU = float(os.environ.get("MU", 0.05)); N = int(os.environ.get("ROUNDS", 1000)); SEEDS = int(os.environ.get("SEEDS", 30))
ra.N = N; K, D, P, M = 10, 200, 4, 35; THETAS = (0.0, 1.0, 10.0)

def one(sd):
    t = generate_tasks(K, 2500, P=P, seed=sd); r = RFFMap(P, D, seed=sd); pr = make_client_probe(t, r, N, N + 300)
    o = {}; L = []; run_mocha(t, r, mu=MU, n_rounds=N, probe=(pr, L)); o["Online FMTL"] = L
    for c, nm in ((True, "c"), (False, "u")):
        L = []; ra.psofed(t, r, M=M, mu=MU, coord=c, probe=(pr, L)); o[f"PSO-Fed|{nm}"] = L
        for th in THETAS:
            L = []; ra.fmtl(t, r, M=M, mu=MU, coord=c, theta=th, probe=(pr, L)); o[f"PSO-FMTL|{nm}|{th}"] = L
    return {k: np.asarray(v) for k, v in o.items()}

if __name__ == "__main__":
    with Pool() as p: runs = p.map(one, range(SEEDS))
    keys = runs[0].keys(); mean = {k: np.mean([r[k] for r in runs], 0) for k in keys}
    fin = {k: np.array([r[k][-100:].mean() for r in runs]) for k in keys}
    json.dump({"mean_curves": {k: v.tolist() for k, v in mean.items()}, "final": {k: v.tolist() for k, v in fin.items()}},
              open("multitask_theta_results.json", "w"))
    def P_(a, b, n):
        x = fin[a] - fin[b]; print(f"{n:48s}{x.mean():+7.2f} dB  p={stats.ttest_rel(fin[a], fin[b]).pvalue:.3g}  better in {(x<0).sum()}/{len(x)}")
    print(f"mu={MU}, {SEEDS} seeds, final = mean of last 100 rounds"); print("Online FMTL %.2f +/- %.2f" % (fin['Online FMTL'].mean(), fin['Online FMTL'].std(ddof=1)))
    for nm in "cu":
        print(nm, "PSO-Fed %.2f" % fin[f"PSO-Fed|{nm}"].mean(), {th: round(float(fin[f"PSO-FMTL|{nm}|{th}"].mean()), 2) for th in THETAS})
        for th in THETAS:
            P_(f"PSO-FMTL|{nm}|{th}", "Online FMTL", f"{nm} PSO-FMTL theta={th} - Online FMTL")
            P_(f"PSO-FMTL|{nm}|{th}", f"PSO-Fed|{nm}", f"{nm} PSO-FMTL theta={th} - PSO-Fed")
        P_(f"PSO-FMTL|{nm}|1.0", f"PSO-FMTL|{nm}|0.0", f"{nm} smoothing effect theta=1 vs 0")
        P_(f"PSO-FMTL|{nm}|10.0", f"PSO-FMTL|{nm}|0.0", f"{nm} smoothing effect theta=10 vs 0")
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    for nm, name in (("c", "coordinated"), ("u", "uncoordinated")):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.plot(mean["Online FMTL"], color="#0000CC", lw=2.2, label="Online FMTL (400 params)")
        ax.plot(mean[f"PSO-Fed|{nm}"], color="#888888", lw=1.4, label="PSO-Fed, $M=35$ (baseline)")
        ax.plot(mean[f"PSO-FMTL|{nm}|0.0"], color="#E00000", lw=1.4, label=r"PSO-FMTL, $M=35$, $\theta=0$")
        ax.plot(mean[f"PSO-FMTL|{nm}|1.0"], color="#00A000", lw=1.8, ls="--", label=r"PSO-FMTL, $M=35$, $\theta=1$")
        ax.plot(mean[f"PSO-FMTL|{nm}|10.0"], color="#E08000", lw=1.4, ls=":", label=r"PSO-FMTL, $M=35$, $\theta=10$")
        ax.set_xlabel("Round index ($m$)"); ax.set_ylabel("Testing MSE (dB)"); ax.set_xlim(0, N); ax.grid(alpha=.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.3, frameon=True, edgecolor="k", fancybox=False)
        fig.tight_layout(); fig.savefig(f"fig_multitask_{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
