"""Coordinated / uncoordinated learning-performance figures in the layout requested for the paper:
one panel per mask type; Baseline [10] = PSO-Fed (dashed blue, M = 1, 5, 40); PSO-FMTL (solid red, M = 1, 5, 35);
Online FL = FedAvg (dashed orange); Online FMTL (solid green) = the same PSO-FMTL update with M = D (full sharing, stable clipped Omega).
All curves are simulated (K = 10 tasks, D = 200, held-out test MSE in dB, mean over SEEDS seeds), nothing is drawn by hand.
Env: MU (default 0.75), ROUNDS (500), SEEDS (30).  Env SHIFT=M uses the rotation step sigma = M for every PSO method (default sigma = 1).  Output: fig_learning_coordinated_mu*.png, fig_learning_uncoordinated_mu*.png, user_style_results_mu*.json"""
import json, os, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import run_all_paper_experiments as ra
from fed_simulation import generate_tasks, RFFMap, run_fedavg, run_mocha, make_client_probe
MU = float(os.environ.get("MU", 0.75)); N = int(os.environ.get("ROUNDS", 500)); SEEDS = int(os.environ.get("SEEDS", 30))
SH = os.environ.get("SHIFT", "1"); ra.N = N; K, D, P = 10, 200, 4; MB, MF = (1, 5, 40), (1, 5, 35)

def one(sd):
    t = generate_tasks(K, N + 1000, P=P, seed=sd); r = RFFMap(P, D, seed=sd); pr = make_client_probe(t, r, N, N + 300); o = {}
    def run(name, f):
        L = []; f((pr, L)); o[name] = L
    run("Online FL", lambda p: run_fedavg(t, r, mu=MU, n_rounds=N, probe=p))
    run("MOCHA-style", lambda p: run_mocha(t, r, mu=MU, n_rounds=N, probe=p))
    run("Online FMTL", lambda p: ra.fmtl(t, r, M=D, mu=MU, coord=False, probe=p))
    for c, nm in ((True, "c"), (False, "u")):
        for M in MB: run(f"PSO-Fed|{nm}|{M}", lambda p: ra.psofed(t, r, M=M, mu=MU, coord=c, shift=(M if SH == "M" else 1), probe=p))
        for M in MF: run(f"PSO-FMTL|{nm}|{M}", lambda p: ra.fmtl(t, r, M=M, mu=MU, coord=c, shift=(M if SH == "M" else 1), probe=p))
    return {k: np.asarray(v) for k, v in o.items()}

if __name__ == "__main__":
    with Pool() as p: runs = p.map(one, range(SEEDS))
    mean = {k: np.mean([r[k] for r in runs], 0) for k in runs[0]}
    fin = {k: np.array([r[k][-30:].mean() for r in runs]) for k in runs[0]}
    sfx = f"_mu{MU}" + (f"_N{N}" if N != 500 else "") + ("_sigmaM" if SH == "M" else ""); json.dump({"mean": {k: v.tolist() for k, v in mean.items()}, "final": {k: v.tolist() for k, v in fin.items()}}, open(f"user_style_results{sfx}.json", "w"))
    print(f"mu={MU}, {N} rounds, {SEEDS} seeds, final = mean of last 30 rounds")
    for k in fin: print(f"{k:22s}{fin[k].mean():8.2f} +/- {fin[k].std(ddof=1):.2f}")
    def P_(a, b, n):
        x = fin[a] - fin[b]; print(f"{n:50s}{x.mean():+7.2f} dB p={stats.ttest_rel(fin[a], fin[b]).pvalue:.3g} better in {(x<0).sum()}/{len(x)}")
    for nm in "cu":
        P_(f"PSO-FMTL|{nm}|35", "Online FMTL", f"{nm}: PSO-FMTL M=35 - Online FMTL"); P_(f"PSO-Fed|{nm}|40", "Online FL", f"{nm}: PSO-Fed M=40 - Online FL")
        P_(f"PSO-FMTL|{nm}|35", f"PSO-Fed|{nm}|40", f"{nm}: PSO-FMTL M=35 - PSO-Fed M=40"); P_(f"PSO-FMTL|{nm}|35", f"PSO-Fed|{nm}|35" if False else f"PSO-Fed|{nm}|40", "")
    P_("Online FMTL", "Online FL", "Online FMTL - Online FL"); P_("MOCHA-style", "Online FMTL", "MOCHA-style - Online FMTL")
    BL, RD = ["#3C8DC9", "#2F75B5", "#3E9FE0"], ["#E8584A", "#C0392B", "#D14B3C"]
    plt.rcParams.update({"font.size": 10})
    for nm, name in (("c", "Coordinated"), ("u", "Uncoordinated")):
        fig, ax = plt.subplots(figsize=(10.5, 5.6)); ax.set_facecolor("#F7F7F7"); mk = np.arange(0, N, max(N // 10, 1)); LAB = []
        ax.plot(mean["Online FL"], color="#F39C12", ls="--", lw=2.6, label="Online FL"); ax.plot(mean["Online FMTL"], color="#27AE60", lw=3, label="Online FMTL")
        for i, M in enumerate(MB):
            y = mean[f"PSO-Fed|{nm}|{M}"]; ax.plot(y, color=BL[i], ls="--", lw=2.4, label="Baseline [10] (PSO-Fed)" if i == 0 else None); ax.plot(mk, y[mk], "o", color="#2C3E50", mfc="#2C3E50", mec="white", ms=7)
            LAB.append([y[-1], f"M={M}", BL[i]])
        for i, M in enumerate(MF):
            y = mean[f"PSO-FMTL|{nm}|{M}"]; ax.plot(y, color=RD[i], lw=2.4, label="PSO-FMTL" if i == 0 else None); ax.plot(mk, y[mk], "s", color="#2C3E50", mec="white", ms=7)
            LAB.append([y[-1], f"M={M}", RD[i]])
        LAB.sort(key=lambda a: a[0]); gap = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]); pos = [a[0] for a in LAB]
        for j in range(1, len(pos)): pos[j] = max(pos[j], pos[j - 1] + gap)
        for (y0, lab, c), yy in zip(LAB, pos): ax.text(N + 8, yy, lab, color=c, weight="bold", va="center")
        ax.set_xlim(0, N); ax.set_xlabel("Training Iterations"); ax.set_ylabel("Test MSE (dB)"); ax.grid(ls="--", alpha=.35)
        ax.set_title(f"{name} Learning Performance ($\\mu={MU}$, $K={K}$ tasks, {SEEDS} seeds)", weight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.07, 1.0), frameon=True); fig.tight_layout(); fig.savefig(f"fig_learning_{name.lower()}{sfx}.png", dpi=200); plt.close(fig)
