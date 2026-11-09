"""Re-draws the learning-performance figures from user_style_results_*.json (no re-simulation).
Usage: python plot_user_style.py user_style_results_mu0.05_sigmaM.png-source.json  -> fig_learning_*_fine.png
Curves: 11-point centred moving average of the 30-seed mean (display only; all numbers in the paper come from the raw runs)."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else "user_style_results_mu0.05_sigmaM.json"; tag = sys.argv[2] if len(sys.argv) > 2 else "mu0.05_sigmaM"
d = json.load(open(src)); mean = {k: np.asarray(v) for k, v in d["mean"].items()}; N = len(mean["Online FL"])
def sm(y, w=11):
    p = np.pad(y, (w // 2, w // 2), mode="edge"); return np.convolve(p, np.ones(w) / w, mode="valid")
fin = lambda k: float(np.mean(mean[k][-30:]))
MB, MF = (1, 5, 40), (1, 5, 35)
BL = {1: "#9CC3EA", 5: "#4F93D6", 40: "#0B4F9E"}; RD = {1: "#F2A19A", 5: "#E0463A", 35: "#9E1B12"}; MKB = {1: "o", 5: "D", 40: "^"}; MKF = {1: "s", 5: "v", 35: "P"}
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
for nm, name in (("c", "Coordinated"), ("u", "Uncoordinated")):
    fig, ax = plt.subplots(figsize=(11.5, 6.4)); ax.set_facecolor("#FBFBFB"); mk = np.arange(0, N, N // 10)
    yl, yf = sm(mean["Online FL"]), sm(mean["Online FMTL"])
    ax.plot(yl, color="#F39C12", lw=6.5, alpha=.55, solid_capstyle="round", label="Online FL (400 params / round)", zorder=2)
    ax.plot(yl, color="#F39C12", lw=2.2, ls=(0, (5, 3)), zorder=3)
    ax.plot(yf, color="#1E8449", lw=2.2, label="Online FMTL (400 params / round)", zorder=4); ax.plot(mk, yf[mk], "*", color="#1E8449", ms=13, mec="white", zorder=5)
    L = []
    for M in MB:
        y = sm(mean[f"PSO-Fed|{nm}|{M}"]); ax.plot(y, color=BL[M], lw=2.2, ls="--", zorder=3); ax.plot(mk, y[mk], MKB[M], color=BL[M], ms=8, mec="white", zorder=5)
        ax.plot([], [], color=BL[M], ls="--", marker=MKB[M], ms=8, lw=2.2, label=f"PSO-Fed (baseline), M={M}  [{2*M} params]"); L.append([y[-1], f"PSO-Fed M={M}", BL[M]])
    for M in MF:
        y = sm(mean[f"PSO-FMTL|{nm}|{M}"]); ax.plot(y, color=RD[M], lw=2.2, zorder=4); ax.plot(mk, y[mk], MKF[M], color=RD[M], ms=8, mec="white", zorder=5)
        ax.plot([], [], color=RD[M], marker=MKF[M], ms=8, lw=2.2, label=f"PSO-FMTL, M={M}  [{2*M} params]"); L.append([y[-1], f"PSO-FMTL M={M}", RD[M]])
    L.append([yl[-1], "Online FL", "#C77700"]); L.append([yf[-1], "Online FMTL", "#1E8449"])
    ylo, yhi = ax.get_ylim(); gap = 0.038 * (yhi - ylo); L.sort(key=lambda a: a[0]); pos = [a[0] for a in L]
    for j in range(1, len(pos)): pos[j] = max(pos[j], pos[j - 1] + gap)
    for (y0, lab, c), yy in zip(L, pos): ax.text(N * 1.012, yy, lab, color=c, weight="bold", va="center", fontsize=11)
    box = "Final test MSE (dB)\n" + "\n".join(f"{lab:<18}{v:7.2f}" for lab, v in (("Online FL", fin("Online FL")), ("Online FMTL", fin("Online FMTL")), ("PSO-Fed  M=40", fin(f"PSO-Fed|{nm}|40")), ("PSO-FMTL M=35", fin(f"PSO-FMTL|{nm}|35"))))
    ax.text(0.985, 0.97, box, transform=ax.transAxes, ha="right", va="top", family="monospace", fontsize=10.5, bbox=dict(fc="white", ec="#999", boxstyle="round,pad=0.4"))
    if nm == "u":                                   # zoom on the four close curves (last 50 iterations)
        ins = ax.inset_axes([0.07, 0.07, 0.30, 0.25]); a, b = N - 50, N
        for k, c, ls, w in (("Online FL", "#F39C12", "-", 6.5), ("Online FMTL", "#1E8449", "-", 2.2), ("PSO-Fed|u|40", BL[40], "--", 2.2), ("PSO-FMTL|u|35", RD[35], "-", 2.2)):
            ins.plot(np.arange(a, b), sm(mean[k])[a:b], color=c, ls=ls, lw=w, alpha=.6 if w > 5 else 1)
        ins.set_title("zoom, last 50 iterations", fontsize=8.5, pad=2); ins.tick_params(labelsize=8); ins.grid(ls="--", alpha=.3)
        ax.indicate_inset_zoom(ins, edgecolor="#555")
    ax.set_xlim(0, N); ax.set_xlabel("Training iterations"); ax.set_ylabel("Test MSE (dB)"); ax.grid(ls="--", alpha=.35)
    ax.set_title(f"{name} learning performance (K=10 tasks, μ=0.05, 30 seeds)", weight="bold", fontsize=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10.5, columnspacing=1.6)
    fig.subplots_adjust(right=0.86, bottom=0.27); fig.savefig(f"fig_learning_{name.lower()}_{tag}_fine.png", dpi=220, bbox_inches="tight"); plt.close(fig)
