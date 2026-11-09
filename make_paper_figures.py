"""Build all figures of the paper from paper_results_v2.json and ntasks_results.json.
Output: fig1_framework.png, fig3_M_and_rotation.png, fig4_ntasks.png
(the convergence figure comes from make_gogineni_style_figs.py)"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from scipy import stats

plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "savefig.dpi": 300})
BL, OR, GR, VE, GY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"
R = json.load(open("paper_results_v2.json")); S = sorted(R, key=int)
g = lambda f: np.array([f(R[s]) for s in S])
sm = lambda x, w=15: np.convolve(x, np.ones(w) / w, mode="valid")

# ---------------- Fig. 1: framework ----------------
fig, ax = plt.subplots(figsize=(3.45, 2.75)); ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
def bar(x, y, w, h, hl_start, hl_len, col, D=20):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=GY, lw=0.6))
    for i in range(hl_len):
        j = (hl_start + i) % D
        ax.add_patch(Rectangle((x + j * w / D, y), w / D, h, fc=col, ec="none", alpha=0.9))
ax.add_patch(FancyBboxPatch((2.2, 5.6), 5.6, 2.1, boxstyle="round,pad=0.05", fc="#F2F2F2", ec=GY, lw=0.7))
ax.text(5, 7.35, "Server", ha="center", fontsize=8, weight="bold")
ax.text(2.45, 6.65, r"global model $\mathbf{w}_m$", fontsize=6.5, va="center"); bar(5.0, 6.45, 2.5, 0.4, 0, 20, "#BBBBBB")
ax.text(2.45, 5.95, r"average of $M$ uploaded entries", fontsize=6.3, va="center")
for i, (x, col, st) in enumerate([(0.2, BL, 0), (3.6, OR, 7), (7.0, GR, 14)]):
    ax.add_patch(FancyBboxPatch((x, 0.4), 2.8, 3.2, boxstyle="round,pad=0.05", fc="white", ec=GY, lw=0.7))
    ax.text(x + 1.4, 3.25, f"Client {['1','2','K'][i]}" if i < 2 else "Client K", ha="center", fontsize=7.5, weight="bold")
    ax.text(x + 0.15, 2.6, r"local $\mathbf{w}_k$", fontsize=6.0, va="center"); bar(x + 1.35, 2.45, 1.3, 0.3, st, 4, col)
    ax.text(x + 0.15, 1.85, "RFF + LMS step", fontsize=6.0, va="center")
    ax.text(x + 0.15, 1.2, r"mask $\mathbf{S}_{k,m}$ shifts" + "\nby $\\sigma$ each round", fontsize=5.9, va="center")
    ax.add_patch(FancyArrowPatch((x + 0.9, 3.7), (x + 0.9 + (3.6 - x) * 0.0 + 0.0, 5.55), arrowstyle="-|>", mutation_scale=6, color=col, lw=0.9))
    ax.add_patch(FancyArrowPatch((x + 1.9, 5.55), (x + 1.9, 3.7), arrowstyle="-|>", mutation_scale=6, color=col, lw=0.9, ls="--"))
ax.text(0.15, 4.65, "up: $M$", fontsize=6, color=GY); ax.text(1.55, 4.65, "down: $M$", fontsize=6, color=GY)
ax.text(5.0, 0.0, r"Optional: $\Omega_\theta$ coupling (needs other clients' full models)", fontsize=6.2, ha="center", color=VE, style="italic")
fig.savefig("fig1_framework.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig. 3: M sweep and rotation sweep ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
Ms = [1, 5, 10, 20, 35, 50, 70, 100]; ax = axs[0]
for nm, col, lab in (("c", OR, "coord."), ("u", BL, "uncoord.")):
    for who, ls in (("PSO-Fed", "-"), ("PSO-FMTL", "--")):
        y = [g(lambda r: r["Msweep"][f"{who}|{nm}|{M}"]).mean() for M in Ms]
        ax.plot(Ms, y, color=col, ls=ls, marker="o", ms=2.5, lw=1.0, label=f"{who}, {lab}")
ax.axhline(g(lambda r: r["methods"]["FedAvg"]).mean(), color=GY, lw=1.4, label="FedAvg")
ax.axhline(g(lambda r: r["methods"]["Online FMTL"]).mean(), color="k", ls=":", lw=1.0, label="Online FMTL")
ax.set_xscale("log"); ax.set_xlabel("Shared parameters per round $M$ (of $D{=}200$)"); ax.set_ylabel("Final test MSE (dB)")
ax.set_title("(a) Accuracy versus communication", fontsize=8); ax.grid(color="#E3E3E3", lw=0.5); ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.27), columnspacing=1.0)
ax = axs[1]; sg = [0, 1, 2, 5, 10, 17, 35, 70, 100]
for nm, col, lab in (("c", OR, "coordinated"), ("u", BL, "uncoordinated")):
    m = np.array([g(lambda r: r["shift"][f"{nm}|{s}"]).mean() for s in sg]); e = np.array([g(lambda r: r["shift"][f"{nm}|{s}"]).std(ddof=1) / np.sqrt(len(S)) for s in sg])
    ax.errorbar(range(len(sg)), m, yerr=e, color=col, marker="o", ms=2.5, lw=1.0, capsize=1.5, label=lab)
ax.set_xticks(range(len(sg))); ax.set_xticklabels([str(s) for s in sg]); ax.set_xlabel(r"Shift step $\sigma$ per round (0 = frozen mask)")
ax.set_ylabel("Final test MSE (dB)"); ax.set_title(r"(b) Rotation speed ($M{=}35$)", fontsize=8); ax.grid(color="#E3E3E3", lw=0.5); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("fig3_M_and_rotation.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig. 4: number of tasks vs Online FMTL ----------------
d = json.load(open("ntasks_results.json"))
fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.2), sharey=True)
for ax, mu, title in ((axs[0], 0.05, r"$\mu{=}0.05$"), (axs[1], 0.5, r"$\mu{=}0.5$")):
    for M, col, off in ((10, VE, -0.12), (20, GR, 0.0), (35, BL, 0.12)):
        xs, ys, es = [], [], []
        for i, K in enumerate((5, 10, 20)):
            k = f"K{K}|mu{mu}"; x = np.array(d[k][f"PSO-FMTL|th1.0|M{M}"]) - np.array(d[k]["Online-FMTL(full)"])
            xs.append(i + off); ys.append(x.mean()); es.append(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=3, color=col, capsize=1.5, lw=0.9, label=f"$M{{=}}{M}$")
    ax.axhline(0, color="k", lw=0.7); ax.set_xticks(range(3)); ax.set_xticklabels(["5", "10", "20"]); ax.set_xlabel("Number of tasks $K$"); ax.set_title(title, fontsize=8)
    ax.grid(color="#E3E3E3", lw=0.5)
axs[0].set_ylabel("PSO-FMTL $-$ Online FMTL (dB)"); axs[0].legend(frameon=False, loc="lower left", fontsize=6)
fig.tight_layout(); fig.savefig("fig4_ntasks.png", bbox_inches="tight"); plt.close(fig)
print("figures written")
