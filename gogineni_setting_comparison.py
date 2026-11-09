"""
Online-Fed vs PSO-Fed vs PSO-FMTL in the experimental setting of
  V. C. Gogineni, S. Werner, Y.-F. Huang, A. Kuh,
  "Communication-efficient online federated learning framework for nonlinear
   regression" (arXiv:2110.06556, ICASSP version).
for COORDINATED and UNCOORDINATED partial sharing.

Self-contained (numpy + matplotlib only). Works as a script or one Colab cell.
Every number comes from the simulation; nothing is hardcoded.

What follows the reference paper (Sec. 4 and Algorithm 1)
  * K = 100 clients, D = 200 cosine RFF features, learning rate 0.75 for all
  * synthetic non-IID inputs: AR(1)  x_n = th_k x_{n-1} + sqrt(1-th_k^2) u_n,
    th_k~U(0.2,0.9), u_n~N(mu_k, s2_k), mu_k~U(-0.2,0.2), s2_k~U(0.2,1.2);
    regressor = last L=4 samples; noise variance ~ U(0.005, 0.03)
  * target f(x) = sqrt(x1^2 + sin^2(pi x4)) + (0.8 - 0.5 exp(-x2^2)) x3 + noise
  * every global iteration the server picks |S_n| = 4 clients uniformly at random
  * Online-Fed: selected clients restart from the global model, others idle
  * PSO-Fed (Algorithm 1): ALL clients update their local models every
    iteration; selected clients receive S_{k,n} w_n (M entries), upload
    S_{k,n+1} w_{k,n+1}, and the server updates
        w_{n+1} = 1/|S_n| sum_{k in S_n} [ S_{k,n+1} w_{k,n+1} + (I - S_{k,n+1}) w_n ]
  * coordinated: all clients start from the same selection matrix;
    uncoordinated: random initial selection matrices; both then circularly
    shift every iteration (tau = 1)
  * metric: test MSE (dB) of the GLOBAL model w_n on held-out data from all
    clients, averaged over independent runs (the paper uses 500 runs, 1000
    iterations; the defaults below use fewer runs so it finishes quickly)

What is NOT stated in the reference and is therefore my choice
  RFF kernel bandwidth (sigma=1), test-set size (50 samples per client),
  and, for PSO-FMTL, everything (it is not part of that paper): a MOCHA-style
  task-coupling pull toward an Omega-weighted average of the other clients'
  local models, Omega refreshed every 20 iterations with clipped non-negative
  weights, plus optional theta mixing of the global update.
  In this single-task setting (all clients observe the same function) PSO-FMTL
  with the coupling off and theta=1 is exactly PSO-Fed. The coupling reads
  other clients' FULL local models, so its traffic is not in the 2*M count.

Outputs: fig_gogineni_coordinated.png, fig_gogineni_uncoordinated.png,
         gogineni_results.json, plus a printed steady-state table.
"""

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Configuration (edit here)
# ----------------------------------------------------------------------------
K = 100              # clients
D = 200              # RFF dimension
L = 4                # regressor length
N_IT = 1000          # global iterations
N_SEL = 4            # clients selected per iteration
MU = 0.75            # learning rate (reference value)
M_LIST = [1, 5, 40]  # entries exchanged per iteration
TAU = 1              # circular shift per iteration
RUNS = 10            # independent runs (reference: 500). Use 3 for a quick test.
N_TEST = 50          # test samples per client
LAM1 = 0.05          # PSO-FMTL coupling strength
THETA = 1.0          # PSO-FMTL mixing (1.0 = none). Also run theta=0.6 at M_LIST[-1].
OMEGA_EVERY = 20
TAIL = 100           # steady state = mean test MSE over the last TAIL iterations
OUT_DIR = "."


# ----------------------------------------------------------------------------
# Data and features (reference Sec. 4)
# ----------------------------------------------------------------------------
def make_data(rng, n_it):
    """Return X (K, n_it, L), y (K, n_it), Xte (K, N_TEST, L), yte (K, N_TEST)."""
    T = n_it + N_TEST + L
    th = rng.uniform(0.2, 0.9, K)
    mu_k = rng.uniform(-0.2, 0.2, K)
    s2 = rng.uniform(0.2, 1.2, K)
    nv = rng.uniform(0.005, 0.03, K)
    x = np.zeros((K, T))
    u = mu_k[:, None] + np.sqrt(s2)[:, None] * rng.standard_normal((K, T))
    for t in range(1, T):
        x[:, t] = th * x[:, t - 1] + np.sqrt(1 - th ** 2) * u[:, t]
    # regressor at time t: [x_t, x_{t-1}, ..., x_{t-L+1}]
    idx = np.arange(L - 1, T)
    R = np.stack([x[:, idx - j] for j in range(L)], axis=2)          # (K, T-L+1, L)
    f = (np.sqrt(R[..., 0] ** 2 + np.sin(np.pi * R[..., 3]) ** 2)
         + (0.8 - 0.5 * np.exp(-R[..., 1] ** 2)) * R[..., 2])
    y = f + np.sqrt(nv)[:, None] * rng.standard_normal(f.shape)
    Xtr, ytr = R[:, :n_it], y[:, :n_it]
    Xte, yte = R[:, n_it:n_it + N_TEST], f[:, n_it:n_it + N_TEST]    # noiseless test target
    return Xtr, ytr, Xte, yte


class RFF:
    def __init__(self, rng, sigma=1.0):
        self.V = rng.normal(0, 1.0 / sigma, size=(D, L))
        self.b = rng.uniform(0, 2 * np.pi, size=D)

    def __call__(self, X):                       # (..., L) -> (..., D)
        return np.sqrt(2.0 / D) * np.cos(X @ self.V.T + self.b)


def test_mse_db(w, Zte, yte):
    err = yte - Zte @ w                          # Zte: (K*N_TEST, D), yte flat
    return 10 * np.log10(np.mean(err ** 2))


def masks_at(starts, n, M):
    """Selection masks (K, D) at iteration n: M consecutive ones starting at
    (start_k + n*TAU) mod D, i.e. a circular shift of TAU per iteration."""
    pos = (starts + n * TAU) % D
    cols = (np.arange(M)[None, :] + pos[:, None]) % D
    mask = np.zeros((K, D))
    np.put_along_axis(mask, cols, 1.0, axis=1)
    return mask


# ----------------------------------------------------------------------------
# Algorithms. Each returns the test-MSE-in-dB curve of the global model (N_IT,)
# ----------------------------------------------------------------------------
def run_online_fed(Z, y, Zte, yte, sel):
    w = np.zeros(D)
    curve = np.empty(N_IT)
    for n in range(N_IT):
        S = sel[n]
        z = Z[S, n]                                           # (4, D)
        e = y[S, n] - z @ w
        wk = w + MU * e[:, None] * z
        w = wk.mean(axis=0)
        curve[n] = test_mse_db(w, Zte, yte)
    return curve


def run_pso(Z, y, Zte, yte, sel, M, starts, coupling=False, theta=1.0):
    """PSO-Fed (coupling=False, theta=1) or PSO-FMTL (coupling and/or theta)."""
    w = np.zeros(D)
    Wl = np.zeros((K, D))
    Omega = np.eye(K) / K
    curve = np.empty(N_IT)
    mask_now = masks_at(starts, 0, M)
    for n in range(N_IT):
        S = sel[n]
        mask_next = masks_at(starts, n + 1, M)
        part = np.zeros(K, bool)
        part[S] = True
        # local view: participants splice the M received global entries in
        view = Wl.copy()
        view[S] = mask_now[S] * w + (1 - mask_now[S]) * Wl[S]
        z = Z[:, n]
        e = y[:, n] - np.einsum("kd,kd->k", view, z)
        grad = -e[:, None] * z
        if coupling:
            wts = np.maximum(Omega, 0.0)
            np.fill_diagonal(wts, 0.0)
            s = wts.sum(axis=1, keepdims=True)
            nb = np.divide(wts @ Wl, s, out=np.zeros_like(Wl), where=s > 1e-8)
            grad = grad + np.where(s > 1e-8, LAM1 * (view - nb), 0.0)
        Wnew = view - MU * grad                               # all clients learn locally
        # server: participants upload S_{k,n+1} w_{k,n+1}; the rest of the
        # coordinates are filled with the previous global model (Alg. 1)
        contrib = mask_next[S] * Wnew[S] + (1 - mask_next[S]) * w
        agg = contrib.mean(axis=0)
        w = theta * agg + (1 - theta) * w
        Wl = Wnew
        mask_now = mask_next
        if coupling and (n + 1) % OMEGA_EVERY == 0:
            A = Wl @ Wl.T
            ev, U = np.linalg.eigh(A)
            sq = (U * np.sqrt(np.clip(ev, 1e-8, None))) @ U.T
            tr = np.trace(sq)
            Omega = sq / tr if tr > 1e-8 else np.eye(K) / K
        curve[n] = test_mse_db(w, Zte, yte)
    return curve


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = ["Online-Fed"]
    for mode in ("coord", "uncoord"):
        for M in M_LIST:
            keys += [f"PSO-Fed|{mode}|{M}", f"PSO-FMTL|{mode}|{M}"]
        keys.append(f"PSO-FMTL(theta=0.6)|{mode}|{M_LIST[-1]}")
    curves = {k: [] for k in keys}
    t0 = time.time()
    print(f"K={K} clients, |S_n|={N_SEL}, mu={MU}, D={D}, {N_IT} iterations, {RUNS} runs", flush=True)
    for r in range(RUNS):
        rng = np.random.default_rng(1000 + r)
        Xtr, ytr, Xte, yte = make_data(rng, N_IT)
        rff = RFF(rng)
        Z = rff(Xtr)                                          # (K, N_IT, D)
        Zte = rff(Xte).reshape(-1, D)
        yte_f = yte.reshape(-1)
        sel = [rng.choice(K, N_SEL, replace=False) for _ in range(N_IT)]
        starts_c = np.zeros(K, dtype=int)
        starts_u = rng.integers(0, D, K)
        curves["Online-Fed"].append(run_online_fed(Z, ytr, Zte, yte_f, sel))
        for mode, starts in (("coord", starts_c), ("uncoord", starts_u)):
            for M in M_LIST:
                curves[f"PSO-Fed|{mode}|{M}"].append(
                    run_pso(Z, ytr, Zte, yte_f, sel, M, starts))
                curves[f"PSO-FMTL|{mode}|{M}"].append(
                    run_pso(Z, ytr, Zte, yte_f, sel, M, starts, coupling=True, theta=THETA))
            curves[f"PSO-FMTL(theta=0.6)|{mode}|{M_LIST[-1]}"].append(
                run_pso(Z, ytr, Zte, yte_f, sel, M_LIST[-1], starts, coupling=True, theta=0.6))
        print(f"  run {r + 1}/{RUNS} done ({time.time() - t0:.0f}s)", flush=True)

    mean = {k: np.mean(np.array(v), axis=0) for k, v in curves.items()}
    ss = {k: np.array(v)[:, -TAIL:].mean(axis=1) for k, v in curves.items()}   # per-run steady state

    # ---------------- table ----------------
    print("\nSTEADY-STATE TEST MSE (dB, mean of last %d iterations; mean +/- std over %d runs)" % (TAIL, RUNS))
    print(f"{'method':<40}{'entries/iter':>13}   MSE (dB)")
    f = lambda a: f"{a.mean():7.2f} +/- {a.std(ddof=1) if len(a) > 1 else 0:4.2f}"
    print(f"{'Online-Fed':<40}{D:>13}   {f(ss['Online-Fed'])}")
    for mode in ("coord", "uncoord"):
        for M in M_LIST:
            print(f"{'PSO-Fed ' + mode + ' M=' + str(M):<40}{M:>13}   {f(ss[f'PSO-Fed|{mode}|{M}'])}")
            print(f"{'PSO-FMTL(th=' + str(THETA) + ') ' + mode + ' M=' + str(M):<40}{M:>13}   {f(ss[f'PSO-FMTL|{mode}|{M}'])}")
        k = f"PSO-FMTL(theta=0.6)|{mode}|{M_LIST[-1]}"
        print(f"{'PSO-FMTL(th=0.6) ' + mode + ' M=' + str(M_LIST[-1]):<40}{M_LIST[-1]:>13}   {f(ss[k])}")
    print("\nCoordinated minus uncoordinated (PSO-Fed, negative = coordinated better):")
    for M in M_LIST:
        d = ss[f"PSO-Fed|coord|{M}"] - ss[f"PSO-Fed|uncoord|{M}"]
        print(f"  M={M:<3} {d.mean():+.2f} dB")
    print("PSO-Fed minus Online-Fed (negative = PSO-Fed better):")
    for mode in ("coord", "uncoord"):
        for M in M_LIST:
            d = ss[f"PSO-Fed|{mode}|{M}"] - ss["Online-Fed"]
            print(f"  {mode:<8} M={M:<3} {d.mean():+.2f} dB")

    # ---------------- figures (one per mode, like Fig. 1a / 1b of the reference) ----
    colors = {1: "#2a78d6", 5: "#2a9d8f", 40: "#d98324"}
    for mode, name in (("coord", "Coordinated"), ("uncoord", "Uncoordinated")):
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
        ax.plot(mean["Online-Fed"], color="#8a8983", lw=2.0, label=f"Online-Fed ({D} entries)")
        for M in M_LIST:
            c = colors.get(M, None)
            ax.plot(mean[f"PSO-Fed|{mode}|{M}"], color=c, lw=1.5, label=f"PSO-Fed, M={M}")
            ax.plot(mean[f"PSO-FMTL|{mode}|{M}"], color=c, lw=1.3, ls="--",
                    label=f"PSO-FMTL (theta={THETA}), M={M}")
        ax.set_xlabel("Global iteration")
        ax.set_ylabel("Test MSE of global model (dB)")
        ax.set_title(f"{name} partial sharing  (mu={MU}, K={K}, |S_n|={N_SEL}, {RUNS} runs)", fontsize=10)
        ax.grid(color="#e5e4e0", lw=0.6)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"fig_gogineni_{'coordinated' if mode == 'coord' else 'uncoordinated'}.png"))
        plt.close(fig)

    with open(os.path.join(OUT_DIR, "gogineni_results.json"), "w") as fh:
        json.dump({k: v.tolist() for k, v in ss.items()}, fh)
    print("\nSaved: fig_gogineni_coordinated.png, fig_gogineni_uncoordinated.png, gogineni_results.json")


main()
