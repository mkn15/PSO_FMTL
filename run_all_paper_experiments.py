"""
Run every experiment reported in the paper with ONE consistent configuration.

Configuration shared by all runs
  K = 10 clients (one task each unless stated), D = 200 RFF features, P = 4 AR inputs,
  500 online rounds (one new sample per client per round), mu = 0.05 unless stated,
  M = 35, rotation step sigma = 1, uncoordinated offsets k*floor(D/K),
  PSO-FMTL: lambda1 = 0.05, Omega refreshed every 20 rounds with clipped non-negative
  weights, smoothing theta = 0, server mixing beta = 1 (beta = 0.6 reported separately).
  Metric: TEST MSE in dB of the models the clients use, on held-out samples (rows 500-799 of every
  client's stream, never trained on), averaged over the last 30 of 500 rounds; mean +/- std over 10 seeds.

Needs fed_simulation.py in the same folder. Output: paper_results_v2.json
Run:  python run_all_paper_experiments.py        (about 5-10 minutes on 2 cores)
"""
import json, time, sys, os
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fed_simulation import (generate_tasks, RFFMap, run_local_only, run_fedavg,
                            run_pso_fed, run_mocha, run_pso_fmtl, make_client_probe, run_with_test)

K, D, P, N = 10, 200, 4, 500
SEEDS = list(range(10))


def gen_hetero(K, n, h=1.0, seed=0, P=4, base_noise=0.08):
    """Task generator with a heterogeneity factor h (h = 0: identical task on all clients)."""
    rng = np.random.default_rng(seed)
    base = np.array([0.55, 0.25, 0.12, 0.06])[:P]
    tasks = []
    for k in range(K):
        co = base * (1 + (rng.random(P) - 0.5) * 0.3 * h)
        s = np.abs(co).sum()
        if s >= 0.9:
            co = co * 0.85 / s
        ns = base_noise * (1 + 0.4 * rng.random())
        x = rng.standard_normal(n)
        for i in range(P, n):
            x[i] = np.dot(co, x[i - P:i][::-1]) + rng.standard_normal() * ns
        x = (x - x.mean()) / (x.std() + 1e-8)
        a = 0.8 + 0.1 * h * (rng.random() - 0.5)
        b = 0.5 + 0.1 * h * (rng.random() - 0.5)
        X = np.array([x[i:i + P] for i in range(len(x) - P)])
        y = (np.sqrt(X[:, 0] ** 2 + np.sin(np.pi * X[:, 3]) ** 2)
             + (a - b * np.exp(-X[:, 1] ** 2)) * X[:, 2] + rng.standard_normal(X.shape[0]) * ns)
        tasks.append((X, y))
    return tasks


def fmtl(t, r, M=35, mu=0.05, coord=False, beta=1.0, omega=True, partial=True, shift=1, theta=0.0, probe=None):
    return run_pso_fmtl(t, r, M=M, mu=mu, n_rounds=N, theta=beta, shift_step=shift,
                        coordinated=coord, use_mtl_term=omega, use_partial_sharing=partial,
                        omega_nonneg=True, omega_smooth=theta, probe=probe)


def psofed(t, r, M=35, mu=0.05, coord=False, shift=1, probe=None):
    return run_pso_fed(t, r, M=M, mu=mu, n_rounds=N, shift_step=shift, coordinated=coord, probe=probe)


def one_seed(sd):
    R = {}
    t = generate_tasks(K, 2500, P=P, seed=sd)
    r = RFFMap(P, D, seed=sd)
    pr = make_client_probe(t, r, N)
    tm = {}

    def T(f, probe=None):                      # final test MSE (dB) of a run
        return run_with_test(f, probe or pr)[0]

    def timed(name, f):
        s = time.perf_counter(); v = f(); tm[name] = (time.perf_counter() - s) * 1000; return v

    # ---- 1. method comparison (mu = 0.05) and test-MSE curves
    cur = {}
    def curve(name, f, probe=None):
        v, L = run_with_test(f, probe or pr); cur[name] = L; return v
    meth = {}
    meth["Local-only"] = curve("Local-only", lambda p: run_local_only(t, r, n_rounds=N, probe=p))
    meth["FedAvg"] = timed("FedAvg", lambda: curve("FedAvg", lambda p: run_fedavg(t, r, n_rounds=N, probe=p)))
    meth["FedProx"] = curve("FedProx", lambda p: run_fedavg(t, r, n_rounds=N, prox_mu=0.02, probe=p))
    meth["Online FMTL"] = timed("Online FMTL", lambda: curve("Online FMTL", lambda p: run_mocha(t, r, n_rounds=N, probe=p)))
    for c, nm in ((True, "c"), (False, "u")):
        meth[f"PSO-Fed {nm}"] = timed(f"PSO-Fed {nm}", lambda: curve(f"PSO-Fed {nm}", lambda p: psofed(t, r, coord=c, probe=p)))
        meth[f"PSO-FMTL {nm}"] = timed(f"PSO-FMTL {nm}", lambda: curve(f"PSO-FMTL {nm}", lambda p: fmtl(t, r, coord=c, probe=p)))
        meth[f"PSO-FMTL b0.6 {nm}"] = T(lambda p: fmtl(t, r, coord=c, beta=0.6, probe=p))
    R["methods"] = meth
    R["time_ms"] = tm

    # ---- 2. component ablation (both mask types)
    ab = {}
    for c, nm in ((True, "c"), (False, "u")):
        ab[f"full {nm}"] = T(lambda p: fmtl(t, r, coord=c, probe=p))
        ab[f"no Omega {nm}"] = T(lambda p: fmtl(t, r, coord=c, omega=False, probe=p))
        ab[f"no partial {nm}"] = T(lambda p: fmtl(t, r, coord=c, partial=False, probe=p))
        ab[f"beta0.6 {nm}"] = T(lambda p: fmtl(t, r, coord=c, beta=0.6, probe=p))
        ab[f"frozen {nm}"] = T(lambda p: fmtl(t, r, coord=c, shift=0, probe=p))
        ab[f"theta1 {nm}"] = T(lambda p: fmtl(t, r, coord=c, theta=1.0, probe=p))
    R["ablation"] = ab

    # ---- 3. rotation-step sweep
    R["shift"] = {f"{nm}|{s}": T(lambda p: fmtl(t, r, coord=c, shift=s, probe=p))
                  for c, nm in ((True, "c"), (False, "u")) for s in (0, 1, 2, 5, 10, 17, 35, 70, 100)}

    # ---- 4. M sweep
    Ms = (1, 5, 10, 20, 35, 50, 70, 100)
    R["Msweep"] = {}
    for c, nm in ((True, "c"), (False, "u")):
        for M in Ms:
            R["Msweep"][f"PSO-Fed|{nm}|{M}"] = T(lambda p: psofed(t, r, M=M, coord=c, probe=p))
            R["Msweep"][f"PSO-FMTL|{nm}|{M}"] = T(lambda p: fmtl(t, r, M=M, coord=c, probe=p))

    # ---- 5. step-size sweep
    R["mu"] = {}
    for mu in (0.05, 0.2, 0.5, 0.75):
        d = {"Local-only": T(lambda p: run_local_only(t, r, mu=mu, n_rounds=N, probe=p)),
             "FedAvg": T(lambda p: run_fedavg(t, r, mu=mu, n_rounds=N, probe=p)),
             "Online FMTL": T(lambda p: run_mocha(t, r, mu=mu, n_rounds=N, probe=p))}
        for c, nm in ((True, "c"), (False, "u")):
            d[f"PSO-Fed {nm}"] = T(lambda p: psofed(t, r, mu=mu, coord=c, probe=p))
            d[f"PSO-FMTL {nm}"] = T(lambda p: fmtl(t, r, mu=mu, coord=c, probe=p))
        R["mu"][str(mu)] = d

    # ---- 6. single-task special case (identical task on all clients)
    ts = gen_hetero(K, 2500, 0.0, seed=sd)
    ps = make_client_probe(ts, r, N)
    s = {"Local-only": T(lambda p: run_local_only(ts, r, n_rounds=N, probe=p), ps),
         "FedAvg": T(lambda p: run_fedavg(ts, r, n_rounds=N, probe=p), ps),
         "Online FMTL": T(lambda p: run_mocha(ts, r, n_rounds=N, probe=p), ps)}
    for c, nm in ((True, "c"), (False, "u")):
        s[f"PSO-Fed {nm}"] = T(lambda p: psofed(ts, r, coord=c, probe=p), ps)
        s[f"PSO-FMTL {nm}"] = T(lambda p: fmtl(ts, r, coord=c, probe=p), ps)
    R["single_task"] = s

    # ---- 7. RFF dimension (M = 0.175 D)
    R["rff"] = {}
    for Dd in (50, 100, 200, 400):
        rr = RFFMap(P, Dd, seed=sd); pd_ = make_client_probe(t, rr, N); Md = int(round(0.175 * Dd))
        R["rff"][str(Dd)] = {"M": Md,
                             "PSO-FMTL u": T(lambda p: fmtl(t, rr, M=Md, coord=False, probe=p), pd_),
                             "FedAvg": T(lambda p: run_fedavg(t, rr, n_rounds=N, probe=p), pd_)}
    return sd, R


if __name__ == "__main__":
    out = {}
    with Pool() as p:
        for sd, R in p.imap_unordered(one_seed, SEEDS):
            out[str(sd)] = R
            print("seed", sd, "done", flush=True)
    json.dump(out, open("paper_results_v2.json", "w"))
    print("saved paper_results_v2.json")
