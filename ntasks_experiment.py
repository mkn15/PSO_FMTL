"""Online-FMTL vs PSO-FMTL for K = 5, 10, 20 tasks (one task per client).
Needs fed_simulation.py in the same folder. Metric: TEST MSE (dB) of the client models on held-out samples,
averaged over the last 30 of 500 rounds. Writes ntasks_results.json."""
import json, sys, numpy as np
from multiprocessing import Pool
from fed_simulation import (generate_tasks, RFFMap, run_fedavg, run_pso_fed,
                            run_mocha, run_pso_fmtl, make_client_probe, run_with_test)
D, P, N = 200, 4, 500
KS = [5, 10, 20]
MUS = [0.05, 0.5]
MS = [5, 10, 20, 35, 50, 70]
SEEDS = list(range(10))

def one(args):
    K, mu, sd = args
    t = generate_tasks(K, 2500, P=P, seed=sd); r = RFFMap(P, D, seed=sd); pr = make_client_probe(t, r, N)
    T = lambda f: run_with_test(f, pr)[0]
    o = {"Online-FMTL(full)": T(lambda p: run_mocha(t, r, mu=mu, n_rounds=N, probe=p)),
         "FedAvg(full)": T(lambda p: run_fedavg(t, r, mu=mu, n_rounds=N, probe=p))}
    for M in MS:
        for th in (1.0, 0.6):
            o[f"PSO-FMTL|th{th}|M{M}"] = T(lambda p: run_pso_fmtl(t, r, M=M, mu=mu, n_rounds=N, theta=th,
                shift_step=1, coordinated=False, omega_nonneg=True, probe=p))
    o["PSO-Fed|M35"] = T(lambda p: run_pso_fed(t, r, M=35, mu=mu, n_rounds=N, shift_step=1, coordinated=False, probe=p))
    return (K, mu, sd, o)

if __name__ == "__main__":
    jobs = [(K, mu, sd) for K in KS for mu in MUS for sd in SEEDS]
    with Pool() as p: res = p.map(one, jobs)
    out = {}
    for K, mu, sd, o in res:
        for k, v in o.items(): out.setdefault(f"K{K}|mu{mu}", {}).setdefault(k, []).append(v)
    json.dump(out, open("ntasks_results.json", "w"))
    for key, d in out.items():
        print("\n==", key)
        for k, v in d.items(): print(f"  {k:<24}{np.mean(v):8.2f} +/- {np.std(v, ddof=1):.2f}")
