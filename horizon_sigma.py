"""Gap to full sharing versus training length for rotation step sigma = 1 and sigma = M.
K=10 tasks, D=200, uncoordinated masks, test MSE of the clients' models on rows 4000-4300 of every stream (held out for all horizons).
Online FL = FedAvg; Online FMTL = PSO-FMTL update with M = D.  Env: MU (0.05), SEEDS (20).  Output: horizon_sigma_mu*.json + printed table (dB, + = PSO method worse)."""
import os, sys, json, numpy as np
from multiprocessing import Pool
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_paper_experiments as ra
from fed_simulation import generate_tasks, RFFMap, run_fedavg, make_client_probe
K, D, P, NMAX = 10, 200, 4, 4000; MU = float(os.environ.get("MU", 0.05)); SEEDS = int(os.environ.get("SEEDS", 20)); NS = (300, 500, 1000, 2000, 4000); ra.N = NMAX
def one(sd):
    t = generate_tasks(K, 5000, P=P, seed=sd); r = RFFMap(P, D, seed=sd); pr = make_client_probe(t, r, NMAX, NMAX + 300); o = {}
    def run(k, f): L = []; f((pr, L)); o[k] = np.asarray(L)
    run("FL", lambda p: run_fedavg(t, r, mu=MU, n_rounds=NMAX, probe=p)); run("FMTL", lambda p: ra.fmtl(t, r, M=D, mu=MU, coord=False, probe=p))
    for s, sg in (("s1", 1), ("sM", None)):
        run(f"Fed40|{s}", lambda p: ra.psofed(t, r, M=40, mu=MU, coord=False, shift=sg or 40, probe=p))
        run(f"FMTL35|{s}", lambda p: ra.fmtl(t, r, M=35, mu=MU, coord=False, shift=sg or 35, probe=p))
    return o
if __name__ == "__main__":
    with Pool() as p: runs = p.map(one, range(SEEDS))
    out = {}
    print(f"mu={MU}, {SEEDS} seeds; gap (dB) of PSO method to its full-sharing counterpart, mean over seeds [paired p]")
    for n in NS:
        f = lambda k: np.array([r[k][n - 30:n].mean() for r in runs]); row = []
        for lab, a, b in (("Fed40 s=1", "Fed40|s1", "FL"), ("Fed40 s=M", "Fed40|sM", "FL"), ("FMTL35 s=1", "FMTL35|s1", "FMTL"), ("FMTL35 s=M", "FMTL35|sM", "FMTL")):
            x = f(a) - f(b); out[f"{n}|{lab}"] = float(x.mean()); row.append(f"{lab} {x.mean():+.2f} [{stats.ttest_rel(f(a), f(b)).pvalue:.2g}]")
        print(f"N={n:5d}  OnlineFL {f('FL').mean():6.2f} OnlineFMTL {f('FMTL').mean():6.2f} | " + " | ".join(row))
    json.dump(out, open(f"horizon_sigma_mu{MU}.json", "w"))
