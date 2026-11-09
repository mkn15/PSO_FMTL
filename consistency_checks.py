"""Consistency checks between Fig. 2 / Table (test-MSE metric). K=10, D=200, mu=0.05, 500 rounds, M=35, 30 seeds.
Check 1: |PSO-FMTL - PSO-Fed| <= 0.15 dB (uncoordinated).  Check 2: gap to FedAvg ~ 0.1 dB at sigma = M.
Check 3: coordinated loses about 3 dB versus uncoordinated."""
import os, sys, numpy as np
from multiprocessing import Pool
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_paper_experiments as ra
from fed_simulation import generate_tasks, RFFMap, run_fedavg, make_client_probe, run_with_test
K, D, P, N, M, MU = 10, 200, 4, 500, 35, 0.05; ra.N = N
def one(sd):
    t = generate_tasks(K, 2500, P=P, seed=sd); r = RFFMap(P, D, seed=sd); pr = make_client_probe(t, r, N); o = {}
    o["FedAvg"] = run_with_test(lambda p: run_fedavg(t, r, mu=MU, n_rounds=N, probe=p), pr)[0]
    for sg in (1, M):
        for nm, c in (("c", True), ("u", False)):
            o[f"Fed|{nm}|{sg}"] = run_with_test(lambda p: ra.psofed(t, r, M=M, mu=MU, coord=c, shift=sg, probe=p), pr)[0]
            o[f"FMTL|{nm}|{sg}"] = run_with_test(lambda p: ra.fmtl(t, r, M=M, mu=MU, coord=c, shift=sg, probe=p), pr)[0]
    return o
if __name__ == "__main__":
    with Pool() as p: runs = p.map(one, range(30))
    f = {k: np.array([r[k] for r in runs]) for k in runs[0]}
    def d(a, b): x = f[a] - f[b]; return f"{x.mean():+.2f} dB (p={stats.ttest_rel(f[a], f[b]).pvalue:.2g}, a better on {(x<0).sum()}/30)"
    print("FedAvg %.2f" % f["FedAvg"].mean())
    for sg in (1, M):
        print(f"--- sigma={sg}")
        for nm in "cu": print(f"  {nm}: PSO-Fed {f[f'Fed|{nm}|{sg}'].mean():.2f}  PSO-FMTL {f[f'FMTL|{nm}|{sg}'].mean():.2f}")
        print("  check1  PSO-FMTL - PSO-Fed (u):", d(f"FMTL|u|{sg}", f"Fed|u|{sg}"), "| (c):", d(f"FMTL|c|{sg}", f"Fed|c|{sg}"))
        print("  check2  PSO-Fed - FedAvg (u):", d(f"Fed|u|{sg}", "FedAvg"), "| PSO-FMTL - FedAvg (u):", d(f"FMTL|u|{sg}", "FedAvg"))
        print("  check3  coordinated - uncoordinated: PSO-Fed", d(f"Fed|c|{sg}", f"Fed|u|{sg}"), "| PSO-FMTL", d(f"FMTL|c|{sg}", f"FMTL|u|{sg}"))
