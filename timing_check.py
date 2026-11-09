"""Simulator run time (ms per 500-round run, K = 10, no test probe), mean over 5 seeds. Hardware-dependent."""
import time, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fed_simulation as fs
from run_all_paper_experiments import fmtl, psofed
res = {}
for sd in range(5):
    t = fs.generate_tasks(10, 2500, seed=sd); r = fs.RFFMap(4, 200, seed=sd)
    for name, f in (("FedAvg", lambda: fs.run_fedavg(t, r, n_rounds=500)), ("Online FMTL", lambda: fs.run_mocha(t, r, n_rounds=500)),
                    ("PSO-Fed", lambda: psofed(t, r)), ("PSO-FMTL, Omega off", lambda: fmtl(t, r, omega=False)), ("PSO-FMTL, Omega on", lambda: fmtl(t, r))):
        s = time.perf_counter(); f(); res.setdefault(name, []).append((time.perf_counter() - s) * 1000)
for k, v in res.items(): print(f"{k:<22}{np.mean(v):7.0f} ms")
