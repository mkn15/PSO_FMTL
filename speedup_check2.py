"""Can the gap to Online FMTL be closed in fewer rounds? K=10 tasks, D=200, mu=0.05, uncoordinated masks, 10 seeds.
Variants of PSO-FMTL: shift sigma, M, server over-relaxation beta>1 (w <- w + beta (avg - w)). Reports gap (dB, + = worse) to Online FMTL (M=D) at 300/500/1000 rounds."""
import os, sys, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_paper_experiments as ra
from fed_simulation import generate_tasks, RFFMap, make_client_probe
K, D, P, N = 10, 200, 4, 1000; ra.N = N; MU = float(os.environ.get("MU", 0.05))
VARX = {"base M35 s1 b1": dict(M=35), "sigma=35": dict(M=35, shift=35), "sigma=17": dict(M=35, shift=17), "beta=1.5": dict(M=35, beta=1.5), "beta=2": dict(M=35, beta=2.0),
       "M=50": dict(M=50), "M=50 sigma=50 beta=1.5": dict(M=50, shift=50, beta=1.5), "M=35 sigma=35 beta=1.5": dict(M=35, shift=35, beta=1.5), "M=70": dict(M=70)}
VAR = {"sigma=35 beta=1 (u)": dict(M=35, shift=35), "sigma=35 beta=1.5 (u)": dict(M=35, shift=35, beta=1.5), "sigma=35 beta=2 (u)": dict(M=35, shift=35, beta=2.0),
       "sigma=35 beta=1 (c)": dict(M=35, shift=35, coord=True), "sigma=35 beta=1.5 (c)": dict(M=35, shift=35, beta=1.5, coord=True)}
FULL = {"Online FMTL beta=1": 1.0, "Online FMTL beta=1.5": 1.5, "Online FMTL beta=2": 2.0}
def one(sd):
    t = generate_tasks(K, N + 1000, P=P, seed=sd); r = RFFMap(P, D, seed=sd); pr = make_client_probe(t, r, N, N + 300); o = {}
    for k, b in FULL.items():
        L = []; ra.fmtl(t, r, M=D, mu=MU, coord=False, beta=b, probe=(pr, L)); o[k] = np.asarray(L)
    for k, kw in VAR.items():
        kw = dict(kw); kw.setdefault("coord", False); L = []; ra.fmtl(t, r, mu=MU, probe=(pr, L), **kw); o[k] = np.asarray(L)
    return o
if __name__ == "__main__":
    with Pool() as p: runs = p.map(one, range(10))
    f = lambda k, n: np.mean([r[k][n - 30:n].mean() for r in runs])
    print(f"mu={MU}  (absolute test MSE dB at 300/500/1000 rounds)")
    for k in list(FULL) + list(VAR): print(f"{k:26s}" + "  ".join(f"{f(k,n):8.2f}" for n in (300, 500, 1000)))
