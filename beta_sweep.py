"""Server-mixing sweep: final TEST MSE vs beta for coordinated / uncoordinated masks (K=10, M=35, mu=0.05, 10 seeds)."""
import json, sys, os, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fed_simulation import generate_tasks, RFFMap, run_pso_fmtl, make_client_probe, run_with_test
BETAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
def one(sd):
    t = generate_tasks(10, 2500, seed=sd); r = RFFMap(4, 200, seed=sd); pr = make_client_probe(t, r, 500)
    return {f"{nm}|{b}": run_with_test(lambda p: run_pso_fmtl(t, r, M=35, n_rounds=500, theta=b, shift_step=1, coordinated=c, omega_nonneg=True, probe=p), pr)[0]
            for c, nm in ((True, "c"), (False, "u")) for b in BETAS}
if __name__ == "__main__":
    with Pool() as p: R = p.map(one, range(10))
    out = {k: [r[k] for r in R] for k in R[0]}; json.dump(out, open("beta_sweep.json", "w"))
    for nm in ("c", "u"): print(nm, {b: round(float(np.mean(out[f'{nm}|{b}'])), 2) for b in BETAS})
