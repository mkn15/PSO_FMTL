"""Quantized FedAvg baseline (reviewer request), compared with PSO-FMTL in transmitted BITS.
Stochastic uniform b-bit quantization (max-scaled, QSGD-style) of the model sent in BOTH directions;
cost per direction = D*b + 32 bits (one float32 scale). Writes quant_results.json."""
import json, sys, os, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fed_simulation import generate_tasks, RFFMap, run_pso_fmtl, mse_db, make_client_probe, run_with_test
K, D, P, N = 10, 200, 4, 500

def quant(v, bits, rng):
    L = 2 ** (bits - 1) - 1
    s = np.max(np.abs(v))
    if s == 0: return v
    x = v / s * L
    lo = np.floor(x); x = lo + (rng.random(v.shape) < (x - lo))
    return x * s / L

def fedavg_q(tasks, rff, bits, mu=0.05, seed=0, probe=None):
    rng = np.random.default_rng(seed + 77); w = np.zeros(D); curve = []
    for m in range(N):
        wl, errs = [], []
        for k in range(K):
            X, y = tasks[k]; i = m % len(y); z = rff.map(X[i:i + 1])[0]
            wk = quant(w, bits, rng) if bits else w.copy()          # downlink
            e = y[i] - wk @ z; wk = wk + mu * e * z
            wl.append(quant(wk, bits, rng) if bits else wk)         # uplink
            errs.append(e)
        w = np.mean(wl, axis=0); curve.append(mse_db(np.array(errs)))
        if probe is not None: probe[1].append(probe[0](w, None))
    return np.array(curve)

def one(sd):
    t = generate_tasks(K, 2500, P=P, seed=sd); r = RFFMap(P, D, seed=sd); pr = make_client_probe(t, r, N)
    T = lambda f: run_with_test(f, pr)[0]
    o = {f"FedAvg float32": T(lambda p: fedavg_q(t, r, 0, seed=sd, probe=p))}
    for b in (8, 4, 2): o[f"FedAvg {b}-bit"] = T(lambda p: fedavg_q(t, r, b, seed=sd, probe=p))
    o["PSO-FMTL M=35 float32"] = T(lambda p: run_pso_fmtl(t, r, M=35, n_rounds=N, theta=1.0, shift_step=1, coordinated=False, omega_nonneg=True, probe=p))
    return o
if __name__ == "__main__":
    with Pool() as p: R = p.map(one, range(10))
    bits = {"FedAvg float32": 2 * D * 32, "FedAvg 8-bit": 2 * (D * 8 + 32), "FedAvg 4-bit": 2 * (D * 4 + 32),
            "FedAvg 2-bit": 2 * (D * 2 + 32), "PSO-FMTL M=35 float32": 2 * 35 * 32}
    out = {k: {"mse": [r[k] for r in R], "bits_per_client_round": bits[k]} for k in R[0]}
    json.dump(out, open("quant_results.json", "w"))
    for k, v in out.items(): print(f"{k:<24}{np.mean(v['mse']):7.2f} +/- {np.std(v['mse'], ddof=1):.2f} dB   {v['bits_per_client_round']:6d} bits")
