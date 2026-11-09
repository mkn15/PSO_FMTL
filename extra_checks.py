"""Two supporting experiments (test-MSE metric, K = 10, M = 35, uncoordinated masks, clipped weights):
 (1) smoothing of Omega: effect of switching the coupling on, for theta in {0, 0.1, 1, 10, decaying 10 -> 0};
     heterogeneity h in {1,3,6}, mu in {0.05, 0.5}, lambda1 in {0.05, 0.5}; 6 seeds, 500 rounds
 (2) server mixing: constant beta = 0.6 and decaying schedules versus beta = 1, 500 and 2000 rounds,
     mu in {0.05, 0.5}; 6 seeds
Writes extra_checks.json"""
import json, sys, os, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fed_simulation as fs
from fed_simulation import make_client_probe, run_with_test
from run_all_paper_experiments import gen_hetero

SM = {"no coupling": None, "theta=0": 0.0, "theta=0.1": 0.1, "theta=1": 1.0, "theta=10": 10.0,
      "theta decaying 10->0": (lambda m: 10 * np.exp(-m / 150))}
SCH = {"beta=1": 1.0, "beta=0.6": 0.6, "0.3+0.7exp(-m/300)": (lambda m: 0.3 + 0.7 * np.exp(-m / 300)),
       "0.3+0.7exp(-m/100)": (lambda m: 0.3 + 0.7 * np.exp(-m / 100)),
       "max(0.2,1/(1+m/200))": (lambda m: max(0.2, 1 / (1 + m / 200)))}

def job_a(a):
    sd, name, h, mu, lam = a
    t = gen_hetero(10, 2500, h, seed=sd); r = fs.RFFMap(4, 200, seed=sd); pr = make_client_probe(t, r, 500)
    th = SM[name]
    v = run_with_test(lambda p: fs.run_pso_fmtl(t, r, M=35, mu=mu, n_rounds=500, theta=1.0, shift_step=1,
            coordinated=False, omega_nonneg=True, lam1=lam, use_mtl_term=th is not None,
            omega_smooth=0.0 if th is None else th, probe=p), pr)[0]
    return ("A", name, h, mu, lam, sd, v)

def job_b(a):
    sd, name, mu, N = a
    t = fs.generate_tasks(10, 2500, seed=sd); r = fs.RFFMap(4, 200, seed=sd); pr = make_client_probe(t, r, N)
    v = run_with_test(lambda p: fs.run_pso_fmtl(t, r, M=35, mu=mu, n_rounds=N, theta=SCH[name], shift_step=1,
            coordinated=False, omega_nonneg=True, probe=p), pr)[0]
    return ("B", name, mu, N, sd, v)

if __name__ == "__main__":
    ja = [(s, n, h, mu, l) for s in range(6) for n in SM for h in (1, 3, 6) for mu in (0.05, 0.5) for l in (0.05, 0.5)]
    jb = [(s, n, mu, N) for s in range(6) for n in SCH for mu in (0.05, 0.5) for N in (500, 2000)]
    with Pool() as p: A = p.map(job_a, ja); B = p.map(job_b, jb)
    json.dump({"A": A, "B": B}, open("extra_checks.json", "w"))
    print("(1) change in test MSE (dB) when the coupling is on (negative = coupling helps), mean over h and seeds")
    for mu in (0.05, 0.5):
        for l in (0.05, 0.5):
            base = np.array([r[6] for r in A if r[1] == "no coupling" and r[3] == mu and r[4] == l])
            print(f" mu={mu} lam1={l}: no coupling {base.mean():6.2f} |", " ".join(
                f"{n}: {np.mean(np.array([r[6] for r in A if r[1]==n and r[3]==mu and r[4]==l])-base):+.2f}" for n in list(SM)[1:]))
    print("(2) beta schedules: change vs beta=1 (positive = worse)")
    for mu in (0.05, 0.5):
        for N in (500, 2000):
            base = np.array([r[5] for r in B if r[1] == "beta=1" and r[2] == mu and r[3] == N])
            print(f" mu={mu} rounds={N}: beta=1 {base.mean():6.2f} |", " ".join(
                f"{n}: {np.mean(np.array([r[5] for r in B if r[1]==n and r[2]==mu and r[3]==N])-base):+.2f}" for n in list(SCH)[1:]))
