"""Summaries and paired t-tests from paper_results_v2.json (used for the paper's tables)."""
import json, numpy as np
from scipy import stats
R = json.load(open("paper_results_v2.json")); S = sorted(R, key=int)
g = lambda f: np.array([f(R[s]) for s in S])
ms = lambda a: f"{a.mean():7.2f} +/- {a.std(ddof=1):4.2f}"
def pdiff(a, b):
    d = a - b; p = stats.ttest_1samp(d, 0).pvalue
    return f"{d.mean():+.2f} dB (p={p:.1e}, {int((d<0).sum())}/{len(d)} seeds a<b)"
if __name__ == "__main__":
    print("== methods (mu=0.05)")
    for k in R[S[0]]["methods"]: print(f"  {k:<20}{ms(g(lambda r: r['methods'][k]))}")
    print("  time ms:", {k: round(float(np.mean(g(lambda r: r['time_ms'][k]))),0) for k in R[S[0]]['time_ms']})
    m = lambda k: g(lambda r: r["methods"][k])
    for a, b in [("PSO-Fed u","FedAvg"),("PSO-FMTL u","FedAvg"),("PSO-FMTL u","PSO-Fed u"),("PSO-FMTL u","Online FMTL"),("PSO-Fed c","FedAvg"),("PSO-FMTL c","PSO-Fed c"),("PSO-Fed c","PSO-Fed u"),("PSO-FMTL c","PSO-FMTL u"),("PSO-FMTL b0.6 u","PSO-FMTL u"),("Online FMTL","FedAvg"),("Local-only","FedAvg")]:
        print(f"  {a} - {b}: {pdiff(m(a), m(b))}")
    print("== ablation")
    for k in R[S[0]]["ablation"]: print(f"  {k:<18}{ms(g(lambda r: r['ablation'][k]))}")
    for nm in ("c","u"):
        f = g(lambda r: r["ablation"][f"full {nm}"])
        for k in ("no Omega","no partial","beta0.6","frozen","theta1"):
            print(f"  [{nm}] {k} - full: {pdiff(g(lambda r: r['ablation'][f'{k} {nm}']), f)}")
    print("== shift")
    for nm in ("c","u"):
        print(" ", nm, {s: round(float(np.mean(g(lambda r: r['shift'][f'{nm}|{s}']))),2) for s in (0,1,2,5,10,17,35,70,100)})
    print("== Msweep")
    for nm in ("c","u"):
        for M in (1,5,10,20,35,50,70,100):
            a = g(lambda r: r["Msweep"][f"PSO-Fed|{nm}|{M}"]); b = g(lambda r: r["Msweep"][f"PSO-FMTL|{nm}|{M}"])
            print(f"  {nm} M={M:<4} PSO-Fed {ms(a)}  PSO-FMTL {ms(b)}  diff {pdiff(b,a)}")
    print("== mu")
    for mu in ("0.05","0.2","0.5","0.75"):
        print(" mu", mu, {k: round(float(np.mean(g(lambda r: r['mu'][mu][k]))),2) for k in R[S[0]]['mu'][mu]})
        fa = g(lambda r: r["mu"][mu]["FedAvg"])
        print("    PSO-Fed u - FedAvg:", pdiff(g(lambda r: r["mu"][mu]["PSO-Fed u"]), fa),"| PSO-FMTL u - PSO-Fed u:", pdiff(g(lambda r: r["mu"][mu]["PSO-FMTL u"]), g(lambda r: r["mu"][mu]["PSO-Fed u"])), "| PSO-FMTL u - OnlineFMTL:", pdiff(g(lambda r: r["mu"][mu]["PSO-FMTL u"]), g(lambda r: r["mu"][mu]["Online FMTL"])))
    print("== single task")
    for k in R[S[0]]["single_task"]: print(f"  {k:<14}{ms(g(lambda r: r['single_task'][k]))}")
    st = lambda k: g(lambda r: r["single_task"][k])
    print("  PSO-Fed u - FedAvg:", pdiff(st("PSO-Fed u"), st("FedAvg")), "| PSO-FMTL u - FedAvg:", pdiff(st("PSO-FMTL u"), st("FedAvg")), "| PSO-FMTL u - PSO-Fed u:", pdiff(st("PSO-FMTL u"), st("PSO-Fed u")))
    print("== rff")
    for d in ("50","100","200","400"): print(" ", d, R[S[0]]["rff"][d]["M"], ms(g(lambda r: r["rff"][d]["PSO-FMTL u"])), "| FedAvg", ms(g(lambda r: r["rff"][d]["FedAvg"])))
