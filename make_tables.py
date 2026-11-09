"""Generate the LaTeX table fragments of the paper from the result JSON files (no hand-typed numbers)."""
import json, numpy as np
from scipy import stats
R = json.load(open("paper_results_v2.json")); S = sorted(R, key=int)
NT = json.load(open("ntasks_results.json")); Q = json.load(open("quant_results.json"))
g = lambda f: np.array([f(R[s]) for s in S])
def pm(a): return f"${np.mean(a):.2f}\\pm{np.std(a, ddof=1):.2f}$"
def pstr(p): return "$<\\!10^{-3}$" if p < 1e-3 else f"${p:.3f}$"
def d_p(a, b):
    d = np.asarray(a) - np.asarray(b); return d.mean(), stats.ttest_1samp(d, 0).pvalue, int((d < 0).sum())
def w(name, s): open(f"paper/{name}.tex", "w").write(s)

# ---- Table: methods (accuracy + bits)
m = lambda k: g(lambda r: r["methods"][k]); st = lambda k: g(lambda r: r["single_task"][k])
rows = [("Local-only", 0, 0, m("Local-only"), st("Local-only")),
        ("FedAvg", 400, Q["FedAvg float32"]["bits_per_client_round"], m("FedAvg"), st("FedAvg")),
        ("FedProx", 400, Q["FedAvg float32"]["bits_per_client_round"], m("FedProx"), None),
        ("Online FMTL", 400, 12800, m("Online FMTL"), st("Online FMTL")),
        ("FedAvg, 8-bit", 400, Q["FedAvg 8-bit"]["bits_per_client_round"], np.array(Q["FedAvg 8-bit"]["mse"]), None),
        ("FedAvg, 4-bit", 400, Q["FedAvg 4-bit"]["bits_per_client_round"], np.array(Q["FedAvg 4-bit"]["mse"]), None),
        ("PSO-Fed, coord.", 70, 2240, m("PSO-Fed c"), st("PSO-Fed c")),
        ("PSO-Fed, uncoord.", 70, 2240, m("PSO-Fed u"), st("PSO-Fed u")),
        ("PSO-FMTL, coord.", 70, 2240, m("PSO-FMTL c"), st("PSO-FMTL c")),
        ("PSO-FMTL, uncoord.", 70, 2240, m("PSO-FMTL u"), st("PSO-FMTL u")),
        ("PSO-FMTL $\\beta{=}0.6$, uncoord.", 70, 2240, m("PSO-FMTL b0.6 u"), None)]
fmtb = lambda b: f"{b:,}".replace(",", "\\,")
body = "".join(f"{n} & {p} & {fmtb(b)} & {pm(a)} & {pm(s) if s is not None else '--'} \\\\\n" for n, p, b, a, s in rows)
w("tab_methods", "\\begin{table}[t]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}\n\\caption{Final MSE (dB, mean $\\pm$ std over 10 seeds, $\\mu=0.05$, $K=10$, $M=35$). Params/bits are per client per round, uplink plus downlink (float32 unless stated). Single task: all clients observe the same function.}\n\\label{tab:methods}\n\\begin{tabular}{@{}lccrc@{}}\n\\toprule\nMethod & Params & Bits & $K{=}10$ tasks & Single task \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---- Table: step size
cols = ["Local-only", "FedAvg", "Online FMTL", "PSO-Fed c", "PSO-Fed u", "PSO-FMTL c", "PSO-FMTL u"]
hdr = "$\\mu$ & Local & FedAvg & Online FMTL & PSO-Fed c & PSO-Fed u & PSO-FMTL c & PSO-FMTL u"
body = ""
for mu in ("0.05", "0.2", "0.5", "0.75"):
    vals = [g(lambda r: r["mu"][mu][c]).mean() for c in cols]; best = min(vals)
    body += f"{mu} & " + " & ".join((f"\\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}") for v in vals) + " \\\\\n"
w("tab_mu", "\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{5pt}\n\\caption{Step-size sensitivity: final MSE (dB, mean of 10 seeds, $M=35$, $\\sigma=1$, $K=10$; best value in bold; ``c'' coordinated, ``u'' uncoordinated, PSO-FMTL with $\\beta=1$). Seed-to-seed standard deviations are 1--2.5~dB; paired tests are in the text.}\n\\label{tab:mu}\n\\begin{tabular}{@{}cccccccc@{}}\n\\toprule\n" + hdr + " \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---- Table: ablation
A = lambda k: g(lambda r: r["ablation"][k])
items = [("Full PSO-FMTL ($\\beta{=}1$, $\\Omega$ on, $\\theta{=}0$)", "full"), ("No $\\Omega$ coupling", "no Omega"), ("Smoothed $\\Omega$, $\\theta{=}1$", "theta1"),
         ("No partial sharing ($M{=}D$)", "no partial"), ("Server mixing $\\beta{=}0.6$", "beta0.6"), ("Frozen mask ($\\sigma{=}0$)", "frozen")]
body = ""
for lab, k in items:
    row = lab
    for nm in ("c", "u"):
        a = A(f"{k} {nm}")
        if k == "full": row += f" & {pm(a)} & -- & --"
        else:
            dm, p, _ = d_p(a, A(f"full {nm}")); row += f" & {pm(a)} & ${dm:+.2f}$ & {pstr(p)}"
    body += row + " \\\\\n"
w("tab_ablation", "\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\caption{Component ablation ($K=10$, $\\mu=0.05$, $M=35$, $\\sigma=1$, 10 seeds). $\\Delta$: paired mean difference to the full model in dB (positive = worse), $p$: paired $t$-test.}\n\\label{tab:ablation}\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{Coordinated masks} & \\multicolumn{3}{c}{Uncoordinated masks} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nConfiguration & MSE (dB) & $\\Delta$ & $p$ & MSE (dB) & $\\Delta$ & $p$ \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---- Table: number of tasks
body = ""
for mu in (0.05, 0.5):
    for K in (5, 10, 20):
        d = NT[f"K{K}|mu{mu}"]; on = np.array(d["Online-FMTL(full)"])
        c35 = d_p(d["PSO-FMTL|th1.0|M35"], on); c20 = d_p(d["PSO-FMTL|th1.0|M20"], on)
        body += (f"{mu} & {K} & {pm(on)} & {pm(d['FedAvg(full)'])} & {pm(d['PSO-Fed|M35'])} & {pm(d['PSO-FMTL|th1.0|M35'])} "
                 f"& ${c35[0]:+.2f}$ ({pstr(c35[1])}) & ${c20[0]:+.2f}$ ({pstr(c20[1])}) \\\\\n")
    if mu == 0.05: body += "\\midrule\n"
w("tab_ntasks", "\\begin{table*}[t]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\caption{Number of tasks $K$ (one task per client): final MSE (dB) of Online FMTL (400 params), FedAvg (400), PSO-Fed (70) and PSO-FMTL (70, $M{=}35$, $\\beta{=}1$, uncoordinated, clipped $\\Omega$ weights); 10 seeds. Last two columns: paired difference PSO-FMTL $-$ Online FMTL in dB (negative = PSO-FMTL better) with the paired $p$-value, for $M{=}35$ and $M{=}20$ (40 params).}\n\\label{tab:ntasks}\n\\begin{tabular}{@{}cccccccc@{}}\n\\toprule\n$\\mu$ & $K$ & Online FMTL & FedAvg & PSO-Fed & PSO-FMTL & $\\Delta_{M=35}$ ($p$) & $\\Delta_{M=20}$ ($p$) \\\\\n\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# print the facts used in the text
print("quant vs PSO-FMTL u (paired):", d_p(Q["PSO-FMTL M=35 float32"]["mse"], Q["FedAvg 4-bit"]["mse"]))
print("PSO-FMTL u vs FedAvg 8-bit:", d_p(Q["PSO-FMTL M=35 float32"]["mse"], Q["FedAvg 8-bit"]["mse"]))
for k in ("PSO-Fed u","PSO-FMTL u"): print(k,"vs Local", d_p(m(k), m("Local-only")))
print("tables written")
