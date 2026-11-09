import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
names=["FedAvg","FedProx","Online FMTL","FedSS\n($\\rho$=0.6)","PSO-FMTL\n(M=35)"]; v=[400,400,400,240,70]
c=["#8d99ae","#8d99ae","#2a9d8f","#e9c46a","#d62828"]
f,a=plt.subplots(figsize=(4.6,2.8)); b=a.bar(names,v,color=c,width=.62)
for r,x in zip(b,v): a.text(r.get_x()+r.get_width()/2,x+8,f"{x}\n({x/400:.1%})".replace("100.0%","100%"),ha="center",fontsize=7)
a.set_ylabel("params / client / round\n(up + down)",fontsize=8); a.set_ylim(0,500); a.tick_params(labelsize=7)
for s in("top","right"): a.spines[s].set_visible(False)
f.tight_layout(); f.savefig("fig_params_bar.png",dpi=300)
