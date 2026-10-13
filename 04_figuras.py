"""Tablas y figuras comparativas finales (ULB e IEEE-CIS)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from common import OUT

NOMBRES = {"ulb": "ULB (partición aleatoria)", "ieee_cis": "IEEE-CIS (partición temporal)"}
MODELOS = ["Regresión logística", "Árbol de decisión", "Random Forest", "XGBoost"]
BAL = ["sin_balanceo", "SMOTE", "pesos"]
COL = {"sin_balanceo": "#8C8C8C", "SMOTE": "#C44E52", "pesos": "#4C72B0"}

res = pd.concat([pd.read_csv(OUT / f"{d}_resultados_crudos.csv") for d in NOMBRES])
met = ["AUC_PR", "AUC_ROC", "Precision", "Recall", "F1", "Costo"]
tabla = res.groupby(["dataset", "modelo", "balanceo"])[met].agg(["mean", "std"]).round(4)
tabla.to_csv(OUT / "tabla_final_comparativa.csv")

for metrica, etiqueta in (("AUC_PR", "AUC-PR (media ± desv. entre semillas)"), ("F1", "F1-score")):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, ds in zip(axs, NOMBRES):
        d = res[res.dataset == ds].groupby(["modelo", "balanceo"])[metrica].agg(["mean", "std"]).fillna(0)
        w = 0.26
        for j, b in enumerate(BAL):
            m = [d.loc[(mo, b), "mean"] for mo in MODELOS]
            s = [d.loc[(mo, b), "std"] for mo in MODELOS]
            ax.bar([i + (j - 1) * w for i in range(4)], m, w, yerr=s, capsize=2,
                   label=b.replace("_", " "), color=COL[b])
        ax.set_xticks(range(4)); ax.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELOS])
        ax.set_title(NOMBRES[ds]); ax.set_ylabel(etiqueta); ax.set_ylim(0, 1)
    axs[0].legend(title="Balanceo")
    plt.tight_layout(); plt.savefig(OUT / f"comparativa_{metrica}.png", dpi=150); plt.close()

# Impacto de SMOTE: diferencia de AUC-PR frente a no balancear
fig, ax = plt.subplots(figsize=(7, 4))
for k, ds in enumerate(NOMBRES):
    d = res[res.dataset == ds].groupby(["modelo", "balanceo"])["AUC_PR"].mean()
    dif = [d[(m, "SMOTE")] - d[(m, "sin_balanceo")] for m in MODELOS]
    ax.bar([i + k * 0.38 for i in range(4)], dif, 0.38, label=NOMBRES[ds])
ax.axhline(0, color="k", lw=0.8); ax.set_xticks([i + 0.19 for i in range(4)])
ax.set_xticklabels([m.replace(" ", "\n", 1) for m in MODELOS])
ax.set_ylabel("Δ AUC-PR (SMOTE − sin balanceo)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "efecto_smote.png", dpi=150); plt.close()
print(res.groupby(["dataset", "modelo", "balanceo"])[["AUC_PR", "F1"]].mean().round(3).to_string())
