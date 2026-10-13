"""Análisis complementario de las predicciones de prueba (no reentrena): calibración, captura en las K alertas más
altas, fraudes no detectados por banda de monto, curvas precisión-exhaustividad y costo con umbral elegido por costo.

El último análisis (umbral por costo) es EXPLORATORIO: el umbral se elige con el propio conjunto de prueba porque no se
guardaron predicciones de validación; por eso es una cota optimista del ahorro posible y no una evaluación válida.
Uso: python 08_analisis_errores.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, brier_score_loss
from common import OUT

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PRED = OUT / "predicciones"
MOD = ["Regresión_logística", "Árbol_de_decisión", "Random_Forest", "XGBoost"]
BAL = ["sin_balanceo", "SMOTE", "pesos"]
KS = [0.001, 0.005, 0.01, 0.02, 0.05]
BANDAS = [0, 50, 100, 250, 1000, 1e12]
BANDAN = ["≤50", "50–100", "100–250", "250–1 000", ">1 000"]
ESC = [("Ca=5", 5.0, 0.0), ("Ca=5 + 25 % del monto", 5.0, 0.25), ("Ca=5 + 100 % del monto", 5.0, 1.0)]

cal, topk, bandas, orac = [], [], [], []
for ds in ("ulb", "ieee_cis"):
    for f in sorted(PRED.glob(f"{ds}__*.npz")):
        _, mod, bal, seed = f.stem.split("__")
        z = np.load(f)
        y, p, m, u = z["y"], z["p"], z["monto"], float(z["umbral"])
        base = dict(dataset=ds, modelo=mod.replace("_", " "), balanceo=bal, semilla=int(seed))
        cal.append(dict(base, tasa_real=y.mean(), p_media=p.mean(), brier=brier_score_loss(y, p)))
        o = np.argsort(-p)
        for k in KS:
            n = max(1, int(round(len(y) * k)))
            sel = o[:n]
            topk.append(dict(base, k=k, alertas=n, fraudes_capt=int(y[sel].sum()), fraudes_tot=int(y.sum()),
                             monto_capt=float(m[sel][y[sel] == 1].sum()), monto_tot=float(m[y == 1].sum()),
                             precision=float(y[sel].mean())))
        pred = p >= u
        b = pd.cut(m, BANDAS, labels=BANDAN, right=True)
        for nb in BANDAN:
            msk = (y == 1) & (b == nb)
            bandas.append(dict(base, banda=nb, fraudes=int(msk.sum()), detectados=int((msk & pred).sum()),
                               monto=float(m[msk].sum()), monto_no_detectado=float(m[msk & ~pred].sum())))
        # umbral elegido por costo en el propio test (cota optimista)
        sin_modelo = float(m[y == 1].sum())
        for nom, ca, lam in ESC:
            fpc = np.where(y == 0, ca + lam * m, 0.0)
            fnc = np.where(y == 1, m, 0.0)
            c_f1 = float(fnc[p < u].sum() + fpc[p >= u].sum())
            # costo para cada número k de alertas (las k puntuaciones más altas); k = 0 equivale a no usar modelo
            costos = sin_modelo - np.cumsum(fnc[o]) + np.cumsum(fpc[o])
            c_min = float(min(costos.min(), sin_modelo, c_f1))
            orac.append(dict(base, escenario=nom, costo_umbral_f1=c_f1, costo_umbral_costo=c_min, sin_modelo=sin_modelo))

cal = pd.DataFrame(cal); topk = pd.DataFrame(topk); bandas = pd.DataFrame(bandas); orac = pd.DataFrame(orac)
cal.groupby(["dataset", "modelo", "balanceo"])[["tasa_real", "p_media", "brier"]].mean().round(5).reset_index().to_csv(
    OUT / "analisis_calibracion.csv", index=False)
t = topk.groupby(["dataset", "modelo", "balanceo", "k"])[["alertas", "fraudes_capt", "fraudes_tot", "monto_capt", "monto_tot", "precision"]].mean()
t["pct_fraudes"] = t.fraudes_capt / t.fraudes_tot * 100
t["pct_monto"] = t.monto_capt / t.monto_tot * 100
t.round(3).reset_index().to_csv(OUT / "analisis_topk.csv", index=False)
bn = bandas.groupby(["dataset", "modelo", "balanceo", "banda"])[["fraudes", "detectados", "monto", "monto_no_detectado"]].mean()
bn.round(2).reset_index().to_csv(OUT / "analisis_bandas_monto.csv", index=False)
oa = orac.groupby(["dataset", "modelo", "balanceo", "escenario"])[["costo_umbral_f1", "costo_umbral_costo", "sin_modelo"]].mean()
oa.round(1).reset_index().to_csv(OUT / "analisis_umbral_costo.csv", index=False)

# curvas PR (semilla 42, sin balanceo)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (ds, nom) in zip(axs, (("ulb", "ULB"), ("ieee_cis", "IEEE-CIS"))):
    for mod in MOD:
        z = np.load(PRED / f"{ds}__{mod}__sin_balanceo__42.npz")
        pr, rc, _ = precision_recall_curve(z["y"], z["p"])
        ax.plot(rc, pr, label=mod.replace("_", " "), lw=1.6)
    ax.axhline(z["y"].mean(), color="gray", ls=":", lw=1, label="Azar (tasa de fraude)")
    ax.set_xlabel("Exhaustividad"); ax.set_ylabel("Precisión"); ax.set_title(f"{nom} (sin balanceo, semilla 42)")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "curvas_pr.png", dpi=150); plt.close()

pd.set_option("display.width", 250)
print("## calibración (sin balanceo / SMOTE / pesos)")
print(pd.read_csv(OUT / "analisis_calibracion.csv").to_string())
print("\n## top-K XGBoost y LR sin balanceo")
x = pd.read_csv(OUT / "analisis_topk.csv")
print(x[(x.balanceo == "sin_balanceo") & x.modelo.isin(["XGBoost", "Regresión logística", "Random Forest", "Árbol de decisión"])][
    ["dataset", "modelo", "k", "alertas", "pct_fraudes", "pct_monto", "precision"]].to_string())
print("\n## bandas XGBoost sin balanceo")
b = pd.read_csv(OUT / "analisis_bandas_monto.csv")
print(b[(b.modelo == "XGBoost") & (b.balanceo == "sin_balanceo")].to_string())
print("\n## umbral por costo (exploratorio) mejores combinaciones")
o = pd.read_csv(OUT / "analisis_umbral_costo.csv")
for (ds, esc), g in o.groupby(["dataset", "escenario"]):
    print(ds, esc, "sin modelo", round(g.sin_modelo.iloc[0]))
    print(g.sort_values("costo_umbral_costo").head(3)[["modelo", "balanceo", "costo_umbral_f1", "costo_umbral_costo"]].to_string())
