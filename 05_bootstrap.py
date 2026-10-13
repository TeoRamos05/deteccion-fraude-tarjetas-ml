"""Incertidumbre y significancia: bootstrap sobre el conjunto de prueba (requiere predicciones guardadas).

Para cada dataset y semilla, remuestrea B veces las filas de test (las MISMAS filas para todos los modelos,
lo que permite comparaciones pareadas) y calcula:
  - IC 95 % percentil del AUC-PR de cada combinación modelo x balanceo.
  - Diferencias pareadas con IC 95 % y proporción de remuestras con diferencia <= 0 (pseudo p-valor
    unilateral) para los contrastes de H1 (ensambles > referencia) y H2 (SMOTE > sin balanceo).
Nota: con partición temporal el remuestreo i.i.d. ignora la dependencia temporal; declararlo en limitaciones.
Uso: python 05_bootstrap.py [B=1000]
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from common import OUT

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
PRED = OUT / "predicciones"
MODELOS = ["Regresión_logística", "Árbol_de_decisión", "Random_Forest", "XGBoost"]
BAL = ["sin_balanceo", "SMOTE", "pesos"]
rng = np.random.default_rng(12345)

# Contrastes: (nombre, A, B)  ->  diferencia = AUC-PR(A) - AUC-PR(B)
CONTRASTES = []
for ens in ("Random_Forest", "XGBoost"):
    for ref in ("Regresión_logística", "Árbol_de_decisión"):
        for b in BAL:
            CONTRASTES.append((f"H1: {ens} - {ref} [{b}]", (ens, b), (ref, b)))
CONTRASTES.append(("XGBoost - Random_Forest [sin_balanceo]", ("XGBoost", "sin_balanceo"), ("Random_Forest", "sin_balanceo")))
CONTRASTES.append(("XGBoost - Random_Forest [pesos]", ("XGBoost", "pesos"), ("Random_Forest", "pesos")))
for m in MODELOS:
    CONTRASTES.append((f"H2: {m} SMOTE - sin_balanceo", (m, "SMOTE"), (m, "sin_balanceo")))
    CONTRASTES.append((f"{m} pesos - sin_balanceo", (m, "pesos"), (m, "sin_balanceo")))

filas_ic, filas_dif = [], []
for ds in ("ulb", "ieee_cis"):
    semillas = sorted({int(f.stem.split("__")[-1]) for f in PRED.glob(f"{ds}__*.npz")})
    for seed in semillas:
        d = {(m, b): np.load(PRED / f"{ds}__{m}__{b}__{seed}.npz") for m in MODELOS for b in BAL}
        y = d[(MODELOS[0], BAL[0])]["y"]
        n = len(y)
        idx = [rng.integers(0, n, n) for _ in range(B)]
        idx = [i for i in idx if y[i].sum() > 0]
        ap = {k: np.array([average_precision_score(y[i], v["p"][i]) for i in idx]) for k, v in d.items()}
        for (m, b), a in ap.items():
            filas_ic.append(dict(dataset=ds, semilla=seed, modelo=m, balanceo=b,
                                 AUC_PR=average_precision_score(y, d[(m, b)]["p"]),
                                 IC95_inf=np.percentile(a, 2.5), IC95_sup=np.percentile(a, 97.5)))
        for nombre, A, Bk in CONTRASTES:
            dif = ap[A] - ap[Bk]
            filas_dif.append(dict(dataset=ds, semilla=seed, contraste=nombre, dif_media=dif.mean(),
                                  IC95_inf=np.percentile(dif, 2.5), IC95_sup=np.percentile(dif, 97.5),
                                  prop_dif_menor_igual_0=(dif <= 0).mean()))
        print(ds, seed, "ok", flush=True)

pd.DataFrame(filas_ic).round(4).to_csv(OUT / "bootstrap_IC_AUC_PR.csv", index=False)
dif = pd.DataFrame(filas_dif).round(4)
dif.to_csv(OUT / "bootstrap_diferencias.csv", index=False)
print(dif[dif.semilla == dif.groupby("dataset").semilla.transform("min")].to_string())
