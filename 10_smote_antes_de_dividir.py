"""Experimento de contraste: SMOTE aplicado a TODO el conjunto ANTES de dividir (protocolo con fuga de información).

Replica lo que hacen algunos antecedentes (p. ej., Khalid et al., 2024; Ileberi et al., 2021): se aplica SMOTE a los 283 726
registros de ULB, y después se divide en entrenamiento, validación y prueba (70/15/15 estratificada). Se entrena cada algoritmo con
la misma grilla de 02_modelado.py y se mide en dos pruebas:
  - 'completa': la prueba tal como queda tras el SMOTE (contiene fraudes sintéticos; es lo que reportan esos antecedentes);
  - 'solo reales': solo las filas originales de la prueba (los fraudes reales), aunque el modelo se entrenó con sintéticos
    interpolados a partir de ellos (fuga indirecta).
Se compara con el protocolo correcto del estudio (SMOTE solo sobre entrenamiento), en resultados/ulb_resultados_crudos.csv.
Uso: python 10_smote_antes_de_dividir.py [semillas]   (por defecto 42 7 2024)
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from common import cargar, OUT
from experimento import modelos

SEEDS = [int(s) for s in sys.argv[1:]] or [42, 7, 2024]
df, info = cargar()
X, y = df.drop(columns="Class"), df["Class"].astype(int)
print(info, flush=True)


def umbral_f1(yv, p):
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.9999, 400)))   # prueba balanceada: el umbral ya no está en el 10 % superior
    f = [f1_score(yv, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def medir(yt, p, u):
    pred = (p >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return dict(AUC_PR=average_precision_score(yt, p), AUC_ROC=roc_auc_score(yt, p), Precision=precision_score(yt, pred, zero_division=0),
                Recall=recall_score(yt, pred, zero_division=0), F1=f1_score(yt, pred, zero_division=0), TP=tp, FP=fp, FN=fn, TN=tn,
                n_pos=int(yt.sum()), n_neg=int((yt == 0).sum()))


filas = []
for seed in SEEDS:
    t0 = time.time()
    Xr, yr = SMOTE(k_neighbors=5, random_state=seed).fit_resample(X, y)          # ANTES de dividir (a propósito)
    yr = np.asarray(yr)
    sint = np.zeros(len(Xr), bool); sint[len(X):] = True                           # imblearn agrega los sintéticos al final
    idx = np.arange(len(Xr))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=yr, random_state=seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=yr[i_tmp], random_state=seed)
    cols = [c for c in ["Amount", "Time"] if c in Xr.columns]
    sc = StandardScaler().fit(Xr.iloc[i_tr][cols])
    Xs = Xr.copy(); Xs[cols] = sc.transform(Xr[cols])
    Xtr, Xva, Xte = Xs.iloc[i_tr], Xs.iloc[i_va], Xs.iloc[i_te]
    ytr, yva, yte = yr[i_tr], yr[i_va], yr[i_te]
    real_te = ~sint[i_te]
    print(f"\n=== semilla {seed} | tras SMOTE: {len(Xr)} filas ({int(yr.sum())} fraudes, {int(sint.sum())} sintéticos) | "
          f"train={len(Xtr)} val={len(Xva)} test={len(Xte)} (fraudes en test: {int(yte.sum())}; sintéticos en test: {int((~real_te & (yte == 1)).sum())}; "
          f"reales en test: {int((real_te & (yte == 1)).sum())})", flush=True)
    for mod, (mk, grid) in modelos(seed, "SMOTE", 1.0, False).items():
        t1 = time.time()
        best = max(((average_precision_score(yva, mk(**g).fit(Xtr, ytr).predict_proba(Xva)[:, 1]), g) for g in grid), key=lambda x: x[0])
        m = mk(**best[1]).fit(Xtr, ytr)
        u = umbral_f1(yva, m.predict_proba(Xva)[:, 1])
        p = m.predict_proba(Xte)[:, 1]
        for prueba, msk in (("completa", np.ones(len(yte), bool)), ("solo reales", real_te)):
            r = medir(yte[msk], p[msk], u)
            r.update(modelo=mod, semilla=seed, prueba=prueba, umbral=u, params=str(best[1]))
            filas.append(r)
        a = filas[-2]; b = filas[-1]
        print(f"{mod:20s} completa: AUC-PR={a['AUC_PR']:.3f} F1={a['F1']:.3f} P={a['Precision']:.3f} R={a['Recall']:.3f} | "
              f"solo reales: AUC-PR={b['AUC_PR']:.3f} F1={b['F1']:.3f} P={b['Precision']:.3f} R={b['Recall']:.3f} [{time.time()-t1:.0f}s]", flush=True)
    pd.DataFrame(filas).to_csv(OUT / "smote_antes_de_dividir.csv", index=False)
    print(f"semilla {seed} lista en {time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(filas)
print("\nResumen (media de semillas):")
print(res.groupby(["prueba", "modelo"])[["AUC_PR", "AUC_ROC", "Precision", "Recall", "F1"]].mean().round(3).to_string())
