"""Utilidades comunes: carga de datos, partición y métricas."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)

BASE = Path(__file__).parent
DATA = BASE / "data" / "ulb" / "creditcard.csv"
OUT = BASE / "resultados"
OUT.mkdir(exist_ok=True)

# Costos supuestos (ajustar con datos reales de la operación):
# un fraude no detectado (FN) cuesta en promedio el monto de la transacción;
# un falso positivo (FP) cuesta un valor fijo de fricción/revisión manual.
COSTO_FP = 5.0


def cargar():
    """Carga el CSV. Si trae 'Time' (versión Kaggle) elimina duplicados exactos.

    La copia de OpenML no trae 'Time'; allí los 'duplicados' son un artefacto
    de haber quitado esa columna, por lo que NO se eliminan.
    """
    df = pd.read_csv(DATA)
    n0 = len(df)
    if "Time" in df.columns:
        df = df.drop_duplicates().reset_index(drop=True)
    info = {"tiene_time": "Time" in df.columns, "filas_original": n0,
            "filas_finales": len(df), "fraudes": int(df["Class"].sum())}
    return df, info


def particionar(df, seed=42):
    """Partición estratificada 70/15/15; el escalado se ajusta SOLO con train."""
    X, y = df.drop(columns="Class"), df["Class"].astype(int)
    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(
        Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=seed)
    cols = [c for c in ["Amount", "Time"] if c in X.columns]
    sc = StandardScaler().fit(Xtr[cols])
    out = []
    for Xs in (Xtr, Xva, Xte):
        Xs = Xs.copy()
        Xs[cols] = sc.transform(Xs[cols])
        out.append(Xs)
    return out[0], out[1], out[2], ytr, yva, yte


def costo_total(y, pred, monto):
    """Costo = monto de cada fraude no detectado + COSTO_FP por falso positivo."""
    y, pred, monto = np.asarray(y), np.asarray(pred), np.asarray(monto)
    fn = (y == 1) & (pred == 0)
    fp = (y == 0) & (pred == 1)
    return float(monto[fn].sum() + COSTO_FP * fp.sum())


def mejor_umbral_f1(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.90, 0.9999, 400)))
    f = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f))])


def metricas(y, p, umbral):
    pred = (p >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"AUC_ROC": roc_auc_score(y, p), "AUC_PR": average_precision_score(y, p),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "umbral": umbral}
