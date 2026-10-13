"""Validación externa: IEEE-CIS Fraud Detection con partición TEMPORAL 70/15/15.

Preparación:
  - Unión transaction + identity por TransactionID (left join).
  - Se descartan columnas con >60 % de nulos; se añade hora del día (de TransactionDT).
  - Categóricas -> códigos ordinales (categorías ajustadas solo con train).
  - Imputación por mediana (ajustada con train) + escalado para los modelos lineales.
  - Orden cronológico: train = primeros 70 %, val = siguientes 15 %, test = últimos 15 %.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from common import BASE
from experimento import correr

D = BASE / "data" / "ieee_cis"
SEEDS = [int(s) for s in sys.argv[1:]] or [42, 7]
CATS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]

tx = pd.read_csv(D / "train_transaction.csv")
idt = pd.read_csv(D / "train_identity.csv", usecols=["TransactionID", "DeviceType"])
df = tx.merge(idt, on="TransactionID", how="left").sort_values("TransactionDT").reset_index(drop=True)
print("IEEE-CIS:", df.shape, f"fraude={df.isFraud.mean():.3%}")

y = df["isFraud"].astype(int)
monto = df["TransactionAmt"].values
df["hora"] = (df["TransactionDT"] // 3600) % 24
X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
n = len(X)
i1, i2 = int(n * 0.70), int(n * 0.85)
keep = X.columns[X.iloc[:i1].isna().mean() <= 0.60]  # criterio con train solamente
X = X[keep]
cats = [c for c in CATS if c in X.columns]
for c in cats:
    cat = pd.Categorical(X[c].iloc[:i1]).categories
    X[c] = pd.Categorical(X[c], categories=cat).codes.astype("float32")
    X.loc[X[c] < 0, c] = np.nan
X = X.astype("float32")
med = X.iloc[:i1].median()
X = X.fillna(med)
sc = StandardScaler().fit(X.iloc[:i1])
X = pd.DataFrame(sc.transform(X), columns=X.columns, dtype="float32")
print("Predictores finales:", X.shape[1], "| categóricas codificadas:", len(cats))
print(f"Fraude por tramo: train={y[:i1].mean():.3%} val={y[i1:i2].mean():.3%} test={y[i2:].mean():.3%}")


def particion(seed):
    return (X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:], y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:],
            monto[i2:])


# SMOTE parcial (fraude -> 20 % de la clase mayoritaria): el SMOTE completo generaría ~800 mil filas.
correr("ieee_cis", particion, SEEDS, smote_strategy=0.2, ligero=True)
