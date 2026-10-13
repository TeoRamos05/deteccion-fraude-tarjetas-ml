"""Sensibilidad de la grilla de hiperparámetros en IEEE-CIS.

En 03_ieee_cis.py los cuatro algoritmos eligieron el valor más alto de la grilla (C = 10, profundidad 12 del árbol con hoja mínima 20,
profundidad 20 de Random Forest y profundidad 6 de XGBoost). Aquí se repite el experimento completo (4 modelos x 3 estrategias) con una
grilla ampliada hacia esos extremos, con la misma partición temporal y el mismo protocolo, para una semilla (por defecto 42), y se
guarda aparte (nombre 'ieee_cis_grillaamp') para no mezclar con los resultados principales.
Uso: python 11_grilla_ampliada_ieee.py [semillas]   (por defecto 42). Si se corta, relanzar: reanuda desde las predicciones guardadas.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from common import BASE
import experimento
from experimento import correr

D = BASE / "data" / "ieee_cis"
SEEDS = [int(s) for s in sys.argv[1:]] or [42]
# ---- preparación idéntica a 03_ieee_cis.py ----
tx = pd.read_csv(D / "train_transaction.csv")
idt = pd.read_csv(D / "train_identity.csv", usecols=["TransactionID", "DeviceType"])
df = tx.merge(idt, on="TransactionID", how="left").sort_values("TransactionDT").reset_index(drop=True)
y = df["isFraud"].astype(int)
monto = df["TransactionAmt"].values
df["hora"] = (df["TransactionDT"] // 3600) % 24
X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
n = len(X)
i1, i2 = int(n * 0.70), int(n * 0.85)
keep = X.columns[X.iloc[:i1].isna().mean() <= 0.60]
X = X[keep]
cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
for c in cats:
    cat = pd.Categorical(X[c].iloc[:i1]).categories
    X[c] = pd.Categorical(X[c], categories=cat).codes.astype("float32")
    X.loc[X[c] < 0, c] = np.nan
X = X.astype("float32")
X = X.fillna(X.iloc[:i1].median())
sc = StandardScaler().fit(X.iloc[:i1])
X = pd.DataFrame(sc.transform(X), columns=X.columns, dtype="float32")
print("Predictores finales:", X.shape[1], flush=True)


def particion(seed):
    return (X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:], y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:], monto[i2:])


# ---- grilla ampliada hacia los extremos observados ----
_modelos_orig = experimento.modelos


def modelos_ampliados(seed, balanceo, spw, ligero):
    base = _modelos_orig(seed, balanceo, spw, ligero)
    mk = {k: v[0] for k, v in base.items()}
    return {
        "Regresión logística": (mk["Regresión logística"], [{"C": c} for c in (10, 100, 1000)]),
        "Árbol de decisión": (mk["Árbol de decisión"], [{"max_depth": d, "min_samples_leaf": m} for d in (12, 20, 30) for m in (20, 50)]),
        "Random Forest": (mk["Random Forest"], [{"max_depth": d, "min_samples_leaf": 5} for d in (20, 30, 40)]),
        "XGBoost": (mk["XGBoost"], [{"max_depth": d, "reg_lambda": 10} for d in (6, 8, 10)]),
    }


experimento.modelos = modelos_ampliados
correr("ieee_cis_grillaamp", particion, SEEDS, smote_strategy=0.2, ligero=True)
