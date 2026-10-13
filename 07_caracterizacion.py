"""Caracterización de los conjuntos de datos (OE1): estructura, calidad, desbalance, montos y estabilidad temporal.

Genera CSV en resultados/caracterizacion_*.csv y dos figuras. No modifica los datos ni entrena modelos.
Uso: python 07_caracterizacion.py
"""
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import cargar, OUT, BASE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

D = BASE / "data" / "ieee_cis"
res = {}

# ------------------------------------------------------------------ ULB
raw = pd.read_csv(BASE / "data" / "ulb" / "creditcard.csv")
df, info = cargar()
vs = [c for c in df.columns if c.startswith("V")]
res["ulb"] = dict(
    filas_original=len(raw), duplicados=int(len(raw) - len(df)), filas=len(df),
    fraudes_original=int(raw.Class.sum()), fraudes=int(df.Class.sum()),
    tasa_original=float(raw.Class.mean()), tasa=float(df.Class.mean()),
    nulos=int(raw.isna().sum().sum()), columnas=int(raw.shape[1]), predictores=int(df.shape[1] - 1),
    horas=float(raw.Time.max() / 3600),
)
amt = []
for c, nom in ((0, "Legítima"), (1, "Fraude")):
    a = df[df.Class == c].Amount
    amt.append(dict(clase=nom, n=len(a), media=a.mean(), mediana=a.median(), p90=a.quantile(.9), p99=a.quantile(.99),
                    maximo=a.max(), suma=a.sum()))
pd.DataFrame(amt).round(2).to_csv(OUT / "caracterizacion_ulb_monto.csv", index=False)
sep = ((df[df.Class == 1][vs].mean() - df[df.Class == 0][vs].mean()).abs() / df[vs].std()).sort_values(ascending=False)
sep.head(8).round(2).rename("separacion").to_csv(OUT / "caracterizacion_ulb_variables.csv")
corrabs = df[vs + ["Amount", "Time"]].corr().abs().to_numpy(copy=True)
np.fill_diagonal(corrabs, 0)
res["ulb"]["max_corr_entre_predictores"] = float(corrabs.max())
res["ulb"]["corr_amount_class"] = float(df[["Amount", "Class"]].corr().iloc[0, 1])
# fraude por mitad del período (24 h)
mit = np.where(df.Time < 86400, "Primeras 24 h", "Últimas 24 h")
t = df.groupby(mit).Class.agg(["size", "sum", "mean"]).reset_index().rename(columns={"index": "tramo", "level_0": "tramo"})
t.columns = ["tramo", "transacciones", "fraudes", "tasa"]
t.round(5).to_csv(OUT / "caracterizacion_ulb_tiempo.csv", index=False)

# ------------------------------------------------------------------ IEEE-CIS
tx = pd.read_csv(D / "train_transaction.csv")
idt = pd.read_csv(D / "train_identity.csv")
res["ieee"] = dict(
    filas=len(tx), fraudes=int(tx.isFraud.sum()), tasa=float(tx.isFraud.mean()),
    columnas_tx=int(tx.shape[1]), columnas_id=int(idt.shape[1]), filas_id=len(idt),
    cobertura_id=float(tx.TransactionID.isin(idt.TransactionID).mean()),
    dias=float(tx.TransactionDT.max() / 86400), duplicados_id=int(tx.duplicated("TransactionID").sum()),
)
con_id = tx.TransactionID.isin(idt.TransactionID)
res["ieee"]["tasa_con_id"] = float(tx[con_id].isFraud.mean())
res["ieee"]["tasa_sin_id"] = float(tx[~con_id].isFraud.mean())
nul = tx.isna().mean()
res["ieee"]["cols_nulos_gt60"] = int((nul > 0.60).sum())
res["ieee"]["cols_sin_nulos"] = int((nul == 0).sum())
res["ieee"]["tipos_num"] = int(tx.select_dtypes("number").shape[1])
res["ieee"]["tipos_cat"] = int(tx.select_dtypes("object").shape[1])
tx = tx.sort_values("TransactionDT").reset_index(drop=True)
n = len(tx)
i1, i2 = int(n * .7), int(n * .85)
tr = []
for nom, a, b in (("Entrenamiento", 0, i1), ("Validación", i1, i2), ("Prueba", i2, n)):
    s = tx.iloc[a:b]
    tr.append(dict(tramo=nom, transacciones=len(s), fraudes=int(s.isFraud.sum()), tasa=s.isFraud.mean(),
                   dia_ini=s.TransactionDT.min() / 86400, dia_fin=s.TransactionDT.max() / 86400,
                   monto_medio=s.TransactionAmt.mean()))
pd.DataFrame(tr).round(4).to_csv(OUT / "caracterizacion_ieee_tramos.csv", index=False)
amt = []
for c, nom in ((0, "Legítima"), (1, "Fraude")):
    a = tx[tx.isFraud == c].TransactionAmt
    amt.append(dict(clase=nom, n=len(a), media=a.mean(), mediana=a.median(), p90=a.quantile(.9), p99=a.quantile(.99),
                    maximo=a.max(), suma=a.sum()))
pd.DataFrame(amt).round(2).to_csv(OUT / "caracterizacion_ieee_monto.csv", index=False)
filas = []
tx["hora"] = (tx.TransactionDT // 3600) % 24
tx["franja"] = pd.cut(tx.hora, [-1, 5, 11, 17, 23], labels=["00–05 h", "06–11 h", "12–17 h", "18–23 h"])
tx = tx.merge(idt[["TransactionID", "DeviceType"]], on="TransactionID", how="left")
tx["monto_banda"] = pd.cut(tx.TransactionAmt, [0, 50, 100, 250, 1000, 1e9], labels=["≤50", "50–100", "100–250", "250–1 000", ">1 000"])
for col in ("ProductCD", "card4", "card6", "DeviceType", "franja", "monto_banda"):
    g = tx.groupby(col, observed=True).isFraud.agg(["size", "sum", "mean"])
    for k, r in g.iterrows():
        filas.append(dict(variable=col, categoria=str(k), transacciones=int(r["size"]), fraudes=int(r["sum"]), tasa=float(r["mean"])))
pd.DataFrame(filas).round(4).to_csv(OUT / "caracterizacion_ieee_categorias.csv", index=False)
# nulos de las columnas más vacías
nul.sort_values(ascending=False).head(5).round(3).rename("proporcion_nulos").to_csv(OUT / "caracterizacion_ieee_nulos_top.csv")

# ------------------------------------------------------------------ figuras
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (d, nom) in zip(axs, (("ulb", "ULB"), ("ieee", "IEEE-CIS"))):
    tasa = res[d]["tasa"] * 100
    ax.bar(["Legítima", "Fraude"], [100 - tasa, tasa], color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log"); ax.set_ylabel("% de transacciones (escala log)"); ax.set_title(nom)
    for i, v in enumerate([100 - tasa, tasa]):
        ax.text(i, v, f"{v:.3f} %", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(OUT / "car_desbalance.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.6))
tt = pd.DataFrame(tr)
ax.bar(tt.tramo, tt.tasa * 100, color="#4C72B0")
ax.set_ylabel("% de fraude"); ax.set_title("IEEE-CIS: tasa de fraude por tramo cronológico")
for i, v in enumerate(tt.tasa * 100):
    ax.text(i, v, f"{v:.2f} %", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(OUT / "car_ieee_tramos.png", dpi=150); plt.close()

(OUT / "caracterizacion_resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=1, ensure_ascii=False))
for f in ("caracterizacion_ulb_monto", "caracterizacion_ulb_variables", "caracterizacion_ulb_tiempo", "caracterizacion_ieee_tramos",
          "caracterizacion_ieee_monto", "caracterizacion_ieee_categorias", "caracterizacion_ieee_nulos_top"):
    print("\n##", f); print(pd.read_csv(OUT / f"{f}.csv").to_string())
