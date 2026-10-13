"""Sensibilidad del costo económico al costo de falso positivo (usa las predicciones guardadas; no reentrena).

Costo de una combinación = sum(monto de fraudes no detectados) + sum(costo de cada falso positivo), con
    costo de FP_i = Ca + lambda * monto_i
donde Ca es el costo administrativo fijo (revisión y contacto, Bahnsen et al., 2013) y lambda es la fracción
del monto de la transacción legítima rechazada que se pierde (costo comercial: venta no recuperada).
Escenarios:
  - administrativos: Ca en {1, 2,5, 5, 10}, lambda = 0 (el valor 5 es el del análisis principal).
  - comerciales: Ca = 5 y lambda en {0,25; 1,0}. 0,25 = proporción de compradores rechazados por error que se
    van a la competencia (The Paypers, citado por Ravelin; fuente secundaria); 1,0 = cota superior (venta perdida).
El umbral es el elegido en validación maximizando F1 (el mismo del análisis principal), por lo que no se
reoptimiza para cada costo: con costos de FP altos esto sobrestima el costo de los modelos poco precisos.
Los montos están en la unidad monetaria de cada conjunto (IEEE-CIS: USD; ULB: la moneda no está documentada).
Uso: python 06_sensibilidad_costo.py
"""
import numpy as np
import pandas as pd
from common import OUT

PRED = OUT / "predicciones"
ESCENARIOS = [("Ca=1", 1.0, 0.0), ("Ca=2,5", 2.5, 0.0), ("Ca=5 (base)", 5.0, 0.0), ("Ca=10", 10.0, 0.0),
              ("Ca=5 + 25 % del monto", 5.0, 0.25), ("Ca=5 + 100 % del monto", 5.0, 1.0)]

filas = []
for f in sorted(PRED.glob("*.npz")):
    ds, mod, bal, seed = f.stem.split("__")
    z = np.load(f)
    y, p, monto, umbral = z["y"], z["p"], z["monto"], float(z["umbral"])
    pred = p >= umbral
    fn = (y == 1) & ~pred
    fp = (y == 0) & pred
    perd_fn = float(monto[fn].sum())
    sin_modelo = float(monto[y == 1].sum())
    for nombre, ca, lam in ESCENARIOS:
        costo = perd_fn + ca * fp.sum() + lam * float(monto[fp].sum())
        filas.append(dict(dataset=ds, modelo=mod.replace("_", " "), balanceo=bal, semilla=int(seed),
                          escenario=nombre, costo=costo, sin_modelo=sin_modelo, FP=int(fp.sum()), FN=int(fn.sum())))

d = pd.DataFrame(filas)
if d.empty:
    print("No hay predicciones disponibles en", PRED)
    print("Ejecuta 02_modelado.py y, si corresponde, 03_ieee_cis.py antes de calcular la sensibilidad.")
    raise SystemExit(0)

g = d.groupby(["dataset", "escenario", "modelo", "balanceo"]).agg(
    costo=("costo", "mean"), sd=("costo", "std"), sin_modelo=("sin_modelo", "mean"), n_semillas=("semilla", "nunique")).reset_index()
g["reduccion_vs_sin_modelo_pct"] = (1 - g.costo / g.sin_modelo) * 100
g["ranking"] = g.groupby(["dataset", "escenario"]).costo.rank(method="min").astype(int)
g.round(2).to_csv(OUT / "sensibilidad_costo.csv", index=False)

orden = [e[0] for e in ESCENARIOS]
for ds in ("ulb", "ieee_cis"):
    x = g[g.dataset == ds]
    if x.empty:
        print(f"\n=== {ds}: sin predicciones locales; se omite")
        continue
    print(f"\n=== {ds}: costo sin modelo = {x.sin_modelo.iloc[0]:,.0f}")
    top = x[x.ranking == 1].set_index("escenario").loc[orden]
    print("Mejor combinación por escenario:")
    for e, r in top.iterrows():
        print(f"  {e:26s} {r.modelo} / {r.balanceo}: {r.costo:,.0f} ({r.reduccion_vs_sin_modelo_pct:.0f} % menos que sin modelo)")
    # ¿los ensambles siguen entre los 4 de menor costo? ¿el orden de los modelos se mantiene?
    for e in orden:
        s = x[x.escenario == e].sort_values("costo")
        mejores4 = s.head(4)
        ens = mejores4.modelo.isin(["Random Forest", "XGBoost"]).all()
        medio = s.groupby("modelo").costo.mean().sort_values()
        print(f"  {e:26s} 4 mejores todas ensambles: {ens} | orden de modelos (costo medio): {' < '.join(medio.index)}")
    base = x[x.escenario == "Ca=5 (base)"].set_index(["modelo", "balanceo"]).ranking
    for e in orden:
        if e == "Ca=5 (base)":
            continue
        otro = x[x.escenario == e].set_index(["modelo", "balanceo"]).ranking
        rho = base.corr(otro.loc[base.index], method="spearman")
        print(f"  Spearman ranking base vs {e:24s}: {rho:.2f}")
