"""Motor de experimentos común: 4 modelos x 3 estrategias de balanceo.

Diseño (controlado):
  - El balanceo (SMOTE o pesos de clase) se aplica SOLO sobre train; nunca ambos a la vez.
  - Hiperparámetros: grilla pequeña evaluada en validación (AUC-PR).
  - Umbral de decisión: elegido en validación (máx. F1); métricas finales en test.
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from common import metricas, mejor_umbral_f1, costo_total, OUT

PRED = OUT / "predicciones"
PRED.mkdir(exist_ok=True)


def modelos(seed, balanceo, spw, ligero):
    cw = "balanced" if balanceo == "pesos" else None
    xs = spw if balanceo == "pesos" else 1.0
    rf_extra = {"max_samples": 0.3} if ligero else {}
    return {
        "Regresión logística": (
            lambda **k: LogisticRegression(max_iter=3000, class_weight=cw, random_state=seed, **k),
            [{"C": c} for c in (0.01, 0.1, 1, 10)]),
        "Árbol de decisión": (
            lambda **k: DecisionTreeClassifier(class_weight=cw, random_state=seed, **k),
            [{"max_depth": d, "min_samples_leaf": m} for d in (4, 8, 12) for m in (5, 20)]),
        "Random Forest": (
            lambda **k: RandomForestClassifier(n_estimators=100 if ligero else 150, class_weight=cw,
                                               n_jobs=-1, random_state=seed, **rf_extra, **k),
            [{"max_depth": d, "min_samples_leaf": m} for d in (10, 16) for m in (1, 5)]
            if not ligero else [{"max_depth": d, "min_samples_leaf": 5} for d in (12, 20)]),
        "XGBoost": (
            lambda **k: XGBClassifier(n_estimators=300, learning_rate=0.1, scale_pos_weight=xs,
                                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                      eval_metric="aucpr", n_jobs=-1, random_state=seed, **k),
            [{"max_depth": d, "reg_lambda": l} for d in (4, 6) for l in (1, 10)]
            if not ligero else [{"max_depth": d, "reg_lambda": 10} for d in (4, 6)]),
    }


def correr(nombre, particion, seeds, smote_strategy="auto", ligero=False):
    """particion(seed) -> (Xtr, Xva, Xte, ytr, yva, yte, monto_te)."""
    filas = []
    for seed in seeds:
        Xtr, Xva, Xte, ytr, yva, yte, monto_te = particion(seed)
        spw = (ytr == 0).sum() / (ytr == 1).sum()
        print(f"\n=== {nombre} | semilla {seed} | train={len(Xtr)} (fraudes {ytr.sum()}) "
              f"val={len(Xva)} (fraudes {yva.sum()}) test={len(Xte)} (fraudes {yte.sum()})", flush=True)
        for bal in ("sin_balanceo", "SMOTE", "pesos"):
            if bal == "SMOTE":
                Xb, yb = SMOTE(k_neighbors=5, sampling_strategy=smote_strategy,
                               random_state=seed).fit_resample(Xtr, ytr)
            else:
                Xb, yb = Xtr, ytr
            for mod, (mk, grid) in modelos(seed, bal, spw, ligero).items():
                fn = PRED / f"{nombre}__{mod}__{bal}__{seed}.npz".replace(" ", "_")
                if fn.exists():  # reanudación: combinación ya calculada y guardada con sus parámetros
                    z = np.load(fn)
                    if "params" in z.files:
                        r = metricas(z["y"], z["p"], float(z["umbral"]))
                        r["Costo"] = costo_total(z["y"], z["p"] >= float(z["umbral"]), z["monto"])
                        r["Costo_sin_modelo"] = float(z["monto"][z["y"] == 1].sum())
                        r.update(dataset=nombre, modelo=mod, balanceo=bal, semilla=seed,
                                 params=str(z["params"]))
                        filas.append(r)
                        print(f"{bal:13s} {mod:20s} (reanudado) AUC-PR={r['AUC_PR']:.3f}", flush=True)
                        continue
                t0 = time.time()
                best = max(((average_precision_score(yva, mk(**g).fit(Xb, yb).predict_proba(Xva)[:, 1]), g)
                            for g in grid), key=lambda x: x[0])
                m = mk(**best[1]).fit(Xb, yb)
                umbral = mejor_umbral_f1(yva, m.predict_proba(Xva)[:, 1])
                pte = m.predict_proba(Xte)[:, 1]
                r = metricas(yte, pte, umbral)
                r["Costo"] = costo_total(yte, pte >= umbral, monto_te)
                r["Costo_sin_modelo"] = float(monto_te[yte.values == 1].sum())
                np.savez_compressed(fn, y=yte.values, p=pte, monto=monto_te, umbral=umbral,
                                    params=str(best[1]))
                r.update(dataset=nombre, modelo=mod, balanceo=bal, semilla=seed, params=str(best[1]))
                filas.append(r)
                print(f"{bal:13s} {mod:20s} AUC-PR={r['AUC_PR']:.3f} AUC-ROC={r['AUC_ROC']:.3f} "
                      f"P={r['Precision']:.3f} R={r['Recall']:.3f} F1={r['F1']:.3f} "
                      f"Costo={r['Costo']:,.0f} [{time.time()-t0:.0f}s]", flush=True)
        pd.DataFrame(filas).to_csv(OUT / f"{nombre}_resultados_crudos.csv", index=False)
    res = pd.DataFrame(filas)
    cols = ["AUC_PR", "AUC_ROC", "Precision", "Recall", "F1", "FP", "FN", "Costo"]
    res.groupby(["modelo", "balanceo"])[cols].agg(["mean", "std"]).round(4).to_csv(
        OUT / f"{nombre}_resumen_modelos.csv")
    print("\n", res.groupby(["modelo", "balanceo"])[cols[:5] + ["Costo"]].mean().round(3).to_string())
    return res
