"""Figuras conceptuales de los capítulos II y III (no usan los datos del estudio):
fig_flujo.png (procedimiento experimental), fig_decision.png (decisión de un detector) y fig_smote.png (ilustración de SMOTE).
Uso: python 09_figuras_conceptuales.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from common import OUT

AZUL, GRIS, ROJO, VERDE = "#4C72B0", "#E6ECF5", "#C44E52", "#55A868"


def caja(ax, x, y, w, h, texto, fc="white", ec=AZUL, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def flecha(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color="#444444", lw=1.2))


# ---------------------------------------------------------------- flujo experimental
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
caja(ax, 0.2, 5.0, 2.8, 0.9, "1. Datos\nULB e IEEE-CIS (Kaggle)", GRIS, bold=True)
caja(ax, 3.6, 5.0, 2.8, 0.9, "2. Preparación y partición\n70 % / 15 % / 15 %", GRIS, bold=True)
caja(ax, 7.0, 5.0, 2.8, 0.9, "Entrenamiento | Validación | Prueba", "white", fs=9)
flecha(ax, 3.0, 5.45, 3.6, 5.45); flecha(ax, 6.4, 5.45, 7.0, 5.45)
caja(ax, 0.2, 3.5, 2.8, 1.0, "3. Balanceo (solo entrenamiento)\nsin balanceo | SMOTE | ponderación", "#DCE6F4", fs=8.5, bold=True)
caja(ax, 3.6, 3.5, 2.8, 1.0, "4. Modelos y grilla\nRL | árbol | RF | XGBoost\nelección por AUC-PR en validación", "#DCE6F4", fs=8.5, bold=True)
caja(ax, 7.0, 3.5, 2.8, 1.0, "5. Umbral de decisión\nmáximo F1 en validación", "#DCE6F4", fs=8.5, bold=True)
flecha(ax, 1.6, 5.0, 1.6, 4.5); flecha(ax, 3.0, 4.0, 3.6, 4.0); flecha(ax, 6.4, 4.0, 7.0, 4.0)
caja(ax, 0.2, 1.9, 2.8, 1.0, "6. Evaluación única en prueba\nAUC-PR, AUC-ROC, P, R, F1", GRIS, bold=True, fs=8.5)
caja(ax, 3.6, 1.9, 2.8, 1.0, "7. Bootstrap pareado\n1 000 remuestras, IC 95 %", GRIS, bold=True, fs=8.5)
caja(ax, 7.0, 1.9, 2.8, 1.0, "8. Costo económico\ny análisis de sensibilidad", GRIS, bold=True, fs=8.5)
flecha(ax, 8.4, 3.5, 8.4, 3.2); ax.plot([8.4, 8.4], [3.2, 3.2], color="#444")
flecha(ax, 8.4, 3.2, 1.6, 3.2) if False else None
ax.plot([8.4, 8.4, 1.6], [3.5, 3.2, 3.2], color="#444444", lw=1.2); flecha(ax, 1.6, 3.2, 1.6, 2.9)
flecha(ax, 3.0, 2.4, 3.6, 2.4); flecha(ax, 6.4, 2.4, 7.0, 2.4)
caja(ax, 1.2, 0.2, 7.6, 0.9, "Repetición: 4 algoritmos × 3 estrategias × 2 conjuntos × 3 semillas = 72 corridas", "white", ec=ROJO, bold=True, fs=9)
flecha(ax, 5.0, 1.9, 5.0, 1.1)
plt.tight_layout(); plt.savefig(OUT / "fig_flujo.png", dpi=170); plt.close()

# ---------------------------------------------------------------- decisión de un detector
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
caja(ax, 0.1, 1.3, 1.9, 1.0, "Transacción\nnueva", GRIS, bold=True)
caja(ax, 2.6, 1.3, 2.1, 1.0, "Modelo de\nclasificación\n(probabilidad p)", "#DCE6F4", bold=True, fs=8.5)
caja(ax, 5.3, 1.3, 1.6, 1.0, "¿p ≥\numbral?", "white", ec=ROJO, bold=True)
caja(ax, 7.6, 2.3, 2.2, 0.9, "Alerta: rechazar,\nautenticar o revisar", "#F5DADA", ec=ROJO, fs=8.5)
caja(ax, 7.6, 0.4, 2.2, 0.9, "Aprobar", "#DDEFE2", ec=VERDE, fs=9)
flecha(ax, 2.0, 1.8, 2.6, 1.8); flecha(ax, 4.7, 1.8, 5.3, 1.8)
flecha(ax, 6.9, 2.05, 7.6, 2.7); flecha(ax, 6.9, 1.55, 7.6, 0.9)
ax.text(7.05, 2.55, "sí", fontsize=9); ax.text(7.05, 0.95, "no", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "fig_decision.png", dpi=170); plt.close()

# ---------------------------------------------------------------- SMOTE
rng = np.random.default_rng(3)
mayor = rng.normal([0, 0], 1.1, (90, 2))
menor = rng.normal([2.2, 2.0], 0.55, (9, 2))
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
for ax in axs:
    ax.scatter(*mayor.T, s=14, c="#9AA9C2", label="Clase mayoritaria (legítima)")
    ax.scatter(*menor.T, s=34, c=ROJO, label="Clase minoritaria (fraude)")
axs[0].set_title("Antes: pocos ejemplos minoritarios")
i = 0
xi = menor[i]
d = np.linalg.norm(menor - xi, axis=1)
vec = menor[np.argsort(d)[1:6]]
nuevos = []
for j in range(len(menor)):
    dj = np.linalg.norm(menor - menor[j], axis=1)
    vj = menor[np.argsort(dj)[1:6]]
    for _ in range(2):
        z = vj[rng.integers(0, 5)]
        nuevos.append(menor[j] + rng.random() * (z - menor[j]))
for ax in axs[1:]:
    nuevos_a = np.array(nuevos)
    ax.scatter(*nuevos_a.T, s=26, marker="^", c="#E29A9C", edgecolors=ROJO, label="Ejemplos sintéticos (SMOTE)")
    for z in vec:
        ax.plot([xi[0], z[0]], [xi[1], z[1]], color=ROJO, lw=0.7, ls="--", alpha=0.6)
    ax.scatter(*xi, s=70, facecolors="none", edgecolors="black", lw=1.5)
axs[1].set_title("Después: ejemplos sintéticos (k = 5)")
axs[1].legend(fontsize=7, loc="lower left")
for ax in axs:
    ax.set_xlabel("Variable 1")
axs[0].set_ylabel("Variable 2")
plt.tight_layout(); plt.savefig(OUT / "fig_smote.png", dpi=170); plt.close()
print("OK")
