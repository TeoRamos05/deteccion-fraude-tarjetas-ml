"""Experimento principal: dataset ULB (partición aleatoria estratificada 70/15/15)."""
import sys
from common import cargar, particionar
from experimento import correr

SEEDS = [int(s) for s in sys.argv[1:]] or [42, 7, 2024]
df, info = cargar()
print(info)


def particion(seed):
    Xtr, Xva, Xte, ytr, yva, yte = particionar(df, seed)
    return Xtr, Xva, Xte, ytr, yva, yte, df.loc[Xte.index, "Amount"].values


correr("ulb", particion, SEEDS)
