"""EDA: desbalance, calidad de datos, distribuciones por clase y correlaciones."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import cargar, OUT

df, info = cargar()
print(info)
print(df.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
print("Nulos:", int(df.isna().sum().sum()))

# 1) Desbalance
fig, ax = plt.subplots(figsize=(4, 4))
vc = df["Class"].value_counts()
ax.bar(["Legítima", "Fraude"], vc.values, color=["#4C72B0", "#C44E52"])
ax.set_yscale("log"); ax.set_ylabel("Transacciones (escala log)")
for i, v in enumerate(vc.values):
    ax.text(i, v, f"{v:,}\n({v/len(df):.3%})", ha="center", va="bottom")
ax.set_title("Desbalance de clases"); plt.tight_layout()
plt.savefig(OUT / "eda_desbalance.png", dpi=150); plt.close()

# 2) Amount por clase
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df[df.Amount < 500], x="Amount", hue="Class", bins=60,
             stat="density", common_norm=False, ax=axs[0])
axs[0].set_title("Amount (<500) por clase")
sns.boxplot(data=df, x="Class", y="Amount", showfliers=False, ax=axs[1])
axs[1].set_title("Amount por clase (sin atípicos)")
plt.tight_layout(); plt.savefig(OUT / "eda_amount.png", dpi=150); plt.close()

# 3) Variables más discriminantes (|diferencia de medias| / desviación)
vs = [c for c in df.columns if c.startswith("V")]
sep = ((df[df.Class == 1][vs].mean() - df[df.Class == 0][vs].mean()).abs()
       / df[vs].std()).sort_values(ascending=False)
print("\nTop variables separadoras:\n", sep.head(8).round(2).to_string())
top = sep.head(6).index.tolist()
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for a, c in zip(axs.ravel(), top):
    sns.violinplot(data=df, x="Class", y=c, ax=a, cut=0)
    a.set_title(c)
plt.tight_layout(); plt.savefig(OUT / "eda_violin_top.png", dpi=150); plt.close()

# 4) Correlación con la clase
corr = df.corr()["Class"].drop("Class").sort_values()
fig, ax = plt.subplots(figsize=(6, 7))
corr.plot.barh(ax=ax, color="#4C72B0"); ax.set_title("Correlación de cada variable con Class")
plt.tight_layout(); plt.savefig(OUT / "eda_corr_class.png", dpi=150); plt.close()
print("\nFiguras guardadas en", OUT)
