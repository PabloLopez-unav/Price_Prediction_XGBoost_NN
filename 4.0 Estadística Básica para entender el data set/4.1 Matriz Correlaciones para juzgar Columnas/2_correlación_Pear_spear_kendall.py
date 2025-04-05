import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("Madrid_Sale.csv")
df.drop(['ASSETID', 'PERIOD', 'geometry'], axis=1, inplace=True)


# Definir los métodos de correlación
methods = ["pearson", "spearman"]

# Crear una figura con 3 subgráficos (uno por método)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for i, method in enumerate(methods):
    corr_matrix = df.corr(method=method)
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5, ax=axes[i])
    axes[i].set_title(f"Matriz de correlación ({method.capitalize()})")

plt.tight_layout()
plt.show()
