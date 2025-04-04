import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("Madrid_Sale.csv")

df.drop(['ASSETID', 'PERIOD', 'geometry'], axis=1, inplace=True)


# Matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de correlación")
plt.show()

correlation = df.corr()["PRICE"].sort_values(ascending=False)
print(correlation)
correlation.to_csv("C:\\Users\\costa\\Desktop\\TFG\\4.0 Matriz Correlaciones para juzgar Columnas\\correlation_price.csv", header=True, encoding='utf-8')



