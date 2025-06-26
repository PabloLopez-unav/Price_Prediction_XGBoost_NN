import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


print("Modelo guardado correctamente.")

# 1. Cargar los datos
data = pd.read_csv(r"6.5 XGBoost con Datos Filtrados\Dataset_Con_Filtrado_y_Cuadrículas.csv")


# ELIMINO PARA LUEGO TESTARLO CON ESTAS COLUMNAS
asset_ids = [
    'A780752105476887286',
    'A9913171220255804177',
    'A15378954940238110703',
    'A5102506203930200985',
    'A13756852451347714092'
]

# Quedarse con las filas cuyo ASSETID NO esté en asset_ids
data = data[~data['ASSETID'].isin(asset_ids)]


data = data.drop(columns=['ASSETID', 'PERIOD', 'UNITPRICE', 'CONSTRUCTIONYEAR', 'CADASTRALQUALITYID', 'LONGITUDE', 'LATITUDE', 'geometry'])


# 2. Preprocesamiento
# Rellenar valores nulos con 0
data.fillna(0, inplace=True)



# Convertir variables categóricas en dummies
data = pd.get_dummies(data, drop_first=True)

# 3. Definir variables predictoras y objetivo
X = data.drop(columns=["PRICE"])  # Todas menos PRICE
y = data["PRICE"]


# 4. Dividir en entrenamiento y validación (80%-20%)
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.5, random_state=70)  # Quedarse con el 50% de los datos por sencillez
X_train, X_valid, y_train, y_valid = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)


"""
print(X_train.isna().sum())
print(y_train.isna().sum())


"""

# 5. Entrenar XGBoost con valores por defecto
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
model.fit(X_train, y_train)

# 6. Predicción y evaluación
y_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_pred)
print(f"Error absoluto medio (MAE): {mae}")

error_porcentual = np.zeros(len(y_pred))

y_valid = y_valid.to_numpy()  

for i in range(0, len(y_pred)):

    dif = y_pred[i] / y_valid[i]
    if dif <= 1:
        error_porcentual[i] = (1 - dif) 
    else:
        error_porcentual[i] = (dif - 1)

mean_porcentual_error = np.mean(error_porcentual)

print(f"Error porcentual medio: {mean_porcentual_error}")


# 7. Guardar resultados en un archivo CSV

# Crear un DataFrame con los valores reales y predichos
df_resultados = pd.DataFrame({'y_valid': y_valid, 'y_pred': y_pred, 'error_porcentual': error_porcentual})
df_resultados['error_porcentual'] = df_resultados['error_porcentual'] * 100  # Convertir a porcentaje

print(f"La media de precio de los pisos es de {y_pred.mean()}")

# Guardar en un archivo CSV
df_resultados.to_csv(r'6.5 XGBoost con Datos Filtrados\resultados_xgboost_60per_NO_CATASQUAL.csv', index=False) 

print("Archivo 'resultados_xgboost.csv' generado correctamente.")


# 8. Guardar el modelo en un archivo
# Guardar el modelo en un archivo
joblib.dump(model, r"6.5 XGBoost con Datos Filtrados\modelo_xgboost_60per_NO_CATASQUAL.pkl")




# 9. Análisis de los errores

# Convertir X_valid a DataFrame si no lo es
X_valid_df = pd.DataFrame(X_valid, columns=X.columns)

# Crear un DataFrame con los valores reales, predichos y el error porcentual
df_analisis = X_valid_df.copy()
df_analisis['y_valid'] = y_valid  # Valores reales
df_analisis['y_pred'] = y_pred  # Predicciones
df_analisis['error_porcentual'] = error_porcentual * 100  # Error porcentual en porcentaje

# Guardar el DataFrame en un archivo CSV para análisis posterior
df_analisis.to_csv(r'6.5 XGBoost con Datos Filtrados\analisis_errores.csv', index=False)

print("Archivo 'analisis_errores.csv' generado correctamente.")


# 10. Aanlizar el peso de cada variable en el modelo

# Obtener las importancias de las características
importances = model.feature_importances_

# Crear un DataFrame para visualizar las importancias
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importances)

# Opcional: Guardar las importancias en un archivo CSV
feature_importances.to_csv(r"6.5 XGBoost con Datos Filtrados\importancias_caracteristicas.csv", index=False)


# 11. Graficar las importancias de las características 

import shap

# 11.1. Crear el explainer (usando TreeExplainer que es más eficiente para XGBoost)
explainer = shap.TreeExplainer(model)

# 11.2. Seleccionar un caso con error alto
errores_altos = df_analisis[df_analisis['error_porcentual'] < 10]
if len(errores_altos) == 0:
    print("No hay casos con error > 100%. Probando con error > 50%...")
    errores_altos = df_analisis[df_analisis['error_porcentual'] > 50]
    
indice_caso = errores_altos.index[1]
# 11.3. Preparar los features del caso (manteniendo estructura original)
caso_features = X_valid_df.loc[indice_caso:indice_caso]

# 11.4. Calcular SHAP values y crear objeto Explanation
shap_values = explainer(caso_features)  # Esto devuelve un objeto Explanation

# 11.5. Visualización Waterfall plot CORRECTA
plt.figure()
shap.plots.waterfall(shap_values[0], max_display=25, show=False)
plt.savefig(r'6.5 XGBoost con Datos Filtrados\waterfall_HighError.png', 
            bbox_inches='tight', dpi=300)
plt.close()


# === GRÁFICAS ADICIONALES DE EVALUACIÓN ===
import seaborn as sns

# Crear carpeta de guardado si fuera necesario
import os
output_dir = r"6.5 XGBoost con Datos Filtrados"
os.makedirs(output_dir, exist_ok=True)

r2 = r2_score(y_valid, y_pred)

plt.figure(figsize=(10, 5))

# 1. Predicciones vs Reales
plt.subplot(1, 2, 1)
plt.scatter(y_valid, y_pred, alpha=0.4)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--', lw=2)
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')
plt.title(f'Predicciones vs Reales (R² = {r2:.3f})')

# 2. Histograma de errores (reales - predichos)
plt.subplot(1, 2, 2)
errores = y_valid - y_pred
plt.hist(errores, bins=30, alpha=0.7)
plt.xlabel('Error (Real - Predicho)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')

ax = plt.gca()
formatter = ScalarFormatter(useMathText=True)   # usa “1×10^6”
formatter.set_powerlimits((6, 6))               # siempre 10^6
ax.xaxis.set_major_formatter(formatter)
ax.ticklabel_format(axis='x', style='sci')      # muestra en científica

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "analisis_graficas_xgboost.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()
#

# === Distribución de precios reales vs predichos ===
plt.figure(figsize=(8, 5))
sns.histplot(y_valid, color='orange', label='Precio Real', kde=True, stat="density", bins=30, alpha=0.6)
sns.histplot(y_pred, color='blue', label='Precio Predicho', kde=True, stat="density", bins=30, alpha=0.6)
plt.xlabel('Precio')
plt.ylabel('Densidad')
plt.title('Distribución de Precios Reales vs Predichos')
plt.legend()
plt.savefig(os.path.join(output_dir, "distribucion_precios_reales_vs_predichos.png"), dpi=300, bbox_inches='tight')
#plt.show()
plt.close()

