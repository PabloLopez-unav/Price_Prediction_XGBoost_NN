import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib

print("Modelo guardado correctamente.")

# 1. Cargar los datos
data = pd.read_csv("Madrid_Sale.csv")  # Cambia "datos.csv" por la ruta real

data = data.drop(columns=['ASSETID', 'PERIOD', 'UNITPRICE', 'CONSTRUCTIONYEAR', 'geometry'])


# 2. Preprocesamiento
# Rellenar valores nulos con 0
data.fillna(0, inplace=True)



# Convertir variables categóricas en dummies
data = pd.get_dummies(data, drop_first=True)

# 3. Definir variables predictoras y objetivo
X = data.drop(columns=["PRICE"])  # Todas menos PRICE
y = data["PRICE"]


# 4. Dividir en entrenamiento y validación (80%-20%)
X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.9, random_state=35)  # Quedarse con el 50% de los datos por sencillez
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


# 7. Guardar resultados en un archivo CSV

# Crear un DataFrame con los valores reales y predichos
df_resultados = pd.DataFrame({'y_valid': y_valid, 'y_pred': y_pred})

print(y_pred.mean())

# Guardar en un archivo CSV
df_resultados.to_csv('resultados_xgboost.csv', index=False)

print("Archivo 'resultados_xgboost.csv' generado correctamente.")


# 8. Guardar el modelo en un archivo
# Guardar el modelo en un archivo
joblib.dump(model, "modelo_xgboost.pkl")

