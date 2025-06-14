import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ------------------------ CARGA Y PREPROCESADO DE DATOS ------------------------

df = pd.read_csv(r"C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\Dataset_Con_Filtrado_y_Cuadrículas.csv")

num_columns = [
    'CONSTRUCTEDAREA', 'ROOMNUMBER', 'BATHNUMBER', 'FLATLOCATIONID', 
    'CADCONSTRUCTIONYEAR', 'CADMAXBUILDINGFLOOR', 'CADDWELLINGCOUNT', 
    'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA'
]

# Escalado robusto para variables numéricas
scaler = RobustScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])

# Preparación de variables
X = pd.get_dummies(df.drop(columns=['ASSETID','PERIOD','PRICE','UNITPRICE','AMENITYID','ISPARKINGSPACEINCLUDEDINPRICE','PARKINGSPACEPRICE','HASNORTHORIENTATION','HASSOUTHORIENTATION','HASEASTORIENTATION','HASWESTORIENTATION','CONSTRUCTIONYEAR','FLATLOCATIONID','CADASTRALQUALITYID','BUILTTYPEID_1','BUILTTYPEID_2','BUILTTYPEID_3','LONGITUDE','LATITUDE','geometry','lon_normalized','lat_normalized'
]), drop_first=True)
y = df['PRICE']

#'ASSETID', 'PERIOD', 'UNITPRICE', 'CONSTRUCTIONYEAR', 'CADASTRALQUALITYID', 'geometry', 'PRICE'
#ASSETID,PERIOD,PRICE,UNITPRICE,AMENITYID,ISPARKINGSPACEINCLUDEDINPRICE,PARKINGSPACEPRICE,HASNORTHORIENTATION,HASSOUTHORIENTATION,HASEASTORIENTATION,HASWESTORIENTATION,CONSTRUCTIONYEAR,FLATLOCATIONID,CADASTRALQUALITYID,BUILTTYPEID_1,BUILTTYPEID_2,BUILTTYPEID_3,LONGITUDE,LATITUDE,geometry,lon_normalized,lat_normalized

print(df['PRICE'].describe())


y_scaled = y.values  # sin escalar


# División en sets
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(X_train, y_train_scaled, test_size=0.15, random_state=42)

# Conversión a datasets TensorFlow
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values.astype(np.float32), y_train_scaled.astype(np.float32))).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values.astype(np.float32), y_val_scaled.astype(np.float32))).batch(batch_size)

# ------------------------ DEFINICIÓN DEL MODELO BÁSICO ------------------------

def crear_modelo_basico(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)  # salida sin activación (lineal)
    ])
    return model

modelo_basico = crear_modelo_basico(X_train.shape[1])
modelo_basico.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# ------------------------ ENTRENAMIENTO ------------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)

history = modelo_basico.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ------------------------ EVALUACIÓN ------------------------

X_test_tensor = tf.convert_to_tensor(X_test.values.astype(np.float32))
y_pred_scaled = modelo_basico.predict(X_test_tensor)
print("Valores únicos en las predicciones escaladas:", np.unique(y_pred_scaled[:50]))
print("Pérdidas (inicio):", history.history['loss'][:5])
print("Pérdidas (final):", history.history['loss'][-5:])

y_pred = y_pred_scaled
y_test = y_test_scaled.reshape(-1, 1)


mae = mean_absolute_error(y_test, y_pred)
print(f"Error Absoluto Medio (MAE): {mae}")

# ------------------------ GUARDADO DE RESULTADOS ------------------------

resultados_df = pd.DataFrame({
    'Precio_real': y_test.flatten(),
    'Precio_predicho': y_pred.flatten()
})
resultados_df.to_csv(r"C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\resultados_modelo_basico.csv", index=False)

modelo_basico.save_weights(r'C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\modelo_basico.weights.h5')


# ------------------------ VISUALIZACIÓN ------------------------

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE del modelo')
plt.ylabel('MAE')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper right')

plt.tight_layout()
plt.savefig(r'C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\modelo_basico_training_history.png')
plt.show()
