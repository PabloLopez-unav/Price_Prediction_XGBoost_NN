import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

# Cargar datos
df = pd.read_csv("Madrid_Sale.csv")

# Columnas numéricas para normalizar
num_columns = ['CONSTRUCTEDAREA', 'ROOMNUMBER', 'BATHNUMBER', 'FLATLOCATIONID', 
               'CADCONSTRUCTIONYEAR', 'CADMAXBUILDINGFLOOR', 'CADDWELLINGCOUNT', 
               'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA']

# Usar RobustScaler para las características (mejor con outliers)
scaler = RobustScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])

# Preparar datos
X = pd.get_dummies(df.drop(columns=['ASSETID', 'PERIOD', 'UNITPRICE', 'CONSTRUCTIONYEAR', 'CADASTRALQUALITYID', 'geometry', 'PRICE']), drop_first=True)
y = df['PRICE']

# Usar RobustScaler para la variable objetivo
y_scaler = RobustScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Dividir en entrenamiento y prueba
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Crear conjunto de validación
X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(X_train, y_train_scaled, test_size=0.15, random_state=42)

# Construcción de un modelo más estable
model = Sequential()
# Capa de entrada - menos neuronas, sin regularización fuerte
model.add(Dense(units=128, activation='relu', input_dim=len(X_train.columns)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Capas ocultas simples
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Capa de salida
model.add(Dense(units=1))

# Configurar optimizador con tasa de aprendizaje más baja y clipping de gradientes
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)  # Añadir clipping para estabilidad

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

# Configurar callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reducción más gradual
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Entrenar el modelo
history = model.fit(
    X_train, y_train_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=300,
    batch_size=64,  # Batch size más grande para estabilidad
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluar el modelo
y_pred_scaled = model.predict(X_test)
# Deshacer la escala para obtener predicciones en euros
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

mae = mean_absolute_error(y_test, y_pred)
print(f"Error Absoluto Medio (MAE): {mae}")

# Guardar modelo
model.save('tfmodel_stable.keras')