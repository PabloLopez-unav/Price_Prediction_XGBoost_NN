import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Cargar datos
df = pd.read_csv("Madrid_Sale.csv")

# Columnas numéricas para normalizar
num_columns = ['CONSTRUCTEDAREA', 'ROOMNUMBER', 'BATHNUMBER', 'FLATLOCATIONID', 
               'CADCONSTRUCTIONYEAR', 'CADMAXBUILDINGFLOOR', 'CADDWELLINGCOUNT', 
               'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA']

# Escalar las características numéricas
scaler = StandardScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])

# Feature engineering - Crear características adicionales
# Puede descomentar estas líneas si tiene sentido para su dataset
# df['AREA_PER_ROOM'] = df['CONSTRUCTEDAREA'] / df['ROOMNUMBER'].replace(0, 1)
# df['BATH_PER_ROOM'] = df['BATHNUMBER'] / df['ROOMNUMBER'].replace(0, 1)

# Preparar datos
X = pd.get_dummies(df.drop(columns=['ASSETID', 'PERIOD', 'UNITPRICE', 'CONSTRUCTIONYEAR', 'CADASTRALQUALITYID', 'geometry', 'PRICE']), drop_first=True)
y = df['PRICE']

# Escalar la variable objetivo (crucial para redes neuronales con regresión)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Dividir en entrenamiento y prueba
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Crear conjunto de validación
X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(X_train, y_train_scaled, test_size=0.15, random_state=42)

# Construcción del modelo mejorado
model = Sequential()
# Capa de entrada - más neuronas
model.add(Dense(units=256, activation='relu', kernel_regularizer=l2(0.001), input_dim=len(X_train.columns)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Capas ocultas
model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# Capa de salida
model.add(Dense(units=1))  # Salida lineal para regresión

# Configurar optimizador con tasa de aprendizaje personalizada
optimizer = Adam(learning_rate=0.001)

# Configurar callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

# Entrenar el modelo
history = model.fit(
    X_train, y_train_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=300,
    batch_size=32,
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
model.save('tfmodel_improved.keras')

# Visualizar el historial de entrenamiento
import matplotlib.pyplot as plt

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
plt.savefig('training_history.png')
plt.show()
