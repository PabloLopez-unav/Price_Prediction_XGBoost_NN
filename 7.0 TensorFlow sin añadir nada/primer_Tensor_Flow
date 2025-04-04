import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import mean_absolute_error

# Cargar datos
df = pd.read_csv("Madrid_Sale.csv")  # Cambia "datos.csv" por la ruta real

num_columns = ['CONSTRUCTEDAREA', 'ROOMNUMBER', 'BATHNUMBER', 'FLATLOCATIONID', 
               'CADCONSTRUCTIONYEAR', 'CADMAXBUILDINGFLOOR', 'CADDWELLINGCOUNT', 
               'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA']

scaler = StandardScaler()

# Aplicar la normalización a las columnas numéricas
df[num_columns] = scaler.fit_transform(df[num_columns])

# Ver los primeros registros del DataFrame después de la normalización
#print(df[num_columns].head())


X = pd.get_dummies(df.drop(columns=['ASSETID', 'PERIOD', 'UNITPRICE', 'CONSTRUCTIONYEAR', 'CADASTRALQUALITYID' ,'geometry']), drop_first=True)  # Quitar 'Price' porque es la variable a predecir

y = df['PRICE']  # Ahora predecimos el precio

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Construcción del modelo
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=len(X_train.columns)))
model.add(Dropout(0.3))  # Dropout del 30% en la primera capa
model.add(Dense(units=32, activation='sigmoid'))
model.add(Dropout(0.3))  # Dropout del 30% en la primera capa
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))  # Dropout del 30% en la primera capa
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1))  # Última capa con activación lineal (por defecto)

early_stopping = EarlyStopping(monitor='mae', patience=10, restore_best_weights=True)


# Compilar el modelo para regresión
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=16, callbacks=[early_stopping])


# Evaluar el modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Error Absoluto Medio (MAE): {mae}")

# Guardar modelo
model.save('tfmodel.keras')
