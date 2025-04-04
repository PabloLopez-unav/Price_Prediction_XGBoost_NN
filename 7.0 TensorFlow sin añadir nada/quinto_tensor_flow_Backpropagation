import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Clase para implementar una red neuronal con backpropagation usando TensorFlow
class NeuralNetworkTF:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.001):
        """
        Inicializar red neuronal con backpropagation usando TensorFlow
        
        Args:
            input_size: Número de características de entrada
            hidden_layers: Lista con el número de neuronas por capa oculta
            output_size: Número de neuronas de salida
            learning_rate: Tasa de aprendizaje para actualización de pesos
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        """Construir el modelo de red neuronal con TensorFlow"""
        model = tf.keras.Sequential()
        
        # Capa de entrada
        model.add(tf.keras.layers.Dense(self.hidden_layers[0], 
                                        input_dim=self.input_size, 
                                        activation='relu'))
        
        # Capas ocultas
        for units in self.hidden_layers[1:]:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            
        # Capa de salida
        model.add(tf.keras.layers.Dense(self.output_size, activation='linear'))
        
        # Compilar el modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=1):
        """
        Entrenar la red neuronal (backpropagation se realiza automáticamente)
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación
            y_val: Etiquetas de validación
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            verbose: Nivel de información durante el entrenamiento
        
        Returns:
            Historial de entrenamiento
        """
        # Configurar callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Entrenar el modelo
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X):
        """Hacer predicciones con el modelo entrenado"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluar el modelo en datos de prueba"""
        return self.model.evaluate(X_test, y_test)
    
    def save(self, filepath):
        """Guardar el modelo"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Cargar un modelo guardado"""
        self.model = tf.keras.models.load_model(filepath)
        
# Función para visualizar el historial de entrenamiento
def plot_training_history(history):
    """Visualizar el historial de entrenamiento"""
    plt.figure(figsize=(12, 4))
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Gráfico de MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    if 'val_mae' in history.history:
        plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso con datos inmobiliarios
def ejemplo_inmobiliario():
    # Cargar datos (sustituir por tu archivo)
    df = pd.read_csv("Madrid_Sale.csv")
    
    # Preprocesamiento
    num_columns = ['CONSTRUCTEDAREA', 'ROOMNUMBER', 'BATHNUMBER', 'FLATLOCATIONID', 
                   'CADCONSTRUCTIONYEAR', 'CADMAXBUILDINGFLOOR', 'CADDWELLINGCOUNT', 
                   'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA']
    
    # Normalizar características
    scaler_X = StandardScaler()
    df[num_columns] = scaler_X.fit_transform(df[num_columns])
    
    # Preparar datos
    X = pd.get_dummies(df.drop(columns=['ASSETID', 'PERIOD', 'UNITPRICE', 'CONSTRUCTIONYEAR', 
                                       'CADASTRALQUALITYID', 'geometry', 'PRICE']), 
                      drop_first=True)
    y = df['PRICE'].values.reshape(-1, 1)
    
    # Normalizar objetivo
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    # Crear y entrenar la red neuronal
    input_size = X_train.shape[1]
    hidden_layers = [64, 32, 16]  # Tres capas ocultas
    output_size = 1
    
    nn = NeuralNetworkTF(input_size, hidden_layers, output_size, learning_rate=0.001)
    history = nn.fit(X_train, y_train, X_val, y_val, epochs=100, batch_size=64)
    
    # Visualizar entrenamiento
    plot_training_history(history)
    
    # Evaluar en datos de prueba
    y_pred = nn.predict(X_test)
    
    # Convertir predicciones a escala original
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # Calcular MAE en escala original
    mae = np.mean(np.abs(y_pred_original - y_test_original))
    print(f"Error Absoluto Medio (MAE): {mae}")
    
    # Guardar modelo
    nn.save('modelo_backpropagation_neural_tf.keras')
    
    return nn

# Si quieres ejecutar el ejemplo, descomenta la siguiente línea
modelo = ejemplo_inmobiliario()