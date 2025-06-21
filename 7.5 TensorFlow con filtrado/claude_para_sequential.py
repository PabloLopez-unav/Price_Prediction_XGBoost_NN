import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Configurar TensorFlow para mayor estabilidad numérica
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

print("=== DIAGNÓSTICO Y CARGA DE DATOS ===")

# ------------------------ CARGA Y ANÁLISIS INICIAL DE DATOS ------------------------

df = pd.read_csv(r"C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\Dataset_Con_Filtrado_y_Cuadrículas.csv")

num_columns = [
    'CONSTRUCTEDAREA', 'ROOMNUMBER', 'BATHNUMBER', 'FLATLOCATIONID', 
    'CADCONSTRUCTIONYEAR', 'CADMAXBUILDINGFLOOR', 'CADDWELLINGCOUNT', 
    'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA'
]

print(f"Forma del dataset: {df.shape}")
print(f"Columnas disponibles: {df.columns.tolist()}")

# Preparación de variables
X = pd.get_dummies(df.drop(columns=['ASSETID','PERIOD','PRICE','UNITPRICE','AMENITYID',
                                   'ISPARKINGSPACEINCLUDEDINPRICE','PARKINGSPACEPRICE',
                                   'HASNORTHORIENTATION','HASSOUTHORIENTATION',
                                   'HASEASTORIENTATION','HASWESTORIENTATION',
                                   'CONSTRUCTIONYEAR','FLATLOCATIONID','CADASTRALQUALITYID',
                                   'BUILTTYPEID_1','BUILTTYPEID_2','BUILTTYPEID_3',
                                   'LONGITUDE','LATITUDE','geometry','lon_normalized','lat_normalized'
]), drop_first=True)
y = df['PRICE']

print(f"\n=== ANÁLISIS DE DATOS ===")
print(f"Forma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# Verificar valores problemáticos
print(f"\nValores NaN en X: {X.isnull().sum().sum()}")
print(f"Valores NaN en y: {y.isnull().sum()}")
print(f"Valores infinitos en X: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()}")
print(f"Valores infinitos en y: {np.isinf(y).sum()}")

# Estadísticas del precio
print(f"\n=== ESTADÍSTICAS DEL PRECIO ===")
print(y.describe())
print(f"Rango de precios: {y.min():,.2f} - {y.max():,.2f}")

# Detectar outliers extremos en y
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR  # Usar 3 * IQR para outliers extremos
upper_bound = Q3 + 3 * IQR

outliers_mask = (y < lower_bound) | (y > upper_bound)
print(f"Outliers extremos detectados: {outliers_mask.sum()} ({outliers_mask.mean()*100:.1f}%)")

if outliers_mask.sum() > 0:
    print(f"Valores outliers - Min: {y[outliers_mask].min():,.2f}, Max: {y[outliers_mask].max():,.2f}")
    
    # Opcional: filtrar outliers extremos
    print("Filtrando outliers extremos...")
    X = X[~outliers_mask]
    y = y[~outliers_mask]
    print(f"Nuevo tamaño del dataset: {X.shape[0]} filas")

# Limpiar datos
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())  # Rellenar NaN con la mediana

print(f"\n=== DIVISIÓN DE DATOS ===")
# División en sets ANTES del escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

print(f"Tamaños - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Identificar columnas numéricas reales en el dataset procesado
numeric_cols_in_X = []
for col in X_train.columns:
    if any(num_col in col for num_col in num_columns):
        numeric_cols_in_X.append(col)

print(f"Columnas numéricas identificadas: {numeric_cols_in_X}")

print(f"\n=== ESCALADO DE DATOS ===")
# Escalado más conservador
scaler_X = RobustScaler()  # Más robusto a outliers
scaler_y = RobustScaler()  # Cambiar a RobustScaler también para y

# Crear copias para escalado
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

# Escalar solo las columnas numéricas
if numeric_cols_in_X:
    X_train_scaled[numeric_cols_in_X] = scaler_X.fit_transform(X_train[numeric_cols_in_X])
    X_val_scaled[numeric_cols_in_X] = scaler_X.transform(X_val[numeric_cols_in_X])
    X_test_scaled[numeric_cols_in_X] = scaler_X.transform(X_test[numeric_cols_in_X])

# Escalar el target de forma más suave
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

print(f"Estadísticas del target escalado:")
print(f"  Min: {y_train_scaled.min():.3f}, Max: {y_train_scaled.max():.3f}")
print(f"  Mean: {y_train_scaled.mean():.3f}, Std: {y_train_scaled.std():.3f}")

# Verificar que no hay valores problemáticos después del escalado
print(f"\nVerificación post-escalado:")
print(f"  NaN en X_train_scaled: {X_train_scaled.isnull().sum().sum()}")
print(f"  NaN en y_train_scaled: {np.isnan(y_train_scaled).sum()}")
print(f"  Inf en X_train_scaled: {np.isinf(X_train_scaled.select_dtypes(include=[np.number])).sum().sum()}")
print(f"  Inf en y_train_scaled: {np.isinf(y_train_scaled).sum()}")

# Convertir a float32 y verificar
X_train_array = X_train_scaled.values.astype(np.float32)
X_val_array = X_val_scaled.values.astype(np.float32)
y_train_array = y_train_scaled.astype(np.float32)
y_val_array = y_val_scaled.astype(np.float32)

print(f"Rango final de X_train: {X_train_array.min():.3f} - {X_train_array.max():.3f}")
print(f"Rango final de y_train: {y_train_array.min():.3f} - {y_train_array.max():.3f}")

# Crear datasets TensorFlow
batch_size = 64  # Batch size más pequeño para mayor estabilidad
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_array, y_train_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_array, y_val_array)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"\n=== DEFINICIÓN DEL MODELO ===")
# ------------------------ DEFINICIÓN DEL MODELO MÁS ESTABLE ------------------------

def crear_modelo_estable(input_dim):
    model = Sequential([
        # Capa de entrada con inicialización conservadora
        Dense(256, input_shape=(input_dim,), 
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros',
              kernel_regularizer=l2(0.0001)),  # Regularización más suave
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.2),
        
        # Segunda capa
        Dense(128, 
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros',
              kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.2),
        
        # Tercera capa
        Dense(64, 
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros',
              kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.1),
        
        # Capa de salida con inicialización muy conservadora
        Dense(1, 
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros')
    ])
    return model

# Crear modelo
modelo_estable = crear_modelo_estable(X_train_scaled.shape[1])

# Optimizador con configuración muy conservadora
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001,  # Learning rate muy bajo
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    clipnorm=1.0  # Gradient clipping importante
)

# Compilar con MSE simple 
modelo_estable.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

print(modelo_estable.summary())

print(f"\n=== ENTRENAMIENTO ===")
# ------------------------ ENTRENAMIENTO ------------------------

# Callbacks muy conservadores
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20,  # Más paciencia
    restore_best_weights=True, 
    verbose=1,
    min_delta=1e-5
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.8,  # Reducción más suave
    patience=10,  # Más paciencia
    min_lr=1e-7, 
    verbose=1,
    min_delta=1e-5
)

# Callback personalizado para verificar NaN
class NaNTerminator(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        if any(np.isnan(v) or np.isinf(v) for v in logs.values()):
            print(f"\n!!! NaN/Inf detectado en el batch {batch}: {logs}")
            self.model.stop_training = True

nan_terminator = NaNTerminator()

# Entrenamiento
print("Iniciando entrenamiento con configuración ultra-conservadora...")
try:
    history = modelo_estable.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,  # Menos épocas inicialmente
        callbacks=[early_stopping, reduce_lr, nan_terminator],
        verbose=1
    )
    
    training_successful = True
    
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    training_successful = False

if training_successful:
    print(f"\n=== EVALUACIÓN ===")
    # ------------------------ EVALUACIÓN ------------------------
    
    try:
        # Predicciones en el conjunto de prueba
        X_test_array = X_test_scaled.values.astype(np.float32)
        y_pred_scaled = modelo_estable.predict(X_test_array, verbose=0)
        
        # Verificar si las predicciones contienen NaN
        if np.isnan(y_pred_scaled).any():
            print("¡ADVERTENCIA! Las predicciones contienen valores NaN")
            # Reemplazar NaN con la media del target
            y_pred_scaled[np.isnan(y_pred_scaled)] = np.nanmean(y_pred_scaled)
        
        # Desescalar predicciones
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = y_test.values.reshape(-1, 1)
        
        # Métricas
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        rmse = np.sqrt(np.mean((y_test_original - y_pred)**2))
        
        print(f"\n=== RESULTADOS DEL MODELO ===")
        print(f"Error Absoluto Medio (MAE): {mae:,.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:,.2f}")
        
        # Verificar diversidad en las predicciones
        print(f"\n=== ANÁLISIS DE PREDICCIONES ===")
        print(f"Rango de predicciones: {y_pred.min():,.2f} - {y_pred.max():,.2f}")
        print(f"Desviación estándar de predicciones: {y_pred.std():,.2f}")
        print(f"Valores únicos en primeras 20 predicciones: {len(np.unique(y_pred[:20]))}")
        
        # Mostrar algunas predicciones vs valores reales
        print(f"\n=== COMPARACIÓN PREDICCIONES VS REALES (primeras 10) ===")
        for i in range(min(10, len(y_test_original))):
            print(f"Real: {y_test_original[i,0]:,.2f}, Predicho: {y_pred[i,0]:,.2f}, Diferencia: {abs(y_test_original[i,0] - y_pred[i,0]):,.2f}")
        
        # Guardar resultados (ahora conservamos el índice original de X_test)
        resultados_df = pd.DataFrame(
            {
                'Precio_real': y_test_original.flatten(),
                'Precio_predicho': y_pred.flatten(),
                'Error_absoluto': np.abs(y_test_original.flatten() - y_pred.flatten())
            },
            index=X_test_scaled.index          #  <-- línea clave
        )

        
        resultados_df.to_csv(r"C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\resultados_modelo_estable.csv", index=False)
        modelo_estable.save_weights(r'C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\modelo_estable.weights.h5')
        
        # Visualización
        plt.figure(figsize=(10, 5))
        
        # Gráfico 1: Pérdida durante entrenamiento
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], label='Validación')
        plt.title('Pérdida del modelo')
        plt.ylabel('Pérdida')
        plt.xlabel('Época')
        plt.legend()
        plt.yscale('log')
        
        # Gráfico 2: MAE durante entrenamiento
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Entrenamiento')
        plt.plot(history.history['val_mae'], label='Validación')
        plt.title('MAE del modelo')
        plt.ylabel('MAE')
        plt.xlabel('Época')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(r'C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\modelo_estable_analisis_ENTRENAMIENTO.png', dpi=300, bbox_inches='tight')
        #plt.show()


        plt.figure(figsize=(10, 5))


        # Gráfico 1: Predicciones vs valores reales
        plt.subplot(1, 2, 1)
        plt.scatter(y_test_original, y_pred, alpha=0.5)
        plt.plot([y_test_original.min(), y_test_original.max()], 
                 [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
        plt.xlabel('Precio Real')
        plt.ylabel('Precio Predicho')
        plt.title(f'Predicciones vs Reales (R² = {r2:.3f})')
        
        # Gráfico 2: Distribución de errores
        plt.subplot(1, 2, 2)
        errors = y_test_original.flatten() - y_pred.flatten()
        plt.hist(errors, bins=30, alpha=0.7)
        plt.xlabel('Error (Real - Predicho)')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Errores')
        
        plt.tight_layout()
        plt.savefig(r'C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\modelo_estable_analisis_R2_Y_DIST.png', dpi=300, bbox_inches='tight')
        #plt.show()
        
        import seaborn as sns

        # === Distribución de precios reales vs predichos ===
        plt.figure(figsize=(8, 5))
        sns.histplot(y_test_original.flatten(), color='orange', label='Precio Real', kde=True, stat="density", bins=30, alpha=0.6)
        sns.histplot(y_pred.flatten(), color='orange', label='Precio Predicho', kde=True, stat="density", bins=30, alpha=0.6)

        plt.xlabel('Precio')
        plt.ylabel('Densidad')
        plt.title('Distribución de Precios Reales vs Predichos')
        plt.legend()
        plt.savefig(r'C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado\distribucion_precios_reales_vs_predichos.png', dpi=300, bbox_inches='tight')
        #plt.show()

        print(f"\nModelo guardado exitosamente!")
        
    except Exception as e:
        print(f"Error durante la evaluación: {e}")
        print("El modelo se entrenó pero hubo problemas en la evaluación")

else:
    print("\n=== DIAGNÓSTICO ADICIONAL ===")
    print("El entrenamiento falló. Algunas sugerencias:")
    print("1. Verificar que no haya outliers extremos en los datos")
    print("2. Revisar la distribución de las variables de entrada")
    print("3. Considerar usar un modelo más simple inicialmente")
    print("4. Verificar la calidad de los datos de entrada")
    
    # Crear un modelo aún más simple para diagnosticar
    print("\nIntentando con un modelo lineal simple...")
    modelo_simple = Sequential([
        Dense(1, input_shape=(X_train_scaled.shape[1],), activation='linear')
    ])
    
    modelo_simple.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    try:
        historia_simple = modelo_simple.fit(
            X_train_array, y_train_array,
            validation_data=(X_val_array, y_val_array),
            epochs=5,
            verbose=1
        )
        print("✓ El modelo lineal simple funciona. El problema está en la complejidad de la red.")
    except Exception as e:
        print(f"✗ Incluso el modelo simple falla: {e}")
        print("El problema está en los datos de entrada.")

print(f"\n=== DIAGNÓSTICO COMPLETADO ===")

# === SHAP WATERFALL PARA UN SOLO CASO =======================================
# Requiere: modelo_estable entrenado, X_train_array, X_test_scaled, resultados_df

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Calcular el error porcentual de cada predicción del test
# ---------------------------------------------------------------------------
resultados_df["error_pct"] = (
    np.abs(resultados_df["Precio_real"] - resultados_df["Precio_predicho"])
    / resultados_df["Precio_real"]
) * 100

# ---------------------------------------------------------------------------
# 2. Elegir el primer caso con error alto (>100 %, o >50 % si no hay)
# ---------------------------------------------------------------------------
errores_altos = resultados_df[resultados_df["error_pct"] > 100]
if errores_altos.empty:
    errores_altos = resultados_df[resultados_df["error_pct"] > 50]

indice_caso = errores_altos.index[0]          # índice del caso elegido
caso_scaled  = X_test_scaled.loc[[indice_caso]]   # DataFrame de UNA fila

# ---------------------------------------------------------------------------
# 3. Crear el explainer (usa automáticamente Deep/Gradient para Keras)
#    -> se toma una muestra pequeña de entrenamiento como background
# ---------------------------------------------------------------------------
np.random.seed(42)
background_idx = np.random.choice(len(X_train_array), size=100, replace=False)
explainer = shap.Explainer(
    modelo_estable.predict,           # función predict del modelo
    X_train_array[background_idx]     # background en formato np.array
)

# ---------------------------------------------------------------------------
# 4. Obtener SHAP values del caso y graficar waterfall
# ---------------------------------------------------------------------------
shap_values = explainer(caso_scaled)     # shap.Explanation
sv_row = shap_values[0]                  # contribuciones de esa fila

# Carpeta de salida
out_dir = Path(r"C:\Users\costa\Desktop\TFG\7.5 TensorFlow con filtrado")
out_dir.mkdir(parents=True, exist_ok=True)

# 4.1. Waterfall plot --------------------------------------------------------
plt.figure()
shap.plots.waterfall(sv_row, max_display=25, show=False)
plt.savefig(out_dir / "waterfall_high_NONO_error_NN.png",
            dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 4-bis. Convertir contribuciones y valor base a EUROS
# ---------------------------------------------------------------------------
target_scale  = scaler_y.scale_[0]     # IQR del precio
target_median = scaler_y.center_[0]    # mediana del precio

sv_row_eur = shap.Explanation(
    values      = sv_row.values * target_scale,            # contribuciones € 
    base_values = sv_row.base_values * target_scale + target_median,  # valor base €
    data        = sv_row.data,
    feature_names = sv_row.feature_names
)

# 4.1. Waterfall plot (ahora en €) ------------------------------------------
plt.figure()
shap.plots.waterfall(sv_row_eur, max_display=25, show=False)
plt.savefig(out_dir / "waterfall_high_error_NN.png",
            dpi=300, bbox_inches="tight")
plt.close()


print("\n=== CONTRIBUCIONES SHAP – CASO DE ERROR ALTO ===")

