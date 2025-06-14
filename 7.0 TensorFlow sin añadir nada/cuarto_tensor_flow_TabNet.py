import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# TabNet implementation using TensorFlow (simplified version)
class TabNet(tf.keras.Model):
    def __init__(
        self,
        feature_columns,
        num_decision_steps=5,
        feature_dim=64,
        output_dim=1,
        relaxation_factor=1.5,
        bn_momentum=0.7,
    ):
        super(TabNet, self).__init__()
        self.feature_columns = feature_columns
        self.num_decision_steps = num_decision_steps
        self.feature_dim = feature_dim
        self.relaxation_factor = relaxation_factor
        self.output_dim = output_dim
        self.bn_momentum = bn_momentum
        
        # Feature transformer - shared across decision steps
        self.transform = tf.keras.Sequential([
            tf.keras.layers.Dense(feature_dim * 2, use_bias=False),
            tf.keras.layers.BatchNormalization(momentum=bn_momentum)
        ])
        
        # First decision step - feature selection
        self.decision_layers = []
        for _ in range(num_decision_steps):
            decision_layer = tf.keras.Sequential([
                tf.keras.layers.Dense(feature_dim, activation='relu'),
                tf.keras.layers.BatchNormalization(momentum=bn_momentum),
                tf.keras.layers.Dense(len(feature_columns), activation='sigmoid')
            ])
            self.decision_layers.append(decision_layer)
        
        # Final output layer
        self.output_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(feature_dim, activation='relu'),
            tf.keras.layers.Dense(output_dim)
        ])
        
        # Attentive transformer
        self.attentive = tf.keras.Sequential([
            tf.keras.layers.Dense(len(feature_columns), activation='sigmoid')
        ])
        
    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        
        # Initialize attention and aggregated features
        prior_scales = tf.ones((batch_size, len(self.feature_columns)))
        aggregated = tf.zeros((batch_size, self.feature_dim))
        
        for step_idx in range(self.num_decision_steps):
            # Apply attention mechanism
            attention = self.attentive(prior_scales)
            masked_features = inputs * attention
            
            # Feature transformation
            features = self.transform(masked_features)
            features, gates = tf.split(features, 2, axis=1)
            features = tf.nn.relu(features)
            gates = tf.nn.sigmoid(gates)
            
            # Apply gates
            decision = features * gates
            
            # Update prior scales for the next step
            decision_mask = self.decision_layers[step_idx](decision)
            
            # Relaxation factor for sparsity
            prior_scales = prior_scales * (self.relaxation_factor - decision_mask)
            
            # Aggregate features
            aggregated += decision
        
        # Final output
        return self.output_layer(aggregated)

# Load data
df = pd.read_csv(r"C:\Users\costa\Desktop\TFG\7.0 TensorFlow sin añadir nada\Madrid_Sale.csv")

# Numeric columns to normalize
num_columns = ['CONSTRUCTEDAREA', 'ROOMNUMBER', 'BATHNUMBER', 'FLATLOCATIONID', 
               'CADCONSTRUCTIONYEAR', 'CADMAXBUILDINGFLOOR', 'CADDWELLINGCOUNT', 
               'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA']

# Use RobustScaler for features (better with outliers)
scaler = RobustScaler()
df[num_columns] = scaler.fit_transform(df[num_columns])

# Feature engineering (optional)
# Create price per square meter in the area if needed
# df['PRICE_PER_SQM'] = df['PRICE'] / df['CONSTRUCTEDAREA']

# Prepare data
X = pd.get_dummies(df.drop(columns=['ASSETID', 'PERIOD', 'UNITPRICE', 'CONSTRUCTIONYEAR', 'CADASTRALQUALITYID', 'geometry', 'PRICE']), drop_first=True)
y = df['PRICE']

# Use RobustScaler for target variable
y_scaler = RobustScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split into training and test sets
X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Create validation set
X_train, X_val, y_train_scaled, y_val_scaled = train_test_split(X_train, y_train_scaled, test_size=0.15, random_state=42)

# Convert to TensorFlow datasets
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values.astype(np.float32), 
                                                   y_train_scaled.astype(np.float32))).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values.astype(np.float32), 
                                                 y_val_scaled.astype(np.float32))).batch(batch_size)

# Create TabNet model
tabnet_model = TabNet(
    feature_columns=X_train.columns,
    num_decision_steps=5,  # Number of decision steps
    feature_dim=64,        # Feature dimension
    output_dim=1,          # Single output for regression
    relaxation_factor=1.5, # For feature selection sparsity
    bn_momentum=0.7        # BatchNorm momentum
)

# Configure optimizer with clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

# Compile model
tabnet_model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

# Configure callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train the model
history = tabnet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate the model
# Convert test data to tensors
X_test_tensor = tf.convert_to_tensor(X_test.values.astype(np.float32))
y_pred_scaled = tabnet_model.predict(X_test_tensor)

# Inverse transform predictions to original scale
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Error Absoluto Medio (MAE): {mae}")

# Save model
tabnet_model.save_weights('tabnet_model_weights.weights.h5')

# Visualize training history
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
plt.savefig('tabnet_training_history.png')
plt.show()