import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# Cargar dataset
data = pd.read_csv('rotation_memory.csv')

# Inicializar listas
state_list, rotation_list, action_list, utility_list = [], [], [], []

# Parsear filas
for index, row in data.iterrows():
    try:
        state = ast.literal_eval(row['state'])
        rotation = float(row['rotation'])
        action = ast.literal_eval(row['action'])  # as list
        utility = float(row['utility'])

        if len(state) == 3 and len(action) == 2:
            state_list.append(state)
            rotation_list.append([rotation])  # keep as 2D
            action_list.append(action)
            utility_list.append(utility)
        else:
            print(f"Fila {index} con longitud incorrecta, ignorada.")
    except Exception as e:
        print(f"Error en fila {index}: {e}")
        continue

# Convertir a arrays
state_array = np.array(state_list)
rotation_array = np.array(rotation_list)
action_array = np.array(action_list)
utility_array = np.array(utility_list).reshape(-1, 1)

# Escalar características
scaler_state = MinMaxScaler()
scaler_rotation = MinMaxScaler()
scaler_action = MinMaxScaler()

state_scaled = scaler_state.fit_transform(state_array)
rotation_scaled = scaler_rotation.fit_transform(rotation_array)
action_scaled = scaler_action.fit_transform(action_array)

# Concatenar entrada
X = np.hstack([state_scaled, rotation_scaled])#, action_scaled])
y = utility_array

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Definir modelo
model = keras.models.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Entrenar
model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2, verbose=1, callbacks=[early_stop])

# Evaluar
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# Predicción de ejemplo
sample_input = X_test[5].reshape(1, -1)
pred = model.predict(sample_input)

print("Predicción (utilidad):", pred)
print("Real (utilidad):", y_test[5])


sample_input = X_test[1].reshape(1, -1)
pred = model.predict(sample_input)

print("Predicción (utilidad):", pred)
print("Real (utilidad):", y_test[1])

# Guardar modelo
model.save('utility_model4.h5')
