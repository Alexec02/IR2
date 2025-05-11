import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping


# Cargar dataset
data = pd.read_csv('world_model_dataset_rotation.csv', header=None, on_bad_lines='skip')

# Inicializar listas
pos_before_list, rot_before_list, actions_list = [], [], []
pos_after_list, rot_after_list = [], []

# Parsear filas con validación
for index, row in data.iterrows():
    try:
        pos_before = ast.literal_eval(row[0])
        rot_before = ast.literal_eval(row[1])
        action = ast.literal_eval(row[2])
        pos_after = ast.literal_eval(row[3])
        rot_after = ast.literal_eval(row[4])

        # Validar longitud de cada parte
        if (len(pos_before) == 3 and len(rot_before) == 3 and
            len(action) == 2 and len(pos_after) == 3 and len(rot_after) == 3):

            pos_before_list.append(pos_before)
            rot_before_list.append([rot_before[1]])
            actions_list.append(action)
            pos_after_list.append(pos_after)
            rot_after_list.append([rot_after[1]])

        else:
            print(f"Fila {index} con longitud incorrecta, ignorada.")

    except Exception as e:
        print(f"Error en fila {index}: {e}")
        continue

# Convertir a arrays
pos_before = np.array(pos_before_list)
rot_before = np.array(rot_before_list)
actions = np.array(actions_list)
pos_after = np.array(pos_after_list)
rot_after = np.array(rot_after_list)

# Verificar formas antes de escalar
print("Shapes:")
print("pos_before:", pos_before.shape)
print("rot_before:", rot_before.shape)
print("actions:", actions.shape)
print("pos_after:", pos_after.shape)
print("rot_after:", rot_after.shape)

# Escaladores por separado
scaler_pos_before = MinMaxScaler()
scaler_rot_before = MinMaxScaler()
scaler_actions = MinMaxScaler()
scaler_pos_after = MinMaxScaler()
scaler_rot_after = MinMaxScaler()

pos_before_scaled = scaler_pos_before.fit_transform(pos_before)
rot_before_scaled = scaler_rot_before.fit_transform(rot_before)
actions_scaled = scaler_actions.fit_transform(actions)
pos_after_scaled = scaler_pos_after.fit_transform(pos_after)
rot_after_scaled = scaler_rot_after.fit_transform(rot_after)

# Entradas y salidas
X = np.hstack([pos_before_scaled, rot_before_scaled, actions_scaled])
y = np.hstack([pos_after_scaled, rot_after_scaled])

# Dividir conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Definir modelo
model = keras.models.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4)
])

model.compile(optimizer='adam', loss='mse')

# Entrenar
model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2, verbose=1, callbacks=[early_stop])

# Evaluar
model.evaluate(X_test, y_test)

# Predicción de ejemplo
sample_input = X_test[1].reshape(1, -1)
pred_scaled = model.predict(sample_input)
true_scaled = y_test[1].reshape(1, -1)

# Separar la predicción en posición y rotación
pred_pos_scaled = pred_scaled[:, :3]
pred_rot_scaled = pred_scaled[:, 3:]
true_pos_scaled = true_scaled[:, :3]
true_rot_scaled = true_scaled[:, 3:]

# Desnormalizar
pred_pos = scaler_pos_after.inverse_transform(pred_pos_scaled)
pred_rot = scaler_rot_after.inverse_transform(pred_rot_scaled)
true_pos = scaler_pos_after.inverse_transform(true_pos_scaled)
true_rot = scaler_rot_after.inverse_transform(true_rot_scaled)

print("Predicción (posición):", pred_pos)
print("Predicción (rotación):", pred_rot)
print("Real (posición):", true_pos)
print("Real (rotación):", true_rot)

# Predicción de ejemplo
sample_input = X_test[13].reshape(1, -1)
pred_scaled = model.predict(sample_input)
true_scaled = y_test[13].reshape(1, -1)

# Separar la predicción en posición y rotación
pred_pos = pred_scaled[:, :3]
pred_rot = pred_scaled[:, 3:]
true_pos = true_scaled[:, :3]
true_rot = true_scaled[:, 3:]

print("Predicción (posición):", pred_pos)
print("Predicción (rotación):", pred_rot)
print("Real (posición):", true_pos)
print("Real (rotación):", true_rot)

# Guardar modelo
model.save('model_rotation.h5')
