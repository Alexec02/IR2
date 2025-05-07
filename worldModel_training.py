import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import ast
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers

# Load the CSV data
data = pd.read_csv('world_model_dataset.csv', header=None, on_bad_lines='skip')
print(data.columns)

X = []
y = []

# Iterate correctly over rows
states = []
actions = []
next_states = []

for index, row in data.iterrows():
    state = ast.literal_eval(row[0])
    action = ast.literal_eval(row[1])
    next_state = ast.literal_eval(row[2])

    states.append(state)
    actions.append(action)
    next_states.append(next_state)

states = np.array(states)
actions = np.array(actions)
next_states = np.array(next_states)

# 2. Normalizar por separado
scaler_state = MinMaxScaler()
scaler_action = MinMaxScaler()
scaler_next_state = MinMaxScaler()

states = scaler_state.fit_transform(states)
actions = scaler_action.fit_transform(actions)
next_states = scaler_next_state.fit_transform(next_states)

# 3. Concatenar para entrada al modelo
X = np.hstack([states, actions])
y = next_states

print(X.shape, y.shape)

print(X[1])
print(y[1])
# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""model = keras.models.Sequential([
    layers.Bidirectional(layers.LSTM(64, activation='relu', input_shape=(X.shape[1], 1), return_sequences=True)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)),
    layers.Dense(y.shape[1])
])"""

model = keras.models.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(3)])

model.compile(optimizer='adam', loss='mse')

# Entrenar
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2,verbose=1)
model.evaluate(X_test,y_test)
print(X_test[1])
print(model.predict(X_test[1].reshape(1, -1)))
print(y_test[1])
# Desnormalizar las predicciones y las etiquetas reales
y_pred_real = scaler_next_state.inverse_transform(model.predict(X_test[1].reshape(1, -1)))
y_test_real = scaler_next_state.inverse_transform(y_test[1].reshape(1, -1))
print(y_pred_real)
print(y_test_real)

# Guardar en .h5
model.save('modelo.h5')