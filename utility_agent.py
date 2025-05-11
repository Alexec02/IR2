import numpy as np
import pandas as pd
import ast
import time
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from perception import getDistance, get_perception, getRotation
from actuation import move
from behavior import wall_avoidance, obstacle_avoidance
from world_test import move_yellow_to_corner, reset_if_needed, reached_red
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

def sample_actions(n=10):
    return [np.array([np.random.uniform(10, 20), np.random.uniform(10, 20)]) for _ in range(n)]


"""def sample_actions():
    actions = []

    # Avance recto
    for _ in range(3):
        speed = np.random.uniform(10, 20)
        variation = np.random.uniform(-1, 1)
        actions.append(np.array([speed, speed + variation]))

    # Giro suave derecha
    for _ in range(3):
        left = np.random.uniform(10, 15)
        right = left + np.random.uniform(2, 5)
        actions.append(np.array([left, right]))

    # Giro suave izquierda
    for _ in range(3):
        right = np.random.uniform(10, 15)
        left = right + np.random.uniform(2, 5)
        actions.append(np.array([left, right]))

    # Giro brusco
    for _ in range(3):
        slow = np.random.uniform(0, 5)
        fast = np.random.uniform(15, 20)
        if np.random.rand() < 0.5:
            actions.append(np.array([slow, fast]))
        else:
            actions.append(np.array([fast, slow]))

    return actions"""

def main_loop(sim, rob):
    # Load rotation_memory.csv
    data = pd.read_csv('rotation_memory.csv', on_bad_lines='skip')

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

    # Load trained utility model
    utility_model = load_model("utility_model.h5")
    # Load trained model
    model = load_model("model_rotation.h5")
    step = 0

    while True:
        reset_if_needed(rob, sim)
        wall_avoidance(sim, rob)
        obstacle_avoidance(sim, rob)

        step += 1
        current_state = getDistance(sim)
        current_rotation = [getRotation(sim)[1]]

        norm_state = scaler_state.transform([current_state])[0]
        norm_rot = scaler_rotation.transform([current_rotation])[0]

        candidate_actions = sample_actions()
        utility_predictions = []

        for action in candidate_actions:
            norm_action = scaler_action.transform([action])[0]
            model_input = np.concatenate([norm_state, norm_rot, norm_action]).reshape(1, -1)
            pred_state = model.predict(model_input,verbose=0)[0]
            
            model_input = np.concatenate([pred_state]).reshape(1, -1)#, norm_rot, norm_action]).reshape(1, -1)
            predicted_utility = utility_model.predict(model_input, verbose=0)[0][0]
            utility_predictions.append(predicted_utility)

        best_idx = np.argmax(utility_predictions)
        selected_action = candidate_actions[best_idx]
        print(f"Step {step} - Best predicted utility: {utility_predictions[best_idx]:.4f}")

        move(rob, selected_action[0], selected_action[1])

if __name__ == "__main__":
    rob = Robobo("localhost")
    rob.connect()
    sim = RoboboSim("localhost")
    sim.connect()
    time.sleep(1)
    rob.moveTiltTo(90, 5)
    move_yellow_to_corner(sim)
    main_loop(sim, rob)
