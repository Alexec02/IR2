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
from keras.src.legacy.saving import legacy_h5_format

def sample_actions(n=5):
    actions = [np.array([np.random.uniform(0, 20), np.random.uniform(0, 20)]) for _ in range(n)]
    for _ in range(n//2):
        speed = np.random.uniform(10, 30)
        variation = np.random.uniform(-1, 1)
        actions.append(np.array([speed, speed + variation]))

    return actions


"""def sample_actions():
    actions = []

    # Avance recto: motores similares
    for _ in range(1):
        speed = np.random.uniform(10, 20)
        variation = np.random.uniform(-1, 1)
        actions.append(np.array([speed, speed + variation]))

    # Giro suave hacia un lado
    for _ in range(1):
        left = np.random.uniform(10, 15)
        right = left + np.random.uniform(2, 5)
        actions.append(np.array([left, right]))

    # Giro suave hacia el otro lado
    for _ in range(1):
        right = np.random.uniform(10, 15)
        left = right + np.random.uniform(2, 5)
        actions.append(np.array([left, right]))

    # Giro brusco (uno muy lento)
    for _ in range(1):
        slow = np.random.uniform(0, 5)
        fast = np.random.uniform(15, 20)
        actions.append(np.array([slow, fast]))
        actions.append(np.array([fast, slow]))

    # Agregar algo de ruido totalmente aleatorio (exploración)
    for _ in range(4):
        actions.append(np.array([np.random.uniform(0, 20), np.random.uniform(0, 20)]))

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
    scaler_state_utility = MinMaxScaler()
    scaler_rotation_utility = MinMaxScaler()
    scaler_action_utility = MinMaxScaler()

    state_scaled = scaler_state_utility.fit_transform(state_array)
    rotation_scaled = scaler_rotation_utility.fit_transform(rotation_array)
    action_scaled = scaler_action_utility.fit_transform(action_array)

    # Load dataset to fit normalizers
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


    # Load trained utility model
    utility_model = legacy_h5_format.load_model_from_hdf5("utility_model2.h5", custom_objects={'mse': 'mse'})
    # Load trained model
    model = legacy_h5_format.load_model_from_hdf5("model_rotation.h5", custom_objects={'mse': 'mse'})

    step = 0

    while True:
        reset_if_needed(rob, sim)
        wall_avoidance(sim, rob)
        obstacle_avoidance(sim, rob)

        step += 1
        current_state = getDistance(sim)
        current_rotation = [getRotation(sim)[1]]

        norm_state = scaler_pos_before.transform([current_state])[0]
        norm_rot = scaler_rot_before.transform([current_rotation])[0]

        candidate_actions = sample_actions()
        utility_predictions = []

        for action in candidate_actions:
            norm_action = scaler_actions.transform([action])[0]
            model_input = np.concatenate([norm_state, norm_rot, norm_action]).reshape(1, -1)
            pred_state = model.predict(model_input,verbose=0)[0]

            new_state = scaler_pos_after.inverse_transform([pred_state[:3]])[0]
            new_rot = scaler_rot_after.inverse_transform([pred_state[3:]])[0]
            new_state = scaler_state_utility.transform([new_state])[0]
            new_rot = scaler_rotation_utility.transform([new_rot])[0]

            model_input = np.concatenate([new_state,new_rot]).reshape(1, -1)#, norm_rot, norm_action]).reshape(1, -1)
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
