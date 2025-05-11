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

def compute_novelty(candidate_state, memory_states, n=2):
    """
    candidate_state: np.array of shape (D,) - the k-th candidate state
    memory_states: np.array of shape (M, D) - list of past states
    n: float - novelty sensitivity coefficient
    """
    if len(memory_states) == 0:
        return float('inf')  # Encourage exploring completely new areas
    
    # Calculate distances to all past states
    distances = np.linalg.norm(memory_states - candidate_state, axis=1)
    
    # Raise distances to power n and average
    novelty = np.mean(distances ** n)
    return novelty

# Define action space (motor1, motor2) range from 0–20
"""def sample_actions(n=10):
    return [np.array([np.random.uniform(0, 20), np.random.uniform(0, 20)]) for _ in range(n)]
"""

def sample_actions():
    actions = []

    # Avance recto: motores similares
    for _ in range(3):
        speed = np.random.uniform(10, 20)
        variation = np.random.uniform(-1, 1)
        actions.append(np.array([speed, speed + variation]))

    # Giro suave hacia un lado
    for _ in range(3):
        left = np.random.uniform(10, 15)
        right = left + np.random.uniform(2, 5)
        actions.append(np.array([left, right]))

    # Giro suave hacia el otro lado
    for _ in range(3):
        right = np.random.uniform(10, 15)
        left = right + np.random.uniform(2, 5)
        actions.append(np.array([left, right]))

    # Giro brusco (uno muy lento)
    for _ in range(3):
        slow = np.random.uniform(0, 5)
        fast = np.random.uniform(15, 20)
        if np.random.rand() < 0.5:
            actions.append(np.array([slow, fast]))
        else:
            actions.append(np.array([fast, slow]))

    # Agregar algo de ruido totalmente aleatorio (exploración)
    #for _ in range(8):
    #    actions.append(np.array([np.random.uniform(0, 20), np.random.uniform(0, 20)]))

    return actions

def main_loop(sim, rob):
    # Load dataset to fit normalizers
    data = pd.read_csv('world_model_dataset_rotation.csv', header=None, on_bad_lines='skip')
    print(data.columns)

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

    # Load trained model
    model = load_model("model_rotation.h5")

    memory = []          # For novelty calculation
    trace = []           # Current episode trace
    trace_memory = []    # Long-term memory for extrinsic model
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
        predicted_states = []
        predicted_rotations = []

        # First, check if any predicted state reaches the red goal
        goal_reached_in_candidates = False
        selected_action = None

        for action in candidate_actions:
            norm_action = scaler_actions.transform([action])[0]
            model_input = np.concatenate([norm_state, norm_rot, norm_action]).reshape(1, -1)
            pred_next_state = model.predict(model_input, verbose=0)[0]

            predicted_states.append(pred_next_state[:3])
            #print(pred_next_state[:4])
            # Check if the predicted state would reach the goal
            simulated_state = scaler_pos_after.inverse_transform([pred_next_state[:3]])[0]
            #simulated_rot
            
            simulated_perception = get_perception(rob)  # Use current perception as approximation
            distance_estimate = simulated_state[0]  # Assuming 0th dimension is relevant for distance

            if reached_red(simulated_perception, distance_estimate):
                goal_reached_in_candidates = True
                selected_action = action
                break

        if goal_reached_in_candidates:
            print("Goal can be reached with one of the predicted actions! Executing it now.")
        else:
            # No immediate goal found, select based on novelty
            novelty_scores = [compute_novelty(s, memory) for s in predicted_states]
            best_idx = np.argmax(novelty_scores)
            selected_action = candidate_actions[best_idx]
            print(f"Step {step} - Best novelty score: {novelty_scores[best_idx]:.4f}")

        move(rob, selected_action[0], selected_action[1])

        # Update state and trace
        current_state = getDistance(sim)
        current_rotation = getRotation(sim)[1]
        
        #print("Difference in prediction: "+current_state-scaler_state.inverse_transform([predicted_states[best_idx]]))
        memory.append(current_state)
        if len(memory) > 15:
            memory.pop(0)  # Keep memory length to 10

        trace.append((current_state, current_rotation, selected_action))
        if len(trace) > 15:
            trace.pop(0)  # Keep trace length to 10


        # Check if red goal is actually reached now
        perception = get_perception(rob)
        distance = current_state[0]
        if reached_red(perception, distance):
            print("Red goal reached in real world. Backpropagating utility trace.")

            max_trace_len = 15
            for i, (state, rotation, action) in enumerate(reversed(trace[-max_trace_len:])):
                utility = 1.0 - (i / max_trace_len)
                trace_memory.append((state, rotation, action, utility))
            print(trace_memory)
            # save trace data
            df = pd.DataFrame([
                {"state": s, "rotation": r, "action": a, "utility": u}
                for s, r, a, u in trace_memory
            ])

            file_path = "rotation_memory.csv"
            file_exists = os.path.isfile(file_path)

            df.to_csv(file_path, mode='a', header=not file_exists, index=False)

            trace = []   # Clear episode trace
            memory = []  # Reset novelty memory for new exploration


        

if __name__ == "__main__":
    rob = Robobo("localhost")
    rob.connect()
    sim = RoboboSim("localhost")
    sim.connect()
    time.sleep(1)
    rob.moveTiltTo(90,5)
    move_yellow_to_corner(sim)
    main_loop(sim,rob)