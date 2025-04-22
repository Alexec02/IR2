# main.py
import csv
import os
import time
import random
from perception import get_perception
from actuation import move_forward, turn_left, turn_right
from behavior import wall_avoidance, obstacle_avoidance
from world_test import move_yellow_to_corner, reset_if_needed
from robobopy.Robobo import Robobo

DATASET_FILE = "world_model_dataset.csv"
ACTIONS = ["forward", "left", "right"]

ACTION_FUNCTIONS = {
    "forward": move_forward,
    "left": turn_left,
    "right": turn_right
}

def log_to_dataset(p_before, action, p_after):
    file_exists = os.path.isfile(DATASET_FILE)

    with open(DATASET_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            headers = list(p_before.keys()) + ['action'] + [f"{k}_next" for k in p_after.keys()]
            writer.writerow(headers)

        row = list(p_before.values()) + [action] + list(p_after.values())
        writer.writerow(row)

def main_loop():
    rob = Robobo("localhost")
    rob.connect()

    move_yellow_to_corner(rob)

    while True:
        #reset_if_needed(rob)

        if wall_avoidance(rob):
            continue
        if obstacle_avoidance(rob):
            continue

        # --- Perception before action ---
        p_before = get_perception(rob)

        # --- Random action ---
        action = random.choice(ACTIONS)
        ACTION_FUNCTIONS[action](rob)
        time.sleep(0.6)

        # --- Perception after action ---
        p_after = get_perception(rob)

        # --- Log step ---
        log_to_dataset(p_before, action, p_after)
        print("Logged:", action, p_before, "->", p_after)

if __name__ == "__main__":
    main_loop()
