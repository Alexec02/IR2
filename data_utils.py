# data_utils.py
import numpy as np

def save_data(filename, actions, perceptions):
    np.savez(filename, actions=actions, perceptions=perceptions)

def load_data(filename):
    data = np.load(filename)
    return data['actions'], data['perceptions']
