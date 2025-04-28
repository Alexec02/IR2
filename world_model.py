# world_model.py
from sklearn.neural_network import MLPRegressor
import numpy as np
from data_utils import load_data

class WorldModel:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)

    def train(self, actions, perceptions):
        self.model.fit(actions, perceptions)
        print("World model training completed.")

    def predict(self, action):
        return self.model.predict([action])[0]

if __name__ == "__main__":
    actions, perceptions = load_data("robot_data.npz")
    wm = WorldModel()
    wm.train(actions, perceptions)
    prediction = wm.predict([45])  # Predict perception after action 45º
    print("Predicted perception:", prediction)
