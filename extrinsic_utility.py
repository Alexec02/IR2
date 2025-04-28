# extrinsic_utility.py
from sklearn.neural_network import MLPRegressor
import numpy as np

class ExtrinsicUtility:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)

    def train(self, states, utilities):
        self.model.fit(states, utilities)
        print("Extrinsic utility model trained.")

    def predict_utility(self, state):
        return self.model.predict([state])[0]

# Example usage:
if __name__ == "__main__":
    states = np.random.rand(10, 3)  # collected states
    utilities = np.linspace(0, 1, 10)  # utilities from 0 (far) to 1 (goal)
    eu = ExtrinsicUtility()
    eu.train(states, utilities)
    utility_prediction = eu.predict_utility([0.5, 0.5, 0.5])
    print("Predicted utility:", utility_prediction)
