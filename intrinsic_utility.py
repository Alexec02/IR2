# intrinsic_utility.py
import numpy as np

class IntrinsicUtility:
    def __init__(self, memory_states):
        self.memory_states = memory_states

    def calculate_novelty(self, candidate_state):
        distances = [np.linalg.norm(candidate_state - state) for state in self.memory_states]
        novelty = np.mean(distances)
        return novelty

if __name__ == "__main__":
    memory_states = np.random.rand(10, 3)  # previously visited states (random example)
    iu = IntrinsicUtility(memory_states)
    candidate_state = np.array([0.5, 0.5, 0.5])
    novelty_score = iu.calculate_novelty(candidate_state)
    print("Novelty score:", novelty_score)
