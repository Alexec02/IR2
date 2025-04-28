# main.py
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
import numpy as np
from robot_actions import RobotActions
from data_utils import save_data
from world_model import WorldModel
from intrinsic_utility import IntrinsicUtility
from extrinsic_utility import ExtrinsicUtility

def collect_data(robobo, sim, num_samples=50):
    actions = []
    perceptions = []

    robot_actions = RobotActions(robobo)
    for _ in range(num_samples):
        action = np.random.choice([-90, -45, 0, 45, 90])
        robot_actions.execute_action(action)

        perception = [
            sim.get_distance_to_object("red"),
            sim.get_distance_to_object("green"),
            sim.get_distance_to_object("blue")
        ]

        actions.append([action])
        perceptions.append(perception)

    save_data("robot_data.npz", actions, perceptions)
    print("Data collection completed.")

def main():
    robobo = Robobo("localhost")
    sim = RoboboSim("localhost")

    collect_data(robobo, sim)

    actions, perceptions = np.array(actions), np.array(perceptions)

    # Train World Model
    wm = WorldModel()
    wm.train(actions, perceptions)

    # Intrinsic Utility setup
    # intrinsic_util = IntrinsicUtility(memory_states=perceptions)

    # Example of evaluating intrinsic utility
    # action_example = [45]
    # predicted_state = wm.predict(action_example)
    # novelty_score = intrinsic_util.calculate_novelty(predicted_state)
    # print("Novelty Score for action", action_example, ":", novelty_score)

    # Placeholder for extrinsic utility training once goal is discovered
    # extrinsic_util = ExtrinsicUtility()
    # extrinsic_util.train(goal_states, utilities)

if __name__ == "__main__":
    main()
