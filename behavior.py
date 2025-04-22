# behavior.py
import random
from robobopy.utils.IR import IR
from robobopy import Robobo
from actuation import turn_left, turn_right

PROX_THRESHOLD = 0.3

def wall_avoidance(rob):
    # Read proximity sensors using robobopy
    front = rob.readIRSensor(IR.FrontC)
    left = rob.readIRSensor(IR.FrontL)
    right = rob.readIRSensor(IR.FrontR)

    # Wall avoidance based on proximity to walls (through IR sensors)
    if front < PROX_THRESHOLD:
        turn_left(rob)
        return True
    elif left < PROX_THRESHOLD:
        turn_right(rob)
        return True
    elif right < PROX_THRESHOLD:
        turn_left(rob)
        return True

    return False

def obstacle_avoidance(rob):
    # Check for obstacles ahead using the front IR sensor
    front = rob.readIRSensor(IR.FrontC)

    if front < PROX_THRESHOLD:
        # Randomly choose to turn left or right to avoid the obstacle
        if random.random() < 0.5:
            turn_left(rob)
        else:
            turn_right(rob)
        return True

    return False
