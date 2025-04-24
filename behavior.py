# behavior.py
import random
from robobopy.utils.IR import IR
from robobopy import Robobo
from actuation import turn_left, turn_right

PROX_THRESHOLD = 60

def wall_avoidance(rob):
    # Read proximity sensors using robobopy
    sensors = {
        'BackR': rob.readIRSensor(IR.BackR),
        'BackC': rob.readIRSensor(IR.BackC),
        'FrontR': rob.readIRSensor(IR.FrontR),
        'FrontRR': rob.readIRSensor(IR.FrontRR),
        'FrontC': rob.readIRSensor(IR.FrontC),
        'FrontL': rob.readIRSensor(IR.FrontL),
        'FrontLL': rob.readIRSensor(IR.FrontLL),
        'BackL': rob.readIRSensor(IR.BackL),
    }

    # Wall avoidance based on proximity to walls (through IR sensors)
    if sensors['FrontL'] > PROX_THRESHOLD or sensors['FrontLL'] > PROX_THRESHOLD:
        print("Wall on left - turning right")
        turn_right(rob)
        return True
    elif sensors['FrontR'] > PROX_THRESHOLD or sensors['FrontRR'] > PROX_THRESHOLD:
        print("Wall on right - turning left")
        turn_left(rob)
        return True
    elif sensors['BackR'] > PROX_THRESHOLD:
        print("Wall on back right - turning left")
        turn_left(rob)
        return True
    elif sensors['BackL'] > PROX_THRESHOLD:
        print("Wall on back left - turning right")
        turn_right(rob)
        return True
        

    return False

def obstacle_avoidance(rob):
    # Check for obstacles ahead using the front IR sensor
    front = rob.readIRSensor(IR.FrontC)

    if front > PROX_THRESHOLD:
        # Randomly choose to turn left or right to avoid the obstacle
        if random.random() < 0.5:
            turn_left(rob)
        else:
            turn_right(rob)
        return True

    return False
