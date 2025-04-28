# behavior.py
import random
from robobopy.utils.IR import IR
from robobopy import Robobo
from actuation import move
from perception import get_perception, getDistance
from world_test import get_safe_random_location


PROX_THRESHOLD = 30

def wall_avoidance(sim,rob):
    
    wall=True
    count=0
    # Wall avoidance based on proximity to walls (through IR sensors)
    while wall:
        #avoid stucked position
        if count==5:
            sim.setRobotLocation(0,get_safe_random_location(sim), {'x': 0.0, 'y': 0.0, 'z': 0.0})


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
        #print(sensors)
        #FORNTL;FRONTR;BACKC always 57
        if  (sensors['FrontLL'] ) > PROX_THRESHOLD and sensors['FrontRR'] > PROX_THRESHOLD:
            print("Corner - turn right")
            move(rob,-20,20)
        elif (sensors['FrontLL'] ) > PROX_THRESHOLD:
            print("Wall on left - turning right")
            move(rob,-20,20)
        elif sensors['FrontRR'] > PROX_THRESHOLD:
            print("Wall on right - turning left")
            move(rob,20,-20)
        elif max(sensors['BackR'],sensors['BackL']) > 60:
            if obstacle_avoidance(sim,rob):
                print("Wall on back, obstacle in front - turning right")
            else:
                print("Wall on back  - Forward")
                move(rob,30,30)
        else:
            wall=False

        if wall:
            print(sensors)

        count += 1
        
def obstacle_avoidance(sim,rob,threshold=330):
    blobs=get_perception(rob)["blobs"]
    distances=getDistance(sim)
    detected=False
    
    for x,obj in enumerate(['red','blue','green']):
        if distances[x]<=threshold and blobs[obj]["seen"] and blobs[obj]["size"]>=70:
            move(rob,-20,20)
            detected=True
            print("obstacle")
    return detected