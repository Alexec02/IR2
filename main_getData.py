# main.py
import csv
import os
import time
import random
from perception import getDistance, getRotation
from actuation import move
from behavior import wall_avoidance, obstacle_avoidance
from world_test import move_yellow_to_corner, reset_if_needed
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

DATASET_FILE = "world_model_dataset_2.csv"
#ACTIONS = [-90,-45,0,45,90] # i tried discretized using motors (really difficult), does the statement mean using sim.setRobotLocation()????



def log_to_dataset(p_before, r_before, action, p_after, r_after):

    with open(DATASET_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        row = [p_before] + [r_before] + [action] + [p_after] + [r_after]
        writer.writerow(row)

def main_loop(sim,rob):
    while True:
        reset_if_needed(rob,sim)
        wall_avoidance(sim,rob)
        obstacle_avoidance(sim,rob)
        # --- Perception before action ---
        p_before = getDistance(sim)
        r_before = getRotation(sim)

        # --- Random action ---
        #action = random.choice(ACTIONS)
        #For creating continuos movement we select different velocity for each wheel
        
        #sign=[-1,1]
        #sign1=sign[random.randint(0,1)]
        #sign2=sign[random.randint(0,1)]

        vel1 = random.random()*20#*sign1
        vel2 = random.random()*20#*sign2
        action=[vel1,vel2]
        move(rob, vel1, vel2)
        
        # --- Perception after action ---
        p_after = getDistance(sim)
        r_after = getRotation(sim)
        
        # --- Log step ---
        log_to_dataset(p_before, r_before, action, p_after, r_after)
        #print("Logged:", action, p_before, "->", p_after)

if __name__ == "__main__":
    rob = Robobo("localhost")
    rob.connect()
    sim = RoboboSim("localhost")
    sim.connect()
    time.sleep(1)
    rob.moveTiltTo(90,5)
    move_yellow_to_corner(sim)
    main_loop(sim,rob)
    
    

