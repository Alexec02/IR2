# robot_actions.py
from robobopy import Robobo
import time

class RobotActions:
    def __init__(self, robobo):
        self.robobo = robobo
    
    def execute_action(self, angle):
        # Discrete actions [-90, -45, 0, 45, 90]
        print(f"Executing action with rotation angle: {angle}")
        if angle != 0:
            self.robobo.turn(angle, speed=50, block=True)
        
        # Move forward at fixed speed
        self.robobo.move_forward_time(speed=30, time=1.5, block=True)
        print("Action completed.")
