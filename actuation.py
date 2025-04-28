# actuation.py
import time

# Define forward speed
MOVE_DURATION = 0.6  # seconds

def move(rob,vel1,vel2):
    rob.moveWheelsByTime(vel1,vel2,MOVE_DURATION)