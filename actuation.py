# actuation.py
import time

# Define forward speed
FORWARD_SPEED = 20
TURN_SPEED = 20
MOVE_DURATION = 0.5  # seconds

def move_forward(rob):
    rob.moveWheelsByTime(FORWARD_SPEED, FORWARD_SPEED, MOVE_DURATION)

def turn_left(rob):
    rob.moveWheelsByTime(-TURN_SPEED, TURN_SPEED, MOVE_DURATION)

def turn_right(rob):
    rob.moveWheelsByTime(TURN_SPEED, -TURN_SPEED, MOVE_DURATION)

def stop(rob):
    rob.moveWheelsByTime(0, 0, MOVE_DURATION)

