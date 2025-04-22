# world_test.py
from perception import get_perception
from actuation import stop
import time

def reached_red(perception, size_threshold=60, position_threshold=45):
    red_blob = perception['blobs']['red']
    
    # Check if the red blob is detected and if its size exceeds the threshold
    if red_blob['seen'] and red_blob['size'] and red_blob['size'] > size_threshold:
        # Also, check if the blob is centered in the camera's field of view
        if abs(red_blob['x'] - 50) < position_threshold:
            print(f"Red blob reached! (size: {red_blob['size']}, position: {red_blob['x']})")
            return True
        
    return False

def detect_collision(perception, threshold=0.8):
    prox = perception['proximity']
    # Check if any sensor reading is below the threshold (indicating a collision)
    for sensor, value in prox.items():
        if value == threshold:
            return True  # Return True if any sensor indicates a collision
    
    return False  # Return False if no collision is detected
    
def reset_if_needed(rob):
    perception = get_perception(rob)
    
    if detect_collision(perception):
        print("Collision detected. Stopping...")
        stop(rob)
        rob.sayText("Collision! Resetting.")
        time.sleep(2)
        #reset

    elif reached_red(perception):
        print("Red goal reached!")
        stop(rob)
        rob.sayText("Goal reached!")
        time.sleep(2)
        #reset

def move_yellow_to_corner(rob):
    return True
    # move yellow to a corner
    