# world_test.py
from perception import get_perception
import time
import random

def get_safe_random_location(sim, buffer=100, min_dist=110):
    objects = sim.getObjects()
    positions = [sim.getObjectLocation(obj)['position'] for obj in objects]

    # Try up to 100 times to find a safe spot
    for _ in range(100):
        x = random.uniform(-1000 + buffer, 1000 - buffer)
        z = random.uniform(-1000 + buffer, 1000 - buffer)
        safe = True

        for pos in positions:
            dx = pos['x'] - x
            dz = pos['z'] - z
            if (dx**2 + dz**2)**0.5 < min_dist:
                safe = False
                break

        if safe:
            return {'x': x, 'y': 10.0, 'z': z}

    raise Exception("Couldn't find a safe location after 100 tries.")

def reached_red(perception, size_threshold=100, position_threshold=45):
    red_blob = perception['blobs']['red']
    sensor = perception['proximity']['FrontC']
    # Check if the red blob is detected and if its size exceeds the threshold
    if red_blob['seen'] and red_blob['size'] and red_blob['size'] >= size_threshold:
        # Also, check if the blob is centered in the camera's field of view
        if abs(red_blob['x'] - 50) < position_threshold and sensor>=60:
            print(f"Red blob reached! (size: {red_blob['size']}, position: {red_blob['x']}, FrontC: {sensor})")
            return True
        
    return False

def detect_collision(perception, threshold=500):
    prox = perception['proximity']
    # Check if any sensor reading is below the threshold (indicating a collision)
    for sensor, value in prox.items():
        if value >= threshold:
            return True  # Return True if any sensor indicates a collision
    
    return False  # Return False if no collision is detected
    
def reset_if_needed(rob,sim):
    perception = get_perception(rob)
    
    if detect_collision(perception):
        print("Collision detected")
        sim.setRobotLocation(0,get_safe_random_location(sim), {'x': 0.0, 'y': 0.0, 'z': 0.0})
        time.sleep(1)
        

    elif reached_red(perception):
        print("Red goal reached!")
        sim.setRobotLocation(0,get_safe_random_location(sim), {'x': 0.0, 'y': 0.0, 'z': 0.0})
        time.sleep(1)
        

def move_yellow_to_corner(sim):
    sim.setObjectLocation("CUSTOMCYLINDER",{'x': -975.0, 'y': 10.0, 'z': 975.0}, {'x': 0.0, 'y': 0.0, 'z': 0.0})