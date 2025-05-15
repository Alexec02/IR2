# world_test.py
from perception import get_perception, getDistance
import time
import random

def get_safe_random_location(sim, buffer=100, min_dist=130):
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

def reached_red(perception, distance, size_threshold=100, distance_threshold=300):
    red_blob = perception['blobs']['red']
    sensor = perception['proximity']['FrontC']
    
    #check if the red blob is detected in the middle of the image
    if red_blob['seen'] and (red_blob["x"]>40 or red_blob["x"]<50):
        # Check if its size exceeds the threshold
        if red_blob['size'] >= size_threshold and (distance <= distance_threshold or sensor>30):
            print(f"Red blob reached! (size: {red_blob['size']}, position: {red_blob['x']}, distance: {distance})")
            return True
            
    return False

def detect_collision(perception, threshold=300):
    prox = perception['proximity']
    # Check if any sensor reading is below the threshold (indicating a collision)
    for sensor, value in prox.items():
        if value >= threshold:
            return True  # Return True if any sensor indicates a collision
    
    return False  # Return False if no collision is detected
    
def reset_if_needed(rob,sim):
    perception = get_perception(rob)
    distance = getDistance(sim)[0]
    if reached_red(perception,distance):
        print("Red goal reached!")
        sim.setRobotLocation(0,get_safe_random_location(sim), {'x': 0.0, 'y': 0.0, 'z': 0.0})
        sim.setObjectLocation("REDCYLINDER",{'x': 600.0, 'y': 10.0, 'z': -600.0},{'x': 0.0, 'y': 0.0, 'z': 0.0})
        #Setobjectlocation not working
        time.sleep(1)
    
    elif detect_collision(perception):
        print("Collision detected")
        sim.setRobotLocation(0,get_safe_random_location(sim), {'x': 0.0, 'y': 0.0, 'z': 0.0})
        
        sim.setObjectLocation("GREENCYLINDER",{'x': -600.0, 'y': 10.0, 'z': -600.0},{'x': 0.0, 'y': 0.0, 'z': 0.0})
        sim.setObjectLocation("BLUECYLINDER",{'x': 600.0, 'y': 10.0, 'z': 600.0},{'x': 0.0, 'y': 0.0, 'z': 0.0})
        sim.setObjectLocation("REDCYLINDER",{'x': 600.0, 'y': 10.0, 'z': -600.0},{'x': 0.0, 'y': 0.0, 'z': 0.0})
        
        time.sleep(1)

def move_yellow_to_corner(sim):
    sim.setObjectLocation("CUSTOMCYLINDER",{'x': -975.0, 'y': 10.0, 'z': 975.0}, {'x': 0.0, 'y': 0.0, 'z': 0.0})