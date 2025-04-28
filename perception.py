# perception.py
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.BlobColor import BlobColor

def get_perception(rob):
    # Reading all IR proximity sensors
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

    # Color blob perception function
    def read_blob(color):
        blob = rob.readColorBlob(color)
        if blob.size > 0:
            return {
                'seen': True,
                'x': blob.posx,      # Horizontal position: 0 (left) → 100 (right)
                'size': blob.size    # Used as a proxy for distance
            }
        else:
            return {'seen': False, 'x': None, 'size': None}

    # Read color blobs for red, green, and blue
    red_blob = read_blob(BlobColor.RED)
    green_blob = read_blob(BlobColor.GREEN)
    blue_blob = read_blob(BlobColor.BLUE)
    
    # Return all sensor data including proximity and blob color information
    return {
        'proximity': sensors,
        'blobs': {'red': red_blob, 'blue': blue_blob, 'green': green_blob}
    }

def getDistance(sim):
    robot=sim.getRobotLocation(0)['position']
    dist=[]

    for obj in ['REDCYLINDER','BLUECYLINDER','GREENCYLINDER']:
        pos=sim.getObjectLocation(obj)['position']
        dist.append( ( (robot['x']-pos['x'])**2 + (robot['z']-pos['z'])**2 )**0.5 )

    return dist