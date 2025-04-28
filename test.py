from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.Emotions import Emotions
from robobopy.utils.Wheels import Wheels

from robobosim.RoboboSim import RoboboSim
import time

rob = Robobo("localhost")
 
rob.connect()
sim = RoboboSim("localhost")
sim.connect()

sim.setRobotLocation(0,{'x': -750.0, 'y': 10.0, 'z': 750.0}, {'x': 0.0, 'y': 0.0, 'z': 0.0})
"""pos_ori = sim.getRobotLocation(0)["rotation"]["y"]
pos=pos_ori
rob.moveWheels(10,-10)
while abs(pos_ori-pos) < 43 or abs(pos_ori-pos) > 47:
    pos=sim.getRobotLocation(0)["rotation"]["y"]
    

rob.stopMotors()"""
"""objects = sim.getObjects()
positions = [sim.getObjectLocation(obj) for obj in objects]
print(positions)
for pos in positions:
    print(pos)"""
"""rob.sayText("I start moving") 
rob.moveWheels(20,20)
while rob.readIRSensor(IR.FrontC) < 30:
    rob.wait(0.1)
rob.stopMotors()
 
rob.setEmotionTo(Emotions.AFRAID)"""
 
#rob.disconnect()
sim.disconnect()


"""
IDEA FOR DEVELOPING ACTIONS: DIVIDE Y VENCERAS 
TAKE 2 rand values
Evaluate 
FOR X iterations 
Take best value and add simmetric noise
evaluate

Take best utility.
"""