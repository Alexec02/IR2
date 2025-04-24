from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.Emotions import Emotions

from robobosim.RoboboSim import RoboboSim

"""rob = Robobo("localhost")
 
rob.connect()"""
sim = RoboboSim("localhost")
sim.connect()

objects = sim.getObjects()
positions = [sim.getObjectLocation(obj) for obj in objects]
print(positions)
for pos in positions:
    print(pos)
"""rob.sayText("I start moving") 
rob.moveWheels(20,20)
while rob.readIRSensor(IR.FrontC) < 30:
    rob.wait(0.1)
rob.stopMotors()
 
rob.setEmotionTo(Emotions.AFRAID)"""
 
#rob.disconnect()
sim.disconnect()