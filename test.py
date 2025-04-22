from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.Emotions import Emotions
 
rob = Robobo("localhost")
 
rob.connect()
 
rob.sayText("I start moving") 
rob.moveWheels(20,20)
while rob.readIRSensor(IR.FrontC) < 30:
    rob.wait(0.1)
rob.stopMotors()
 
rob.setEmotionTo(Emotions.AFRAID)
 
rob.disconnect()