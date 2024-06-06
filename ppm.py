import serial
import pyvesc
#creating the new messsage
class SetServoPos(metaclass=pyvesc.VESCMessage):
    id = 12
    fields = [
        ('servo', 'i')
    ]

def setrpm():
    # function to set the rpm of the motor: This function works properly
    ser = serial.Serial('/dev/ttyACM0')
    my_msg = pyvesc.SetRPM(5000)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)
    print("Hello")
    ser.close()
def steering():
    # function to control the servo: In  this function is the issue.
    ser = serial.Serial('/dev/ttyACM0')
    my_msg = SetServoPos(180)
    print(my_msg.servo)
    my_packet = pyvesc.encode(my_msg)
    ser.write(my_packet)
    ser.close()
#MainFunction
if __name__ == '__main__':
    steering()
