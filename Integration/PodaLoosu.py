from pyfirmata import Arduino,SERVO, util
from time import sleep


port = '/dev/cu.usbmodem144101'

board = Arduino(port)
pin = 10

board.digital[pin].mode = SERVO

def rotateservo(pin,angle):
    
    board.digital[pin].write(angle)
    sleep(0.015)

while True:

    x = int(input("input : "))

    if x == 1:
        for i in range(0,45):
            rotateservo(pin,i)
    elif x == 2:
        for i in range(0,90):
            rotateservo(pin,i)
    elif x == 3:
        for i in range(0,135):
            rotateservo(pin,i)

    x = 0




