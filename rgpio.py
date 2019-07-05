from gpiozero import LED
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import sys

Pi1factory = PiGPIOFactory(host='192.168.2.33')

m1 = LED(17, pin_factory=Pi1factory)
m2 = LED(27, pin_factory=Pi1factory)
try:
    while True:
        m1.on()
        m2.on()
        sleep(1)
        m1.off()
        m2.off()
        sleep(1)
except KeyboardInterrupt:
    m1.off()
    m2.off()
    sys.exit