from keras.models import load_model
import numpy as np
from keras.optimizers import Adam
from keras import metrics
import numpy as np
import PIL
from PIL import Image
import io
import socket
import struct
from gpiozero import LED
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import sys

model = load_model('saves/model.h5')

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy])

Pi1factory = PiGPIOFactory(host='192.168.2.33')

m1 = LED(17, pin_factory=Pi1factory)
m2 = LED(27, pin_factory=Pi1factory)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)
connection = server_socket.accept()[0].makefile('rb')
try:
    while True:

        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break

        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))

        image_stream.seek(0)
        image = Image.open(image_stream)
        image = image.resize((224, 224), PIL.Image.ANTIALIAS)
        image = np.reshape(image,[1,224,224,3])
        classes = model.predict(image)
        classes = np.squeeze(classes)
        print(classes)
        c_comp = np.argmax(classes, axis=0)
        print(c_comp)
        if c_comp == 0:
            m1.on()
            m2.on()
        if c_comp == 1:
            m1.off()
            m2.on()
        if c_comp == 2:
            m1.on()
            m2.off()

finally:
    m1.off()
    m2.off()
    connection.close()
    server_socket.close()
    sys.exit()

"""
print ("Forward: %.2f" % classes[0])
print ("Left: %.2f" % classes[1])
print ("Right: %.2f" % classes[2])
"""