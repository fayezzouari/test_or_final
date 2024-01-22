import cv2
from ClassificationModule import Classifier

# i want to send a request to a raspberry through serial communication

import serial
import time
import requests

ser = serial.Serial("/dev/ttyACM0", 9600)


mydata = Classifier("keras_model.h5", "labels.txt")
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    predict, index = mydata.getPrediction(img, color=(0, 0, 255))
    print(predict, index)
    cv2.imshow("Video Frame", img)
    if index == 1:
        ser.write(b"1")
        print("1")

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
