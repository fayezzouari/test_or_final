import cv2
from ClassificationModule import Classifier

mydata = Classifier("keras_model.h5", "labels.txt")
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    predict, index = mydata.getPrediction(img, color=(0, 0, 255))
    if index == 0:
        print("bottle is detected")
    print(predict, index)
    cv2.imshow("Video Frame", img)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
