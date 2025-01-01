import cv2
import cvzone

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# This is the Webcam Template (often used in cv2/cvzone project)