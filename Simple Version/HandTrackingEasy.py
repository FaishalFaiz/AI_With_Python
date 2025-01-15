from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

detector = HandDetector(maxHands=2, detectionCon=0.5, minTrackCon=0.5)

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img, draw=True, flipType=True)

    cv2.imshow("Hand Tracing", img)
    cv2.waitKey(1)