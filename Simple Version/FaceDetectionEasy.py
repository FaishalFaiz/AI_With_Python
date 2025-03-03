from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)

detector = FaceDetector(minDetectionCon=0.5)

while True:
    success, img = cap.read()

    img, bboxs = detector.findFaces(img, draw=True)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(1)
