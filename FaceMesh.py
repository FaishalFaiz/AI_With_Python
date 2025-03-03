from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.Utils import findDistance
import cv2

cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

while True:
    success, img = cap.read()

    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        for face in faces:
            leftEyeUpPoint = face[159]
            leftEyeDownPoint = face[23]

            leftEyeVerticalPoint, info = findDistance(leftEyeUpPoint, leftEyeDownPoint)

            print(leftEyeVerticalPoint)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
