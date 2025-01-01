import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2

camera = cv2.VideoCapture(0)

detector = FaceMeshDetector(staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5)

while True:

    success, img = camera.read()

    img, faces = detector.findFaceMesh(img, draw=True)

    cv2.imshow("Face Mesh", img)
    cv2.waitKey(1)