# You can also import the whole library of cvzone with "import cvzone"
from cvzone.Utils import putTextRect
from cvzone.Utils import cornerRect
from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)

detector = FaceDetector(minDetectionCon=0.5)

while True:
    success, img = cap.read()

    img, bboxs = detector.findFaces(img, draw=True)

    if bboxs:
        for bbox in bboxs:

            # Get Data
            center = bbox["center"]
            x,y,w,h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            # Draw Data
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            putTextRect(img,  f'{score}%', (x, y - 10))
            cornerRect(img, (x,y,w,h))

    cv2.imshow("Face Detection", img)
    cv2.waitKey(1)
