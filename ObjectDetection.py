from ultralytics import YOLO
import cv2
from cvzone.Utils import cornerRect, putTextRect
import math

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
cap.set(3, 1000)  # Lebar frame
cap.set(4, 720)   # Tinggi frame

# Load model YOLO
model = YOLO("Yolo-Weights/yolov8n.pt")

while True:
    success, img = cap.read()

    # Prediksi menggunakan model YOLO
    result = model(img, stream=True)

    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Gambar bounding box dengan corner rectangle
            cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Tampilkan confidence di atas bounding box
            putTextRect(img, f'{conf}', (max(0, x1), max(30, y1)))

    # Tampilkan gambar
    cv2.imshow("Image", img)
    cv2.waitKey(1)
