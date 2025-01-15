from ultralytics import YOLO
import cv2
from cvzone.Utils import cornerRect, putTextRect
import math

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
cap.set(3, 1080)  # Lebar frame
cap.set(4, 720)   # Tinggi frame

# Load model YOLO
model = YOLO("Yolo-Weights/yolov8n.pt")

# Daftar nama kelas
className = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

while True:
    success, img = cap.read()

    # Prediksi menggunakan YOLO
    result = model(img, stream=True)

    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Gambar bounding box
            cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Nama kelas
            cls = int(box.cls[0])
            putTextRect(img, f'{className[cls]} {conf}', (max(0, x1), max(30, y1)))

    # Tampilkan gambar
    cv2.imshow("Image", img)
    cv2.waitKey(3)
