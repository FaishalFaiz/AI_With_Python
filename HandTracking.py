from cvzone.HandTrackingModule import HandDetector
from cvzone.Utils import findDistance
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1080)  # Lebar frame
cap.set(4, 720)   # Tinggi frame

# Inisialisasi detektor tangan
detector = HandDetector(maxHands=2, detectionCon=0.5, minTrackCon=0.5)

while True:
    success, img = cap.read()

    # Deteksi tangan
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        # Tangan pertama
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        center1 = hand1["center"]
        handType1 = hand1["type"]

        # Status jari tangan pertama
        fingers1 = detector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")

        # Hitung jarak antara ujung jari pertama dan kedua tangan pertama
        tipOfFirstFinger1 = lmList1[4][0:2]
        tipOfSecondFinger1 = lmList1[8][0:2]

        length, info, img = findDistance(tipOfFirstFinger1, tipOfSecondFinger1, img)

    if len(hands) == 2:
        # Tangan kedua
        hand2 = hands[1]
        lmList2 = hand2["lmList"]
        bbox2 = hand2["bbox"]
        center2 = hand2["center"]
        handType2 = hand2["type"]

        # Status jari tangan kedua
        fingers2 = detector.fingersUp(hand2)
        print(f'H2 = {fingers2.count(1)}', end=" ")

        # Hitung jarak antara ujung jari pertama dan kedua tangan kedua
        tipOfFirstFinger2 = lmList2[4][0:2]
        tipOfSecondFinger2 = lmList2[8][0:2]

        length, info, img = findDistance(tipOfFirstFinger2, tipOfSecondFinger2, img)
        print(length)

    # Tampilkan hasil pada jendela
    cv2.imshow("Hand Tracing", img)
    cv2.waitKey(1)
