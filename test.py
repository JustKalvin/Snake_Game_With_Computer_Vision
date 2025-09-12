from ultralytics import YOLO
import cv2

# =========================
# 1️⃣ Load YOLO model
# =========================
model = YOLO("best.pt")  # ganti path sesuai modelmu

# =========================
# 2️⃣ Buka webcam
# =========================
cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    print("❌ Tidak bisa membuka webcam")
    exit()

# =========================
# 3️⃣ Loop realtime
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame")
        break

    # =========================
    # 4️⃣ Prediksi YOLO
    # =========================
    results = model.predict(frame, verbose=False)

    # =========================
    # 5️⃣ Gambar bounding box
    # =========================
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                # rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # label + confidence
                label = f"Class {int(cls)}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # =========================
    # 6️⃣ Tampilkan frame
    # =========================
    cv2.imshow("YOLO Webcam Realtime", frame)

    # =========================
    # 7️⃣ Keluar dengan 'q'
    # =========================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# 8️⃣ Release resources
# =========================
cap.release()
cv2.destroyAllWindows()
