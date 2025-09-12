import cv2
from ultralytics import YOLO

# Muat model YOLOv8 yang sudah dilatih
# Pastikan file 'best.pt' berada di folder yang sama dengan script ini,
# atau berikan path lengkapnya.
try:
    model = YOLO('best2.pt')
except Exception as e:
    print(f"Error memuat model: {e}")
    print("Pastikan file 'best.pt' ada di direktori yang sama atau path sudah benar.")
    exit()

# Inisialisasi webcam
# Angka 0 biasanya merujuk pada webcam internal. Ganti jika Anda punya banyak kamera.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka webcam.")
    exit()

print("✅ Webcam berhasil dibuka. Tekan 'q' untuk keluar.")

# Loop untuk memproses setiap frame dari webcam
while True:
    # Baca frame dari webcam
    success, frame = cap.read()

    if not success:
        print("Gagal membaca frame dari webcam. Keluar...")
        break

    # Lakukan deteksi pada frame menggunakan model
    # stream=True lebih efisien untuk video
    results = model(frame, stream=True)

    # Loop melalui hasil deteksi
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Ambil koordinat kotak (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # konversi ke integer

            # Ambil skor kepercayaan (confidence)
            conf = round(float(box.conf[0]), 2)
            
            # Ambil ID kelas dan nama kelasnya
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Gambar kotak di sekitar objek yang terdeteksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Buat label dengan nama kelas dan skor kepercayaan
            label = f'{class_name} {conf}'
            
            # Tampilkan label di atas kotak
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Cerminkan frame secara horizontal (agar seperti cermin)
    flipped_frame = cv2.flip(frame, 1)

    # Tampilkan frame yang sudah diproses
    cv2.imshow('YOLOv8 Real-time Detection', flipped_frame)

    # Hentikan loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
print("Program dihentikan.")