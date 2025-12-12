import cv2
from ultralytics import YOLO
import os

def detect_realtime(model):
    """
    Deteksi objek secara real-time menggunakan webcam.
    :param model: Model YOLO yang sudah dilatih.
    """
    # Menggunakan webcam (ID kamera 0 biasanya adalah kamera utama laptop)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    # Mendapatkan ukuran frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))qq
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat membaca frame.")
            break

        # Memprediksi dan memplot hasil pada frame
        results = model.predict(source=frame, imgsz=640, conf=0.25)
        for result in results:
            rendered_frame = result.plot()  # Render hasil deteksi pada frame

        # Menampilkan hasil deteksi dalam jendela
        cv2.imshow("Real-Time YOLO Detection", rendered_frame)

        # Tekan 'q' untuk keluar dari deteksi real-time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Load model
trained_model_path = "yolo11_trained.pt"
if os.path.exists(trained_model_path):
    print("Memuat model terlatih...")
    model = YOLO(trained_model_path)
else:
    print("Model tidak ditemukan. Melatih model baru...")
    model = YOLO("yolo11n.pt")
    train_results = model.train(
        data=os.path.abspath("coco/coco8.yaml"),
        epochs=100,
        imgsz=640,
        device="cpu",
    )
    model.save(trained_model_path)

# Menjalankan deteksi real-time menggunakan webcam
detect_realtime(model)
