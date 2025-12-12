import cv2
from ultralytics import YOLO
import os
from pathlib import Path

def detect_video(model, video_path, output_dir="output_videos", output_name="output_video.mp4"):
    """
    Deteksi objek pada video menggunakan YOLO.
    :param model: Model YOLO yang sudah dilatih.
    :param video_path: Path ke video yang akan dideteksi.
    :param output_dir: Direktori untuk menyimpan hasil video.
    :param output_name: Nama file video output.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka video.")
        return

    # Mendapatkan ukuran frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Menggunakan codec 'XVID' untuk video output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Memproses video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Memprediksi dan memplot hasil pada frame
        results = model.predict(source=frame, imgsz=640, conf=0.25)
        for result in results:
            rendered_frame = result.plot()  # Render hasil deteksi pada frame
            out.write(rendered_frame)  # Menulis frame dengan anotasi ke video output

        # Menampilkan video secara langsung (opsional, tekan 'q' untuk keluar)
        cv2.imshow("YOLO Detection", rendered_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Hasil deteksi video disimpan di {output_video_path}.")

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

# Path video (use pathlib to avoid invalid escape sequences on Windows)
script_dir = Path(__file__).parent
video_file = "WhatsApp Video 2025-12-05 at 17.24.34_42c7aeb7.mp4"
video_path = script_dir / "video" / video_file
video_path = str(video_path.resolve())

# Verify video file exists before attempting to open
if not os.path.exists(video_path):
    print(f"Error: Video file not found: {video_path}")
    print("Please check the filename in the `video/` folder or update `video_file` accordingly.")
else:
    # Deteksi objek pada video
    detect_video(model, video_path)
