import os
import time
from pathlib import Path
import cv2
from ultralytics import YOLO


# URL stream CCTV default (ganti jika perlu)
STREAM_URL = "https://atcs-dishub.bandung.go.id:1990/SimpangLima/stream.m3u8"


def load_model(trained_model_path="yolo11_trained.pt"):
    if os.path.exists(trained_model_path):
        print("Memuat model terlatih...")
        return YOLO(trained_model_path)
    else:
        print("Model terlatih tidak ditemukan. Memuat model bawaan...")
        return YOLO("yolo11n.pt")


def run_cctv_stream(model, url, save_output=False, output_dir="output_videos", output_name="cctv_output.avi"):
    print(f"Mencoba membuka stream: {url}")

    while True:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print("Gagal membuka stream. Mencoba lagi dalam 5 detik...")
            cap.release()
            time.sleep(5)
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps == 0:
            fps = 20.0

        writer = None
        if save_output:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_path = str(Path(output_dir) / output_name)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            print(f"Menyimpan hasil ke: {out_path}")

        print("Stream terbuka. Mulai deteksi. Tekan Ctrl+C untuk berhenti.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame tidak tersedia lagi (stream putus). Mencoba reconnect...")
                    break

                # Jalankan prediksi pada frame
                results = model.predict(source=frame, imgsz=640, conf=0.25)
                rendered = None
                for r in results:
                    rendered = r.plot()

                if rendered is None:
                    rendered = frame

                if writer is not None:
                    writer.write(rendered)

                cv2.imshow("CCTV Deteksi)", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Keluar atas perintah keyboard (q).")
                    cap.release()
                    if writer:
                        writer.release()
                    cv2.destroyAllWindows()
                    return

        except KeyboardInterrupt:
            print("Diterima interrupt. Menghentikan...")
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            return

        # loop akan mencoba reconnect otomatis
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Mencoba reconnect dalam 5 detik...")
        time.sleep(5)


if __name__ == '__main__':
    # Bisa mengganti STREAM_URL via environment variable atau argumen jika diperlukan
    url = os.environ.get("CCTV_STREAM_URL", STREAM_URL)
    model = load_model()
    # Jika ingin menyimpan hasil, set SAVE_OUTPUT=1 di environment
    save_flag = os.environ.get("SAVE_OUTPUT", "0") in ("1", "true", "True")
    run_cctv_stream(model, url, save_output=save_flag)
