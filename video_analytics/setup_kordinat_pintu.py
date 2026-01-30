import cv2
import numpy as np

# Ganti dengan path video anda atau 0 untuk webcam
VIDEO_SOURCE = "C:\\Pemrograman\\kuliah\\smt5\\computer_vision\\video_analytics\\video_input\\video1.mp4"
# VIDEO_SOURCE = "video_sample.mp4" 

# Global scale variable
scale_factor = 1.0
points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 2:
            print("Sudah 2 titik (garis), tekan q untuk keluar atau restart script untuk ulang.")
            return

        # Konversi koordinat klik kembali ke koordinat asli
        x_orig = int(x / scale_factor)
        y_orig = int(y / scale_factor)
        
        points.append([x_orig, y_orig])
        print(f"Titik {len(points)}: [{x_orig}, {y_orig}]")
        
        # Visualisasi
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        if len(points) == 2:
            # Gambar garis antar 2 titik
            pt1 = (int(points[0][0] * scale_factor), int(points[0][1] * scale_factor))
            pt2 = (int(points[1][0] * scale_factor), int(points[1][1] * scale_factor))
            cv2.line(img_display, pt1, pt2, (0, 255, 0), 2)
            print("Garis terbentuk! Catat koordinat di bawah ini:")
            print(f"GARIS_PINTU = {points}")

        cv2.imshow('Setup Garis Pintu', img_display)

cap = cv2.VideoCapture(VIDEO_SOURCE)

# Baca 1 frame saja untuk setup
ret, img = cap.read()

if not ret:
    print("Gagal membaca video / kamera tidak terdeteksi.")
    cap.release()
    exit()

# --- RESIZE LOGIC ---
h, w, _ = img.shape
TARGET_WIDTH = 640 
scale_factor = TARGET_WIDTH / w
TARGET_HEIGHT = int(h * scale_factor)

img_display = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))

print("=== SETUP GARIS PINTU ===")
print(f"Resolusi Asli: {w}x{h}")
print(f"Resolusi Display: {TARGET_WIDTH}x{TARGET_HEIGHT}")
print("Klik 2 titik untuk membuat GARIS BATAS Pintu.")
print("Titik 1: Awal Garis, Titik 2: Akhir Garis.")

cv2.imshow('Setup Garis Pintu', img_display)
cv2.setMouseCallback('Setup Garis Pintu', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()