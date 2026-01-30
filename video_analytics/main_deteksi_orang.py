import cv2
from ultralytics import YOLO
import cvzone
import math
import numpy as np

# --- KONFIGURASI ---
VIDEO_SOURCE = "video1.mp4" # Ganti dengan path video, misal: "video.mp4"
MODEL_PATH = "yolov8n-pose.pt" # Model pose untuk deteksi angkat tangan (download dari ultralytics jika belum ada)

# Koordinat GARIS Pintu (Mulai -> Akhir)
# Ganti angka-angka ini dengan hasil dari script setup_kordinat_pintu.py
# Contoh format: [ [x1, y1], [x2, y2] ]
GARIS_PINTU = [
 [624, 1060],
 [1288, 1376]
]

# Kelas yang ingin dideteksi (0 adalah person di COCO dataset)
TARGET_CLASS_ID = 0 

# --- PROSES SCALING KOORDINAT & RESIZE ---
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Baca 1 frame untuk dapat ukuran asli
ret, frame = cap.read()
if ret:
    h, w, _ = frame.shape
    
    # Target lebar agar tidak terlalu besar & lebih cepat (misal 960 pixel)
    TARGET_WIDTH = 640
    scale = TARGET_WIDTH / w
    TARGET_HEIGHT = int(h * scale)
    
    # Update Garis dengan scale baru
    # GARIS_PINTU format: [[x1, y1], [x2, y2]]
    garis_original = np.array(GARIS_PINTU, np.int32)
    garis_scaled = garis_original * scale
    garis_scaled = garis_scaled.astype(np.int32)
    
    LIMIT_GARIS = garis_scaled # Titik [A, B]
else:
    print("Video tidak dapat dibaca")
    exit()

# Dictionary untuk menyimpan posisi sebelumnya: {id: [x, y]}
track_history = {}
# Set ID yang sudah dihitung masuk
counted_ids = set()
# Set ID yang sudah dihitung raise hand
raise_hand_ids = set()

def is_raise_hand(keypoints):
    if keypoints is None or len(keypoints) < 17:
        return False
    # Ambil keypoints yang diperlukan
    nose = keypoints[0]  # [x, y, conf]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    # Cek confidence
    if nose[2] < 0.5 or left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5:
        return False
    # Jika salah satu tangan di atas bahu
    if left_wrist[2] > 0.5 and left_wrist[1] < left_shoulder[1]:
        return True
    if right_wrist[2] > 0.5 and right_wrist[1] < right_shoulder[1]:
        return True
    return False

model = YOLO(MODEL_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame agar lebih cepat dan fit di layar
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
    # TRACKING dengan YOLO pose
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, task='pose')
    
    # Gambar Garis Pintu
    cv2.line(frame, (LIMIT_GARIS[0][0], LIMIT_GARIS[0][1]), (LIMIT_GARIS[1][0], LIMIT_GARIS[1][1]), (0, 255, 255), 2)
    cv2.putText(frame, "Garis Pintu", (LIMIT_GARIS[0][0], LIMIT_GARIS[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints
        for i, box in enumerate(boxes):
            if box.id is not None:
                oid = int(box.id.item())
                
                # Ambil bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cls = int(box.cls[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                
                if cls == TARGET_CLASS_ID:
                    w_box, h_box = x2 - x1, y2 - y1
                    
                    # Titik acuan (kaki tengah)
                    cx, cy = x1 + w_box // 2, y1 + h_box
                    
                    # Cek apakah ID ini ada history posisi sebelumnya
                    if oid in track_history:
                        prev_cx, prev_cy = track_history[oid]
                        
                        # Logic intersection garis (Cross Product Logic)
                        # Garis Pintu: A -> B
                        # Pergerakan Orang: C (prev) -> D (curr)
                        # Kita gunakan cvzone atau logika CCW sederhana
                        
                        line_start = (LIMIT_GARIS[0][0], LIMIT_GARIS[0][1])
                        line_end = (LIMIT_GARIS[1][0], LIMIT_GARIS[1][1])
                        
                        # Cek perpotongan
                        # Kita pakai cara sederhana: apakah garis C-D memotong A-B?
                        
                        # Rumus CCW
                        def ccw(A, B, C):
                            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

                        # Cek intersection
                        A, B = line_start, line_end
                        C, D = (prev_cx, prev_cy), (cx, cy)
                        
                        intersect = ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
                        
                        if intersect and oid not in counted_ids:
                             counted_ids.add(oid)
                             # Flash garis hijau saat ada yang lewat
                             cv2.line(frame, line_start, line_end, (0, 255, 0), 5)
                    
                    # Update posisi terakhir
                    track_history[oid] = (cx, cy)
                    
                    # Cek raise hand
                    if oid not in raise_hand_ids and is_raise_hand(keypoints[i]):
                        raise_hand_ids.add(oid)
                    
                    # Gambar Visual
                    color = (0, 0, 255)
                    if oid in counted_ids:
                        color = (0, 255, 0)
                        
                    cvzone.cornerRect(frame, (x1, y1, w_box, h_box), l=9, rt=2, colorR=color)
                    cvzone.putTextRect(frame, f'ID: {oid}', (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
                    cv2.circle(frame, (cx, cy), 5, color, cv2.FILLED)
                    
                    # Gambar trail history (opsional, untuk debug arah)
                    # if oid in track_history:
                    #      cv2.line(frame, track_history[oid], (cx, cy), (255,255,0), 2)

    # Tampilkan Jumlah Count
    count_text = f"Jumlah Orang Masuk: {len(counted_ids)}"
    cvzone.putTextRect(frame, count_text, (50, 50), scale=1.5, thickness=2, offset=10, colorR=(0,200,0))
    
    raise_text = f"Jumlah Raise Hand: {len(raise_hand_ids)}"
    cvzone.putTextRect(frame, raise_text, (50, 100), scale=1.5, thickness=2, offset=10, colorR=(255,0,0))
    
    cv2.imshow("People Counter", frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()