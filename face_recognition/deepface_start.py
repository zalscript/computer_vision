import cv2
from deepface import DeepFace


# Load Haar Cascade dan model pengenalan wajah
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')


# Load mapping nama
name_mapping = {}
with open('name_mapping.txt', 'r') as f:
    for line in f.readlines():
        name, idx = line.strip().split(":")
        name_mapping[int(idx)] = name


# Inisialisasi webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari webcam.")
        break
    
    # Flip frame agar tidak mirror
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


        # Prediksi ID wajah
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 80:  # Threshold confidence
            name = name_mapping.get(id, "Unknown")
        else:
            name = "Unknown"


        # Deteksi ekspresi wajah menggunakan DeepFace
        face_roi = frame[y:y+h, x:x+w]  # Region of Interest (ROI) untuk wajah
        try:
            # Analisis ekspresi wajah
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            
            # Debugging untuk memeriksa format data
            print(type(analysis), analysis)


            # Jika keluaran berupa daftar
            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]  # Ambil elemen pertama dari daftar
            
            emotion = analysis.get("dominant_emotion", "Unknown") if isinstance(analysis, dict) else "Unknown"
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            emotion = "Unknown"


        # Tampilkan nama dan ekspresi di layar
        text = f"{name} ({emotion})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Tampilkan frame di layar
    cv2.imshow('Face Recognition and Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()