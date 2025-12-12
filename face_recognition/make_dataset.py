import cv2
import os


# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Video capture
cap = cv2.VideoCapture(0)


# Folder untuk menyimpan dataset
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    
# Input nama pengguna
user_name = input("Masukkan nama pengguna: ")
print("Ambil gambar dengan menekan 's', dan tekan 'q' untuk keluar")


count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        # Simpan gambar dengan nama berbasis string
        cv2.imwrite(f"dataset/user_{user_name}_{count}.jpg", gray[y:y+h, x:x+w])
    cv2.imshow("Face", frame)


    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 80:  # Ambil 80 gambar
        break


cap.release()
cv2.destroyAllWindows()