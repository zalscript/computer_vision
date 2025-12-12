import cv2
import numpy as np
import os


def get_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]
    face_samples = []
    names = []
    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        try:
            # Ambil nama pengguna (string) dari nama file
            name = os.path.split(image_path)[-1].split("_")[1]
            if gray_img is not None:
                resized_face = cv2.resize(gray_img, (200, 200))
                face_samples.append(resized_face)
                names.append(name)
        except IndexError as e:
            print(f"Skipping file {image_path}: {e}")
    return face_samples, names


# Load dan latih model
recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_path = 'dataset'


faces, names = get_images_and_labels(dataset_path)


# Buat mapping nama ke ID numerik
unique_names = list(set(names))
name_to_id = {name: idx for idx, name in enumerate(unique_names)}
ids = [name_to_id[name] for name in names]


recognizer.train(np.array(faces), np.array(ids))
recognizer.write('trainer.yml')


# Simpan mapping nama ke file untuk digunakan saat prediksi
with open('name_mapping.txt', 'w') as f:
    for name, idx in name_to_id.items():
        f.write(f"{name}:{idx}\n")


print("Model trained successfully!")