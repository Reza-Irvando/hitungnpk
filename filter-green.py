import cv2
import numpy as np
import os

def filter_green_from_folder(input_folder, output_folder):
    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    # Proses setiap file gambar dalam folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Gagal memuat {filename}")
                continue

            # Konversi ke HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Rentang HSV warna hijau
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])

            # Buat mask dan hasil filter
            mask = cv2.inRange(hsv, lower_green, upper_green)
            result = cv2.bitwise_and(image, image, mask=mask)

            # Simpan hasil
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result)
            print(f"Disimpan: {output_path}")

# Contoh penggunaan
input_dir = 'citra-crop'  # ganti dengan nama folder gambar
output_dir = 'output_green'
filter_green_from_folder(input_dir, output_dir)
