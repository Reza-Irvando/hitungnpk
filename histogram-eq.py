import cv2
import os

def histogram_equalization_folder(input_folder, output_folder):
    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    # Iterasi semua file gambar
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Baca gambar dalam grayscale
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Gagal membaca gambar: {filename}")
                continue

            # Histogram Equalization
            equalized = cv2.equalizeHist(img)

            # Simpan hasil
            cv2.imwrite(output_path, equalized)
            print(f"Disimpan: {output_path}")

# Contoh penggunaan
input_dir = 'citra-grayscale'  # ganti dengan folder sumbermu
output_dir = 'equalized_images'
histogram_equalization_folder(input_dir, output_dir)
