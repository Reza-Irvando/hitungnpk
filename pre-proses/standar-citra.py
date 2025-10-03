import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def standardize_image(image_path, save_path):
    # Baca gambar dalam grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Beri pesan error yang lebih spesifik
        raise Exception(f"Gambar tidak ditemukan di path: {image_path}")

    # Hitung rata-rata dan standar deviasi
    mean = np.mean(img)
    std = np.std(img)

    # Standardisasi z-score
    standardized = (img - mean) / (std + 1e-8)

    # Konversi hasil ke rentang 0–255 agar bisa disimpan dan ditampilkan
    standardized_rescaled = cv2.normalize(standardized, None, 0, 255, cv2.NORM_MINMAX)
    standardized_rescaled = standardized_rescaled.astype(np.uint8)

    # Tampilkan hasil
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Standardized (z-score)")
    plt.imshow(standardized, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Rescaled to 0-255")
    plt.imshow(standardized_rescaled, cmap='gray')
    
    plt.tight_layout()
    plt.show()

    # Simpan hasil ke path yang sudah ditentukan
    cv2.imwrite(save_path, standardized_rescaled)
    print(f"Gambar hasil disimpan di: {save_path}")

# --- BAGIAN UTAMA ---

# 1. Tentukan lokasi skrip ini berada secara otomatis
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Tentukan nama file input dan output
input_filename = 'mask-filter.jpg'
output_filename = 'standardized.jpg'

# 3. Gabungkan lokasi skrip dengan nama file untuk membuat path yang lengkap dan pasti
input_path = os.path.join(script_dir, input_filename)
output_path = os.path.join(script_dir, output_filename)

# 4. Panggil fungsi dengan path yang sudah benar
standardize_image(input_path, output_path)