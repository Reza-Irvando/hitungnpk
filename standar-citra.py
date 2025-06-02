import cv2
import numpy as np
import matplotlib.pyplot as plt

def standardize_image(image_path):
    # Baca gambar dalam grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("Gambar tidak ditemukan.")

    # Hitung rata-rata dan standar deviasi
    mean = np.mean(img)
    std = np.std(img)

    # Standardisasi z-score
    standardized = (img - mean) / (std + 1e-8)  # tambahkan epsilon untuk hindari pembagian nol

    # Konversi hasil ke rentang 0–255 agar bisa ditampilkan
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

    # Simpan jika perlu
    cv2.imwrite('standardized.jpg', standardized_rescaled)

# Contoh penggunaan
standardize_image('mask-filter.jpg')
