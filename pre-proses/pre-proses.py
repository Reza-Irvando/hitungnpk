import cv2
import numpy as np
import os

def apply_3x3_filter(image_path, save_path, kernel):
    # Baca gambar dalam grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception(f"Gambar tidak ditemukan di path: {image_path}")

    # Dapatkan dimensi gambar
    height, width = img.shape
    output = np.zeros((height, width), dtype=np.float32)

    # Padding gambar agar bisa difilter di tepi
    padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)

    # Operasi konvolusi manual
    for i in range(height):
        for j in range(width):
            region = padded[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)

    # Normalisasi output ke 0–255 dan ubah ke uint8
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    # Tampilkan hasil
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Simpan hasil ke path yang sudah ditentukan
    cv2.imwrite(save_path, output)
    print(f"Gambar hasil filter disimpan di: {save_path}")

# --- BAGIAN UTAMA ---

# 1. Tentukan lokasi skrip ini berada secara otomatis
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Tentukan nama file input dan output
input_filename = 'data1.jpg'
output_filename = 'mask-filter.jpg'

# 3. Gabungkan lokasi skrip dengan nama file untuk membuat path yang lengkap dan pasti
input_path = os.path.join(script_dir, input_filename)
output_path = os.path.join(script_dir, output_filename)

# 4. Definisi mask filter 3x3 (contoh: mean filter)
mean_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], dtype=np.float32) / 9.0

# 5. Panggil fungsi dengan path yang sudah benar
apply_3x3_filter(input_path, output_path, mean_kernel)