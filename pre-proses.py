import cv2
import numpy as np

def apply_3x3_filter(image_path, kernel):
    # Baca gambar dalam grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("Gambar tidak ditemukan.")

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

    # Simpan hasil jika diinginkan
    cv2.imwrite("mask-filter.jpg", output)

# Definisi mask filter 3x3 (contoh: mean filter)
mean_kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], dtype=np.float32) / 9.0

# Contoh penggunaan
apply_3x3_filter('data1.jpg', mean_kernel)
