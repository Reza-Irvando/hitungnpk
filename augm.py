import cv2
import numpy as np
import os

# Fungsi: rotasi gambar
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated

# Fungsi: ubah brightness
def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Fungsi: zoom (crop center lalu resize)
def zoom_image(image, zoom_factor=1.2):
    h, w = image.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = image[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h))

# Path gambar input
input_path = "equalized_images/58.jpg"
image = cv2.imread(input_path)
output_folder = "img58"
os.makedirs(output_folder, exist_ok=True)

# Simpan versi asli
cv2.imwrite(f"{output_folder}/original.jpg", image)

# Flip horizontal
cv2.imwrite(f"{output_folder}/flip_h.jpg", cv2.flip(image, 1))

# Flip vertical
cv2.imwrite(f"{output_folder}/flip_v.jpg", cv2.flip(image, 0))

# Rotasi 30 derajat
cv2.imwrite(f"{output_folder}/rotate_30.jpg", rotate_image(image, 30))

# Brightness +40%
cv2.imwrite(f"{output_folder}/bright_up.jpg", adjust_brightness(image, 1.4))

# Brightness -30%
cv2.imwrite(f"{output_folder}/bright_down.jpg", adjust_brightness(image, 0.7))

# Zoom in (1.2x)
cv2.imwrite(f"{output_folder}/zoom.jpg", zoom_image(image, zoom_factor=1.2))

print("✅ Semua augmentasi selesai. Lihat folder 'augmented_manual'")