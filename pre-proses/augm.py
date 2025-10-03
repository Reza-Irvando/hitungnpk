import cv2
import numpy as np
import os

# ===================================================================
# KUMPULAN FUNGSI AUGMENTASI

# ===================================================================

def flip_horizontal(image):
    """Membalik gambar secara horizontal."""
    return cv2.flip(image, 1)

def flip_vertical(image):
    """Membalik gambar secara vertikal."""
    return cv2.flip(image, 0)

def rotate_image(image, angle):
    """Memutar gambar sebesar sudut tertentu."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def adjust_brightness(image, factor):
    """Meningkatkan atau mengurangi kecerahan gambar."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] *= factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def zoom_image(image, zoom_factor=1.2):
    """Melakukan zoom ke tengah gambar."""
    h, w = image.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = image[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (w, h))

def shear_image(image, shear_factor=0.15):
    """Memiringkan gambar secara horizontal."""
    h, w = image.shape[:2]
    matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (w, h))

def add_gaussian_noise(image, sigma=25):
    """Menambahkan noise gaussian pada gambar."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def blur_image(image, kernel_size=(7, 7)):
    """Memberikan efek blur pada gambar."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def adjust_contrast(image, alpha=1.5):
    """Meningkatkan kontras gambar."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def invert_colors(image):
    """Membalik warna gambar (negatif)."""
    return cv2.bitwise_not(image)

# ===================================================================
# BAGIAN UTAMA PROGRAM
# ===================================================================

def main():
    # --- Konfigurasi ---
    INPUT_FOLDER = "cropped-eq"
    OUTPUT_FOLDER = "hasil_augmentasi"

    # Membuat path yang absolut dan andal
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir_path = os.path.join(script_dir, INPUT_FOLDER)
    output_dir_path = os.path.join(script_dir, OUTPUT_FOLDER)

    os.makedirs(output_dir_path, exist_ok=True)

    if not os.path.isdir(input_dir_path):
        print(f"Error: Folder input '{input_dir_path}' tidak ditemukan.")
        print(f"Silakan buat folder '{INPUT_FOLDER}' dan letakkan gambar di dalamnya.")
        return

    print(f"Memulai augmentasi dari folder: '{INPUT_FOLDER}'")
    print(f"Hasil akan disimpan di folder: '{OUTPUT_FOLDER}'")

    generated_files = [] # <-- BARU: 1. Inisialisasi list kosong untuk menampung nama file

    # Melakukan perulangan untuk setiap file di folder input
    for filename in os.listdir(input_dir_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_dir_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Gagal membaca: {filename}")
                continue

            base_name, _ = os.path.splitext(filename)
            print(f"   -> Memproses {filename}...")

            augmentations = {
                "flip_h": flip_horizontal(image), "flip_v": flip_vertical(image),
                "rotate_45": rotate_image(image, 45), "brighten": adjust_brightness(image, 1.5),
                "darken": adjust_brightness(image, 0.6), "zoom": zoom_image(image, 1.2),
                "shear": shear_image(image, 0.15), "noise": add_gaussian_noise(image, 30),
                "blur": blur_image(image, (7, 7)), "contrast": adjust_contrast(image, 1.8)
            }

            for aug_name, aug_image in augmentations.items():
                output_filename = f"{base_name}_{aug_name}.jpg"
                save_path = os.path.join(output_dir_path, output_filename)
                cv2.imwrite(save_path, aug_image)
                generated_files.append(output_filename) # <-- BARU: 2. Kumpulkan nama file

    print("\nProses augmentasi selesai.")

    # <-- BARU: 3. Cetak semua nama file yang terkumpul di akhir
    print("\n--- Daftar File yang Dihasilkan ---")
    for f_name in generated_files:
        print(f_name)
    print(f"\nTotal file dihasilkan: {len(generated_files)}")
    
if __name__ == "__main__":
    main()