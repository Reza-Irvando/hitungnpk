import cv2
import os

def crop_with_shifts(image_path, output_prefix, x_center, y_center, crop_size, shift_pixels=20, num_crops=10):
    """
    Melakukan crop pada gambar sebanyak `num_crops` kali dengan menggeser titik tengah.
    Fungsi ini akan dipanggil berulang kali oleh loop utama.
    """
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Gambar di '{image_path}' tidak ditemukan atau gagal dimuat.")
        return
    
    # Ambil gambar grayscale jika belum
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    half = crop_size // 2
    
    # Daftar 10 pergeseran (dx, dy) dari titik tengah
    shifts = [
        (0, 0),                         # 1. Tengah (Asli)
        (-shift_pixels, 0),             # 2. Kiri
        (shift_pixels, 0),              # 3. Kanan
        (0, -shift_pixels),             # 4. Atas
        (0, shift_pixels),              # 5. Bawah
        (-shift_pixels, -shift_pixels), # 6. Kiri-Atas
        (shift_pixels, -shift_pixels),  # 7. Kanan-Atas
        (-shift_pixels, shift_pixels),  # 8. Kiri-Bawah
        (shift_pixels, shift_pixels),   # 9. Kanan-Bawah
        (0, -shift_pixels * 2)          # 10. Atas (lebih jauh)
    ]
    
    shifts = shifts[:num_crops]

    base_name, ext = os.path.splitext(output_prefix)

    for i, (dx, dy) in enumerate(shifts):
        new_x_center = x_center + dx
        new_y_center = y_center + dy
        
        x_start = max(0, new_x_center - half)
        y_start = max(0, new_y_center - half)
        x_end = x_start + crop_size
        y_end = y_start + crop_size

        if x_end > w:
            x_end = w
            x_start = max(0, w - crop_size)
        if y_end > h:
            y_end = h
            y_start = max(0, h - crop_size)

        cropped_img = img[y_start:y_end, x_start:x_end]
        output_path = f"{base_name}_{i}{ext}"
        
        cv2.imwrite(output_path, cropped_img)

    print(f"Selesai memproses untuk ID {os.path.basename(base_name)}, {num_crops} gambar disimpan.")


# --- BAGIAN UTAMA (YANG DIPERBAIKI) ---

# 1. Definisikan Array Data Anda
crop_data = [
    [7, 79, 283], [10, 253, 1237], [23, 430, 286], [26, 600, 1237], [28, 600, 854],
    [30, 605, 476], [39, 774, 289], [42, 954, 1242], [55, 1126, 286], [58, 1303, 1239]
]

# 2. Definisikan Parameter Umum
INPUT_FILENAME = 'equalized.jpg'
OUTPUT_FOLDER_NAME = 'cropped-eq'
CROP_SIZE = 224
SHIFT_PIXELS = 10
NUM_CROPS_PER_POINT = 10

# 3. Tentukan path absolut berdasarkan lokasi skrip
script_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(script_dir, INPUT_FILENAME)
output_dir_path = os.path.join(script_dir, OUTPUT_FOLDER_NAME)

# 4. Buat folder output jika belum ada
os.makedirs(output_dir_path, exist_ok=True)

# 5. Loop Utama untuk Memproses Setiap Baris Data
print(f"Memulai proses cropping dari gambar: {input_image_path}\n")
print(f"Hasil akan disimpan di folder: '{output_dir_path}'")

for data_point in crop_data:
    file_id, x_coord, y_coord = data_point

    # Buat nama file dasar
    base_filename = f"{file_id}.jpg"
    
    # Gabungkan path folder output dengan nama file untuk membuat path lengkap
    output_path_prefix = os.path.join(output_dir_path, base_filename)
    
    # Panggil fungsi cropping dengan path yang sudah benar
    crop_with_shifts(
        image_path=input_image_path, 
        output_prefix=output_path_prefix, 
        x_center=x_coord, 
        y_center=y_coord, 
        crop_size=CROP_SIZE,
        shift_pixels=SHIFT_PIXELS,
        num_crops=NUM_CROPS_PER_POINT
    )

print("\nSemua proses telah selesai.")