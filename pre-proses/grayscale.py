import cv2
import numpy as np
import os

def convert_image_to_grayscale(input_path, output_path, save_as_npy=False):
    """
    Mengonversi satu file gambar menjadi grayscale dan menyimpannya
    sebagai file gambar atau .npy dengan ukuran asli.
    """
    # 1. Baca gambar dari path yang diberikan
    img = cv2.imread(input_path)
    if img is None:
        print(f"Gagal membaca gambar dari: {input_path}")
        return

    # 2. Lakukan proses konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Baris resize dihapus dari sini

    # 3. Simpan hasilnya sesuai pilihan
    if save_as_npy:
        # Tambahkan dimensi channel agar menjadi (height, width, 1) untuk .npy
        gray_3d = np.expand_dims(gray, axis=-1)
        np.save(output_path, gray_3d)
        print(f"Gambar berhasil diproses dan disimpan sebagai .npy di: {output_path}")
    else:
        # Simpan sebagai file gambar (misal: .jpg, .png)
        cv2.imwrite(output_path, gray)
        print(f"Gambar berhasil diproses dan disimpan sebagai gambar di: {output_path}")

# --- BAGIAN UTAMA (YANG DIPERBAIKI) ---

# 1. Tentukan lokasi skrip ini berada secara otomatis
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Definisikan nama file input dan output
input_filename = 'standardized.jpg'
output_filename_img = 'grayscale.jpg'
# output_filename_npy = 'standardized_grayscale.npy' # Untuk output .npy

# 3. Gabungkan lokasi skrip dengan nama file untuk membuat path yang lengkap dan pasti
input_path = os.path.join(script_dir, input_filename)
output_path_img = os.path.join(script_dir, output_filename_img)
# output_path_npy = os.path.join(script_dir, output_filename_npy) # Untuk output .npy

# 4. Panggil fungsi untuk memproses satu file dan menyimpannya sebagai gambar
print(f"Membaca file dari: {input_path}")
convert_image_to_grayscale(input_path, output_path_img, save_as_npy=False)

# Jika Anda ingin menyimpannya sebagai file .npy, panggil seperti ini:
# convert_image_to_grayscale(input_path, output_path_npy, save_as_npy=True)