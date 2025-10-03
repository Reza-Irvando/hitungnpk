import cv2
import os

def histogram_equalization_image(input_path, output_path):
    """
    Membaca satu gambar grayscale, menerapkan histogram equalization,
    dan menyimpannya ke path output.
    """
    # Baca gambar langsung dari path-nya dalam mode grayscale
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Gagal membaca gambar dari: {input_path}")
        return

    # Terapkan Histogram Equalization
    equalized_img = cv2.equalizeHist(img)

    # Simpan gambar hasil
    cv2.imwrite(output_path, equalized_img)
    print(f"Gambar hasil equalisasi disimpan di: {output_path}")

# --- BAGIAN UTAMA (YANG DIPERBAIKI) ---

# 1. Tentukan lokasi skrip ini berada secara otomatis
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Definisikan nama file input dan output
input_filename = 'grayscale.jpg'
output_filename = 'equalized.jpg'

# 3. Gabungkan lokasi skrip dengan nama file untuk membuat path yang lengkap dan pasti
input_path = os.path.join(script_dir, input_filename)
output_path = os.path.join(script_dir, output_filename)

# 4. Panggil fungsi untuk memproses satu gambar dengan path yang sudah benar
print(f"Membaca file dari: {input_path}")
histogram_equalization_image(input_path, output_path)