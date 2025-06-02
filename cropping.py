import cv2
import matplotlib.pyplot as plt

def show_image_size(image_path):
    # Membaca gambar menggunakan OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        print("Gambar tidak ditemukan atau gagal dimuat.")
        return
    
    # Mendapatkan dimensi gambar (tinggi, lebar, saluran warna)
    height, width, channels = img.shape
    
    # Menampilkan grafik skala piksel (ukuran gambar)
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Mengonversi BGR ke RGB untuk ditampilkan dengan benar
    plt.title(f'Ukuran Gambar: {width} x {height} pixels')
    plt.axis('off')  # Menonaktifkan axis
    plt.show()

# Contoh penggunaan
# show_image_size('data1.jpg')


def crop_image_safe(image_path, output_path, x_center, y_center, crop_size):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Gambar tidak ditemukan atau gagal dimuat.")
        return
    
    h, w, _ = img.shape
    half = crop_size // 2
    
    # Hitung batas awal dan akhir, koreksi jika melewati tepi
    x_start = max(0, x_center - half)
    y_start = max(0, y_center - half)
    x_end = x_start + crop_size
    y_end = y_start + crop_size

    # Jika batas akhir melebihi ukuran citra, geser ke atas/kiri
    if x_end > w:
        x_end = w
        x_start = max(0, w - crop_size)
    if y_end > h:
        y_end = h
        y_start = max(0, h - crop_size)

    # Crop dan simpan
    cropped_img = img[y_start:y_end, x_start:x_end]
    cv2.imwrite(output_path, cropped_img)
    print(f"Gambar berhasil disimpan di {output_path}, ukuran: {cropped_img.shape}")

# Contoh penggunaan
crop_image_safe('standardized.jpg', '58.jpg', 1303, 1239, 224)
