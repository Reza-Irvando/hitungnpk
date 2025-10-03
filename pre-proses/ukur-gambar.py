import cv2

def get_image_size_opencv(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Gambar tidak ditemukan.")
    
    print(f"Ukuran gambar: {img.shape} piksel")

# Contoh penggunaan
get_image_size_opencv("7.jpg")