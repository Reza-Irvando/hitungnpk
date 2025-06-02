import cv2
import numpy as np
import os

def convert_folder_to_grayscale_224x224(input_folder, output_folder, save_as_npy=False):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path_img = os.path.join(output_folder, filename)
            output_path_npy = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '.npy')

            # Baca dan proses gambar
            img = cv2.imread(input_path)
            if img is None:
                print(f"Gagal membaca {filename}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (224, 224))
            gray_3d = np.expand_dims(resized, axis=-1)  # (224, 224, 1)

            if save_as_npy:
                np.save(output_path_npy, gray_3d)
                print(f"Disimpan sebagai .npy: {output_path_npy}")
            else:
                # Simpan sebagai gambar (hilangkan dimensi channel untuk menyimpan .jpg)
                cv2.imwrite(output_path_img, resized)
                print(f"Disimpan sebagai gambar: {output_path_img}")

# Contoh penggunaan
input_folder = 'crop-standar'  # folder input
output_folder = 'citra-grayscale'  # folder output
convert_folder_to_grayscale_224x224(input_folder, output_folder, save_as_npy=False)
