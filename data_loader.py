"""
Modul untuk memuat gambar dan label NPK dari folder dan file CSV.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import config # Impor dari file config.py

def load_and_split_data():
    """
    Memuat gambar dan label NPK, lalu membaginya menjadi set latih, validasi, dan uji.

    Returns:
        Tuple berisi data yang telah dibagi:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print("Memuat data dari folder dan CSV...")
    try:
        df_npk = pd.read_csv(config.CSV_FILE_PATH)
        df_npk.columns = df_npk.columns.str.strip().str.upper()
    except FileNotFoundError:
        print(f"Error: File CSV tidak ditemukan di {config.CSV_FILE_PATH}")
        exit()
    except Exception as e:
        print(f"Error saat memuat CSV: {e}")
        exit()

    all_images = []
    all_npk_values = []

    for index, row in df_npk.iterrows():
        try:
            filename = str(row['FILENAME'])
            image_path = os.path.join(config.IMAGES_SUBFOLDER, filename)

            if not os.path.exists(image_path):
                print(f"Gambar tidak ditemukan: {image_path}. Melewatkan.")
                continue

            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            # Normalisasi awal ke rentang [0, 1]
            img_array /= 255.0
            all_images.append(img_array)

            npk_value = [
                pd.to_numeric(row['N'], errors='coerce'),
                pd.to_numeric(row['P'], errors='coerce'),
                pd.to_numeric(row['K'], errors='coerce')
            ]

            if any(pd.isna(val) for val in npk_value):
                print(f"Nilai NPK tidak valid untuk {filename}. Melewatkan.")
                all_images.pop() # Hapus gambar yang baru saja ditambahkan
                continue

            all_npk_values.append(npk_value)

        except Exception as e:
            print(f"Gagal memproses baris {index} untuk file {row.get('FILENAME', 'N/A')}: {e}")
            if all_images: # Jika error terjadi setelah gambar ditambahkan
                all_images.pop()

    if not all_images:
        print("\nTidak ada gambar yang berhasil dimuat. Proses dihentikan.")
        exit()

    X = np.array(all_images)
    y = np.array(all_npk_values)

    print(f"\nTotal gambar yang berhasil diproses: {len(X)}")
    print(f"Bentuk data gambar: {X.shape}")
    print(f"Bentuk data NPK: {y.shape}")

    # --- Split data berdasarkan urutan ---
    total_samples = len(X)
    train_size = int(config.TRAIN_SPLIT * total_samples)
    val_size = int(config.VALIDATION_SPLIT * total_samples)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    print(f"Jumlah sampel pelatihan: {len(X_train)}")
    print(f"Jumlah sampel validasi: {len(X_val)}")
    print(f"Jumlah sampel pengujian: {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)