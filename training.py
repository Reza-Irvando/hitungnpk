import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import pandas as pd # Import pustaka pandas

# --- Fungsi untuk menghitung MAPE ---
def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """
    Menghitung Mean Absolute Percentage Error (MAPE).
    Menambahkan epsilon ke penyebut untuk menghindari pembagian dengan nol.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Menghindari pembagian dengan nol:
    # Hanya hitung MAPE untuk nilai y_true yang bukan nol.
    # Atau tambahkan epsilon kecil ke penyebut.
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# --- 1. Persiapan Data (Memuat dari Folder dan CSV) ---
# Tentukan path ke folder data Anda dan file CSV
DATA_FOLDER = 'data' # Ganti nama folder dari 'data_npk' menjadi 'data'
IMAGES_SUBFOLDER = DATA_FOLDER # Gambar sekarang langsung di dalam DATA_FOLDER
CSV_FILE_PATH = os.path.join(DATA_FOLDER, 'label.csv') # Ganti nama file CSV dari 'npk_values.csv' menjadi 'label.csv'

# Ukuran gambar yang akan di-resize
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
NUM_CHANNELS = 3 # RGB

print("Memuat data dari folder dan CSV...")

# Memuat data NPK dari file CSV
try:
    df_npk = pd.read_csv(CSV_FILE_PATH)
    # --- Perbaikan: Bersihkan nama kolom dari spasi dan ubah ke uppercase ---
    df_npk.columns = df_npk.columns.str.strip().str.upper()
    print(f"Data NPK berhasil dimuat dari: {CSV_FILE_PATH}")
    print("Header CSV yang dibaca (setelah pembersihan):", df_npk.columns.tolist()) # Menampilkan header yang ditemukan
    print("Info DataFrame:")
    df_npk.info() # Menampilkan informasi tipe data kolom
    print(df_npk.head())
except FileNotFoundError:
    print(f"Error: File CSV tidak ditemukan di {CSV_FILE_PATH}")
    print(f"Pastikan Anda memiliki file 'label.csv' di dalam folder '{DATA_FOLDER}'.")
    exit()
except Exception as e:
    print(f"Error saat memuat CSV: {e} (Tipe: {type(e)})")
    exit()

# List untuk menyimpan gambar dan nilai NPK yang sesuai
all_images = []
all_npk_values = []
processed_count = 0
skipped_count = 0

# Iterasi melalui baris DataFrame untuk memuat gambar dan nilai NPK
for index, row in df_npk.iterrows():
    # Pastikan nama kolom 'FILENAME' ada setelah pembersihan header
    try:
        filename = str(row['FILENAME']) # Menggunakan 'FILENAME' setelah di-uppercase
    except KeyError:
        print(f"Error: Kolom 'FILENAME' tidak ditemukan di CSV setelah pembersihan header. Pastikan ada kolom 'filename' di CSV Anda.")
        skipped_count += 1
        continue

    image_path = os.path.join(IMAGES_SUBFOLDER, filename) # Path gambar sekarang langsung di DATA_FOLDER

    # Tambahkan pengecekan untuk memastikan file CSV tidak mencoba memuat dirinya sendiri sebagai gambar
    if filename.lower().endswith('.csv'):
        print(f"Melewatkan file CSV di dalam daftar gambar: {filename}")
        skipped_count += 1
        continue

    if os.path.exists(image_path):
        try:
            # Memuat gambar dan mengubah ukurannya
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0 # Normalisasi piksel ke rentang 0-1

            all_images.append(img_array)

            # --- Bagian kritis: Mengambil nilai NPK ---
            try:
                # Mengambil nilai N, P, K dari kolom yang sesuai
                # Menggunakan pd.to_numeric untuk memaksa konversi ke numerik dan menangani error
                # Mengakses kolom N, P, K setelah di-uppercase
                n_val = pd.to_numeric(row['N'], errors='coerce')
                p_val = pd.to_numeric(row['P'], errors='coerce')
                k_val = pd.to_numeric(row['K'], errors='coerce')

                # Mengecek apakah ada nilai NaN (Not a Number) setelah konversi
                if pd.isna(n_val) or pd.isna(p_val) or pd.isna(k_val):
                    raise ValueError(f"Nilai NPK non-numerik atau kosong ditemukan: N={row['N']}, P={row['P']}, K={row['K']}")

                npk_value = [n_val, p_val, k_val]
                all_npk_values.append(npk_value)
                processed_count += 1
            except KeyError as ke:
                # Pesan error ini seharusnya tidak lagi muncul jika pembersihan header berhasil
                print(f"Error: Kolom N, P, atau K tidak ditemukan untuk {filename}. Detail: {ke}. Pastikan header di CSV adalah 'N', 'P', 'K' (tanpa spasi tersembunyi).")
                skipped_count += 1
            except ValueError as ve:
                print(f"Error: Data NPK non-numerik atau kosong di CSV untuk {filename}. Detail: {ve}. Pastikan nilai N, P, K adalah angka.")
                print(f"Baris data yang bermasalah: {row.to_dict()}") # Cetak seluruh baris
                skipped_count += 1
            except Exception as inner_e:
                print(f"Error tak terduga saat memproses nilai NPK untuk {filename}: {inner_e} (Tipe: {type(inner_e)})")
                print(f"Baris data yang bermasalah: {row.to_dict()}")
                skipped_count += 1

        except tf.errors.NotFoundError:
            print(f"Gagal memuat gambar {filename}: File gambar tidak ditemukan oleh TensorFlow. (Path: {image_path})")
            skipped_count += 1
        except Exception as e:
            # Catch all other exceptions during image loading/processing
            print(f"Gagal memuat atau memproses gambar {filename}: {e} (Tipe: {type(e)})")
            skipped_count += 1
    else:
        print(f"Gambar tidak ditemukan: {image_path}. Melewatkan baris ini.")
        skipped_count += 1

if processed_count == 0:
    print("\nTidak ada gambar yang berhasil dimuat. Pastikan:")
    print(f"1. Path '{DATA_FOLDER}' sudah benar.")
    print(f"2. File 'label.csv' ada di dalam '{DATA_FOLDER}'.")
    print(f"3. Nama file gambar di kolom 'filename' CSV sesuai dengan nama file gambar di '{DATA_FOLDER}'.")
    print("4. Tidak ada masalah permission untuk membaca file.")
    exit()

# Mengubah list menjadi array NumPy
X = np.array(all_images)
y = np.array(all_npk_values)

print(f"\nTotal gambar yang berhasil diproses: {processed_count}")
print(f"Total baris yang dilewati (gambar tidak ditemukan/error): {skipped_count}")
print(f"Bentuk data gambar: {X.shape}")
print(f"Bentuk data NPK: {y.shape}")

# Memisahkan data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Jumlah sampel pelatihan: {len(X_train)}")
print(f"Jumlah sampel pengujian: {len(X_test)}")

# --- 2. Membangun CNN sebagai Feature Extractor ---
# Arsitektur CNN ini adalah contoh sederhana.
# Anda harus menyesuaikannya berdasarkan arsitektur yang dijelaskan dalam tesis
# atau eksperimen Anda sendiri.

def build_cnn_feature_extractor(input_shape):
    """
    Membangun model CNN untuk ekstraksi fitur dari gambar.
    """
    input_tensor = Input(shape=input_shape)

    # Lapisan Konvolusi pertama
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = MaxPooling2D((2, 2))(x)

    # Lapisan Konvolusi kedua
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Lapisan Konvolusi ketiga
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Meratakan output untuk diumpankan ke lapisan Dense atau SVR
    x = Flatten()(x)

    # Model akan mengembalikan output Flatten sebagai fitur
    model = Model(inputs=input_tensor, outputs=x)
    return model

# Membangun feature extractor
cnn_feature_extractor = build_cnn_feature_extractor(
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)
)
cnn_feature_extractor.summary()

# Kompilasi CNN (meskipun tidak akan dilatih untuk regresi NPK secara langsung,
# kompilasi diperlukan jika Anda ingin melatihnya sebagai bagian dari model yang lebih besar
# atau jika Anda ingin menggunakan pre-trained weights)
cnn_feature_extractor.compile(optimizer='adam', loss='mse')

# --- 3. Ekstraksi Fitur Menggunakan CNN ---
print("\nMengekstrak fitur dari gambar menggunakan CNN...")
X_train_features = cnn_feature_extractor.predict(X_train)
X_test_features = cnn_feature_extractor.predict(X_test)

print(f"Bentuk fitur pelatihan: {X_train_features.shape}")
print(f"Bentuk fitur pengujian: {X_test_features.shape}")

# --- 4. Melatih Model SVR untuk Setiap Unsur NPK ---
# Kita akan melatih tiga model SVR terpisah: satu untuk Nitrogen (N), satu untuk Fosfor (P),
# dan satu untuk Kalium (K).

# Anda dapat menyesuaikan parameter SVR (kernel, C, epsilon, gamma)
# sesuai dengan hasil optimasi atau yang disebutkan dalam tesis.
# Contoh: SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')

svr_n = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_p = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_k = SVR(kernel='rbf', C=100, epsilon=0.1)

print("\nMelatih model SVR untuk Nitrogen (N)...")
svr_n.fit(X_train_features, y_train[:, 0]) # Melatih untuk kolom Nitrogen (indeks 0)

print("Melatih model SVR untuk Fosfor (P)...")
svr_p.fit(X_train_features, y_train[:, 1]) # Melatih untuk kolom Fosfor (indeks 1)

print("Melatih model SVR untuk Kalium (K)...")
svr_k.fit(X_train_features, y_train[:, 2]) # Melatih untuk kolom Kalium (indeks 2)

# --- 5. Prediksi dan Evaluasi ---
print("\nMelakukan prediksi dan mengevaluasi model...")

# Prediksi untuk Nitrogen
y_pred_n = svr_n.predict(X_test_features)
mse_n = mean_squared_error(y_test[:, 0], y_pred_n)
r2_n = r2_score(y_test[:, 0], y_pred_n)
mape_n = mean_absolute_percentage_error(y_test[:, 0], y_pred_n) # Hitung MAPE
print(f"Evaluasi Nitrogen (N):")
print(f"  Mean Squared Error (MSE): {mse_n:.4f}")
print(f"  R-squared (R2): {r2_n:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape_n:.2f}%") # Tampilkan MAPE

# Prediksi untuk Fosfor
y_pred_p = svr_p.predict(X_test_features)
mse_p = mean_squared_error(y_test[:, 1], y_pred_p)
r2_p = r2_score(y_test[:, 1], y_pred_p)
mape_p = mean_absolute_percentage_error(y_test[:, 1], y_pred_p) # Hitung MAPE
print(f"\nEvaluasi Fosfor (P):")
print(f"  Mean Squared Error (MSE): {mse_p:.4f}")
print(f"  R-squared (R2): {r2_p:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape_p:.2f}%") # Tampilkan MAPE

# Prediksi untuk Kalium
y_pred_k = svr_k.predict(X_test_features)
mse_k = mean_squared_error(y_test[:, 2], y_pred_k)
r2_k = r2_score(y_test[:, 2], y_pred_k)
mape_k = mean_absolute_percentage_error(y_test[:, 2], y_pred_k) # Hitung MAPE
print(f"\nEvaluasi Kalium (K):")
print(f"  Mean Squared Error (MSE): {mse_k:.4f}")
print(f"  R-squared (R2): {r2_k:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape_k:.2f}%") # Tampilkan MAPE

# Menggabungkan semua prediksi NPK
y_pred_combined = np.column_stack((y_pred_n, y_pred_p, y_pred_k))
print("\nPrediksi NPK gabungan untuk sampel pengujian pertama:")
print(y_pred_combined[:5]) # Menampilkan 5 prediksi pertama

print("\nNilai NPK aktual untuk sampel pengujian pertama:")
print(y_test[:5]) # Menampilkan 5 nilai aktual pertama

# --- Visualisasi (Opsional) ---
# Plot hasil prediksi vs nilai aktual untuk salah satu unsur (misal Nitrogen)
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:, 0], y_pred_n, alpha=0.7)
plt.plot([min(y_test[:, 0]), max(y_test[:, 0])], [min(y_test[:, 0]), max(y_test[:, 0])], 'r--')
plt.xlabel("Nilai N Aktual")
plt.ylabel("Nilai N Prediksi")
plt.title("Prediksi N vs Aktual N")
plt.grid(True)
plt.show()

# --- Fungsi untuk Prediksi pada Gambar Baru ---
def predict_npk_from_image(image_path, cnn_extractor, svr_n_model, svr_p_model, svr_k_model):
    """
    Fungsi untuk memuat, memproses, dan memprediksi nilai NPK dari gambar baru.
    """
    try:
        # Memuat gambar
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Menambahkan dimensi batch
        img_array = img_array / 255.0 # Normalisasi (jika model dilatih dengan data ternormalisasi)

        # Ekstraksi fitur
        features = cnn_extractor.predict(img_array)

        # Prediksi NPK
        pred_n = svr_n_model.predict(features)[0]
        pred_p = svr_p_model.predict(features)[0]
        pred_k = svr_k_model.predict(features)[0]

        return pred_n, pred_p, pred_k
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses gambar: {e}")
        return None, None, None

# Contoh penggunaan fungsi prediksi (Anda perlu memiliki file gambar nyata)
# Untuk menguji fungsi ini, Anda bisa menunjuk ke salah satu gambar di folder 'data'.
# Misalnya, jika Anda memiliki 'data/test_image.jpg'
# test_image_path = os.path.join(DATA_FOLDER, 'test_image.jpg')
# print(f"\nMencoba prediksi pada gambar baru: {test_image_path}")
# predicted_n, predicted_p, predicted_k = predict_npk_from_image(
#     test_image_path, cnn_feature_extractor, svr_n, svr_p, svr_k
# )

# if predicted_n is not None:
#     print(f"Prediksi NPK untuk gambar baru:")
#     print(f"  Nitrogen (N): {predicted_n:.2f}")
#     print(f"  Fosfor (P): {predicted_p:.2f}")
#     print(f"  Kalium (K): {predicted_k:.2f}")
