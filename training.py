import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import shap # Import SHAP library

# Import pre-trained models
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

# --- Fungsi untuk menghitung MAPE ---
def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """
    Menghitung Mean Absolute Percentage Error (MAPE).
    Menambahkan epsilon ke penyebut untuk menghindari pembagian dengan nol.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# --- Konfigurasi Hyperparameter untuk CNN Training (Disatukan untuk Semua Arsitektur CNN) ---
# Hyperparameter yang akan digunakan untuk melatih 'head' atau seluruh CustomCNN
COMMON_EPOCHS = 30 # Disesuaikan, bisa 10 atau 20, tergantung kebutuhan
COMMON_BATCH_SIZE = 32
COMMON_LEARNING_RATE = 0.001 # Learning rate lebih kecil cocok untuk fine-tuning atau melatih head baru
COMMON_DROPOUT_RATE = 0.3 # Dropout rate yang konsisten

# --- 1. Persiapan Data (Memuat dari Folder dan CSV) ---
DATA_FOLDER = 'data'
IMAGES_SUBFOLDER = DATA_FOLDER
CSV_FILE_PATH = os.path.join(DATA_FOLDER, 'label.csv')

IMAGE_HEIGHT = 224 # VGG16 dan ResNet50 idealnya 224x224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3

print("Memuat data dari folder dan CSV...")

try:
    df_npk = pd.read_csv(CSV_FILE_PATH)
    df_npk.columns = df_npk.columns.str.strip().str.upper()
    print(f"Data NPK berhasil dimuat dari: {CSV_FILE_PATH}")
    print("Header CSV yang dibaca (setelah pembersihan):", df_npk.columns.tolist())
    print("Info DataFrame:")
    df_npk.info()
    print(df_npk.head())
except FileNotFoundError:
    print(f"Error: File CSV tidak ditemukan di {CSV_FILE_PATH}")
    print(f"Pastikan Anda memiliki file 'label.csv' di dalam folder '{DATA_FOLDER}'.")
    exit()
except Exception as e:
    print(f"Error saat memuat CSV: {e} (Tipe: {type(e)})")
    exit()

all_images = []
all_npk_values = []
processed_count = 0
skipped_count = 0

# NOTE: Preprocessing images will be handled by specific model's preprocess_input function later.
# For initial loading, we'll just load them as raw arrays and normalize to 0-1.
for index, row in df_npk.iterrows():
    try:
        filename = str(row['FILENAME'])
    except KeyError:
        print(f"Error: Kolom 'FILENAME' tidak ditemukan di CSV setelah pembersihan header. Pastikan ada kolom 'filename' di CSV Anda.")
        skipped_count += 1
        continue

    image_path = os.path.join(IMAGES_SUBFOLDER, filename)

    if filename.lower().endswith('.csv'):
        print(f"Melewatkan file CSV di dalam daftar gambar: {filename}")
        skipped_count += 1
        continue

    if os.path.exists(image_path):
        try:
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            # Normalisasi awal ke 0-1
            img_array = img_array / 255.0

            all_images.append(img_array)

            try:
                n_val = pd.to_numeric(row['N'], errors='coerce')
                p_val = pd.to_numeric(row['P'], errors='coerce')
                k_val = pd.to_numeric(row['K'], errors='coerce')

                if pd.isna(n_val) or pd.isna(p_val) or pd.isna(k_val):
                    raise ValueError(f"Nilai NPK non-numerik atau kosong ditemukan: N={row['N']}, P={row['P']}, K={row['K']}")

                npk_value = [n_val, p_val, k_val]
                all_npk_values.append(npk_value)
                processed_count += 1
            except KeyError as ke:
                print(f"Error: Kolom N, P, atau K tidak ditemukan untuk {filename}. Detail: {ke}. Pastikan header di CSV adalah 'N', 'P', 'K' (tanpa spasi tersembunyi).")
                skipped_count += 1
            except ValueError as ve:
                print(f"Error: Data NPK non-numerik atau kosong di CSV untuk {filename}. Detail: {ve}. Pastikan nilai N, P, K adalah angka.")
                print(f"Baris data yang bermasalah: {row.to_dict()}")
                skipped_count += 1
            except Exception as inner_e:
                print(f"Error tak terduga saat memproses nilai NPK untuk {filename}: {inner_e} (Tipe: {type(inner_e)})")
                print(f"Baris data yang bermasalah: {row.to_dict()}")
                skipped_count += 1

        except tf.errors.NotFoundError:
            print(f"Gagal memuat gambar {filename}: File gambar tidak ditemukan oleh TensorFlow. (Path: {image_path})")
            skipped_count += 1
        except Exception as e:
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

X = np.array(all_images)
y = np.array(all_npk_values)

print(f"\nTotal gambar yang berhasil diproses: {processed_count}")
print(f"Total baris yang dilewati (gambar tidak ditemukan/error): {skipped_count}")
print(f"Bentuk data gambar: {X.shape}")
print(f"Bentuk data NPK: {y.shape}")

# --- MODIFIKASI: Split data menjadi 60% latih, 20% validasi, 20% uji berdasarkan urutan ---
total_samples = len(X)
train_size = int(0.6 * total_samples)
val_size = int(0.2 * total_samples)
test_size = total_samples - train_size - val_size # Use remaining for test to avoid rounding issues

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"Jumlah sampel pelatihan (sequential): {len(X_train)}")
print(f"Jumlah sampel validasi (sequential): {len(X_val)}")
print(f"Jumlah sampel pengujian (sequential): {len(X_test)}")

# Dictionary untuk menyimpan hasil MAPE setiap model
all_models_mape = {}
# Dictionary untuk menyimpan history training loss CNN/head
all_cnn_histories = {}
# List untuk menyimpan data SVR dan fitur untuk plotting SHAP
all_svr_results_for_shap = []


# --- Fungsi untuk Membangun dan Melatih Model Feature Extractor ---
def build_and_train_feature_extractor(base_model_name, input_shape, output_dim, common_dropout_rate,
                                      X_train_raw, y_train_raw, X_val_raw, y_val_raw):
    """
    Membangun model feature extractor dari arsitektur pre-trained,
    menambahkan lapisan Dense baru di atasnya, dan melatih lapisan Dense tersebut.
    Mengembalikan feature extractor, preprocessor, dan history pelatihan untuk model tersebut.
    """
    print(f"\n--- Membangun model feature extractor dengan arsitektur {base_model_name} ---")

    base_model = None
    preprocess_input_fn = None
    full_model = None # Define full_model here for consistent return logic

    if base_model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_vgg16
    elif base_model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_resnet50
    elif base_model_name == "InceptionV3":
        if input_shape[0] < 299 or input_shape[1] < 299:
            print(f"Peringatan: InceptionV3 secara optimal bekerja dengan input 299x299. Saat ini menggunakan {input_shape[0]}x{input_shape[1]}.")
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_inceptionv3
    elif base_model_name == "CustomCNN": # CNN kustom Anda sebelumnya
        print("Membangun Custom CNN (seperti sebelumnya)...")
        input_tensor = Input(shape=input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(common_dropout_rate)(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(common_dropout_rate)(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(common_dropout_rate)(x)

        x = Flatten()(x)
        feature_output_layer = Dense(128, activation='relu', name='feature_extractor_output')(x)
        feature_output_layer = Dropout(common_dropout_rate)(feature_output_layer)
        full_model = Model(inputs=input_tensor, outputs=Dense(3, activation='linear')(feature_output_layer))
        feature_extractor = Model(inputs=input_tensor, outputs=full_model.get_layer('feature_extractor_output').output)

        preprocess_input_fn = lambda x: x # Untuk CustomCNN, preprocessing cukup normalisasi 0-1

        print(f"\nMelatih CustomCNN sebagai model regresi awal dengan {COMMON_EPOCHS} epoch, batch size {COMMON_BATCH_SIZE}, learning rate {COMMON_LEARNING_RATE}...")
        optimizer = Adam(learning_rate=COMMON_LEARNING_RATE)
        full_model.compile(optimizer=optimizer, loss='mse')

        history = full_model.fit(
            X_train_raw, y_train_raw,
            epochs=COMMON_EPOCHS,
            batch_size=COMMON_BATCH_SIZE,
            validation_data=(X_val_raw, y_val_raw),
            verbose=1
        )
        return feature_extractor, preprocess_input_fn, history
    else:
        raise ValueError("Nama model tidak dikenali.")

    # Freeze base model layers for pre-trained models
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom head for feature extraction
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Better than Flatten for some architectures (e.g., ResNet, Inception)

    # Feature extraction layer (sesuai dengan output 128 dari CustomCNN)
    feature_output = Dense(128, activation='relu', name='feature_extractor_output')(x)
    feature_output = Dropout(common_dropout_rate)(feature_output)

    # Output layer for direct NPK regression (for pre-training/fine-tuning the head)
    npk_output = Dense(output_dim, activation='linear', name='npk_regression_output')(feature_output)

    # Full model for training the new head
    full_model = Model(inputs=base_model.input, outputs=npk_output)

    # Feature extractor model
    feature_extractor = Model(inputs=base_model.input, outputs=full_model.get_layer('feature_extractor_output').output)

    full_model.summary()

    # Compile and train the new head with common hyperparameters
    print(f"\nMelatih head {base_model_name} dengan {COMMON_EPOCHS} epoch, batch size {COMMON_BATCH_SIZE}, learning rate {COMMON_LEARNING_RATE}...")
    optimizer = Adam(learning_rate=COMMON_LEARNING_RATE)
    full_model.compile(optimizer=optimizer, loss='mse')

    # Apply preprocessing to data before training the head
    X_train_processed = preprocess_input_fn(X_train_raw * 255.0) # Inverse normalize then apply model's preprocess
    X_val_processed = preprocess_input_fn(X_val_raw * 255.0)

    history = full_model.fit(
        X_train_processed, y_train_raw,
        epochs=COMMON_EPOCHS,
        batch_size=COMMON_BATCH_SIZE,
        validation_data=(X_val_processed, y_val_raw),
        verbose=1
    )

    return feature_extractor, preprocess_input_fn, history


# --- Looping untuk Setiap Arsitektur Model ---
# model_architectures = ["CustomCNN", "VGG16", "ResNet50", "InceptionV3"]
model_architectures = ["CustomCNN"] # Uncomment this line to run only CustomCNN

for arch_name in model_architectures:
    print(f"\n=======================================================")
    print(f"--- Memulai Proses untuk Arsitektur: {arch_name} ---")
    print(f"=======================================================\n")

    # Build and "train" (fine-tune the head) the feature extractor
    cnn_feature_extractor, preprocess_input_fn, current_cnn_history = build_and_train_feature_extractor(
        arch_name,
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS),
        output_dim=3,
        common_dropout_rate=COMMON_DROPOUT_RATE,
        X_train_raw=X_train, y_train_raw=y_train, X_val_raw=X_val, y_val_raw=y_val
    )
    # Store history for later plotting
    all_cnn_histories[arch_name] = current_cnn_history

    # Apply specific preprocessing for extraction
    # Note: X_train, X_test, X_val were already normalized to 0-1 initially.
    # We convert back to 0-255 range then apply specific preprocessing.
    X_train_preprocessed_for_extractor = preprocess_input_fn(X_train * 255.0)
    X_test_preprocessed_for_extractor = preprocess_input_fn(X_test * 255.0)

    print(f"\nMengekstrak fitur dari gambar menggunakan {arch_name} feature extractor...")
    X_train_features = cnn_feature_extractor.predict(X_train_preprocessed_for_extractor)
    X_test_features = cnn_feature_extractor.predict(X_test_preprocessed_for_extractor)

    print(f"Bentuk fitur pelatihan ({arch_name}): {X_train_features.shape}")
    print(f"Bentuk fitur pengujian ({arch_name}): {X_test_features.shape}")

    # --- Feature Scaling untuk SVR ---
    scaler = StandardScaler()
    X_train_features_scaled = scaler.fit_transform(X_train_features)
    X_test_features_scaled = scaler.transform(X_test_features)

    print(f"\nFitur {arch_name} telah diskalakan menggunakan StandardScaler.")

    # Dictionary sementara untuk MAPE model saat ini
    current_model_mape = {}

    # --- Melatih Model SVR untuk Setiap Unsur NPK dengan Hyperparameter Tuning ---
    def train_and_evaluate_svr_with_tuning(X_train_feats, y_train_tgt, X_test_feats, y_test_tgt, target_name):
        print(f"\n--- Training SVR for {target_name} ({arch_name}) with Hyperparameter Tuning ---")

        svr = SVR()
        param_grid = {
            'kernel': ['rbf', 'linear', 'poly'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(svr, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        grid_search.fit(X_train_feats, y_train_tgt)

        print(f"Best hyperparameters for {target_name} ({arch_name}): {grid_search.best_params_}")
        print(f"Best cross-validation MSE for {target_name} ({arch_name}): {-grid_search.best_score_:.4f}")

        best_svr_model = grid_search.best_estimator_
        y_pred = best_svr_model.predict(X_test_feats)

        mse = mean_squared_error(y_test_tgt, y_pred)
        r2 = r2_score(y_test_tgt, y_pred)
        mape = mean_absolute_percentage_error(y_test_tgt, y_pred)

        print(f"Test MSE for {target_name} ({arch_name}): {mse:.4f}")
        print(f"Test R-squared for {target_name} ({arch_name}): {r2:.4f}")
        print(f"Test MAPE for {target_name} ({arch_name}): {mape:.2f}%")

        current_model_mape[f"{target_name}"] = mape

        return best_svr_model, y_pred

    np.set_printoptions(suppress=True, precision=5)

    best_svr_n, y_pred_n = train_and_evaluate_svr_with_tuning(
        X_train_features_scaled, y_train[:, 0], X_test_features_scaled, y_test[:, 0], "Nitrogen (N)"
    )
    all_svr_results_for_shap.append((arch_name, "Nitrogen (N)", best_svr_n, X_test_features_scaled, X_train_features_scaled))


    best_svr_p, y_pred_p = train_and_evaluate_svr_with_tuning(
        X_train_features_scaled, y_train[:, 1], X_test_features_scaled, y_test[:, 1], "Fosfor (P)"
    )
    all_svr_results_for_shap.append((arch_name, "Fosfor (P)", best_svr_p, X_test_features_scaled, X_train_features_scaled))


    best_svr_k, y_pred_k = train_and_evaluate_svr_with_tuning(
        X_train_features_scaled, y_train[:, 2], X_test_features_scaled, y_test[:, 2], "Kalium (K)"
    )
    all_svr_results_for_shap.append((arch_name, "Kalium (K)", best_svr_k, X_test_features_scaled, X_train_features_scaled))


    # Store the MAPE values for this architecture
    all_models_mape[arch_name] = current_model_mape


# --- Menampilkan Grafik Training/Validation Loss untuk Setiap CNN ---
print("\n=======================================================")
print("--- Menampilkan Grafik Training/Validation Loss untuk Setiap CNN ---")
print("=======================================================\n")
for arch_name, history in all_cnn_histories.items():
    if history: # Pastikan history ada (untuk CustomCNN, history langsung dari full_model)
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{arch_name} Training Loss\nModel: {arch_name} | Data: Training and Validation Set')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.show()

# --- Menampilkan Interpretasi SHAP untuk Setiap SVR Model ---
# print("\n=======================================================")
# print("--- Menampilkan Interpretasi SHAP untuk Setiap SVR Model ---")
# print("=======================================================\n")
# for arch_name, target_name, svr_model, X_test_feats_shap, X_train_feats_shap in all_svr_results_for_shap:
#     print(f"\n--- SHAP Interpretation for {target_name} ({arch_name}) ---")
#     # Ensure background_data is not too large for KernelExplainer
#     # Sample from training features to create background data for SHAP
#     background_data = shap.utils.sample(X_train_feats_shap, 100) # Ambil 100 sampel dari fitur training
#     explainer = shap.KernelExplainer(svr_model.predict, background_data)
#     num_shap_samples = min(50, X_test_feats_shap.shape[0]) # Ambil max 50 sampel dari test untuk SHAP values

#     # Check if X_test_feats_shap has enough samples
#     if num_shap_samples == 0:
#         print(f"Peringatan: Tidak cukup sampel di X_test_feats_shap untuk {target_name} ({arch_name}) untuk plot SHAP.")
#         continue

#     shap_values = explainer.shap_values(X_test_feats_shap[:num_shap_samples])

#     print(f"Generating SHAP summary plot for {target_name} ({arch_name})...")
#     shap.summary_plot(shap_values, X_test_feats_shap[:num_shap_samples], show=False)
#     plt.suptitle(f'SHAP Feature Importance for {target_name}\nModel: SVR with {arch_name} Features | Data: Test Set', y=1.02)
#     plt.show()

# --- Tampilan Grafik MAPE Semua Model NPK dan Arsitektur (Sudah di akhir) ---
print("\n=======================================================")
print("--- Menampilkan Grafik MAPE Komparasi Antar Arsitektur ---")
print("=======================================================\n")

if all_models_mape:
    mape_df = pd.DataFrame(all_models_mape).T # Transpose to have architectures as index
    mape_df.index.name = 'Architecture'
    mape_df.columns.name = 'Nutrient'

    # Reshape for seaborn barplot
    mape_df_melted = mape_df.reset_index().melt(id_vars='Architecture', var_name='Nutrient', value_name='MAPE')

    plt.figure(figsize=(12, 7))
    sns.barplot(data=mape_df_melted, x='Architecture', y='MAPE', hue='Nutrient', palette='viridis')
    plt.title('Mean Absolute Percentage Error (MAPE) Comparison Across Architectures\nModel: SVR with various Feature Extractors | Data: Test Set')
    plt.xlabel('CNN Architecture (Feature Extractor)')
    plt.ylabel('MAPE (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Nutrient Type')

    # Add MAPE values on top of bars
    for container in plt.gca().containers:
        for patch in container.patches:
            height = patch.get_height()
            plt.text(patch.get_x() + patch.get_width() / 2.,
                     height + 0.5,
                     f'{height:.2f}%',
                     ha='center', va='bottom', fontsize=9)
    plt.show()
else:
    print("Tidak ada nilai MAPE yang tersedia untuk diplot komparasi.")


print("\n--- Proses Selesai ---")
print("Model CNN kustom, VGG16, ResNet50, dan InceptionV3 telah dilatih sebagai feature extractor,")
print("dan model SVR telah dilatih dan dievaluasi dengan tuning hyperparameter untuk setiap arsitektur.")
print("Interpretasi model menggunakan SHAP juga telah ditambahkan untuk memahami kontribusi fitur.")
print("Sekarang menampilkan grafik MAPE gabungan untuk ketiga model NPK, dikomparasi antar arsitektur.")
print("Untuk menggunakan fungsi prediksi pada gambar baru, uncomment bagian 'Contoh penggunaan fungsi prediksi' di akhir kode.")

# --- Fungsi untuk Prediksi pada Gambar Baru ---
def predict_npk_from_image(image_path, cnn_extractor, scaler_features, svr_n_model, svr_p_model, svr_k_model, image_height, image_width, preprocess_fn):
    """
    Fungsi untuk memuat, memproses, dan memprediksi nilai NPK dari gambar baru.
    Membutuhkan scaler_features yang telah dilatih pada data pelatihan.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(image_height, IMAGE_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Apply specific preprocessing for the model
        img_array = preprocess_fn(np.expand_dims(img_array, axis=0))

        features = cnn_extractor.predict(img_array)

        features_scaled = scaler_features.transform(features)

        pred_n = svr_n_model.predict(features_scaled)[0]
        pred_p = svr_p_model.predict(features_scaled)[0]
        pred_k = svr_k_model.predict(features_scaled)[0]

        return pred_n, pred_p, pred_k
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses gambar: {e}")
        return None, None, None