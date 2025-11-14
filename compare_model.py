import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Import dari file lokal
import config
from data_loader import load_and_split_data
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def build_and_train_feature_extractor(base_model_name, X_train_raw, y_train_raw, X_val_raw, y_val_raw):
    print(f"\n--- Membangun model feature extractor dengan arsitektur {base_model_name} ---")
    
    input_shape = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.NUM_CHANNELS)
    output_dim = 3
    base_model = None
    preprocess_input_fn = None
    
    if base_model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_vgg16
    elif base_model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_resnet50
    elif base_model_name == "InceptionV3":
        if input_shape[0] < 299 or input_shape[1] < 299:
            print(f"Peringatan: InceptionV3 bekerja optimal dengan input 299x299.")
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_inceptionv3
        
    elif base_model_name == "CustomCNN":
        input_tensor = Input(shape=input_shape)

        # --- ARSITEKTUR BARU BERDASARKAN TABEL ---
        
        # Tahap 1: 2x Conv(8) -> Pool
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_tensor)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        
        # Tahap 2: 2x Conv(16) -> Pool
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        
        # Tahap 3: 3x Conv(16) -> Pool
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        
        # Tahap 4: 3x Conv(32) -> Pool
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        
        # Tahap 5: 3x Conv(32) -> Pool
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        
        # Tahap 6: 1x Conv(16) 7x7 (padding='valid' untuk 7x7 -> 1x1)
        x = Conv2D(16, (7, 7), activation='relu', padding='valid')(x)
        
        # --- AKHIR ARSITEKTUR BARU ---
        
        x = Flatten()(x) # Output: (16)
        
        # Layer output dari feature extractor (Dense 128 dipertahankan)
        feature_output_layer = Dense(128, activation='relu', name='feature_extractor_output')(x)
        feature_output_layer = Dropout(config.COMMON_DROPOUT_RATE)(feature_output_layer)
        
        # Model lengkap untuk regresi end-to-end
        full_model_output = Dense(output_dim, activation='linear')(feature_output_layer)
        full_model = Model(inputs=input_tensor, outputs=full_model_output)

        # Model feature extractor terpisah
        feature_extractor = Model(inputs=input_tensor, outputs=full_model.get_layer('feature_extractor_output').output)
        
        # CustomCNN tidak memerlukan preprocessing input khusus
        preprocess_input_fn = lambda x: x
        
        # Kompilasi dan latih model lengkap
        optimizer = Adam(learning_rate=config.COMMON_LEARNING_RATE)
        full_model.compile(optimizer=optimizer, loss='mse')
        
        print("--- Ringkasan Model CustomCNN ---")
        full_model.summary()
        print("--------------------------------------------------")

        history = full_model.fit(X_train_raw, y_train_raw, 
            epochs=config.COMMON_EPOCHS, batch_size=config.COMMON_BATCH_SIZE, 
            validation_data=(X_val_raw, y_val_raw), verbose=1)
        
        return feature_extractor, preprocess_input_fn, history
    
    else:
        raise ValueError("Nama model tidak dikenali.")

    # (VGG16, ResNet50, InceptionV3)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    feature_output = Dense(128, activation='relu', name='feature_extractor_output')(x)
    feature_output = Dropout(config.COMMON_DROPOUT_RATE)(feature_output)
    npk_output = Dense(output_dim, activation='linear', name='npk_regression_output')(feature_output)
    full_model = Model(inputs=base_model.input, outputs=npk_output)
    
    feature_extractor = Model(inputs=base_model.input, outputs=full_model.get_layer('feature_extractor_output').output)
    
    optimizer = Adam(learning_rate=config.COMMON_LEARNING_RATE)
    full_model.compile(optimizer=optimizer, loss='mse')
    
    # Preprocessing input untuk model transfer learning
    X_train_processed = preprocess_input_fn(X_train_raw * 255.0)
    X_val_processed = preprocess_input_fn(X_val_raw * 255.0)
    
    history = full_model.fit(X_train_processed, y_train_raw, 
                            epochs=config.COMMON_EPOCHS, 
                            batch_size=config.COMMON_BATCH_SIZE, 
                            validation_data=(X_val_processed, y_val_raw), 
                            verbose=1)
    
    return feature_extractor, preprocess_input_fn, history

def train_and_evaluate_svr_with_tuning(X_train_feats, y_train_tgt, X_test_feats, y_test_tgt, target_name, arch_name):
    print(f"\n--- Training SVR for {target_name} ({arch_name}) with Hyperparameter Tuning ---")
    svr = SVR()
    param_grid = {
        'kernel': ['rbf', 'linear', 'poly'], 
        'C': [0.001, 0.01, 0.1],
        'epsilon': [0.1, 0.15, 0.2], 
        'gamma': [0.001, 0.01, 0.1]
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(svr, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_feats, y_train_tgt)
    
    print(f"Best hyperparameters for {target_name} ({arch_name}): {grid_search.best_params_}")
    best_svr_model = grid_search.best_estimator_
    y_pred = best_svr_model.predict(X_test_feats)
    
    # Hitung semua metrik
    mae = mean_absolute_error(y_test_tgt, y_pred)
    mse = mean_squared_error(y_test_tgt, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_tgt, y_pred)
    
    # Cetak semua metrik
    print(f"Test MAE for {target_name} ({arch_name}): {mae:.4f}")
    print(f"Test MSE for {target_name} ({arch_name}): {mse:.4f}")
    print(f"Test RMSE for {target_name} ({arch_name}): {rmse:.4f}")
    print(f"Test MAPE for {target_name} ({arch_name}): {mape:.2f}%")
    
    # Kembalikan dictionary berisi metrik
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }
    
    return best_svr_model, metrics

def plot_metric_comparison(all_metrics_data, metric_key, metric_title, file_name, y_label):
    """
    Membuat bar plot perbandingan untuk satu metrik (MAE, MSE, RMSE, MAPE).
    """
    # 1. Ekstrak data untuk metrik spesifik
    # {arch: {nutrient: metric_value}}
    metric_data_dict = {}
    try:
        for arch, nutrients in all_metrics_data.items():
            metric_data_dict[arch] = {}
            for nutrient, metrics in nutrients.items():
                metric_data_dict[arch][nutrient] = metrics[metric_key]
    except KeyError:
        print(f"Error: Kunci metrik '{metric_key}' tidak ditemukan. Pastikan data terstruktur dengan benar.")
        return
    except Exception as e:
        print(f"Error saat mengekstrak data metrik: {e}")
        return

    if not metric_data_dict:
        print(f"Tidak ada data {metric_title} untuk di-plot.")
        return
    
    # 2. Konversi ke DataFrame dan 'melt'
    df = pd.DataFrame(metric_data_dict).T
    df_melted = df.reset_index().melt(id_vars='index', var_name='Nutrient', value_name=metric_title)
    df_melted.rename(columns={'index': 'Architecture'}, inplace=True)
    
    # 3. Plotting
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df_melted, x='Architecture', y=metric_title, hue='Nutrient', palette='viridis')
    
    plt.title(f'Perbandingan {metric_title}')
    plt.ylabel(y_label)
    plt.xlabel('Arsitektur CNN (Feature Extractor)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Anotasi
    for container in ax.containers:
        # Format label: tambahkan '%' jika MAPE, 4 desimal jika MSE, 2 desimal untuk lainnya
        if metric_key == 'mape':
            label_format = '{:.2f}%'
        elif metric_key == 'mse':
            label_format = '{:.4f}'
        else:
            label_format = '{:.2f}'
        
        labels = [label_format.format(v.get_height()) for v in container]
        
        ax.bar_label(container, 
                    labels=labels, 
                    padding=3, 
                    fontsize=9,
                    fontweight='bold')

    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
    plt.savefig(file_name)
    plt.show()

def main():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    # 1. Muat dan bagi data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data()

    print("\n" + "="*70)
    print("--- MEMULAI EKSPERIMEN DENGAN KONFIGURASI ---")
    print(f"Arsitektur yang Diuji : {config.MODEL_ARCHITECTURES}")
    print(f"Epochs (CNN)         : {config.COMMON_EPOCHS}")
    print(f"Batch Size (CNN)     : {config.COMMON_BATCH_SIZE}")
    print(f"Learning Rate (CNN)  : {config.COMMON_LEARNING_RATE}")
    print(f"Dropout Rate (CNN)   : {config.COMMON_DROPOUT_RATE}")
    print(f"Ukuran Input Gambar  : ({config.IMAGE_HEIGHT}, {config.IMAGE_WIDTH}, {config.NUM_CHANNELS})")
    print("="*70 + "\n")

    # --- BARIS BARU: BUAT STRING UNTUK NAMA FILE ---
    lr_str = f"lr_{config.COMMON_LEARNING_RATE}"

    all_models_metrics = {}
    all_cnn_histories = {}

    # 2. Loop melalui setiap arsitektur
    for arch_name in config.MODEL_ARCHITECTURES:
        print(f"\n{'='*60}\n--- Memulai Proses untuk Arsitektur: {arch_name} ---\n{'='*60}\n")
        
        # 3. Bangun, latih, dan SIMPAN feature extractor CNN
        cnn_extractor, preprocess_fn, history = build_and_train_feature_extractor(
            arch_name, X_train, y_train, X_val, y_val
        )
        all_cnn_histories[arch_name] = history
        
        # Simpan model CNN
        # --- NAMA FILE DIMODIFIKASI ---
        cnn_model_path = os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_feature_extractor_{lr_str}.keras")
        cnn_extractor.save(cnn_model_path)
        print(f"Model feature extractor {arch_name} disimpan di: {cnn_model_path}")
        
        # 4. Ekstrak fitur
        X_train_prep = preprocess_fn(X_train * 255.0)
        X_test_prep = preprocess_fn(X_test * 255.0)
        X_train_features = cnn_extractor.predict(X_train_prep)
        X_test_features = cnn_extractor.predict(X_test_prep)

        # 5. Latih, evaluasi, dan SIMPAN scaler
        scaler = StandardScaler()
        X_train_features_scaled = scaler.fit_transform(X_train_features)
        X_test_features_scaled = scaler.transform(X_test_features)
        # --- NAMA FILE DIMODIFIKASI ---
        scaler_path = os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_scaler_{lr_str}.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler untuk {arch_name} disimpan di: {scaler_path}")
        
        # 6. Latih, evaluasi, dan SIMPAN model SVR
        current_model_metrics = {}
        
        # SVR untuk Nitrogen (N)
        svr_n, metrics_n = train_and_evaluate_svr_with_tuning(
            X_train_features_scaled, y_train[:, 0], X_test_features_scaled, y_test[:, 0], "Nitrogen (N)", arch_name
        )
        # --- NAMA FILE DIMODIFIKASI ---
        joblib.dump(svr_n, os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_svr_n_{lr_str}.joblib"))
        current_model_metrics["Nitrogen (N)"] = metrics_n
        
        # SVR untuk Fosfor (P)
        svr_p, metrics_p = train_and_evaluate_svr_with_tuning(
            X_train_features_scaled, y_train[:, 1], X_test_features_scaled, y_test[:, 1], "Fosfor (P)", arch_name
        )
        # --- NAMA FILE DIMODIFIKASI ---
        joblib.dump(svr_p, os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_svr_p_{lr_str}.joblib"))
        current_model_metrics["Fosfor (P)"] = metrics_p
        
        # SVR untuk Kalium (K)
        svr_k, metrics_k = train_and_evaluate_svr_with_tuning(
            X_train_features_scaled, y_train[:, 2], X_test_features_scaled, y_test[:, 2], "Kalium (K)", arch_name
        )
        # --- NAMA FILE DIMODIFIKASI ---
        joblib.dump(svr_k, os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_svr_k_{lr_str}.joblib"))
        current_model_metrics["Kalium (K)"] = metrics_k

        all_models_metrics[arch_name] = current_model_metrics

    # 7. Plot dan tampilkan hasil evaluasi
    # Plotting Loss
    for arch_name, history in all_cnn_histories.items():
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss Pelatihan untuk {arch_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        # --- NAMA FILE DIMODIFIKASI ---
        plt.savefig(f"{arch_name}_training_loss_{lr_str}.png") # Simpan plot
        plt.show()

    # Plotting Perbandingan Metrik (MAPE, MAE, MSE, RMSE)
    if all_models_metrics:
        
        # Plotting Perbandingan MAPE
        plot_metric_comparison(all_models_metrics, 
                            metric_key='mape', 
                            metric_title='MAPE', 
                            file_name=f"mape_comparison_{lr_str}.png", # <-- DIMODIFIKASI
                            y_label='MAPE (%)')

        # Plotting Perbandingan MAE
        plot_metric_comparison(all_models_metrics, 
                            metric_key='mae', 
                            metric_title='MAE', 
                            file_name=f"mae_comparison_{lr_str}.png", # <-- DIMODIFIKASI
                            y_label='MAE')

        # Plotting Perbandingan MSE
        plot_metric_comparison(all_models_metrics, 
                            metric_key='mse', 
                            metric_title='MSE', 
                            file_name=f"mse_comparison_{lr_str}.png", # <-- DIMODIFIKASI
                            y_label='MSE')

        # Plotting Perbandingan RMSE
        plot_metric_comparison(all_models_metrics, 
                            metric_key='rmse', 
                            metric_title='RMSE', 
                            file_name=f"rmse_comparison_{lr_str}.png", # <-- DIMODIFIKASI
                            y_label='RMSE')

if __name__ == '__main__':
    main()