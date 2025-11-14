"""
Skrip utama untuk melatih feature extractor VGG16 dan model SVR
dengan memvariasikan SATU hyperparameter pada satu waktu.
Menyimpan model yang telah dilatih, scaler, dan menghasilkan plot evaluasi.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

# Diasumsikan file config.py dan data_loader.py berada di direktori yang sama
import config
from data_loader import load_and_split_data

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """Menghitung Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Menambahkan epsilon untuk menghindari pembagian dengan nol
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def build_and_train_vgg16(X_train, y_train, X_val, y_val, params):
    """Membangun dan melatih model VGG16 berdasarkan hyperparameter yang diberikan."""
    
    # Ekstrak hyperparameter dari dictionary 'params'
    epochs = params['epochs']
    batch_size = params['batch_size']
    learning_rate = params['lr']
    dropout_rate = params['dropout']
    run_name = params['run_name']
    
    print(f"\n--- Membangun VGG16: {run_name} ---")

    input_shape = (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.NUM_CHANNELS)
    output_dim = 3 # N, P, K

    # 1. Muat model VGG16 pre-trained tanpa layer klasifikasi (include_top=False)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # 2. Bekukan bobot dari base_model agar tidak ikut terlatih (transfer learning)
    for layer in base_model.layers:
        layer.trainable = False

    # 3. Tambahkan layer custom di atas VGG16
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Lebih efisien daripada Flatten untuk model pre-trained
    
    # Layer output dari feature extractor
    feature_output_layer = Dense(128, activation='relu', name='feature_extractor_output')(x)
    feature_output_layer = Dropout(dropout_rate)(feature_output_layer)
    
    # Model lengkap untuk regresi end-to-end
    full_model_output = Dense(output_dim, activation='linear')(feature_output_layer)
    full_model = Model(inputs=base_model.input, outputs=full_model_output)

    # Model feature extractor terpisah
    feature_extractor = Model(inputs=base_model.input, outputs=full_model.get_layer('feature_extractor_output').output)
    
    # 4. Preprocess data sesuai standar VGG16
    # VGG16 mengharapkan input dalam rentang warna tertentu, bukan [0, 1]
    X_train_prep = vgg16_preprocess(X_train * 255.0)
    X_val_prep = vgg16_preprocess(X_val * 255.0)

    # 5. Kompilasi dan latih model lengkap
    optimizer = Adam(learning_rate=learning_rate)
    full_model.compile(optimizer=optimizer, loss='mse')
    
    history = full_model.fit(
        X_train_prep, y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_val_prep, y_val), 
        verbose=1
    )
    
    return feature_extractor, history

def train_and_evaluate_svr_with_tuning(X_train_feats, y_train_tgt, X_test_feats, y_test_tgt, target_name, run_name):
    """Melatih SVR untuk target spesifik dengan hyperparameter tuning."""
    print(f"\n--- Training SVR for {target_name} ({run_name}) ---")
    
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
    
    print(f"Best hyperparameters for {target_name} ({run_name}): {grid_search.best_params_}")
    best_svr_model = grid_search.best_estimator_
    y_pred = best_svr_model.predict(X_test_feats)
    
    mape = mean_absolute_percentage_error(y_test_tgt, y_pred)
    print(f"Test MAPE for {target_name} ({run_name}): {mape:.2f}%")
    
    return best_svr_model, mape

def main():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # ==================== PENGATURAN EKSPERIMEN ====================
    # 1. Tentukan hyperparameter dasar (baseline) untuk layer custom
    baseline_params = {
        'epochs': 10,
        'batch_size': 32,
        'lr': 0.001,
        'dropout': 0.25
    }

    # 2. PILIH SATU hyperparameter yang ingin divariasikan
    # Pilihan: 'epochs', 'batch_size', 'lr', 'dropout'
    hyperparameter_to_tune = 'lr'

    # 3. Tentukan nilai-nilai yang akan diuji untuk setiap hyperparameter
    tuning_options = {
        'epochs': [10, 20, 30],
        'batch_size': [16, 32, 64],
        'lr': [0.01, 0.001, 0.0001],
        'dropout': [0.2, 0.4, 0.6]
    }
    # ===============================================================

    # Membuat daftar eksperimen secara otomatis
    hyperparameter_grid = []
    values_to_test = tuning_options[hyperparameter_to_tune]

    for value in values_to_test:
        new_params = baseline_params.copy()
        new_params[hyperparameter_to_tune] = value
        hyperparameter_grid.append(new_params)
    
    print(f"Memulai eksperimen dengan VGG16 dan memvariasikan: '{hyperparameter_to_tune}'")
    print("Konfigurasi yang akan dijalankan:")
    for p in hyperparameter_grid:
        print(p)

    # Muat dan bagi data (cukup sekali)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data()

    all_runs_mape = {}
    all_cnn_histories = {}

    # Loop melalui setiap kombinasi hyperparameter yang sudah dibuat
    for params in hyperparameter_grid:
        # Buat nama unik untuk setiap run eksperimen
        run_name = f"VGG16_E{params['epochs']}_B{params['batch_size']}_LR{params['lr']}_D{params['dropout']}"
        params['run_name'] = run_name
        
        print(f"\n{'='*70}\n--- Memulai Eksperimen: {run_name} ---\n{'='*70}\n")
        
        # Bangun, latih, dan SIMPAN feature extractor VGG16
        cnn_extractor, history = build_and_train_vgg16(X_train, y_train, X_val, y_val, params)
        all_cnn_histories[run_name] = history
        
        # Simpan model CNN
        cnn_model_path = os.path.join(config.MODEL_SAVE_DIR, f"{run_name}_feature_extractor.keras")
        cnn_extractor.save(cnn_model_path)
        print(f"Model feature extractor {run_name} disimpan di: {cnn_model_path}")
        
        # Ekstrak fitur
        X_train_prep = vgg16_preprocess(X_train * 255.0)
        X_test_prep = vgg16_preprocess(X_test * 255.0)
        X_train_features = cnn_extractor.predict(X_train_prep)
        X_test_features = cnn_extractor.predict(X_test_prep)

        # Latih, evaluasi, dan SIMPAN scaler
        scaler = StandardScaler()
        X_train_features_scaled = scaler.fit_transform(X_train_features)
        X_test_features_scaled = scaler.transform(X_test_features)
        scaler_path = os.path.join(config.MODEL_SAVE_DIR, f"{run_name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler untuk {run_name} disimpan di: {scaler_path}")
        
        # Latih, evaluasi, dan SIMPAN model SVR untuk setiap target
        current_run_mape = {}
        targets = {"Nitrogen (N)": 0, "Fosfor (P)": 1, "Kalium (K)": 2}

        for target_name, target_idx in targets.items():
            svr_model, mape_val = train_and_evaluate_svr_with_tuning(
                X_train_features_scaled, y_train[:, target_idx], 
                X_test_features_scaled, y_test[:, target_idx], 
                target_name, run_name
            )
            joblib.dump(svr_model, os.path.join(config.MODEL_SAVE_DIR, f"{run_name}_svr_{target_name[0]}.joblib"))
            current_run_mape[target_name] = mape_val

        all_runs_mape[run_name] = current_run_mape

    # Plot dan tampilkan hasil evaluasi
    # ... (Sisa kode untuk plotting tetap sama) ...
    for run_name, history in all_cnn_histories.items():
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Loss Pelatihan untuk {run_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.MODEL_SAVE_DIR, f"{run_name}_training_loss.png"))
        plt.show()

    if all_runs_mape:
        mape_df = pd.DataFrame(all_runs_mape).T
        mape_df_melted = mape_df.reset_index().melt(id_vars='index', var_name='Nutrient', value_name='MAPE')
        mape_df_melted.rename(columns={'index': 'Hyperparameter_Set'}, inplace=True)
        
        plt.figure(figsize=(16, 9))
        ax = sns.barplot(data=mape_df_melted, x='Hyperparameter_Set', y='MAPE', hue='Nutrient', palette='viridis')
        
        plt.title(f'Perbandingan MAPE VGG16 dengan Variasi {hyperparameter_to_tune.upper()}')
        plt.ylabel('MAPE (%)')
        plt.xlabel(f'Konfigurasi (Baseline + Variasi {hyperparameter_to_tune.upper()})')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for container in ax.containers:
            ax.bar_label(container, 
                labels=[f'{v.get_height():.2f}%' for v in container], 
                padding=3, 
                fontsize=9)

        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        plt.tight_layout()
        plt.savefig(os.path.join(config.MODEL_SAVE_DIR, f"mape_comparison_vgg16_tune_{hyperparameter_to_tune}.png"))
        plt.show()

if __name__ == '__main__':
    main()