"""
Skrip untuk melatih model SVR (satu target)
menggunakan fitur yang sudah diekstraksi.
Anda HANYA perlu menentukan nama run model CNN yang ingin digunakan.
"""
import os
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    """Menghitung Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Menambahkan epsilon untuk menghindari pembagian dengan nol
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def train_and_evaluate_svr_with_tuning(X_train_feats, y_train_tgt, X_test_feats, y_test_tgt, target_name, run_name):
    """
    Melatih SVR untuk target spesifik dengan hyperparameter tuning
    dan menyimpan model SVR yang terkait dengan 'run_name' CNN.
    """
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
    
    # --- METRIK EVALUASI (TAMBAHAN) ---
    mape = mean_absolute_percentage_error(y_test_tgt, y_pred)
    mae = mean_absolute_error(y_test_tgt, y_pred)
    mse = mean_squared_error(y_test_tgt, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n--- Metrik Evaluasi Test untuk {target_name} ({run_name}) ---")
    print(f"Test MAPE : {mape:.2f}%")
    print(f"Test MAE  : {mae:.4f}")
    print(f"Test MSE  : {mse:.4f}")
    print(f"Test RMSE : {rmse:.4f}")
    print("--------------------------------------------------")
    # --- AKHIR TAMBAHAN ---
    
    # Simpan model SVR
    # Nama file SVR akan menyertakan nama run CNN yang digunakan
    svr_model_path = os.path.join(config.MODEL_SAVE_DIR, f"{run_name}_svr_{target_name[0]}.joblib")
    joblib.dump(best_svr_model, svr_model_path)
    print(f"Model SVR {target_name} ({run_name}) disimpan di: {svr_model_path}")
    
    return mape

def main():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # ==================== PENGATURAN MODEL ====================
    
    # 1. Tentukan NAMA RUN model CNN yang fiturnya ingin Anda gunakan.
    #    Nama ini harus sama persis dengan yang dihasilkan oleh 'train_cnn_features.py'
    #    Contoh: "E10_B32_LR0.001_D0.25"
    #    (Salin-tempel nama run dari output skrip CNN Anda di sini)
    
    cnn_model_run_name = "CustomCNN_feature_extractor" 
    
    
    # 2. PILIH TARGET YANG INGIN DILATIH
    #    "Nitrogen (N)" -> 0
    #    "Fosfor (P)"   -> 1
    #    "Kalium (K)"   -> 2
    
    TARGET_NAME_TO_TRAIN = "Nitrogen (N)"
    TARGET_IDX_TO_TRAIN = 0
    
    # ==========================================================

    # Gunakan nama run yang sudah ditentukan
    run_name = cnn_model_run_name 
    
    print(f"Memulai training SVR untuk: {TARGET_NAME_TO_TRAIN}")
    print(f"Mencari file fitur untuk konfigurasi: {run_name}")

    print(f"\n{'='*70}\n--- Memulai Eksperimen SVR: {run_name} ---\n{'='*70}\n")
    
    # 1. CARI dan MUAT file fitur .npz yang sesuai
    #    Path ini dibuat secara dinamis dari 'cnn_model_run_name'
    features_save_path = os.path.join(config.MODEL_SAVE_DIR, f"{run_name}_features.npz")
    
    if not os.path.exists(features_save_path):
        print(f"ERROR: File fitur {features_save_path} tidak ditemukan.")
        print(f"Pastikan Anda sudah menjalankan 'train_cnn_features.py' untuk run '{run_name}'.")
        print("Pastikan variabel 'cnn_model_run_name' sudah benar.")
        return # Keluar dari skrip jika file tidak ada
        
    print(f"Memuat fitur dari {features_save_path}...")
    data = np.load(features_save_path)
    X_train_features_scaled = data['X_train']
    y_train = data['y_train']
    X_test_features_scaled = data['X_test']
    y_test = data['y_test']
    
    # 2. Latih, evaluasi, dan SIMPAN model SVR untuk SATU target
    
    mape_val = train_and_evaluate_svr_with_tuning(
        X_train_features_scaled, y_train[:, TARGET_IDX_TO_TRAIN], 
        X_test_features_scaled, y_test[:, TARGET_IDX_TO_TRAIN], 
        TARGET_NAME_TO_TRAIN, 
        run_name  # Teruskan nama run untuk penamaan file SVR
    )

    print(f"\n--- Selesai Eksperimen SVR untuk {run_name} ---")
    # Catatan: Nilai yang lain sudah dicetak di dalam fungsi
    print(f"MAPE Akhir ({TARGET_NAME_TO_TRAIN}): {mape_val:.2f}%") 

if __name__ == '__main__':
    main()