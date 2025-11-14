"""
Skrip utama untuk melatih feature extractor CNN dan model SVR.
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
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
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    feature_output = Dense(128, activation='relu', name='feature_extractor_output')(x)
    feature_output = Dropout(config.COMMON_DROPOUT_RATE)(feature_output)
    npk_output = Dense(output_dim, activation='linear', name='npk_regression_output')(feature_output)
    full_model = Model(inputs=base_model.input, outputs=npk_output)
    feature_extractor = Model(inputs=base_model.input, outputs=full_model.get_layer('feature_extractor_output').output)
    optimizer = Adam(learning_rate=config.COMMON_LEARNING_RATE)
    full_model.compile(optimizer=optimizer, loss='mse')
    X_train_processed = preprocess_input_fn(X_train_raw * 255.0)
    X_val_processed = preprocess_input_fn(X_val_raw * 255.0)
    history = full_model.fit(X_train_processed, y_train_raw, epochs=config.COMMON_EPOCHS, 
        batch_size=config.COMMON_BATCH_SIZE, validation_data=(X_val_processed, y_val_raw), verbose=1)
    
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
    
    mape = mean_absolute_percentage_error(y_test_tgt, y_pred)
    print(f"Test MAPE for {target_name} ({arch_name}): {mape:.2f}%")
    
    return best_svr_model, mape

def main():
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    # 1. Muat dan bagi data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_data()

    all_models_mape = {}
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
        cnn_model_path = os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_feature_extractor.keras")
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
        scaler_path = os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler untuk {arch_name} disimpan di: {scaler_path}")
        
        # 6. Latih, evaluasi, dan SIMPAN model SVR
        current_model_mape = {}
        
        # SVR untuk Nitrogen (N)
        svr_n, mape_n = train_and_evaluate_svr_with_tuning(
            X_train_features_scaled, y_train[:, 0], X_test_features_scaled, y_test[:, 0], "Nitrogen (N)", arch_name
        )
        joblib.dump(svr_n, os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_svr_n.joblib"))
        current_model_mape["Nitrogen (N)"] = mape_n
        
        # SVR untuk Fosfor (P)
        svr_p, mape_p = train_and_evaluate_svr_with_tuning(
            X_train_features_scaled, y_train[:, 1], X_test_features_scaled, y_test[:, 1], "Fosfor (P)", arch_name
        )
        joblib.dump(svr_p, os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_svr_p.joblib"))
        current_model_mape["Fosfor (P)"] = mape_p
        
        # SVR untuk Kalium (K)
        svr_k, mape_k = train_and_evaluate_svr_with_tuning(
            X_train_features_scaled, y_train[:, 2], X_test_features_scaled, y_test[:, 2], "Kalium (K)", arch_name
        )
        joblib.dump(svr_k, os.path.join(config.MODEL_SAVE_DIR, f"{arch_name}_svr_k.joblib"))
        current_model_mape["Kalium (K)"] = mape_k

        all_models_mape[arch_name] = current_model_mape

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
        plt.savefig(f"{arch_name}_training_loss.png") # Simpan plot
        plt.show()

    # Plotting Perbandingan MAPE
    if all_models_mape:
        mape_df = pd.DataFrame(all_models_mape).T
        mape_df_melted = mape_df.reset_index().melt(id_vars='index', var_name='Nutrient', value_name='MAPE')
        mape_df_melted.rename(columns={'index': 'Architecture'}, inplace=True)
        
        plt.figure(figsize=(14, 8))
        # Simpan Axes object dari barplot untuk anotasi
        ax = sns.barplot(data=mape_df_melted, x='Architecture', y='MAPE', hue='Nutrient', palette='viridis')
        
        plt.title('Perbandingan Mean Absolute Percentage Error (MAPE)')
        plt.ylabel('MAPE (%)')
        plt.xlabel('Arsitektur CNN (Feature Extractor)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Loop melalui setiap 'container' (setiap set batang untuk N, P, dan K)
        for container in ax.containers:
            ax.bar_label(container, 
                        labels=[f'{v.get_height():.2f}%' for v in container], 
                        padding=3, # Jarak vertikal antara batang dan teks
                        fontsize=9,
                        fontweight='bold')

        # Atur batas atas sumbu y agar ada ruang untuk label
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

        plt.savefig("mape_comparison.png") # Simpan plot
        plt.show()

if __name__ == '__main__':
    main()