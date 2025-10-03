# predict.py
"""
Skrip untuk memuat model yang telah dilatih dan membuat prediksi NPK pada gambar baru.
"""
import os
import numpy as np
import joblib
import tensorflow as tf
import argparse

# Import konfigurasi untuk dimensi gambar
import config
# Import fungsi preprocessing spesifik model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3


def get_preprocessing_function(architecture_name):
    """Mengembalikan fungsi preprocessing yang sesuai berdasarkan nama arsitektur."""
    if architecture_name == "VGG16":
        return preprocess_input_vgg16
    elif architecture_name == "ResNet50":
        return preprocess_input_resnet50
    elif architecture_name == "InceptionV3":
        return preprocess_input_inceptionv3
    elif architecture_name == "CustomCNN":
        # CustomCNN kita hanya butuh normalisasi 0-1, yang sudah dilakukan saat load
        return lambda x: x / 255.0
    else:
        raise ValueError(f"Arsitektur '{architecture_name}' tidak dikenali.")

def predict_npk(image_path, architecture_name):
    """
    Memuat model yang telah dilatih dan memprediksi nilai NPK dari satu gambar.

    Args:
        image_path (str): Path ke file gambar.
        architecture_name (str): Nama arsitektur model yang akan digunakan
                                 ("CustomCNN", "VGG16", dll.).

    Returns:
        dict: Berisi nilai prediksi untuk N, P, dan K.
    """
    # 1. Tentukan path ke model-model yang disimpan
    model_dir = config.MODEL_SAVE_DIR
    cnn_path = os.path.join(model_dir, f"{architecture_name}_feature_extractor.keras")
    scaler_path = os.path.join(model_dir, f"{architecture_name}_scaler.joblib")
    svr_n_path = os.path.join(model_dir, f"{architecture_name}_svr_n.joblib")
    svr_p_path = os.path.join(model_dir, f"{architecture_name}_svr_p.joblib")
    svr_k_path = os.path.join(model_dir, f"{architecture_name}_svr_k.joblib")

    # Periksa apakah semua file model ada
    for path in [cnn_path, scaler_path, svr_n_path, svr_p_path, svr_k_path]:
        if not os.path.exists(path):
            print(f"Error: File model tidak ditemukan di {path}")
            print(f"Pastikan Anda telah menjalankan 'train.py' untuk arsitektur '{architecture_name}'.")
            return None

    # 2. Muat semua model dan scaler
    print("Memuat model...")
    cnn_extractor = tf.keras.models.load_model(cnn_path)
    scaler = joblib.load(scaler_path)
    svr_n = joblib.load(svr_n_path)
    svr_p = joblib.load(svr_p_path)
    svr_k = joblib.load(svr_k_path)
    print("Model berhasil dimuat.")

    # 3. Muat dan proses gambar input
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Dapatkan dan terapkan fungsi preprocessing yang benar
        preprocess_fn = get_preprocessing_function(architecture_name)
        img_preprocessed = preprocess_fn(img_array_expanded)

    except Exception as e:
        print(f"Gagal memuat atau memproses gambar: {e}")
        return None

    # 4. Lakukan pipeline prediksi
    # Ekstrak fitur menggunakan CNN
    features = cnn_extractor.predict(img_preprocessed)
    
    # Skalakan fitur menggunakan scaler yang telah dilatih
    features_scaled = scaler.transform(features)
    
    # Prediksi setiap nilai NPK menggunakan model SVR
    pred_n = svr_n.predict(features_scaled)[0]
    pred_p = svr_p.predict(features_scaled)[0]
    pred_k = svr_k.predict(features_scaled)[0]

    return {"N": pred_n, "P": pred_p, "K": pred_k}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prediksi NPK dari gambar menggunakan model terlatih.")
    parser.add_argument("--image", type=str, required=True, help="Path ke gambar yang akan diprediksi.")
    parser.add_argument("--architecture", type=str, required=True, choices=config.MODEL_ARCHITECTURES,
                        help="Arsitektur CNN yang modelnya akan digunakan.")
    
    args = parser.parse_args()

    predictions = predict_npk(args.image, args.architecture)

    if predictions:
        print("\n--- Hasil Prediksi ---")
        print(f"Gambar: {os.path.basename(args.image)}")
        print(f"Model: {args.architecture}")
        print(f"  > Prediksi Nitrogen (N): {predictions['N']:.2f}")
        print(f"  > Prediksi Fosfor (P):   {predictions['P']:.2f}")
        print(f"  > Prediksi Kalium (K):   {predictions['K']:.2f}")
        print("------------------------")

# Menggunakan model CustomCNN untuk memprediksi gambar dari folder data
# python predict.py --image data/image1.jpg --architecture CustomCNN

# Menggunakan model VGG16
# python predict.py --image /path/to/another/test_image.jpg --architecture VGG16