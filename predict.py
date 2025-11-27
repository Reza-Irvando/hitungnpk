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
        # CustomCNN kita hanya butuh normalisasi 0-1
        return lambda x: x / 255.0
    else:
        raise ValueError(f"Arsitektur '{architecture_name}' tidak dikenali.")

def predict_npk(image_path, architecture_name):
    """
    Memuat model yang telah dilatih dan memprediksi nilai NPK dari satu gambar.
    Menggunakan extractor dan scaler terpisah untuk N, P, dan K.

    Args:
        image_path (str): Path ke file gambar.
        architecture_name (str): Nama arsitektur model yang akan digunakan
                                ("CustomCNN", "VGG16", dll.).

    Returns:
        dict: Berisi nilai prediksi untuk N, P, dan K.
    """
    # 1. Tentukan path ke model-model yang disimpan
    model_dir = config.BEST_MODEL_DIR
    
    # Definisikan mapping untuk setiap unsur (N, P, K)
    model_components = {}
    elements = ['n', 'p', 'k']
    
    all_paths_exist = True
    
    for element in elements:
        # Tentukan path untuk Extractor, Scaler, dan SVR
        cnn_path = os.path.join(model_dir, f"{architecture_name}_feature_extractor_{element}.keras")
        scaler_path = os.path.join(model_dir, f"{architecture_name}_scaler_{element}.joblib")
        svr_path = os.path.join(model_dir, f"{architecture_name}_svr_{element}.joblib")

        # Periksa keberadaan file
        for path in [cnn_path, scaler_path, svr_path]:
            if not os.path.exists(path):
                print(f"Error: File model tidak ditemukan di {path}")
                print(f"Pastikan Anda telah menjalankan 'train.py' yang menghasilkan model per unsur untuk arsitektur '{architecture_name}'.")
                all_paths_exist = False
        
        # Simpan path
        model_components[element] = {
            'cnn_path': cnn_path,
            'scaler_path': scaler_path,
            'svr_path': svr_path
        }
        
    if not all_paths_exist:
        return None

    # 2. Muat semua model dan scaler
    print("Memuat model...")
    loaded_models = {}
    try:
        for element in elements:
            paths = model_components[element]
            loaded_models[f'cnn_{element}'] = tf.keras.models.load_model(paths['cnn_path'])
            loaded_models[f'scaler_{element}'] = joblib.load(paths['scaler_path'])
            loaded_models[f'svr_{element}'] = joblib.load(paths['svr_path'])
        print("Model berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return None


    # 3. Muat dan proses gambar input (Preprocessing dasar tetap sama)
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Dapatkan fungsi preprocessing yang benar
        preprocess_fn = get_preprocessing_function(architecture_name)
        img_preprocessed = preprocess_fn(img_array_expanded)

    except Exception as e:
        print(f"Gagal memuat atau memproses gambar: {e}")
        return None

    # 4. Lakukan pipeline prediksi (Iterasi per unsur)
    predictions = {}
    
    for element in elements:
        # Ekstrak fitur, skalakan, dan prediksi menggunakan model spesifik unsur
        cnn_extractor = loaded_models[f'cnn_{element}']
        scaler = loaded_models[f'scaler_{element}']
        svr = loaded_models[f'svr_{element}']
        
        # Ekstrak fitur menggunakan CNN spesifik unsur
        features = cnn_extractor.predict(img_preprocessed)
        
        # Skalakan fitur menggunakan scaler spesifik unsur
        features_scaled = scaler.transform(features)
        
        # Prediksi menggunakan SVR spesifik unsur
        pred = svr.predict(features_scaled)[0]
        predictions[element.upper()] = pred # Simpan hasil dengan kunci N, P, K

    return predictions


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
        print(f"  > Prediksi Nitrogen (N): {predictions['N']:.2f}")
        print(f"  > Prediksi Fosfor (P):   {predictions['P']:.2f}")
        print(f"  > Prediksi Kalium (K):   {predictions['K']:.2f}")
        print("------------------------")

# python predict.py --image img_predict/7.jpg --architecture CustomCNN
# python predict.py --image img_predict/7.jpg --architecture VGG16
# python predict.py --image img_predict/7.jpg --architecture ResNet50
# python predict.py --image img_predict/7.jpg --architecture InceptionV3