import os
import numpy as np
import joblib
import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt

# Import konfigurasi untuk dimensi gambar
import config
# Import fungsi preprocessing spesifik model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

# python predict.py --image img_predict/7.jpg --architecture CustomCNN
# python predict.py --image img_predict/7.jpg --architecture VGG16
# python predict.py --image img_predict/7.jpg --architecture ResNet50
# python predict.py --image img_predict/7.jpg --architecture InceptionV3

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

def generate_gradcam_on_feature(model_extractor, img_array_expanded, element_name, target_feature_index, last_conv_layer_name, architecture_name):
    element = element_name.upper()
    print(f"\nGenerasi Grad-CAM untuk {element} (Target Fitur Laten: {target_feature_index})...")

    grad_model = tf.keras.models.Model(
        [model_extractor.inputs],
        [model_extractor.get_layer(last_conv_layer_name).output, model_extractor.output]
    )

    with tf.GradientTape() as tape:
        last_conv_output, feature_output = grad_model(img_array_expanded)
        target_output = feature_output[:, target_feature_index]
    
    grads = tape.gradient(target_output, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    max_heatmap = tf.math.reduce_max(heatmap)
    if max_heatmap > 0:
        heatmap = tf.maximum(heatmap, 0) / max_heatmap
    else:
        heatmap = tf.zeros_like(heatmap)
        
    heatmap = heatmap.numpy()

    img = np.squeeze(img_array_expanded)
    
    if img.max() > 1.0:
        img_display = img.astype('float32') / 255.0
    else:
        img_display = img
    
    heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    
    heatmap_resized = np.uint8(255 * heatmap_resized) 
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    superimposed_img = heatmap_colored * 0.4 + img_display * 255 * 0.6 
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Grad-CAM {element} (Aktivasi Fitur Laten {target_feature_index}) - {architecture_name}")
    plt.axis('off')
    plt.show()

def predict_npk(image_path, architecture_name):
    """
    Memuat model yang telah dilatih dan memprediksi nilai NPK dari satu gambar.
    Menggunakan extractor dan scaler terpisah untuk N, P, dan K.

    Args:
        image_path (str): Path ke file gambar.
        architecture_name (str): Nama arsitektur model yang akan digunakan
                                 ("CustomCNN", "VGG16", dll.).

    Returns:
        tuple: (dict prediksi NPK, dict data fitur dan gambar)
    """
    # 1. Tentukan path ke model-model yang disimpan
    model_dir = config.BEST_MODEL_DIR
    
    model_components = {}
    elements = ['n', 'p', 'k']
    
    all_paths_exist = True
    
    for element in elements:
        cnn_path = os.path.join(model_dir, f"{architecture_name}_feature_extractor_{element}.keras")
        scaler_path = os.path.join(model_dir, f"{architecture_name}_scaler_{element}.joblib")
        svr_path = os.path.join(model_dir, f"{architecture_name}_svr_{element}.joblib")

        for path in [cnn_path, scaler_path, svr_path]:
            if not os.path.exists(path):
                print(f"Error: File model tidak ditemukan di {path}")
                print(f"Pastikan Anda telah menjalankan 'train.py' yang menghasilkan model per unsur untuk arsitektur '{architecture_name}'.")
                all_paths_exist = False
        
        model_components[element] = {
            'cnn_path': cnn_path,
            'scaler_path': scaler_path,
            'svr_path': svr_path
        }
        
    if not all_paths_exist:
        return None, None

    print("Memuat model...")
    loaded_models = {}
    try:
        for element in elements:
            paths = model_components[element]
            # Harus menggunakan compile=False jika model tidak memiliki optimizer/loss
            loaded_models[f'cnn_{element}'] = tf.keras.models.load_model(paths['cnn_path'], compile=False) 
            loaded_models[f'scaler_{element}'] = joblib.load(paths['scaler_path'])
            loaded_models[f'svr_{element}'] = joblib.load(paths['svr_path'])
        print("Model berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return None, None


    # 3. Muat dan proses gambar input
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # img_array_expanded adalah gambar sebelum preprocessing
        img_array_expanded = np.expand_dims(img_array, axis=0) 
        
        preprocess_fn = get_preprocessing_function(architecture_name)
        # img_preprocessed adalah gambar yang sudah diproses (siap masuk CNN)
        img_preprocessed = preprocess_fn(img_array_expanded) 

    except Exception as e:
        print(f"Gagal memuat atau memproses gambar: {e}")
        return None, None

    # 4. Lakukan pipeline prediksi (Iterasi per unsur)
    predictions = {}
    interpretation_data = {} # Simpan data untuk Grad-CAM

    for element in elements:
        cnn_extractor = loaded_models[f'cnn_{element}']
        scaler = loaded_models[f'scaler_{element}']
        svr = loaded_models[f'svr_{element}']
        
        # Ekstrak fitur
        features = cnn_extractor.predict(img_preprocessed)
        features_scaled = scaler.transform(features)
        
        # Prediksi
        pred = svr.predict(features_scaled)[0]
        predictions[element.upper()] = pred 
        
        # --- LOGIKA BARU: MENCARI FITUR LATEN AKTIVASI TERTINGGI (PENGGANTI SHAP) ---
        # Kita menggunakan fitur yang BELUM diskala (features)
        # np.argmax(np.abs(features[0])): mencari indeks fitur dengan nilai absolut tertinggi
        target_feature_index = np.argmax(np.abs(features[0]))
        # --------------------------------------------------------------------------
        
        # Simpan data untuk interpretasi
        # Dapatkan nama lapisan konvolusi terakhir. Ini sangat tergantung pada arsitektur!
        if architecture_name == "CustomCNN":
            # Asumsi 4 lapisan dari akhir, ganti jika berbeda
            last_conv_layer_name = cnn_extractor.layers[-4].name 
        elif architecture_name == "VGG16":
            last_conv_layer_name = 'block5_conv3'
        elif architecture_name == "ResNet50":
            last_conv_layer_name = 'conv5_block3_out'
        elif architecture_name == "InceptionV3":
            last_conv_layer_name = 'mixed10'
        else:
            last_conv_layer_name = None 

        interpretation_data[element] = {
            'cnn_extractor': cnn_extractor,
            # Gunakan data yang sudah diproses (img_preprocessed) untuk dimasukkan ke grad_model
            'img_array_expanded': img_preprocessed, 
            'last_conv_layer_name': last_conv_layer_name,
            'target_feature_index': target_feature_index, # Simpan indeks yang baru ditemukan
        }

    return predictions, interpretation_data