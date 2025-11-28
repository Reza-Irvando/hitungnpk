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

# --- FUNGSI BARU UNTUK VISUALISASI GRAD-CAM (TERMASUK REVISI) ---
def generate_gradcam_on_feature(model_extractor, img_array_expanded, element_name, target_feature_index, last_conv_layer_name, architecture_name):
    """
    Menghasilkan heatmap Grad-CAM yang menunjukkan area gambar mana
    yang paling mengaktifkan fitur laten tertentu (target_feature_index).
    """
    element = element_name.upper()
    print(f"\nGenerasi Grad-CAM untuk {element} (Target Fitur Laten: {target_feature_index})...")

    # 1. Dapatkan model yang menghasilkan output dari lapisan konvolusi terakhir dan fitur laten
    grad_model = tf.keras.models.Model(
        [model_extractor.inputs],
        [model_extractor.get_layer(last_conv_layer_name).output, model_extractor.output]
    )

    # 2. Hitung gradien
    with tf.GradientTape() as tape:
        last_conv_output, feature_output = grad_model(img_array_expanded)
        # Targetkan output yang ingin dijelaskan (fitur laten spesifik)
        target_output = feature_output[:, target_feature_index]
    
    # Gradien dari fitur laten spesifik terhadap output lapisan konvolusi terakhir
    grads = tape.gradient(target_output, last_conv_output)

    # Vektor rata-rata gradien (bobot aktivasi)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 3. Hitung Heatmap
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisasi heatmap (0-1) dan penanganan RuntimeWarning
    max_heatmap = tf.math.reduce_max(heatmap)
    if max_heatmap > 0:
        heatmap = tf.maximum(heatmap, 0) / max_heatmap
    else:
        # Jika semua nilai nol, set heatmap ke nol
        heatmap = tf.zeros_like(heatmap)
        
    heatmap = heatmap.numpy()

    # 4. Visualisasi (TERMASUK PERBAIKAN PENANGANAN GAMBAR)
    
    # Muat gambar input yang sudah diproses
    img = np.squeeze(img_array_expanded)
    
    # Normalisasi untuk tampilan (skala 0-1)
    if img.max() > 1.0:
         # Asumsi input adalah [0, 255] jika max > 1.0
         img_display = img.astype('float32') / 255.0
    else:
         # Asumsi input sudah [0, 1] jika max <= 1.0
         img_display = img
    
    # Resize heatmap ke ukuran gambar asli
    heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    
    # Ubah skala heatmap ke 0-255 dan aplikasikan colormap
    heatmap_resized = np.uint8(255 * heatmap_resized) 
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Overlay heatmap pada gambar asli
    # Kedua elemen harus diskalakan ke 0-255 untuk operasi overlay
    superimposed_img = heatmap_colored * 0.4 + img_display * 255 * 0.6 
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    plt.figure(figsize=(8, 8))
    # Konversi dari BGR (output OpenCV colormap) ke RGB (ekspektasi Matplotlib)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    # 'architecture_name' sekarang tersedia
    plt.title(f"Grad-CAM {element} (Aktivasi Fitur Laten {target_feature_index}) - {architecture_name}")
    plt.axis('off')
    plt.show()

def get_preprocessing_function(architecture_name):
    if architecture_name == "VGG16":
        return preprocess_input_vgg16
    elif architecture_name == "ResNet50":
        return preprocess_input_resnet50
    elif architecture_name == "InceptionV3":
        return preprocess_input_inceptionv3
    elif architecture_name == "CustomCNN":
        return lambda x: x / 255.0
    else:
        raise ValueError(f"Arsitektur '{architecture_name}' tidak dikenali.")
    
def predict_npk(image_path, architecture_name):
    model_dir = config.BEST_MODEL_DIR
    model_components = {}
    elements = ['n', 'p', 'k']
    
    all_paths_exist = True
    
    for element in elements:
        cnn_path = os.path.join(model_dir, f"{architecture_name}_feature_extractor_{element}.keras")
        scaler_path = os.path.join(model_dir, f"{architecture_name}_scaler_{element}.joblib")
        svr_path = os.path.join(model_dir, f"{architecture_name}_svr_{element}.joblib")

        model_components[element] = {
            'cnn_path': cnn_path,
            'scaler_path': scaler_path,
            'svr_path': svr_path
        }
        
    if not all_paths_exist:
        return None, None

    print("Memuat model...")
    loaded_models = {}
    for element in elements:
        paths = model_components[element]
        loaded_models[f'cnn_{element}'] = tf.keras.models.load_model(paths['cnn_path'], compile=False) 
        loaded_models[f'scaler_{element}'] = joblib.load(paths['scaler_path'])
        loaded_models[f'svr_{element}'] = joblib.load(paths['svr_path'])

    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0) 
    
    preprocess_fn = get_preprocessing_function(architecture_name)
    img_preprocessed = preprocess_fn(img_array_expanded) 

    for element in elements:
        cnn_extractor = loaded_models[f'cnn_{element}']
        scaler = loaded_models[f'scaler_{element}']
        svr = loaded_models[f'svr_{element}']
        features = cnn_extractor.predict(img_preprocessed)
        features_scaled = scaler.transform(features)
        pred = svr.predict(features_scaled)[0]