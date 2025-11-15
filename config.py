"""
File konfigurasi untuk menyimpan semua variabel global dan hyperparameter.
"""
import os

# --- Konfigurasi Path ---
DATA_FOLDER = 'data'
IMAGES_SUBFOLDER = DATA_FOLDER
CSV_FILE_PATH = os.path.join(DATA_FOLDER, 'label.csv')
MODEL_SAVE_DIR = 'saved_models' # Folder untuk menyimpan semua model yang telah dilatih

# --- Konfigurasi Gambar ---
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3

# --- Konfigurasi Pelatihan Model ---
# Hyperparameter yang akan digunakan untuk melatih 'head' CNN atau seluruh CustomCNN
COMMON_EPOCHS = 10
COMMON_BATCH_SIZE = 32
COMMON_LEARNING_RATE = 0.0001
COMMON_DROPOUT_RATE = 0.25

# --- Konfigurasi Arsitektur ---
# Ganti list ini untuk melatih arsitektur yang berbeda.
# Pilihan: ["CustomCNN", "VGG16", "ResNet50", "InceptionV3"]
MODEL_ARCHITECTURES = ["ResNet50"]

# --- Konfigurasi Split Data ---
TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.2
# Sisa data akan menjadi test split