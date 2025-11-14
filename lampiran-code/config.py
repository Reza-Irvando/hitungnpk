# --- Konfigurasi Path ---
DATA_FOLDER = 'data'
IMAGES_SUBFOLDER = DATA_FOLDER
CSV_FILE_PATH = os.path.join(DATA_FOLDER, 'label.csv')
MODEL_SAVE_DIR = 'saved_models'

# --- Konfigurasi Gambar ---
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3

# --- Konfigurasi Pelatihan Model ---
COMMON_EPOCHS = 10
COMMON_BATCH_SIZE = 32
COMMON_LEARNING_RATE = 0.001
COMMON_DROPOUT_RATE = 0.25

# --- Konfigurasi Arsitektur ---
MODEL_ARCHITECTURES = ["CustomCNN", "VGG16", "ResNet50", "InceptionV3"]

# --- Konfigurasi Split Data ---
TRAIN_SPLIT = 0.6
VALIDATION_SPLIT = 0.2
# Sisa data akan menjadi test split