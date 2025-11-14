def load_and_split_data():
    df_npk = pd.read_csv(config.CSV_FILE_PATH)
    df_npk.columns = df_npk.columns.str.strip().str.upper()

    all_images = []
    all_npk_values = []

    for index, row in df_npk.iterrows():
        try:
            filename = str(row['FILENAME'])
            image_path = os.path.join(config.IMAGES_SUBFOLDER, filename)

            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array /= 255.0
            all_images.append(img_array)

            npk_value = [
                pd.to_numeric(row['N'], errors='coerce'),
                pd.to_numeric(row['P'], errors='coerce'),
                pd.to_numeric(row['K'], errors='coerce')
            ]

            all_npk_values.append(npk_value)

    X = np.array(all_images)
    y = np.array(all_npk_values)

    # --- Split data berdasarkan urutan ---
    total_samples = len(X)
    train_size = int(config.TRAIN_SPLIT * total_samples)
    val_size = int(config.VALIDATION_SPLIT * total_samples)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)