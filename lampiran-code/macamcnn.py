if base_model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_vgg16
    elif base_model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_resnet50
    elif base_model_name == "InceptionV3":
        if input_shape[0] < 299 or input_shape[1] < 299:
            print(f"Peringatan: InceptionV3 bekerja optimal dengan input 299x299.")
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess_input_fn = preprocess_input_inceptionv3
    elif base_model_name == "CustomCNN":
        input_tensor = Input(shape=input_shape)
        # Tahap 1: 2x Conv(8) -> Pool
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_tensor)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        # Tahap 2: 2x Conv(16) -> Pool
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        # Tahap 3: 3x Conv(16) -> Pool
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        # Tahap 4: 3x Conv(32) -> Pool
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        # Tahap 5: 3x Conv(32) -> Pool
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x) 
        x = Dropout(config.COMMON_DROPOUT_RATE)(x)
        # Tahap 6: 1x Conv(16) 7x7 (padding='valid' untuk 7x7 -> 1x1)
        x = Conv2D(16, (7, 7), activation='relu', padding='valid')(x)
        x = Flatten()(x) # Output
        # Layer output dari feature extractor
        feature_output_layer = Dense(128, activation='relu', name='feature_extractor_output')(x)
        feature_output_layer = Dropout(config.COMMON_DROPOUT_RATE)(feature_output_layer)
        # Model lengkap untuk regresi end-to-end
        full_model_output = Dense(output_dim, activation='linear')(feature_output_layer)
        full_model = Model(inputs=input_tensor, outputs=full_model_output)

        # Model feature extractor terpisah
        feature_extractor = Model(inputs=input_tensor, outputs=full_model.get_layer('feature_extractor_output').output)

        preprocess_input_fn = lambda x: x

        optimizer = Adam(learning_rate=config.COMMON_LEARNING_RATE)
        full_model.compile(optimizer=optimizer, loss='mse')
        
        history = full_model.fit(X_train_raw, y_train_raw, 
            epochs=config.COMMON_EPOCHS, batch_size=config.COMMON_BATCH_SIZE, 
            validation_data=(X_val_raw, y_val_raw), verbose=1)
        
        return feature_extractor, preprocess_input_fn, history