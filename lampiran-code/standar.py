def standardize_image(image_path, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception(f"Gambar tidak ditemukan di path: {image_path}")

    mean = np.mean(img)
    std = np.std(img)
    standardized = (img - mean) / (std + 1e-8)
    standardized_rescaled = cv2.normalize(standardized, None, 0, 255, cv2.NORM_MINMAX)
    standardized_rescaled = standardized_rescaled.astype(np.uint8)

    cv2.imwrite(save_path, standardized_rescaled)
    print(f"Gambar hasil disimpan di: {save_path}")