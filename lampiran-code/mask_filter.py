def apply_3x3_filter(image_path, save_path, kernel):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception(f"Gambar tidak ditemukan di path: {image_path}")
    height, width = img.shape
    output = np.zeros((height, width), dtype=np.float32)
    padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)

    # Operasi konvolusi manual
    for i in range(height):
        for j in range(width):
            region = padded[i:i+3, j:j+3]
            output[i, j] = np.sum(region * kernel)

    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    # Tampilkan hasil
    cv2.imshow("Original", img)
    cv2.imshow("Filtered", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(save_path, output)
    print(f"Gambar hasil filter disimpan di: {save_path}")