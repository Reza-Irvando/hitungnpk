def convert_image_to_grayscale(input_path, output_path, save_as_npy=False):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Gagal membaca gambar dari: {input_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)