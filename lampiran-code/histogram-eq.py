def histogram_equalization_image(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    equalized_img = cv2.equalizeHist(img)
    cv2.imwrite(output_path, equalized_img)