def crop_with_shifts(image_path, output_prefix, x_center, y_center, crop_size, shift_pixels=20, num_crops=10):
    img = cv2.imread(image_path)
    h, w = img.shape
    half = crop_size // 2
    
    # Daftar 10 pergeseran (dx, dy) dari titik tengah
    shifts = [
        (0, 0),                         # 1. Tengah (Asli)
        (-shift_pixels, 0),             # 2. Kiri
        (shift_pixels, 0),              # 3. Kanan
        (0, -shift_pixels),             # 4. Atas
        (0, shift_pixels),              # 5. Bawah
        (-shift_pixels, -shift_pixels), # 6. Kiri-Atas
        (shift_pixels, -shift_pixels),  # 7. Kanan-Atas
        (-shift_pixels, shift_pixels),  # 8. Kiri-Bawah
        (shift_pixels, shift_pixels),   # 9. Kanan-Bawah
        (0, -shift_pixels * 2)          # 10. Atas (lebih jauh)
    ]
    
    shifts = shifts[:num_crops]

    base_name, ext = os.path.splitext(output_prefix)

    for i, (dx, dy) in enumerate(shifts):
        new_x_center = x_center + dx
        new_y_center = y_center + dy
        
        x_start = max(0, new_x_center - half)
        y_start = max(0, new_y_center - half)
        x_end = x_start + crop_size
        y_end = y_start + crop_size

        if x_end > w:
            x_end = w
            x_start = max(0, w - crop_size)
        if y_end > h:
            y_end = h
            y_start = max(0, h - crop_size)

        cropped_img = img[y_start:y_end, x_start:x_end]
        output_path = f"{base_name}_{i}{ext}"
        
        cv2.imwrite(output_path, cropped_img)