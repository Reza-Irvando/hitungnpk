import cv2
import numpy as np

# Baca citra
image = cv2.imread('data1.jpg')

# Pisahkan komponen RGB
B, G, R = cv2.split(image)

# Buat masker untuk warna hijau
green_mask = (G > R) & (G > B)

print(green_mask)
# Terapkan masker ke citra asli
green_only = np.zeros_like(image)
green_only[green_mask] = image[green_mask]

cv2.imshow('Green Color', green_only)
cv2.waitKey(0)
cv2.destroyAllWindows()