# Median filtering without OpenCV
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# You can define functions here
# NO OPENCV FUNCTION IS ALLOWED HERE

def median_filter_gray(img, size):
    height = img.shape[0]
    wide = img.shape[1]
    img1 = np.zeros((height, wide), np.uint8)  #for new image
    #为每一个pixel创建一个他的临接点的表
    for i in range(int(size/2), height - int(size/2)):
        for j in range(int(size/2), wide - int(size/2)):
            Adjacent_pixels = np.zeros(size * size, np.uint8)

            #store all pixel in a list
            s = 0
            for k in range(-1 * int(size / 2), int(size / 2)+1):
                for l in range(-1 * int(size / 2), int(size / 2)+1):
                    Adjacent_pixels[s] = img[i + k, j + l]
                    s += 1
            Adjacent_pixels.sort()  # find for median value
            median = Adjacent_pixels[int((size * size - 1) / 2)]  # 将中值代替原来的中心值
            img1[i, j] = median
    return img1

##########################################################################################

im_gray = cv2.imread('../inputs/cat.png', 0)
im_gray = cv2.resize(im_gray, (256, 256))

gaussian_noise = np.zeros((im_gray.shape[0], im_gray.shape[1]), dtype=np.uint8)  #
gaussian_noise = cv2.randn(gaussian_noise, 128, 20)

uniform_noise = np.zeros((im_gray.shape[0], im_gray.shape[1]), dtype=np.uint8)
uniform_noise = cv2.randu(uniform_noise, 0, 255)
ret, impulse_noise = cv2.threshold(uniform_noise, 220, 255, cv2.THRESH_BINARY)

gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
impulse_noise = impulse_noise.astype(np.uint8)

imnoise_gaussian = cv2.add(im_gray, gaussian_noise)
imnoise_impulse = cv2.add(im_gray, impulse_noise)

cv2.imwrite('../results/ex2c_gnoise.jpg', np.uint8(imnoise_gaussian))
cv2.imwrite('../results/ex2c_inoise.jpg', np.uint8(imnoise_impulse))

result_original_mf = median_filter_gray(im_gray, 5)
result_gaussian_mf = median_filter_gray(imnoise_gaussian, 5)
result_impulse_mf = median_filter_gray(imnoise_impulse, 5)

cv2.imwrite('../results/ex2c_original_median_5.jpg', np.uint8(result_original_mf))
cv2.imwrite('../results/ex2c_gnoise_median_5.jpg', np.uint8(imnoise_gaussian))
cv2.imwrite('../results/ex2c_inoise_median_5.jpg', np.uint8(result_impulse_mf))

#
result_original_mf = median_filter_gray(im_gray, 11)
result_gaussian_mf = median_filter_gray(imnoise_gaussian, 11)
result_impulse_mf = median_filter_gray(imnoise_impulse, 11)

cv2.imwrite('../results/ex2c_original_median_11.jpg', np.uint8(result_original_mf))
cv2.imwrite('../results/ex2c_gnoise_median_11.jpg', np.uint8(result_gaussian_mf))
cv2.imwrite('../results/ex2c_inoise_median_11.jpg', np.uint8(result_impulse_mf))

