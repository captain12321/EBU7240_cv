# Bilateral filtering without OpenCV
import numpy as np
import cv2
import sys
import math


#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# You can define functions here
# NO OPENCV FUNCTION IS ALLOWED HERE
def bilateral_filter_gray(image, W_size, sigma_color, sigma_space):
    radius=W_size//2
    image=cv2.resize(image, (128, 128))
    H, W = image.shape
    C = 1 if len(image.shape) == 2 else image.shape[2]
    image = image.reshape(H, W, C)
    output_image = image.copy()
    for i in range(radius, H - radius):
        for j in range(radius, W - radius):
            for k in range(C):
                weight_sum = 0.0
                pixel_sum = 0.0
                for x in range(-radius, radius + 1):
                    for y in range(-radius, radius + 1):
                        spatial_weight = -(x ** 2 + y ** 2) / (2 * (sigma_space ** 2))
                        color_weight = -(int(image[i][j][k]) - int(image[i + x][j + y][k])) ** 2 / (2 * (sigma_color ** 2))
                        weight = np.exp(spatial_weight + color_weight)
                        weight_sum += weight
                        pixel_sum += (weight * image[i + x][j + y][k])
                # normalize
                value = pixel_sum / weight_sum
                output_image[i][j][k] = value
                print(output_image[i][j][k], '->', value)
    return output_image.astype(np.uint8)



##########################################################################################


im_gray = cv2.imread('../inputs/cat.png',0)

result_bf1 = bilateral_filter_gray(im_gray, 11, 30.0, 3.0)
result_bf2 = bilateral_filter_gray(im_gray, 11, 30.0, 30.0)
result_bf3 = bilateral_filter_gray(im_gray, 11, 100.0, 3.0)
result_bf4 = bilateral_filter_gray(im_gray, 11, 100.0, 30.0)
result_bf5 = bilateral_filter_gray(im_gray, 5, 100.0, 30.0)

result_bf1 = np.uint8(result_bf1)
result_bf2 = np.uint8(result_bf2)
result_bf3 = np.uint8(result_bf3)
result_bf4 = np.uint8(result_bf4)
result_bf5 = np.uint8(result_bf5)


cv2.imwrite('../results/ex2b_bf_11_30_3.jpg', result_bf1)
cv2.imwrite('../results/ex2b_bf_11_30_30.jpg', result_bf2)
cv2.imwrite('../results/ex2b_bf_11_100_3.jpg', result_bf3)
cv2.imwrite('../results/ex2b_bf_11_100_30.jpg', result_bf4)
cv2.imwrite('../results/ex2b_bf_5_100_30.jpg', result_bf5)