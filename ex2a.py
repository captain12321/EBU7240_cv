# Gaussian filtering without OpenCV
import numpy as np
import cv2

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# You can define functions here
# NO OPENCV FUNCTION IS ALLOWED HERE
# Gaussian filter

def gaussian_filter_gray(img, K_size, sigma):
    H, W= img.shape
    ## Zero padding

    pad = K_size // 2

    out = np.zeros((H + pad * 2, W + pad * 2), dtype=float)

    out[pad: pad + H, pad: pad + W] = img.copy().astype(float)

    ## prepare Kernel

    K = np.zeros((K_size, K_size), dtype=float)

    for x in range(-pad, -pad + K_size):

        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    K /= (2 * np.pi * sigma * sigma)

    K /= K.sum()

    tmp = out.copy()

    # filtering

    for y in range(H):

        for x in range(W):
                out[pad + y, pad + x] = np.sum(K * tmp[y: y + K_size, x: x + K_size])

    out = np.clip(out, 0, 255)
    #remove padding
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out

##########################################################################################


im_gray = cv2.imread('../inputs/lena.jpg', 0)
im_gray = cv2.resize(im_gray, (256, 256))

result_gf1 = gaussian_filter_gray(im_gray, 5, 1.0)
result_gf2 = gaussian_filter_gray(im_gray, 5, 10.0)
result_gf3 = gaussian_filter_gray(im_gray, 11, 1.0)
result_gf4 = gaussian_filter_gray(im_gray, 11, 10.0)

result_gf1 = np.uint8(result_gf1)
result_gf2 = np.uint8(result_gf2)
result_gf3 = np.uint8(result_gf3)
result_gf4 = np.uint8(result_gf4)

cv2.imwrite('../results/ex2a_gf_5_1.jpg', result_gf1)
cv2.imwrite('../results/ex2a_gf_5_10.jpg', result_gf2)
cv2.imwrite('../results/ex2a_gf_11_1.jpg', result_gf3)
cv2.imwrite('../results/ex2a_gf_11_10.jpg', result_gf4)
