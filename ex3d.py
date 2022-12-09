import numpy as np
import cv2
import math


def distance(x, y, i, j):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def matching_cost_computation(matching_cost, left_img, right_img, d_max):
    # Matching cost computation: SSD
    h, w = left_img.shape
    for y in range(h):
        for x in range(w):
            for d in range(d_max):
                if x - d >= 0:
                    temp_cost = float(left_img[y, x]) - float(right_img[y, x - d])
                    matching_cost[y, x, d] = temp_cost * temp_cost


def cost_aggregation_adaptiveweight(aggregated_cost, matching_cost, left_img, kernel, sigma_r, sigma_s, d_max):
# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
# NO OPENCV FUNCTION IS ALLOWED HERE
    radius=kernel//2
    C=1 #one channel
    H, W, _ = matching_cost.shape
    for d in range(d_max):
        input_image_cp = matching_cost[:, :, d]
        output_image = np.copy(input_image_cp)  # 输出图像
        for i in range(radius, H - radius):
            for j in range(radius, W - radius):
                    weight_sum = 0.0
                    pixel_sum = 0.0
                    for x in range(-radius, radius + 1):
                        for y in range(-radius, radius + 1):
                            spatial_weight = -(x ** 2 + y ** 2) / (2 * (sigma_s ** 2))
                            color_weight = -(int(input_image_cp[i][j]) - int(input_image_cp[i + x][j + y])) ** 2 / (
                                        2 * (sigma_r ** 2))
                            weight = np.exp(spatial_weight + color_weight)
                            weight_sum += weight
                            pixel_sum += (weight * input_image_cp[i + x][j + y])
                    # normalize
                    value = pixel_sum / weight_sum
                    output_image[i][j] = value
        aggregated_cost[:, :, d] = output_image

##########################################################################################


def disparity_estimation(disparity, aggregated_cost, d_max):
    offset_adjust = 255 / d_max  # this is used to map disparity map output to 0-255 range
    h, w = disparity.shape
    for y in range(h):
        for x in range(w):
            best_offset = 0
            prev_ssd = 65534
            for d in range(d_max):
                if aggregated_cost[y, x, d] < prev_ssd:
                    prev_ssd = aggregated_cost[y, x, d]
                    best_offset = d

            # set disparity output for this x,y location to the best match
            disparity[y, x] = best_offset * offset_adjust


if __name__ == '__main__':
    # Load left and right images and convert to grayscale for simplicity
    left_img = cv2.imread('../inputs/teddy_im2.png', 0)
    right_img = cv2.imread('../inputs/teddy_im6.png', 0)

    h, w = left_img.shape
    d_max, kernel_size = 32, 3
    sigma_r, sigma_s = 100.0, 30.0  # parameters used for bilateral filter's intensity and spatial distance

    matching_cost = np.zeros((h, w, d_max), np.float64)
    aggregated_cost = np.zeros((h, w, d_max), np.float64)
    disparity = np.zeros((h, w), np.uint8)

    matching_cost_computation(matching_cost, left_img, right_img, d_max)
    ## COMPLETE THIS FUNCTION
    cost_aggregation_adaptiveweight(aggregated_cost, matching_cost, left_img, kernel_size, sigma_r, sigma_s,d_max)
    ##
    disparity_estimation(disparity, aggregated_cost, d_max)
    cv2.imwrite('../results/ex3d_aw_3.png', disparity)
    print('done1')

    d_max, kernel_size = 32, 11

    matching_cost = np.zeros((h, w, d_max), np.float64)
    aggregated_cost = np.zeros((h, w, d_max), np.float64)
    disparity = np.zeros((h, w), np.uint8)

    matching_cost_computation(matching_cost, left_img, right_img, d_max)
    ## COMPLETE THIS FUNCTION
    cost_aggregation_adaptiveweight(aggregated_cost, matching_cost, left_img, kernel_size, sigma_r, sigma_s, d_max)
    ##
    disparity_estimation(disparity, aggregated_cost, d_max)
    cv2.imwrite('../results/ex3d_aw_11.png', disparity)
    print('done2')
