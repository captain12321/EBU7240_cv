# Window-based stereo matching
import numpy as np
import cv2


def matching_cost_computation(matching_cost, left_img, right_img, d_max):
    # Matching cost computation: SSD
    h, w = left_img.shape

    for y in range(h):
        for x in range(w):
            for d in range(d_max):
                if x - d >= 0:
                    temp_cost = float(left_img[y, x]) - float(right_img[y, x - d])
                    matching_cost[y, x, d] = temp_cost * temp_cost


def cost_aggregation_window(aggregated_cost, matching_cost, kernel, d_max):
    # --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
    # NO OPENCV FUNCTION IS ALLOWED HERE
    filter_template = np.ones((kernel, kernel))  # 空间滤波器模板 卷积核
    h,w,_=matching_cost.shape
    input_image_cp=np.zeros((h,w),np.float64)
    pad_num = int((kernel - 1) / 2)  # 输入图像需要填充的尺寸
    for d in range(d_max):
        input_image_cp=matching_cost[:,:,d]

        input_image_cp =np.pad(input_image_cp, (pad_num, pad_num), mode="edge")  # 以边缘值扩展填充输入图像
        m, n = input_image_cp.shape  # 获取填充后的输入图像的大小
        output_image = np.copy(input_image_cp)  # 输出图像
        for i in range(pad_num, m - pad_num):
            for j in range(pad_num, n - pad_num):
                #kernel内相加
                output_image[i, j] = np.sum(
                    filter_template * input_image_cp[i - pad_num:i + pad_num + 1, j - pad_num:j + pad_num + 1])
        output_image = output_image[pad_num:m - pad_num, pad_num:n - pad_num]  # 还原填充之前的图像形状大小
        aggregated_cost[:,:,d]=output_image

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

    matching_cost = np.zeros((h, w, d_max), np.float64)
    aggregated_cost = np.zeros((h, w, d_max), np.float64)
    disparity = np.zeros((h, w), np.uint8)

    matching_cost_computation(matching_cost, left_img, right_img, d_max)
    ## COMPLETE THIS FUNCTION
    cost_aggregation_window(aggregated_cost, matching_cost, kernel_size, d_max)
    ##
    disparity_estimation(disparity, aggregated_cost, d_max)
    cv2.imwrite('../results/ex3c_w_3.png', disparity)
    print('done1')

    d_max, kernel_size = 32, 11

    matching_cost = np.zeros((h, w, d_max), np.float64)
    aggregated_cost = np.zeros((h, w, d_max), np.float64)
    disparity = np.zeros((h, w), np.uint8)

    matching_cost_computation(matching_cost, left_img, right_img, d_max)
    ## COMPLETE THIS FUNCTION
    cost_aggregation_window(aggregated_cost, matching_cost, kernel_size, d_max)
    ##
    disparity_estimation(disparity, aggregated_cost, d_max)
    cv2.imwrite('../results/ex3c_w_11.png', disparity)
    print('done2')