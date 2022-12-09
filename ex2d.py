# SIFT matching using OpenCV
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im_gray1 = cv2.imread('../inputs/sift_input1.jpg', 0)
im_gray2 = cv2.imread('../inputs/sift_input2.jpg', 0)

# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
def sift_descriptor(gray,num): #num tells which picture it is
    if num==1:
        img=cv2.imread('../inputs/sift_input1.jpg')
    elif num==2:
        img=cv2.imread('../inputs/sift_input2.jpg')
    else:
        print('sorry, wrong number')
    # create SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()
    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # draw the detected key points
    sift_image = cv2.drawKeypoints(gray, keypoints,img)
    # show the image
    cv2.imshow('input'+str(num), sift_image)
    cv2.waitKey()
    return keypoints,sift_image,descriptors
keypoints_1,img_sift_kp_1,descriptor_1=sift_descriptor(im_gray1,1)
keypoints_2,img_sift_kp_2,descriptor_2=sift_descriptor(im_gray2,2)

#match between features
# create feature matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
# match descriptors of both images
matches = bf.match(descriptor_1,descriptor_2)
# sort matches by distance
matches = sorted(matches, key = lambda x:x.distance)
# draw first 50 matches
img_most50 = cv2.drawMatches(im_gray1, keypoints_1, im_gray2, keypoints_2, matches[:50], im_gray2, flags=2)
cv2.imshow('img_most50', img_most50)
cv2.waitKey()
#draw last 50 matches
img_least50 = cv2.drawMatches(im_gray1, keypoints_1, im_gray2, keypoints_2, matches[-50:], im_gray2, flags=2)
cv2.imshow('img_least50', img_least50)
cv2.waitKey()
##########################################################################################

# Keypoint maps
cv2.imwrite('../results/ex2d_sift_input1.jpg', np.uint8(img_sift_kp_1))
cv2.imwrite('../results/ex2d_sift_input2.jpg', np.uint8(img_sift_kp_2))

# Feature Matching outputs
cv2.imwrite('../results/ex2d_matches_least50.jpg', np.uint8(img_least50))
cv2.imwrite('../results/ex2d_matches_most50.jpg', np.uint8(img_most50))
