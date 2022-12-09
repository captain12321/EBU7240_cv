# Image stitching using affine transform
import numpy as np
import cv2
import sys
import math
from matplotlib import pyplot as plt

im1 = cv2.imread('../inputs/Img01.jpg')
im2 = cv2.imread('../inputs/Img02.jpg')


im_gray1 = cv2.imread('../inputs/Img01.jpg', 0)
im_gray2 = cv2.imread('../inputs/Img02.jpg', 0)

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
def cv_show(name, image):
 cv2.imshow(name, image)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

#SIFT features
def detectAndDescribe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return kps, features

#use K-NN algorithm
def matchKeypoints(kpsA, kpsB, featuresA, featuresB,ran_flag, ratio = 0.75):
    matcher = cv2.BFMatcher()
    # use 2-NN
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        #choose patches
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        # get coordinates
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # H is affine transform matrix
        if ran_flag==1:  #use ransac
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)
        else:
            (H, status) = cv2.findHomography(ptsA, ptsB)
        # 返回结果
        return matches, H, status
    # 如果匹配对小于4时，返回None
    return None

def stitch(imageA,imageB, ran_flag,ratio=0.75, reprojThresh=4.0):
    #检测A、B图片的SIFT关键特征点，并计算特征描述子
    (kpsA, featuresA) = detectAndDescribe(imageA)
    (kpsB, featuresB) = detectAndDescribe(imageB)
    # pair results
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ran_flag,ratio)
    if M is None:
        return None
    # H is transformation matrix
    (matches, H, status) = M
    # A after transformation
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    cv_show('Affine_A', result)
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    cv_show('result', result)
    return result
panorama_RANSAC=stitch(im2, im1,1)
panorama_noRANSAC=stitch(im2,im1,0)

##########################################################################################

cv2.imwrite('../results/ex3a_stitched_noRANSAC.jpg', panorama_noRANSAC)
cv2.imwrite('../results/ex3a_stitched_RANSAC.jpg', panorama_RANSAC)