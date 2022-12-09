# Object tracking with your image

import numpy as np
import cv2

cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
a=input("Track paper enter 1,Track name enter 2:")
if a== '1':
    img_array=[]
    _,fisrt_frame=cap.read()
    x=0
    y=120
    width=340
    height=230
    roi=fisrt_frame[y:y+height,x:x+width]
    #covert to HSV
    hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    roi_hist=cv2.calcHist([hsv_roi],[0],None,[256],[0,255])
    roi_hist=cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
else:
    img_array=[]
    _,fisrt_frame=cap.read()
    x=0
    y=200
    width=250
    height=70
    roi=fisrt_frame[y:y+height,x:x+width]
    cv2.imshow('roi',roi)
    cv2.waitKey()
    #covert to HSV
    hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    roi_hist=cv2.calcHist([hsv_roi],[0],None,[256],[0,255])
    roi_hist=cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)

count=0
while count<90:
    success,frame=cap.read()
    count+=1
    if success:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #灰度图相似部分显示为白色
    mask=cv2.calcBackProject([hsv],[0],roi_hist,[0,256],1)
    _,track_window=cv2.meanShift(mask,(x,y,width,height),term_criteria)
    x,y,w,h=track_window
    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    img_array.append(frame)


cv2.namedWindow('1st frame', cv2.WINDOW_NORMAL)
cv2.imshow('1st frame',img_array[0])
cv2.namedWindow('20th frame', cv2.WINDOW_NORMAL)
cv2.imshow('20th frame',img_array[19])
cv2.namedWindow('40th frame', cv2.WINDOW_NORMAL)
cv2.imshow('40th frame', img_array[39])
cv2.namedWindow('60th frame', cv2.WINDOW_NORMAL)
cv2.imshow('60th frame', img_array[59])
cv2.namedWindow('90th frame', cv2.WINDOW_NORMAL)
cv2.imshow('90th frame', img_array[88])
cv2.waitKey()


##########################################################################################
if a== '1':
    out_paper = cv2.VideoWriter('../results/ex4a_meanshift_track_paper.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (360,640))
    for i in range(len(img_array)):
        out_paper.write(img_array[i])
    out_paper.release()
else:
    out_name = cv2.VideoWriter('../results/ex4a_meanshift_track_name.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (360,640))
    for i in range(len(img_array)):
        out_name.write(img_array[i])
    out_name.release()

