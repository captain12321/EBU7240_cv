import cv2

cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

img_array = []

if (cap.isOpened() == False):
    print("Error opening video stream or file")

im_myname = cv2.imread('../inputs/my_name.png')
# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
for a in range(0, frame_num):
    ret, frame = cap.read()
    frame=cv2.resize(frame,(360,640))
    img_array.append(frame)
index = 0
for b in range(len(img_array)):
    img1 = img_array[b]
    img1[550:640,index:index+180] = im_myname
    img_array[b] = img1
    index += 2
# show all the desired frames
cv2.namedWindow('1st frame', cv2.WINDOW_NORMAL)
cv2.imshow('1st frame', img_array[0])
cv2.namedWindow('21st frame', cv2.WINDOW_NORMAL)
cv2.imshow('21st frame', img_array[20])
cv2.namedWindow('31st frame', cv2.WINDOW_NORMAL)
cv2.imshow('31st frame', img_array[30])
cv2.namedWindow('61st frame', cv2.WINDOW_NORMAL)
cv2.imshow('61st frame', img_array[60])
cv2.namedWindow('90th frame', cv2.WINDOW_NORMAL)
cv2.imshow('90th frame', img_array[89])
cv2.waitKey()
##########################################################################################


out = cv2.VideoWriter('../results/ex1_b_hand_composition.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (360,640))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()