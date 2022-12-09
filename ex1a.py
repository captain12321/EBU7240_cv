import cv2

cap = cv2.VideoCapture('../inputs/ebu7240_hand.mp4')

img_array = []

if (cap.isOpened() == False):
    print("Error opening video stream or file")

#--------------------------------- WRITE YOUR CODE HERE ---------------------------------#
count=0 #use counter
size=(360,640)
while count<90:
    success, frame = cap.read()
    if success and count<30:
        count+=1
        img = cv2.resize(frame, size)
        img_array.append(img)
    elif success and 30<=count<50:
        count += 1
        img = cv2.resize(frame, size)
        #set green and red channel to zero
        img[:,:,1]=0
        img[:,:,2]=0
        img_array.append(img)
    elif success and 50<=count<70:
        count += 1
        img = cv2.resize(frame, size)
        # set green and red channel to zero
        img[:, :, 0] = 0
        img[:, :, 1] = 0
        img_array.append(img)
    elif success and 70 <= count < 90:
        count += 1
        img = cv2.resize(frame, size)
        # set green and red channel to zero
        img[:, :, 0] = 0
        img[:, :, 2] = 0
        img_array.append(img)
    else:
        print('something wrong')
        break
#show all the desired frames
cv2.namedWindow('1st frame', cv2.WINDOW_NORMAL)
cv2.imshow('1st frame',img_array[0])
cv2.namedWindow('21st frame', cv2.WINDOW_NORMAL)
cv2.imshow('21st frame',img_array[20])
cv2.namedWindow('31st frame', cv2.WINDOW_NORMAL)
cv2.imshow('31st frame', img_array[30])
cv2.namedWindow('61st frame', cv2.WINDOW_NORMAL)
cv2.imshow('61st frame', img_array[60])
cv2.namedWindow('90th frame', cv2.WINDOW_NORMAL)
cv2.imshow('90th frame', img_array[89])
cv2.waitKey()
##########################################################################################
out = cv2.VideoWriter('../results/ex1_a_hand_rgbtest.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (360,640))
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
