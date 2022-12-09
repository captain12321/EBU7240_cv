#This code is for generating the required format video from raw.mp4 to ebu7240_hand.mp4
import cv2

cap = cv2.VideoCapture('../inputs/raw.mp4')
if not cap.isOpened():
    print("Error opening video stream or file")
fps=30 #required fps
size=(360,640) #required size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_cat = cv2.VideoWriter("../inputs/ebu7240_hand.mp4", fourcc, fps, size, True)  # 保存位置/格式
count=0
while count<90:
    success, frame = cap.read()
    if success:
        count+=1
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        img = cv2.resize(frame, size)
        out_cat.write(img)
        #print(count)
    else:
        print('break')
        break

cap.release()
print('done')




