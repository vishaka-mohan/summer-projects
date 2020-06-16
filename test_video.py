
##IMPORT LIBRARIES
import cv2
import time
import numpy as np


##VIDEO CAPTURE --WAIT FOR 5 SEC 
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
time.sleep(5)



##CAPTURE BACKGROUND FIRST
bg=0
for i in range(20):
    r,bg= cap.read()



##RANGE FOR BLUE HSV -- 
blue_low = np.array([94, 80, 2])
blue_high = np.array([125, 255, 255])



##PLAY VIDEO

while(True):

    #read frame by frame
    _, frame = cap.read()

    #convert current frame to hsv 
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #blue mask on hsv frame-- 1 if pixel falls in range , else 0
    blue_mask = cv2.inRange(frame_hsv, blue_low, blue_high)

    
    blue = cv2.bitwise_and(frame_hsv, frame_hsv, mask=blue_mask)

    #invert blue mask-- 1 if pixel not in range,else 0
    blue_mask1 = cv2.bitwise_not(blue_mask)
    

    #inverted mask on original frame-- 0 if blue color found
    frame1 = cv2.bitwise_and(frame,frame, mask= blue_mask1)
    

    #original blue mask on background-- 1 if blue found
    frame2 = cv2.bitwise_and(bg,bg,mask= blue_mask)
    

    #superposition of res_1 on res_2
    final = cv2.addWeighted(frame1,1,frame2,1,0)


    #display original frame

    cv2.imshow("Original", frame)

    
    #cv2.imshow("Blue frame", blue)
    #cv2.imshow("Blue mask", blue_mask)

    #display final frame after masking
    cv2.imshow("Final",final)



    #press esc to quit
    key = cv2.waitKey(1)
    if key==27:
        break








