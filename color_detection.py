import cv2
import numpy as np
import pandas as pd

image = cv2.imread('F:\INVISIBILITY CLOAK-- COLOR DETECTION--OPENCV\Team 7.png')
dataset = pd.read_csv('color_names.csv')
print(dataset.head())
#cv2.imshow('frame',image)


click = True
colorR = 0
colorG = 0
colorB = 0
def get_color(colorR, colorG, colorB):
    min_dist_color = 195076
    for i in range(len(dataset['Color'])):
        r = (colorR - dataset['R'][i]) ** 2
        g = (colorG - dataset['G'][i]) ** 2
        b = (colorB - dataset['B'][i]) ** 2
        dist= r+g+b
        if dist<min_dist_color:
            min_dist_color = dist
            my_color = dataset["Color_name"][i]
    print(my_color)
    return my_color


def capture_rgb(event, x,y,flags,param):

    if event== cv2.EVENT_LBUTTONDBLCLK:
        global colorR, colorB, colorG, click

        click = True
        colorB = image[y,x,0]
        colorG = image[y,x,1]
        colorR = image[y,x,2]





cv2.namedWindow('frame')





while(True):
    

    cv2.imshow('frame', image)
    
    cv2.setMouseCallback('frame', capture_rgb)
    if click:
        
        disp_color = get_color(colorR, colorG, colorB)
        cv2.rectangle(image,(20,20), (750,60), (255,255,255), -1)
        cv2.putText(image, disp_color, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        click = False

        
    cv2.waitKey(0)  
    if  0xFF==27:
        
        cv2.destroyAllWindows()
        break

        



