# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 04:27:10 2019

@author: anvita
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
#lower and upper bounds of each color in hsv space
lower = {'red':([166, 84, 141]), 'green':([50, 50, 120]), 'blue':([97, 100, 117]),'yellow':([23, 59, 119]), 'orange':([0, 50, 80]), 'purple':([130, 80, 80])} #assign new item lower['blue'] = (93, 10, 0)
upper = {'red':([186,255,255]), 'green':([70, 255, 255]), 'blue':([117,255,255]), 'yellow':([54,255,255]), 'orange':([20,255,255]), 'purple':([150, 255, 255])}

#BGR tuple of each color 
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255), 'purple':(211,0,148)}

frame = cv2.imread("Image2.jpg")
frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

 
blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

mlist=[]
clist=[]
ks=[]

for (key, value) in upper.items():
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.inRange(hsv, np.array(lower[key]), np.array(upper[key]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mlist.append(mask)
    _,cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts)>=1:
        clist.append(cnts[-1])
        ks.append(key)
        
print(ks)
    
for i,cnt in enumerate(clist):
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(frame, [approx], 0, (0), 2)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
     
        if len(approx) == 3:
            cv2.putText(frame, ks[i] + " Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[ks[i]],2)
            
        elif len(approx) == 4:
            x2 = approx.ravel()[2]
            y2 = approx.ravel()[3]
            x4 = approx.ravel()[6]
            y4 = approx.ravel()[7]
            side1 = abs(x2-x)
            side2 = abs(y4-y)
            
            if abs(side1-side2) <= 2:
                cv2.putText(frame, ks[i] + " Square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[ks[i]],2)
            else:
                cv2.putText(frame, ks[i] + " Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[ks[i]],2)
                
        elif len(approx) == 5:
            cv2.putText(frame, ks[i] + " Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[ks[i]],2)
        
        #elif 6 < len(approx) < 15:
            #cv2.putText(res, "Ellipse", (x, y), font, 1, (255,255,255))
        
        elif len(approx) > 10:
            cv2.putText(frame, ks[i] + "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[ks[i]],2)
    
    
cv2.imshow("Frame", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
    