# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time



#define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
lowerHSV = (19,63,114)
upperHSV = (60,179,255)

tomatoMin = (25,79,81)
tomatoMax = (37,255,255)

tennisMin = (26, 70, 84)
tennisMax = (35, 225, 236)




def trackHSV(frame):

    center = (0,0)
    radius = 0
    
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, tennisMin, tennisMax)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 20:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
            return [center[0], center[1], radius]
    

    return [0, 0, 0]