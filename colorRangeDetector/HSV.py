# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


#FPS
counter = 0
FPS = 0
startTime = time.time()



	
    	


#define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
lowerHSV = (23, 141, 160)
upperHSV = (32, 251, 237)
trailLength = 32
pts = deque(maxlen=trailLength)

time.sleep(2.0)
cap = cv2.VideoCapture(0)

# keep looping
while True:


	# grab the current frame
	ret, frame = cap.read()



	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=1000)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, lowerHSV, upperHSV)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)




	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# update the points queue
	pts.appendleft(center)




	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(trailLength / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	
	


	#CALC FPS
	counter+=1
	if (time.time() - startTime) > 1 :
		FPS = int(counter / (time.time() - startTime))
		counter = 0
		startTime = time.time()
		
	cv2.putText(frame, 'FPS:', (10,50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
	cv2.putText(frame, str(FPS), (100,50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)

	
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break



cap.release()
cv2.destroyAllWindows()