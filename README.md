# Robotics Project 2020 - Drone tracking
##### Contributors: Anton Vo - Einar Tomter - Saodat Mansurova

## Introduction & overview

This project uses the drone **Tello** and showcases three different object detection method (HAAR, YOLO, HSV).
The system controlling the drone is equipped with a PID-controller and Kalman filter for smoother movements and reactions.

## Features

##### Controls

We have implemented a control system which allow us to switch between tracking and manual control in addition to tracking mode and object.
The image below have highlighted all the keys available.

<img src=/images/keyboardLayout.png height="300">

* :blue_square: altitude and rotation (WASD)
* :blue_square: translation - forward/backward/left/right (arrow keys)
* :green_square:Take off (F) - Land (L)
* :red_square: Object tracking on (T) - Manual control on (M) - Quit program (Q)
* :purple_square: Track mode rotation/translation (1 & 2) - Object detection method (B) - class change (C)

##### On screen display (OSD)

Data from the drones sensors are neatly displayed on the frame captured from the video feed sent from the drone.
![OSD](/images/OSD.png)






###### ObjectTracking HSV
https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

###### Enable CUDA on openCV
https://jamesbowley.co.uk/accelerate-opencv-4-5-0-on-windows-build-with-cuda-and-python-bindings/#visual_studio_cmake_cmd 

###### Open Images V6 - Free labeled training data
https://storage.googleapis.com/openimages/web/index.html

