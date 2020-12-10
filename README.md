# Robotics Project 2020 - Drone tracking
##### Contributors: Anton Vo - Einar Tomter - Saodat Mansurova

## Introduction & overview

This project uses the drone **Tello** and showcases three different object detection method (HAAR, YOLO, HSV).
The system controlling the drone is equipped with a PID-controller and Kalman filter for smoother movements and reactions.
Below is a simple flowchart of the system. We first connect to the drone, take off then use the frames from drone to detect and object. The data from detection goes through kalman filter and enters PID control calculations. Lastly send movement commands based on PID to track an object.

![Flow chart](/images/FlowChart.png)

## Features


### Controls

We have implemented a control system which allow us to switch between tracking and manual control in addition to tracking mode and object.<br/>
The image below have highlighted all the keys available.

<img src=/images/keyboardLayout.png height="300"><br/>

* :blue_square: altitude and rotation (WASD) - translation forward/backward/left/right (arrow keys)
* :green_square:Take off (F) - Land (L)
* :red_square: Object tracking on (T) - Manual control on (M) - Quit program (Q)
* :purple_square: Track mode rotation/translation (1 & 2) - Object detection method (B) - class change (C)
* :yellow_square: enable/disable OSD

<br/>

### On screen display (OSD)

Data from the drones sensors are neatly displayed on the frame captured from the video feed sent from the drone.<br/>

<img src=/images/OSD.png width="800"><br/>

* Left panel show altitude (tof: measured distance from infrared sensor, h: height above takeoff-level, baro: height above sea-level, temperature and pitch roll yaw
* Right panel show velocity and acceleration in XYZ
* Top indicators: Time since motor start and battery level
* Bottom indicators: Tracking mode, detection mode, FPS.

<br/>

### 



###### ObjectTracking HSV
https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

###### Enable CUDA on openCV
https://jamesbowley.co.uk/accelerate-opencv-4-5-0-on-windows-build-with-cuda-and-python-bindings/#visual_studio_cmake_cmd 

###### Open Images V6 - Free labeled training data
https://storage.googleapis.com/openimages/web/index.html

