# Robotics Project 2020 (Oslo Metropolitan University)  
# Object tracking with Drone
##### Contributors: Anton Vo - Einar Tomter - Saodat Mansurova

<img src=/images/Tello.jpg width="800"><br/>

## Showcase of project (Gif)
### Full video can be seen on YouTube: [Link](https://youtu.be/zmb9cjAXg5U)

![Gif](/images/demo.gif)

## Introduction & overview

This project uses the drone **Tello** and showcases three different object detection method (HAAR, YOLO, HSV).
The system controlling the drone is equipped with a PID-controller and Kalman filter for smoother movements and reactions.
Below is a simple flowchart of the system. We first connect to the drone, take off then use the frames from drone to detect an object. The data from detection goes through kalman filter and enters PID control calculations. Lastly, send movement commands based on PID data to track the object.

![Flow chart](/images/FlowChart.png)

## Features


### Controls

We have implemented a control system which allow us to switch between tracking and manual control in addition to tracking mode and object.<br/>
There are two main reasons for this:
* First reason is safety. As we are testing tracking and tuning parameters, the drone sometimes can perform unexpectedly.  Being able to regain control of the drones movement is crucial.
* Second reason is just for 'Quality of Life' reasons. It is nice to be able to control the drone however you like, whenever you would like.

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

## Detection Methods

### HAAR (Face detect)

This method is simple to setup but does not provide a very good detection, it often draws boundary boxes on random objects.

<img src=/images/HAAR.jpg width="800"><br/>


### YOLO (Multiple classes)

This method is more advanced but is also more computationally heavy to run and is more complex in terms of setting up and training the AI model.
Training is done with [*darknet*](https://pjreddie.com/darknet/yolo/) framework with pretrained weigths. We've also enabled OpenCV to run with CUDA (GPU) to increase performance in terms of FPS. 

<img src=/images/YOLOCombi.png width="800"><br/>
                                      
### HSV (Computer vision technique)

This method is by far the simplest and runs really quickly, but does not recognize object. It only detects by a given color through creating a mask.
Trackbars show minimum and maximum threshold values for detecting the tennis ball.

<img src=/images/HSVCombi.png width="600"><br/>


## Kalman filter

Here is some plots showcasing how the kalman filter works. The GUI is made in python using PyQt5 and pyqtgraph.
Blue line is kalman values, red line is measured values. The measured values is accompanied with some noise which is partially 'suppressed' by the filter.

![QtGui](/images/QtGui.png)




### References
**References to all the sources are included in the *Final_report.pdf*.**  
**Youtube videos and general random google searches are not included, but is an essential resource.**

#### Note
The code is **not** written with *"Best practices"* in mind due to time limitations.  
Source code is neither *Pythonic* or *Object oriented*.
