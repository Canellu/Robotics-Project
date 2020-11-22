from djitellopy import Tello
from cv2 import cv2
import numpy as np

def initializeTello():

    drone = Tello()
    drone.connect()
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0

    print(drone.get_battery())
    drone.streamoff()
    drone.streamon()

    return drone


def telloGetFrame(drone, frameWidth=360, frameHeight=240):

    telloFrame = drone.get_frame_read()
    telloFrame = telloFrame.frame
    img = cv2.resize(telloFrame,(frameWidth,frameHeight))
    
    return img

def findFace(img):

    # prediction
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2,4)

    myFaceListC = []
    myFaceListArea = []

    for(x,y,w,h) in faces:

        # drawing face boundary
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

        # finding face center coordinate
        cx = x + w//2
        cy = y + h//2
        area = w*h
        myFaceListArea.append(area)
        myFaceListC.append([cx,cy])

    if len(myFaceListArea) != 0:

        # finding closest face (biggest area)
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    
    else:
        return img,[[0,0],0]

def trackFace(drone, info, w, pid, pError):

    # PID
    error = (info[0][0] - w//2)//w * 120
    speed = pid[0]*error + pid[1]*(error-pError)
    speed = int(np.clip(speed,-100, 100))
    print(speed)

    if info[0][0] != 0:
        drone.yaw_velocity = speed
    else:
        drone.for_back_velocity = 0
        drone.left_right_velocity = 0
        drone.up_down_velocity = 0
        drone.yaw_velocity = 0
        drone.speed = 0
        error = 0

    if drone.send_rc_control:
        drone.send_rc_control(drone.left_right_velocity,
                              drone.for_back_velocity,
                              drone.up_down_velocity,
                              drone.yaw_velocity)

    return error

def drawOSD(img, drone):


    # shape = (height, width, channels)
    print(img.shape) 
    dataToDisplay = []
    stateDict = drone.get_current_state()
    print(stateDict)
