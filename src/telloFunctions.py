from djitellopy import Tello
import cv2
import numpy as np


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


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
    print("initialize done")

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


def trackFace(drone, info, w, h, pidYaw, pidZ, pidX, pError):

    error = [0,0,0] # yaw, height, distance
    speed = [0,0,0]
    percentArea = 1/25

    area = w * h * percentArea

    # PID
    error[0] = (info[0][0] - w//2) #//w * 120
    error[1] = (info[0][1] - h//2)
    error[2] = (info[1] - area)/25

    speed[0] = pidYaw[0]*error[0] + pidYaw[1]*(error[0]-pError[0])
    speed[0] = int(np.clip(speed[0],-100, 100))

    speed[1] = (pidZ[0]*error[1] + pidZ[1]*(error[1]-pError[1]))*(-1)
    speed[1] = int(np.clip(speed[1],-100, 100))

    # speed[2] = (pidX[0]*error[2] + pidX[1]*(error[2]-pError[2]))*(-1)
    # speed[2] = int(np.clip(speed[2],-100, 100))

    # print(f"error: {error[0]}\t speed: {speed[0]}") # yaw
    # print(f"error: {error[1]}\t speed: {speed[1]}") # height
    #print(f"error: {error[2]}\t speed: {speed[2]}") # distance

    if info[0][0] != 0:
        drone.yaw_velocity = speed[0]
    else:
        drone.left_right_velocity = 0
        drone.yaw_velocity = 0
        error[0] = 0

    if info[0][1] != 0:
        drone.up_down_velocity = speed[1]
    else:
        drone.up_down_velocity = 0
        error[1] = 0

    if info[1] != 0:
        drone.for_back_velocity = speed[2]
    else:
        drone.for_back_velocity = 0
        error[2] = 0

    if drone.send_rc_control:
        drone.send_rc_control(drone.left_right_velocity,
                              drone.for_back_velocity,
                              drone.up_down_velocity,
                              drone.yaw_velocity)

    return error


def drawOSD(frame):

    cv2.putText(frame, "OSD HERE!", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    dataToDisplay = [0] * 7

    # shape = (height, width, channels)
 

# Call before main-loop to create the slider (starts from 0 to maxVal)
# @Parameter is the window to place the slider on.
def distanceSlider(frame):

    def nothing(var):
        pass
    
    img = cv2.namedWindow(frame)
    
    cv2.createTrackbar("Distance", frame, 50, 100, nothing)
    cv2.setTrackbarPos("Distance", frame, 239)
    print(cv2.getWindowImageRect(frame)[3])
    


# Call inside loop to read slider
# @name = name of trackbar
# @frame = the window trackbar resides in
def readSlider(name, frame):
    return cv2.getTrackbarPos(name, frame)    
    