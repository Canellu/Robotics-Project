from djitellopy import Tello
import cv2
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


def initializeTello():

    drone = Tello()
    connection = drone.connect()

    if connection:
        drone.for_back_velocity = 0
        drone.left_right_velocity = 0
        drone.up_down_velocity = 0
        drone.yaw_velocity = 0
        drone.speed = 0


        drone.streamoff()
        drone.streamon()
        print(f"BATTERY: {drone.get_battery()}")
        print("---- Connecting to drone Succeeded ----\n")

    else:
        print("\n---- Connecting to drone Failed ----\n")

    return connection, drone



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
        myFaceListC.append([cx, cy, w, h])

    if len(myFaceListArea) != 0:

        # finding closest face (biggest area)
        i = myFaceListArea.index(max(myFaceListArea))
        return img, myFaceListC[i]
    else:

        return img,[0,0,0,0]


def findFaceYolo(outputs, img, classNames):


    # Neural Network Params
    confThreshold = 0.1 # Lower value, more boxes (but worse confidence per box)
    nmsThreshold = 0.1 # Lower value, less overlaps

    hT, wT, _ = img.shape
    bbox = [] 
    classIndices = []
    confs = []
    returnArea = []

    for output in outputs: # Go through each output layer (3 layers)
        for det in output: # Go through each detection in layers (rows per layer: 300 first layer, 1200 second layer, 4800 third layer)
            scores = det[5:] # List of confidence scores/probability for each class
            classIndex = np.argmax(scores) # Returns index of highest score
            confidence = scores[classIndex] # Get the highest score.
            if confidence > confThreshold:
                w, h = int(det[2]*wT) , int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIndices.append(classIndex)
                confs.append(float(confidence))


    # Returns indices of boxes to keep when multiple box on same object. Finds box with highest probability, suppress rest.
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0] # Flatten indices, comes as a value in list. [val] --> val
        box = bbox[i] # Get a box
        x, y, w, h = box[0], box[1], box[2], box[3] # Extract x, y, width, height
        area = w * h

        if(classNames[classIndices[i]] == 'Human face'):
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2) # Draw bounding box
            cv2.putText(img, f'{classNames[classIndices[i]].upper()} {int(confs[i]*100)}%', 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2) #Write class name and % on bounding box
            returnArea.append(area)


    if len(returnArea) != 0:
        # finding closest face (biggest area)
        i = returnArea.index(max(returnArea))
        bbox[i][0] = bbox[i][0] + bb[i][2]
        bbox[i][1] = bbox[i][1] + bb[i][3]

        return img, (bbox[i])   
    else:
        return img, ([0,0,0,0])

def trackFace(drone, info, pInfo, w, h, pidYaw, pidX, pidZ, pError):

    error = [0,0,0] # yaw, height, distance (pixels)
    speed = [0,0,0] # yaw, height, distance (cm/s)
    wait = 0 # s

    # current info
    cx = info[0]
    cy = info[1]
    bw = info[2]
    bh = info[3]

    # previous info
    pcx = pInfo[0]
    pcy = pInfo[1]
    pbw = pInfo[2]
    pbh = pInfo[3]

    # editable variables
    percentArea = 1/25
    edgeDetecion = False
    sideDetecThreshold = 2


    # calculations

    area = w * h * percentArea
    error[0] = (cx - w//2) #//w * 120
    error[1] = (cy - h//2)
    error[2] = ((bw * bh) - area)/25

    # PID

    speed[0] = pidYaw[0]*error[0] + pidYaw[1]*(error[0]-pError[0])
    speed[0] = int(np.clip(speed[0],-100, 100))

    speed[1] = (pidZ[0]*error[1] + pidZ[1]*(error[1]-pError[1]))*(-1)
    speed[1] = int(np.clip(speed[1],-100, 100))

    # speed[2] = (pidX[0]*error[2] + pidX[1]*(error[2]-pError[2]))*(-1)
    # speed[2] = int(np.clip(speed[2],-100, 100))

    # checking values

    # print(f"error: {error[0]}\t speed: {speed[0]}") # yaw
    # print(f"error: {error[1]}\t speed: {speed[1]}") # height
    # print(f"error: {error[2]}\t speed: {speed[2]}") # distance
    # print(f"center x: {cx} center y: {cy}") # center coordinate
    # print(f"current info: {info}\t previous info: {pInfo}")


    if cx != 0:
        drone.yaw_velocity = speed[0]
    else:
        # testing if object went out of frame left/right
        if pcx != 0 and edgeDetecion == True:
            if pcx < (w*sideDetecThreshold)//10: #left
               drone.yaw_velocity = -100
               wait = 0.15
            #    drone.rotate_counter_clockwise(360)
               print("chasing left")
            elif pcx > (w*(10-sideDetecThreshold))//10: #right
               drone.yaw_velocity = 100
               wait = 0.15
            #    drone.rotate_clockwise(360)
               print("chasing right")
        # if nothing is being tracked
        else:
            drone.left_right_velocity = 0
            drone.yaw_velocity = 0
            error[0] = 0

    if cy != 0:
        drone.up_down_velocity = speed[1]
    else:
        # testing if object went out of frame up/down
        if pcy != 0 and edgeDetecion == True:
            if pcy < (h*sideDetecThreshold)//10: #up
               drone.up_down_velocity = 100
               wait = 0.15
               print("chasing up")
            elif pcy > (h*(10-sideDetecThreshold))//10: #down
               drone.up_down_velocity = -100
               wait = 0.15
               print("chasing down")
        # if nothing is being tracked
        else:
            drone.up_down_velocity = 0
            error[1] = 0

    if (bw * bh) != 0:
        drone.for_back_velocity = speed[2]
    else:
        drone.for_back_velocity = 0
        error[2] = 0

    if drone.send_rc_control:
        drone.send_rc_control(drone.left_right_velocity,
                              drone.for_back_velocity,
                              drone.up_down_velocity,
                              drone.yaw_velocity)

    time.sleep(wait)

    pInfo = info

    return pInfo, error

def drawOSD(drone, frame, frameWidth, frameHeight):

    # Data retrieved from drone
    data = [0] * 10
    data[0] = str(drone.send_command_with_return('speed?')) # Speed in cm/s 1-100
    data[1] = str(drone.send_command_with_return('battery?')) # Battery in percentage 0 -100
    # data[2] = str(drone.send_command_with_return('time?')) # Time since motor on seconds
    # data[3] = str(drone.send_command_with_return('height?')) # Height in cm 0 - 3000
    # data[4] = str(drone.send_command_with_return('temp?')) # Temperature in celcius 0 - 90
    # data[5] = str(drone.send_command_with_return('attitude?')) # Inertial measurement unit (IMU) pitch roll yaw
    # data[6] = str(drone.send_command_with_return('baro?')) # Absolute height in meters
    # data[7] = str(drone.send_command_with_return('acceleration?')) # Angular acceleration (0.001g) x y z
    # data[8] = str(drone.send_command_with_return('tof?')) # Distance from time of flight (TOF) in cm  30 - 1000
    # data[9] = str(drone.send_command_with_return('wifi?')) # Wi-Fi signal to noise ratio (SNR) dB? Higher better?
    
    
    cv2.putText(frame, (data[0]) + (data[1]), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
    # cv2.putText(frame, (data[2]) + (data[3]), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
    # cv2.putText(frame, (data[4]) + (data[5]), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
    # cv2.putText(frame, (data[6]) + (data[7]), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
    # cv2.putText(frame, (data[8]) + (data[9]), (0,70), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255))
    
    


# Call before main-loop to create the slider (starts from 0 to maxVal)
# @Parameter is the window to place the slider on.
def distanceSlider(frame, frameWidth, frameHeight):

    maxVal = 100
    startVal = 50

    def nothing(var):
        pass
    sliderWindow = cv2.namedWindow(frame)
    cv2.resizeWindow(sliderWindow, frameWidth, 300)
    cv2.createTrackbar("Distance", frame, startVal, maxVal, nothing)
    

# Call inside loop to read slider
# @name = name of trackbar
# @frame = the window trackbar resides in
def readSlider(name, frame):
    return cv2.getTrackbarPos(name, frame)    



def plot(x,y,t):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()

    t = 0

    ax.plot(x, t, color='b')

    ani = FuncAnimation(plt.gcf, animate, )

    # x, y = [], []

    #     x.append(i)
    #     y.append(inputVar)

    #     ax.plot(x, y, color = 'b')
    #     fig.canvas.draw()
    #     ax.set_xlim(left=max(0, i-50), right=i+50)

    #     time.sleep(0.1)
    #     i += 1