from djitellopy import Tello
import cv2
import socket
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



def telloGetFrame(drone):

    telloFrame = drone.get_frame_read()
    telloFrame = telloFrame.frame

    return telloFrame


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
    confThreshold = 0.9 # Lower value, more boxes (but worse confidence per box)
    nmsThreshold = 0.3 # Lower value, less overlaps

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

        if(classNames[classIndices[i]] == 'Face'):
            returnArea.append(area)


    if len(returnArea) != 0:
        # finding closest face (biggest area)
        i = returnArea.index(max(returnArea))
        bbox[i][0] = bbox[i][0] + bbox[i][2]//2
        bbox[i][1] = bbox[i][1] + bbox[i][3]//2

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2) # Draw bounding box
        cv2.putText(img, f'{classNames[classIndices[i]].upper()} {int(confs[i]*100)}%',
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2) #Write class name and % on bounding box


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

def droneData(droneStates):

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', 8890))
    count = 0
    while True:
        try:
            data, server = sock.recvfrom(1518)
            if count == 100:
                droneStates.pop(0)
            droneStates.append(data.decode(encoding="utf-8"))
        except Exception as err:
            print(err)
            sock.close
            break

def drawOSD(droneStates, frame):
    # pitch:0;roll:0;yaw:0;vgx:0;vgy:0;vgz:0;templ:82;temph:85;tof:48;h:0;bat:20;baro:163.98;time:0;agx:6.00;agy:-12.00;agz:-1003.00;
    
    states = droneStates[len(droneStates)-1].split(";")
    pitch = states[0][5:]
    roll = states[1][4:]
    yaw = states[2][3:]
    vgx = states[3][3:]
    vgy = states[4][3:]
    vgz = states[5][3:]
    templ = states[6][5:]
    temph = states[7][5:]
    tof = states[8][3:]
    h = states[9][1:]
    bat = states[10][3:]
    baro = states[11][4:]
    time = states[12][4:]
    agx = states[13][3:]
    agy = states[14][3:]
    agz = states[15][3:]


    windowWidth = frame.shape[1]
    windowHeight = frame.shape[0]
    posy = 0
    cv2.putText(frame, f"States: {len(states)}", (150, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
    for i in range(len(states)-1):
        posy += 30
        cv2.rectangle(frame, (0, windowHeight-10), (windowWidth, windowHeight), (0, 255, 0), -1)
        cv2.putText(frame, states[i], (0,posy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv2.putText(frame, states[i], (50,posy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))


# Call before main-loop to create the slider (starts from 0 to maxVal)
# @Parameter is the window to place the slider on.
def distanceSlider(frame):

    maxVal = 100
    startVal = 50

    def nothing(var):
        pass
    
    sliderWindow = cv2.namedWindow(frame)
    cv2.createTrackbar("Distance", frame, startVal, maxVal, nothing)
    

# Call inside loop to read slider
# @name = name of trackbar
# @frame = the window trackbar resides in
def readSlider(name, frame):
    return cv2.getTrackbarPos(name, frame)    


def plot(frameWidth, frameHeight, fig, ax, info, loop, plotInfo):

    # defining axes
    x_axis = np.linspace(0, frameWidth, num=5)
    y_axis = np.linspace(frameHeight, 0, num=5)

    # limiting to 100 points in array
    if len(plotInfo[2]) == 100:
        plotInfo[0].pop(0)
        plotInfo[1].pop(0)
        plotInfo[2].pop(0)

    # appending new values
    if info[0] == 0:
        plotInfo[0].append(frameWidth//2)
    else:
        plotInfo[0].append(info[0])
    
    if info[1] == 0:
        plotInfo[1].append(frameHeight//2)
    else:
        plotInfo[1].append(info[1])

    plotInfo[2].append(loop)
  
    # Plotting

    # x-axis vs loop iteration
    ax[0].cla()
    ax[0].plot(plotInfo[0], plotInfo[2], color='b')
    
    ax[0].set_xticks(x_axis)
    ax[0].set_ylim(bottom=max(0, loop-100), top=loop+100)

    # y-axis vs loop iteration
    ax[1].cla()
    ax[1].plot(plotInfo[2], plotInfo[1], color='b')
    ax[1].invert_yaxis()
    
    ax[1].set_xlim(left=max(0, loop-100), right=loop+100)
    ax[1].set_yticks(y_axis)
    
    fig.canvas.draw()

    # Return updated x,y,t array
    return plotInfo

def kalmanVideo():

    varMatrix = np.array([[0,1],[1,0]]) # Variance-covariance matrix

    varMeasured
    varProcess
