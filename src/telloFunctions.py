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
    confThreshold = 0.65 # Lower value, more boxes (but worse confidence per box)
    nmsThreshold = 0.3 # Lower value, less overlaps

    hT, wT, _ = img.shape
    bbox = [] 
    classIndices = []
    confs = []
    returnIndices = []
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

        
        if(classNames[classIndices[i]] == 'Anton'):
            returnIndices.append(i)


    if len(returnIndices) != 0:
        # finding closest face (biggest area)
        for i in returnIndices:
            area = bbox[i][2] * bbox[i][3]
            returnArea.append(area)

        maxVal = returnArea.index(max(returnArea))
        bbox[maxVal][0] = bbox[maxVal][0] + bbox[maxVal][2]/2
        bbox[maxVal][1] = bbox[maxVal][1] + bbox[maxVal][3]/2

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2) # Draw bounding box
        cv2.putText(img, f'{classNames[classIndices[i]].upper()} {int(confs[i]*100)}%',
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2) #Write class name and % on bounding box


        return img, (bbox[i])
    else:
        return img, ([0,0,0,0])

def trackFace(drone, info, pInfo, w, h, pidY, pidX, pidZ, pidYaw, pError, sliderVal, frame, mode):

    # mode default (True): True = Rotation, False = Translation

    # w = 960
    # h = 720
    # Aspect ratio from tello 4:3
    

    error = [0,0,0] # yaw, height, distance (pixels)
    speed = [0,0,100,0] # leftright, forwardback, updown, rotate


    # current info
    cx = info[0]
    cy = info[1]
    bh = info[2]

    # previous info
    pcx = pInfo[0]
    pcy = pInfo[1]
    pbh = pInfo[2]

    # editable variables
    percentH = 1/6 * h + (sliderVal-50)*4 + h/10


    # calculations
    error[0] = (cx - w//2) / (w/2) * 100
    error[1] = (cy - h//2) / (h/2) * 100
    error[2] = (bh - percentH)/percentH * 100
    # PID
    if mode:
        # rotation - Yaw
        speed[3] = pidYaw[0]*error[0] + pidYaw[1]*(error[0]-pError[0])
        speed[3] = int(np.clip(speed[3],-100, 100))
    else:
        # Y - left/right
        speed[0] = pidY[0]*error[0] + pidY[1]*(error[0]-pError[0])
        speed[0] = int(np.clip(speed[0],-100, 100))
    
    # X - forward/back
    speed[1] = (pidX[0]*error[2] + pidX[1]*(error[2]-pError[2]))*(-1)
    speed[1] = int(np.clip(speed[1],-100, 100))
    
    # Z - up/down
    speed[2] = (pidZ[0]*error[1] + pidZ[1]*(error[1]-pError[1]))*(-1)
    speed[2] = int(np.clip(speed[2],-100, 100))



    # TEST PRINTS ---------------
    cv2.putText(frame, str(mode), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    cv2.rectangle(frame, (0, frame.shape[0]-190), (frame.shape[1], frame.shape[0]), (211, 211, 211), -1)
    cv2.putText(frame, (f"eRotation       : {int(error[0])}\t speed: {speed[3]}") , (10, frame.shape[0]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    cv2.putText(frame, (f"eLeftRight       : {int(error[0])}\t speed: {speed[0]}") , (10, frame.shape[0]-40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    cv2.putText(frame, (f"eUpdown         : {int(error[1])}\t speed: {speed[2]}") , (10, frame.shape[0]-70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    cv2.putText(frame, (f"eForwardBackward: {int(error[2])}\t speed: {speed[1]}") , (10, frame.shape[0]-100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    cv2.putText(frame, (f"center x: {int(cx)} center y: {int(cy)}") , (10, frame.shape[0]-130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    cv2.putText(frame, (f"current info: {info}\n previous info: {pInfo}") , (10, frame.shape[0]-160), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))





    # Rotation / Translation
    if mode:
        # Rotation
        if cx != 0:
            drone.yaw_velocity = speed[3]
        else:
            drone.yaw_velocity = 0
            error[0] = 0
    else:
        # Translation
        if cx != 0:
            drone.left_right_velocity = speed[0]
        else:
            drone.left_right_velocity = 0
            error[0] = 0

    # # Up - down
    if cy != 0:
        drone.up_down_velocity = speed[2]
    else:
        drone.up_down_velocity = 0
        error[2] = 0

    # Forward - Back
    if bh != 0:
        drone.for_back_velocity = speed[1]
    else:
        drone.for_back_velocity = 0
        error[2] = 0


    # Update movement
    if drone.send_rc_control:
        drone.send_rc_control(drone.left_right_velocity,
                              drone.for_back_velocity,
                              drone.up_down_velocity,
                              drone.yaw_velocity)


    
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

def qSlider(frame):
    maxVal = 200
    startVal = 100

    def nothing(var):
        pass
    sliderWindow = cv2.namedWindow(frame)
    cv2.createTrackbar("Q Value", frame, startVal, maxVal, nothing)


# Call inside loop to read slider
# @name = name of trackbar
# @frame = the window trackbar resides in
def readSlider(name, frame):
    return cv2.getTrackbarPos(name, frame)    


def plot(frameWidth, frameHeight, fig, ax, info, X, loop, plotInfo, plotKalman):

    #plotInfo[0] = x
    #plotInfo[1] = y
    #plotInfo[2] = h
    #plotInfo[3] = loop

    #plotKalman[0] = x
    #plotKalman[1] = y
    #plotKalman[2] = h

    # defining axes
    x_axis = np.linspace(0, frameWidth, num=5)
    y_axis = np.linspace(frameHeight, 0, num=5)

    # limiting to 100 points in array
    if len(plotInfo[3]) > 100:
        plotInfo[0].pop(0)
        plotInfo[1].pop(0)
        plotInfo[2].pop(0)
        plotInfo[3].pop(0)
        plotKalman[0].pop(0)
        plotKalman[1].pop(0)
        plotKalman[2].pop(0)

    # appending new values
    if info[0] == 0: # x
        plotInfo[0].append(frameWidth//2)
        plotKalman[0].append(frameWidth//2)
    else:
        plotInfo[0].append(info[0])
        plotKalman[0].append(X[0])
    
    if info[1] == 0: # y
        plotInfo[1].append(frameHeight//2)
        plotKalman[1].append(frameHeight//2)
    else:
        plotInfo[1].append(info[1])
        plotKalman[1].append(X[1])

    if info[3] == 0: # h
        plotInfo[2].append(200)
        plotKalman[2].append(200)
    else:
        plotInfo[2].append(info[3])
        plotKalman[2].append(X[2])

    plotInfo[3].append(loop)
  
    # Plotting

    # x-axis vs loop iteration
    ax[0].cla()
    ax[0].plot(plotInfo[0], plotInfo[3], color='r')
    ax[0].plot(plotKalman[0], plotInfo[3], color='b')
    
    ax[0].set_xticks(x_axis)
    ax[0].set_ylim(bottom=max(0, loop-100), top=loop+100)

    # y-axis vs loop iteration
    ax[1].cla()
    ax[1].plot(plotInfo[3], plotInfo[1], color='r')
    ax[1].plot(plotInfo[3], plotKalman[1], color='b')
    ax[1].invert_yaxis()
    
    ax[1].set_xlim(left=max(0, loop-100), right=loop+100)
    ax[1].set_yticks(y_axis)

    # forwardback vs loop iteration
    
    ax[2].cla()
    ax[2].plot(plotInfo[3], plotInfo[2], color='r')
    ax[2].plot(plotInfo[3], plotKalman[2], color='b')
    
    ax[2].set_xlim(left=max(0, loop-100), right=loop+100)
    ax[2].set_yticks(y_axis)
    
    fig.canvas.draw()

    # Return updated x,y,t array
    return plotInfo, plotKalman

def kalman(info, XOld, POld, Q, R):

    # reminders
    # t: transpose

    # state matrix measurements
    XM = np.array([info[0], info[1], info[3]]) # X measured

    # Transform matrices: A,B,C,I = I3
    # A = np.eye(3)
    # C = np.eye(3)
    I = np.eye(3)

    # Predict cycle
    X = XOld                          # X = A*X-1; predicted state estimate
    P = POld + Q                      # P = A*P-1*At + Q; predicted state variance

    # Update cycle
    K = P.dot(np.linalg.inv(P + R))   # K = P*Ct / (C*P*Ct + R); kalman gain
    XNew = X + K.dot(XM - X)         # X = X + K*(XM - C*X); new estimate
    PNew = (I - K).dot(P)             # P = (I - K * C) * P; new variance
    # print(f'Measured: {XM}\nPrediction: {XNew}')

    XNew = XNew.astype(int)
    return XNew, PNew