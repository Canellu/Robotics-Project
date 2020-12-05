from djitellopy import Tello
from matplotlib.animation import FuncAnimation
import cv2
import socket
import numpy as np
import time
import datetime
import psutil
import matplotlib.pyplot as plt


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



def initYOLO():
    # YOLO STUFF
    whT = 320 # A parameter for image to blob conversion

    # Import class names to list from coco.names
    classesFile = "../YOLOv3/anton.names"
    classNames = []
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # Set up model and network
    modelConfig = "../YOLOv3/yolov3_only_anton.cfg"
    modelWeights = "../YOLOv3/yolov3_only_anton.weights" 
    net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return classNames, net, whT


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
    percentH = 180 - (sliderVal-50)*4 # 1/6 * h + (sliderVal-50)*4 + h/10; exact value, 180 is an estimate for easier computation
                   # estimated head size = 25 cm, desired bounding box height = 1/6 frame height
                   # 720/6 = 120, 25 cm corresponds to 120 px, therefore: 1 cm = 4.8 px

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
    # testPrint()
    def testPrint():
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

    #  Up - down
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
            if count >= 100:
                droneStates.pop(0)
            droneStates.append(data.decode(encoding="utf-8"))
        except Exception as err:
            print(err)
            sock.close
            break
        
def drawOSD(droneStates, frame, pulse, mode, trackOn):
    # pitch:0;roll:0;yaw:0;vgx:0;vgy:0;vgz:0;templ:82;temph:85;tof:48;h:0;bat:20;baro:163.98;time:0;agx:6.00;agy:-12.00;agz:-1003.00;  
    
    
    states = droneStates[len(droneStates)-1].split(";")
    pitch = states[0][6:] 
    roll = states[1][5:] 
    yaw = states[2][4:] 
    vgx = states[3][4:]
    vgy = states[4][4:]
    vgz = states[5][4:]
    templ = states[6][6:] #
    temph = states[7][6:] #
    tof =  float(states[8][4:])/100 
    height = float(states[9][2:])/100 
    bat = states[10][4:]
    baro = int(float(states[11][5:])) 
    time = states[12][5:]
    agx = float(states[13][4:])//10
    agy = float(states[14][4:])//10
    agz = float(states[15][4:])//10
    avgTemp = (int(templ) + int(temph))//2

    def battFill(percent, img):

        percent = int(percent)
        width = 66 / 100 * percent
        start = 960
        end = int(start + width)
        if percent > 50:
            color = (0,255,0)
        elif percent > 20:
            color = (0, 195, 255)
        else:
            color = (0,0,255)

        if percent == 100:
            batStart = 958
        elif percent >= 10:
            batStart = 968
        else:
            batStart = 980

        cv2.rectangle(img, (start, 70), (end, 96), color, -1)
        cv2.putText(img, str(percent)+"%" , (batStart, 91), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)


    def placeIcon(img, iconPath, location, scale):
        icon = cv2.imread(iconPath)
        h, w, _ = icon.shape
        icon = cv2.resize(icon, (int(h*scale),int(w*scale)))
        h, w, _ = icon.shape
        xStart = location[0] - int(w/2)
        yStart = location[1] - int(h/2)
        xEnd = xStart + w
        yEnd = yStart + h
        # ROI
        img[yStart:yEnd, xStart:xEnd] = icon


    w = frame.shape[1]
    h = frame.shape[0]

    
    

    # REC 
    cv2.putText(frame, "LIVE" , (204,90), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    if pulse:
        cv2.circle(frame, (180,78), 12, (0,0,255), -1)

    # Battery
    cv2.rectangle(frame, (958, 68), (1028, 98), (255,255,255), 2)
    cv2.rectangle(frame, (1028, 78), (1032, 88), (255,255,255), 2)
    battFill(bat, frame)

    # Crosshair
    cv2.circle(frame, (w//2, h//2), 12, (255,255,255), 1)

     # TIME
    cv2.putText(frame, ('0'+str(datetime.timedelta(seconds=int(time)))) , (522, 83), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)

    # FPS
    cv2.putText(frame, '720p', (166,684), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

    # MODE
    cv2.putText(frame, 'Mode:', (954,650), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)
    if trackOn:
        if mode:
            cv2.putText(frame, 'rotation', (930,684), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
        else:
            cv2.putText(frame, 'translation', (894,684), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
    else:
        cv2.putText(frame, 'manual', (942,684), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

    # Line top right
    cv2.line(frame, (1050, 50), (917, 50), (255,255,255), 2)
    cv2.line(frame, (1050, 50), (1050, 150), (255,255,255), 2)

    # Line top left
    cv2.line(frame, (150, 50), (283, 50), (255,255,255), 2)
    cv2.line(frame, (150, 50), (150, 150), (255,255,255), 2)

    # Line bottom left
    cv2.line(frame, (150, 710), (283, 710), (255,255,255), 2)
    cv2.line(frame, (150, 710), (150, 610), (255,255,255), 2)

    # Line bottom right
    cv2.line(frame, (1050, 710), (917, 710), (255,255,255), 2)
    cv2.line(frame, (1050, 710), (1050, 610), (255,255,255), 2)


   

    # RIGHT PANEL

    # VELOCITY
    cv2.putText(frame, 'Velocity', (w-90,190), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, '(cm/s)', (w-90,210), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    placeIcon(frame, '../images/velocity.png', (w-60,242), 0.4)
    cv2.putText(frame, f'X: {vgx}', (w-94,286), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f'Y: {vgy}', (w-94,316), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f'Z: {vgz}', (w-94,346), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    # ACCELERATION
    cv2.putText(frame, 'Acceleration', (w-110,390), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, '(0.01g)', (w-90,410), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    placeIcon(frame, '../images/acceleration.png', (w-60,444), 0.4)
    cv2.putText(frame, f'X: {int(agx)}', (w-94,486), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f'Y: {int(agy)}', (w-94,516), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f'Z: {int(agz)}', (w-94,546), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

    
    # LEFT PANEL

    # ROTATION AXES
    cv2.putText(frame, 'Rotation axes', (8,210), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    placeIcon(frame, '../images/axes.png', (60,242), 0.5)
    cv2.putText(frame, f'X: {int(roll)}', (20,286), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f'Y: {int(pitch)}', (20,316), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f'Z: {int(yaw)}', (20,346), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

    # HEIGHT
    cv2.putText(frame, 'Height(m)', (24,390), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    placeIcon(frame, '../images/height.png', (60,430), 0.5)
    cv2.putText(frame, f'h: {height}', (16,480), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f'tof: {tof}', (16,510), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f'baro: {baro}', (16,540), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

    # TEMPERATURE
    cv2.putText(frame, 'Temp(c)', (24,600), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
    placeIcon(frame, '../images/temperature.png', (30,634), 0.5)
    cv2.putText(frame, f'{avgTemp}C', (50,640), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
    


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

    # # x-axis vs loop iteration
    # ax[0].cla()
    # ax[0].plot(plotInfo[0], plotInfo[3], color='r')
    # ax[0].plot(plotKalman[0], plotInfo[3], color='b')
    # ax[0].set_xticks(x_axis)
    # ax[0].set_ylim(bottom=max(0, loop-100), top=loop+100)

    # y-axis vs loop iteration
    ax.cla()
    ax.plot(plotInfo[3], plotInfo[1], color='r')
    ax.plot(plotInfo[3], plotKalman[1], color='b')
    ax.invert_yaxis()
    ax.set_xlim(left=max(0, loop-100), right=loop+100)
    ax.set_yticks(y_axis)

    # # forwardback vs loop iteration   
    # ax[2].cla()
    # ax[2].plot(plotInfo[3], plotInfo[2], color='r')
    # ax[2].plot(plotInfo[3], plotKalman[2], color='b')
    # ax[2].set_xlim(left=max(0, loop-100), right=loop+100)
    # ax[2].set_yticks(y_axis)
    
    fig.canvas.draw()

    # Return updated x,y,t array
    return plotInfo, plotKalman

def kalman(info, XOld, POld, Q, R):

    # reminders
    # t: transpose

    if info[3] == 0: # Testing if object is within view
        XNew = [480, 360, 180]
        PNew = POld

    else:
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



