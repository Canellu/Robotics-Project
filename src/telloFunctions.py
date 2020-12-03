from djitellopy import Tello
import cv2
import socket
import numpy as np
import time

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

        
        if(classNames[classIndices[i]] == 'Face'):
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
    bw = info[2]
    bh = info[3]

    # previous info
    pcx = pInfo[0]
    pcy = pInfo[1]
    pbw = pInfo[2]
    pbh = pInfo[3]

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
    # cv2.putText(frame, (f"eRotation       : {error[0]}\t speed: {speed[3]}") , (10, frame.shape[0]-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    # cv2.putText(frame, (f"eLeftRight       : {error[0]}\t speed: {speed[0]}") , (10, frame.shape[0]-40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    # cv2.putText(frame, (f"eUpdown         : {error[1]}\t speed: {speed[2]}") , (10, frame.shape[0]-70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    cv2.putText(frame, (f"eForwardBackward: {error[2]}\t speed: {speed[1]}") , (10, frame.shape[0]-100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    # cv2.putText(frame, (f"center x: {cx} center y: {cy}") , (10, frame.shape[0]-130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
    # cv2.putText(frame, (f"current info: {info}\n previous info: {pInfo}") , (10, frame.shape[0]-160), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))





    # Rotation / Translation
    if mode:
        # Rotation
        if cx != 0:
            drone.yaw_velocity = speed[3]
        else:
            drone.yaw_velocity = 0
            error[0] = 0
    else:
        print("CALCULATING TRANSLATION \n\n\n\n\n")
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
    if (bw * bh) != 0:
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
    

# Call inside loop to read slider
# @name = name of trackbar
# @frame = the window trackbar resides in
def readSlider(name, frame):
    return cv2.getTrackbarPos(name, frame)    
    