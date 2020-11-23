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
        myFaceListC.append([cx,cy])

    if len(myFaceListArea) != 0:

        # finding closest face (biggest area)
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    
    else:
        return img,[[0,0],0]




def findFaceYolo(outputs, img, classNames):


    # Neural Network Params
    confThreshold = 0.9 # Lower value, more boxes (but worse confidence per box)
    nmsThreshold = 0.3 # Lower value, less overlaps

    hT, wT, _ = img.shape
    bbox = [] 
    classIndices = []
    confs = []
    cupArea = []

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
        
        if(classNames[classIndices[i]] == 'cup'):
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2) # Draw bounding box
            cv2.putText(img, f'{classNames[classIndices[i]].upper()} {int(confs[i]*100)}%', 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2) #Write class name and % on bounding box
            cupArea.append(area)
            
        
    if len(cupArea) != 0:

        # finding closest face (biggest area)
        i = cupArea.index(max(cupArea))
        cx = bbox[i][0] + bbox[i][2]/2
        cy = bbox[i][1] + bbox[i][3]/2
        return img, ([[cx,cy], cupArea[i]])
       
    else:
        return img, ([[0,0],0])


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
    # print(f"error: {error[2]}\t speed: {speed[2]}") # distance

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
    