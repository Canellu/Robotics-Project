from telloFunctions import *
from pynput.keyboard import Key, Listener
import time
import cv2
import threading
import numpy as np


# VARIABLES # ----------------------------------------

# Control panel
listener = None # To check wether listener thread is created
keyPressed = None # Value of pressed keys
trackOn = False # True to track object, false otherwise
mode = True  # True = Rotation, False = Translation
OSDon = True # Turn on and off OSD


# Plotting variables, parameters
countArray = []
plotInfo = [[],[],[],[]] # [x,y,h,loopCount] Do not change value
plotKalman = [[],[],[]] # Do not change value
plotPID = [[],[],[]] # Do not change value
plotError = [[],[],[]] # Do not change value
loopCount = 0
updateCycle = 3

# FPS
counter = 0
FPS = 0
startTime = time.time()
pulse = True # For Red dot on OSD to pulse

# Drone data
droneStates = []
S = 50
classNumber = 0
trackMethod = 0

# Kalman variables, declarations
Q = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 1.4]]) # Process noise
R = np.array([[50, 0, 0], [0, 100, 0],[0, 0, 90]]) # Measurement noise
X = np.array([480, 360, 180])
P = np.array([[15, 0, 0],[0, 35, 0], [0, 0, 15]])

# PID data
pidY = [0.25, 0.02, 0.2] # Left right
pidX = [0.5, 0, 0.2] # Forward back
pidZ = [0.6, 0.2, 0.2] # Up down
pidYaw = [0.4, 0.1, 0.4] # Rotate
info = [0,0,0] # x, y, width, height
pInfo = [0, 0, 0] # x, y, height
pError = [0, 0, 0] # yaw, height, distance


# VARIABLES END # -----------------------------------------------


# Keyboard listener
def on_release(key):
    global keyPressed

    try:
        if key.char == 'q':
            keyPressed = key.char
        elif key.char == 'c':
            keyPressed = key.char
        elif key.char == 'b':
            keyPressed = key.char
        elif key.char == '1':
            keyPressed = key.char
        elif key.char == '2':
            keyPressed = key.char    
        elif key.char == 'f':
            keyPressed = key.char
        elif key.char == 'l':
            keyPressed = key.char
        elif key.char == 'm':
            keyPressed = key.char
        elif key.char == 't':
            keyPressed = key.char
        elif key.char == 'o':
            keyPressed = key.char
        elif key.char == 'w' or key.char == 's':
            drone.up_down_velocity = 0
        elif key.char == 'a' or key.char == 'd':
            drone.yaw_velocity = 0
    except AttributeError:
        if key == Key.left:
            drone.left_right_velocity = 0
        elif key == Key.right:
            drone.left_right_velocity = 0
        elif key == Key.up:
            drone.for_back_velocity = 0
        elif key == Key.down:
            drone.for_back_velocity = 0


def on_press(key):
    global keyPressed

    try:
        if key.char == 'w':
            drone.up_down_velocity = drone.speed
        elif key.char == 's':
            drone.up_down_velocity = -drone.speed
        elif key.char == 'a':
            drone.yaw_velocity = -drone.speed
        elif key.char == 'd':
            drone.yaw_velocity = drone.speed
    except AttributeError:
        if key == Key.left:
            drone.left_right_velocity = -drone.speed
        elif key == Key.right:
            drone.left_right_velocity = drone.speed
        elif key == Key.up:
            drone.for_back_velocity = drone.speed
        elif key == Key.down:
            drone.for_back_velocity = -drone.speed


def CheckWhichKeyIsPressed():  
    global listener 
    if listener == None:  
        listener = Listener(on_release=on_release, on_press=on_press, daemon=True)
        listener.start()
        print("CREATING LISTENER THREAD\n\n")


## MAIN PROGRAM STARTS HERE ## ----------------------------------

# Get drone object
connection, drone = initializeTello()

# Create objects 
if connection:
    print("----- Connection to drone succeeded -----")

    # Drone var
    drone.speed = S

    # YOLO variables
    classNames, net, whT = initYOLO()

    # OSD Thread
    dataThread = threading.Thread(target=droneData, args=(droneStates,), daemon = True)
    dataThread.start()




    # Create Distance slider
    distanceSlider("Display") 
    
    #qSlider("Display")

    # Function that creates listener on different thread that detects a key press/release
    CheckWhichKeyIsPressed()
    
else:
    print("----- Connection to drone failed -----")


# Loop
while connection:
    
    # Get frame and size of frame from Tello
    img = telloGetFrame(drone)
    frameWidth, frameHeight = img.shape[1], img.shape[0]

    #Check wether to track object or not
    if trackOn:

        

        # Tracking methods: HAAR, YOLO, HSV

        if trackMethod == 0:
            outputs = progYOLO(img, net, whT)
            info = findObjectYOLO(outputs, img, classNames, classNumber) # YOLO

        elif trackMethod == 1:
            info = findObjectHaar(img) # HAAR

        else:
            info = findObjectHSV(img) # HSV
        


        
        distance = readSlider('Distance', 'Display') # Read slider data
        XInit[2] = 180 - (distance-50)*2 # Reset init values based on slider

        # Kalman
       
        X, P = kalman(info, X, P, Q, R)
        
        # Control drone movement to track object
        pInfo, pError, infoPID = trackObject(drone, X, pInfo, frameWidth, frameHeight, pidY, pidX, pidZ, pidYaw, pError, distance, img, mode)


       

    else:

        updateMovement(drone)
    

    

    if OSDon:
        # FPS
        counter+=1
        if (time.time() - startTime) > 1 :
            FPS = int(counter / (time.time() - startTime))
            # if FPS > 30:
            #     FPS = 30
            counter = 0
            startTime = time.time()
            pulse = not pulse
        
    
        img = cv2.copyMakeBorder(img, 20, 20, 120, 120, cv2.BORDER_CONSTANT, value=(0,0,0))
        cv2.putText(img, 'FPS:', (166,650), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
        cv2.putText(img, str(FPS), (228,650), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)
        
        drawOSD(droneStates, img, pulse, mode, trackOn, classNames, classNumber, trackMethod)

    # Show frames on window for 1ms
    cv2.imshow('Display', img)
    cv2.waitKey(1)

    

    # To land and end connection
    if keyPressed == 'q':
        cv2.destroyAllWindows()
        drone.end()
        
        break
    
    # To take off
    elif keyPressed == 'f':
       drone.takeoff()
   
    # To land drone
    elif keyPressed == 'l':
        drone.land()


    # Enable/Disable tracking
    elif keyPressed == 't':
        trackOn = True
    elif keyPressed == 'm':
        trackOn = False
        updateMovement(drone, resetMove=True)

    # Change track mode
    elif keyPressed == '1': # Rotation     
        mode = True
    elif keyPressed == '2': # Translation
        mode = False
    
    # Enable/Disable OSD
    elif keyPressed == 'o':
        OSDon = not OSDon

    elif keyPressed == 'c':
        if classNumber == len(classNames)-1:
            classNumber = 0
        else:
            classNumber += 1

    elif keyPressed == 'b':
        if trackMethod == 2:
            trackMethod = 0
        else:
            trackMethod += 1

    keyPressed = None



drone.end()
cv2.destroyAllWindows()