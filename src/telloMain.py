from telloFunctions import *
from pynput.keyboard import Key, Listener, Controller
import time
import cv2
import threading
import numpy as np

# VARIABLES # ----------------------------------------

# Control panel (ALL STATIC, DONT CHANGE)
listener = None # To check wether listener thread is created
keyPressed = None # Value of pressed keys
trackOn = False # True to track object, false otherwise
mode = True  # True = Rotation, False = Translation
safeQuit = False # Do not change value. Safety measures.
plotOn = False # True to draw plots of X Y H.
OSDon = True # Turn on and off OSD


# Kalman variables, declarations
Q = np.array([[1.5, 0, 0], [0, 5, 0], [0, 0, 1.4]]) # Process noise
R = np.array([[80, 0, 0], [0, 200, 0],[0, 0, 90]]) # Measurement noise
X = np.array([480, 360, 180])
P = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])


# Plotting variables, parameters
countArray = []
plotInfo = [[],[],[],[]] # [x,y,h,loopCount] Do not change value
plotKalman = [[],[],[]] # Do not change value
loopCount = 0
updateCycle = 3


# Drone data
droneStates = []

# FPS
counter = 0
FPS = 0
startTime = time.time()
pulse = True # For Red dot on OSD to pulse


# PID data
pidY = [0.4, 0.6, 0] # Left right
pidX = [0.6, 0.75, 0] # Forward back
pidZ = [0.9, 1.2, 0] # Up down
pidYaw = [0.7, 0.2, 0] # Rotate
info = [0,0,0,0] # x, y, width, height
pInfo = [0, 0, 0] # x, y, height
pError = [0, 0, 0] # yaw, height, distance


# VARIABLES END # -----------------------------------------------

# Keyboard listener
def on_release(key):
    global keyPressed
    keyPressed = key.char
    print(f"KEY PRESSED: {keyPressed}")

def CheckWhichKeyIsPressed():  
    global listener 
    if listener == None:  
        listener = Listener(on_release=on_release,suppress=True)
        listener.start()
        print("CREATING LISTENER THREAD\n\n")


## MAIN PROGRAM STARTS HERE ## ----------------------------------

# Get drone object
connection, drone = initializeTello()

# Create objects 
if connection:

    # YOLO variables
    classNames, net, whT = initYOLO()

    # OSD Thread
    dataThread = threading.Thread(target=droneData, args=(droneStates,), daemon=True)
    dataThread.start()

    # Create plot
    if plotOn:
        fig, ax = plt.subplots(1)
        fig.show()


    # Create Distance slider
    distanceSlider("Display") 
    
    #qSlider("Display")




# Loop
while connection:
    
    # Get frame and size of frame from Tello
    img = telloGetFrame(drone)
    frameWidth, frameHeight = img.shape[1], img.shape[0]


    #Check wether to track object or not
    if trackOn:

        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False) #Convert Image to a format the network can take (blobs)
        net.setInput(blob) #Input the image to the network
        layerNames = net.getLayerNames() #Names of all layers in network (relu, conv etc...)
        outputNames = [(layerNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()] #Find names of output layers (3 layers)
        outputs = net.forward(outputNames)# Returns outputs as numpy.2darray from the layernames forwarded.
                                        # (rows , columns) = (boxnumber , ( 0-4 = cx,cy,w,h, score of how 
                                        # likely the box contain an object and how accurate the boundary 
                                        # box is, rest is probability per classes) )

        # Tracking methods: HAAR, YOLO
        # img, info = findFace(img) # HAAR
        img, info = findFaceYolo(outputs, img, classNames) # YOLO

        # Kalman
        # qVal = readSlider('Q Value', 'Display') # For testing purposes
        # Q = np.array([[(qVal/100),0],[0,(qVal/100)]])
        X, P = kalman(info, X, P, Q, R)

        # Plotting center coordinates
        if plotOn:
            if (loopCount % updateCycle) == 0:
                plotInfo, plotKalman = plot(frameWidth, frameHeight, fig, ax, info, X, loopCount, plotInfo, plotKalman)
            loopCount += 1


        # Read slider data
        distance = readSlider('Distance', 'Display')
        
        
        # Control drone movement to track object
        pInfo, pError = trackFace(drone, X, pInfo, frameWidth, frameHeight, pidY, pidX, pidZ, pidYaw, pError, distance, img, mode)


    

    if OSDon:
        # FPS
        counter+=1
        if (time.time() - startTime) > 1 :
            FPS = int(counter / (time.time() - startTime))
            if FPS > 30:
                FPS = 30
            counter = 0
            startTime = time.time()
            pulse = not pulse
        
    
        img = cv2.copyMakeBorder(img, 20, 20, 120, 120, cv2.BORDER_CONSTANT, value=(0,0,0))
        cv2.putText(img, 'FPS:', (166,650), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
        cv2.putText(img, str(FPS), (228,650), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)
        
        drawOSD(droneStates, img, pulse, mode, trackOn)

    # Show frames on window for 1ms
    cv2.imshow('Display', img)
    cv2.waitKey(1)

    # Function that detects a key-release and assign it to keyPressed
    CheckWhichKeyIsPressed()

    # To land and end connection
    if keyPressed == 'q':
        drone.end()
        plt.close('all')
        # print(f"VARIANCE X: {np.var(plotInfo[0])} LEN: {len(plotInfo[0])}") # Measurement variance in X
        # print(f"VARIANCE Y: {np.var(plotInfo[1])} LEN: {len(plotInfo[1])}") # Measurement variance in Y
        # print(f"VARIANCE FB: {np.var(plotInfo[2])} LEN: {len(plotInfo[2])}") # Measurement variance in Forward-Back
        safeQuit = True
        break
    
    # To take off
    if keyPressed == 'f':
        drone.takeoff()
   
    # To land drone
    if keyPressed == 'l':
        drone.land()

    # Enable/Disable tracking
    if keyPressed == 't':
        keyPressed = None
        if trackOn == True:
            trackOn = False
        else:
            trackOn = True

    # Change track mode
    if keyPressed == '1': # Rotation     
        mode = True
    if keyPressed == '2': # Translation
        mode = False
    
    # Enable/Disable OSD
    if keyPressed == 'o':
        OSDon = not OSDon

    
    keyPressed = None



# Safety measure
if not safeQuit:
    plt.close('all')
    drone.end() # If program ended without 'q'

cv2.destroyAllWindows()


