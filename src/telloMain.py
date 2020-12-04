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
flight = False # True = takeoff, false = land
mode = True  # True = Rotation, False = Translation
safeQuit = False # Do not change value. Safety measures.
plotOn = False # True to draw plots of X Y H.


# Kalman variables, declarations
Q = np.array([[1.5, 0, 0], [0, 2, 0], [0, 0, 1.4]]) # Process noise
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


# PID data
pidY = [0.4, 0.6, 0] # Left right
pidX = [0.6, 0.75, 0] # Forward back
pidZ = [0.9, 1.2, 0] # Up down
pidYaw = [0.7, 0.9, 0] # Rotate
info = [0,0,0,0] # x, y, width, height
pInfo = [0, 0, 0] # x, y, height
pError = [0, 0, 0] # yaw, height, distance


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


## MAIN PROGRAM STARTS HERE ##

# Get drone object
connection, drone = initializeTello()

# Start data recieve thread
if connection:
    # OSD
    dataThread = threading.Thread(target=droneData, args=(droneStates,))
    dataThread.start()

    # Create plot
    if plotOn:
        fig, ax = plt.subplots(3)
        fig.show()

    # Distance slider
    distanceSlider("Display") # Creates a slider
    #qSlider("Display")


start_time = time.time()
x = 1
counter = 0
FPS = 0
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


    # drawOSD(droneStates, img)


    # FPS CALCS!
    counter+=1
    if (time.time() - start_time) > x :
        FPS = int(counter / (time.time() - start_time))
        if FPS > 30:
            FPS = 30
        counter = 0
        start_time = time.time()
    
    cv2.putText(img, (f'FPS: {FPS}'), (50,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    # Show frames on window for 1ms
    cv2.imshow('Display', img)
    cv2.waitKey(1)

    # Function that detects a key-release and assign it to keyPressed
    CheckWhichKeyIsPressed()

    # To land and end connection
    if keyPressed == 'q':
        drone.land()
        drone.end()
        plt.close('all')
        # print(f"VARIANCE X: {np.var(plotInfo[0])} LEN: {len(plotInfo[0])}") # Measurement variance in X
        # print(f"VARIANCE Y: {np.var(plotInfo[1])} LEN: {len(plotInfo[1])}") # Measurement variance in Y
        # print(f"VARIANCE FB: {np.var(plotInfo[2])} LEN: {len(plotInfo[2])}") # Measurement variance in Forward-Back
        safeQuit = True
        break
    
    # To take off or land
    if keyPressed == 'f':
        keyPressed = None
        if flight == False:
            flight = True
            drone.takeoff()
        else:
            flight = False
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





# Safety measure
if not safeQuit:
    plt.close('all')
    drone.end() # If program ended without 'q'

cv2.destroyAllWindows()


