from telloFunctions import *
from pynput.keyboard import Key, Listener, Controller
import time
import cv2
import threading
import numpy as np




# Control panel (ALL STATIC, DONT CHANGE)
listener = None # To check wether listener thread is created
keyPressed = None # Value of pressed keys
trackOn = False # True to track object, false otherwise
flight = False # True = takeoff, false = land
mode = True  # True = Rotation, False = Translation
safeQuit = False # Do not change value. Safety measures.

# Drone data
droneStates = []


# PID data
pidY = [0.4, 0.6, 0] # Left right
pidX = [0.6, 0.75, 0] # Forward back
pidZ = [0.9, 1.2, 0] # Up down
pidYaw = [0.7, 0.9, 0] # Rotate
pInfo = [0, 0, 0, 0] # x, y, width, height
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
whT = 416 # A parameter for image to blob conversion

# Import class names to list from coco.names
classesFile = "../YOLOv3/face.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Set up model and network
modelConfig = "../YOLOv3/face.cfg"
modelWeights = "../YOLOv3/face.weights" 
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)




## MAIN PROGRAM STARTS HERE ##

# Get drone object
connection, drone = initializeTello()


# Start data recieve thread   
if connection:
    dataThread = threading.Thread(target=droneData, args=(droneStates,))
    dataThread.start()
    
     
# Distance slider
distanceSlider("Display") # Creates a slider


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

        # Tracking methods: HAAR, YOLO, 
        # img, info = findFace(img) # HAAR
        img, info = findFaceYolo(outputs, img, classNames) # YOLO

        
        # Read slider data
        distance = readSlider('Distance', 'Display')
        
        
        # Control drone movement to track object
        pInfo, pError = trackFace(drone, info, pInfo, frameWidth, frameHeight, pidY, pidX, pidZ, pidYaw, pError, distance, img, mode)

    
    # drawOSD(droneStates, img)

    
    # Show frames on window for 1ms
    cv2.imshow('Display', img)
    cv2.waitKey(1)

    # Function that detects a key-release and assign it to keyPressed
    CheckWhichKeyIsPressed()

    # To land and end connection
    if keyPressed == 'q':
        drone.land()
        drone.end()
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
    if keyPressed == '1':
        mode = True
    if keyPressed == '2':
        mode = False





# Safety measure
if not safeQuit:
    drone.end() # If program ended without 'q'

cv2.destroyAllWindows()






