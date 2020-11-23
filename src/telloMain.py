from telloFunctions import *
import time
import cv2
import numpy as np

whT = 320 # A parameter for image to blob conversion

# Import class names to list from coco.names
classesFile = "../YOLOv3/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Set up model and network
modelConfig = "../YOLOv3/yolov3.cfg"
modelWeights = "../YOLOv3/yolov3.weights" 
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)





frameWidth, frameHeight = 360, 240
# automatic scaling
# frameWidth = 1.5 * frameHeight

# PID data
pidYaw = [0.5, 0.5, 0]
pidZ = [0.7, 0.6, 0]
pidX = [0.5, 0.2, 0]
pInfo = [[0, 0], 0] # [x,y], area
pError = [0, 0, 0] # yaw, height, distance


# Control Panel Parameters
startFlight = False # Controls takeoff at start. True to takeoff.
manualControl = True
safeQuit = False # Do not change value.

# Get drone object
connection, drone = initializeTello()

# Flight
if startFlight:
    drone.takeoff()
    # drone.move_up(40) # Uncomment if starting from ground/floor


# Display info
distanceSlider("Display", frameWidth, frameHeight) # Creates a slider


while connection:
    
    # Step 1
    img = telloGetFrame(drone, frameWidth, frameHeight)

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False) #Convert Image to a format the network can take (blobs)
    net.setInput(blob) #Input the image to the network
    layerNames = net.getLayerNames() #Names of all layers in network (relu, conv etc...)
    outputNames = [(layerNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()] #Find names of output layers (3 layers)
    outputs = net.forward(outputNames)# Returns outputs as numpy.2darray from the layernames forwarded.
                                      # (rows , columns) = (boxnumber , ( 0-4 = cx,cy,w,h, score of how 
                                      # likely the box contain an object and how accurate the boundary 
                                      # box is, rest is probability per classes) )

    # Step 2 
    # img, info = findFace(img)
    img, info = findFaceYolo(outputs, img, classNames)

    # Step 3 Control drone movement to track object
    pInfo, pError = trackFace(drone, info, pInfo, frameWidth, frameHeight, pidYaw, pidX, pidZ, pError)

    

    # # Draw OSD and Slider
    # distance = readSlider('Distance', 'Display') # Read value from slider
    # cv2.putText(img, str(distance), (0,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
    
    # drawOSD(drone, img, frameWidth, frameHeight)

    img = rescale_frame(img, percent=300)
    cv2.imshow('Display', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.end()
        safeQuit = True
        break
 

if not safeQuit:
    print(drone.get_battery()) # temp to check battery
    drone.end() # If program ended without 'q'

cv2.destroyAllWindows()






