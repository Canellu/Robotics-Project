import cv2
import numpy as np



cap = cv2.VideoCapture(0) # Get Webcam feed
whT = 416 # A parameter for image to blob conversion
confThreshold = 0.7 # Lower value, more boxes (but worse confidence per box)
nmsThreshold = 0.3 # Lower value, less overlaps

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


# Import class names to list from coco.names
classesFile = "anv.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Set up model and network
modelConfig = "yolov3_anv.cfg"
modelWeights = "yolov3_anv.weights" 
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Detect object and draw bounding boxes
def findObjects(outputs, img):
    hT, wT, _ = img.shape
    bbox = [] 
    classIndices = []
    confs = []

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
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2) # Draw bounding box
        cv2.putText(img, f'{classNames[classIndices[i]].upper()} {int(confs[i]*100)}%', 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2) #Write class name and % on bounding box


# Capture webcam
while True:

    ret, img = cap.read() #Reading image from Webcam
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False) #Convert Image to a format the network can take (blobs)
    net.setInput(blob) #Input the image to the network
    layerNames = net.getLayerNames() #Names of all layers in network (relu, conv etc...)
    outputNames = [(layerNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()] #Find names of output layers (3 layers)
    outputs = net.forward(outputNames)# Returns outputs as numpy.2darray from the layernames forwarded.
                                      # (rows , columns) = (boxnumber , ( 0-4 = cx,cy,w,h, score of how 
                                      # likely the box contain an object and how accurate the boundary 
                                      # box is, rest is probability per classes) )
    findObjects(outputs,img)
    img = rescale_frame(img, percent=150)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





















