from djitellopy import Tello
from cv2 import cv2

def initializeTello():

    drone = Tello()
    drone.connect()
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0

    print(drone.get_battery())
    drone.streamoff()
    drone.streamon()

    return drone


def telloGetFrame(drone, frameWidth=360, frameHeight=240):

    telloFrame = drone.get_frame_read()
    telloFrame = telloFrame.frame
    img = cv2.resize(telloFrame,(frameWidth,frameHeight))
    
    return img

def findFace(img):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2,4)

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)


def drawOSD(img, drone):


    # shape = (height, width, channels)
    print(img.shape) 
    dataToDisplay = []
    stateDict = drone.get_current_state()
    print(stateDict)