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