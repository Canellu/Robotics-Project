from telloFunctions import *

frameWidth, frameHeight = 360, 240
# automatic scaling
# frameWidth = 1.5 * frameHeight

pidYaw = [0.5, 0.5, 0]
pidZ = [0.7, 0.6, 0]
pidX = [0.5, 0.2, 0]
pInfo = [[0, 0], 0] # [x,y], area
pError = [0, 0, 0] # yaw, height, distance

startCounter = 0 # fly = 1, no fly = 0
manualControl = True

drone = initializeTello()

# Flight
if startCounter == 1:
    drone.takeoff()

while True:
    
    # Step 1
    img = telloGetFrame(drone, frameWidth, frameHeight)

    # Step 2
    img, info = findFace(img)

    # Step 3
    pInfo, pError = trackFace(drone, info, pInfo, frameWidth, frameHeight, pidYaw, pidX, pidZ, pError)

    # switch manual/tracking WIP

    img = rescale_frame(img, percent=200)
    cv2.imshow('The Window', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        break
    # elif cv2.waitKey(1) & 0xFF == ord('m'):
    #     manualControl = !manualControl

print(drone.get_battery()) # temp to check battery
drone.end()