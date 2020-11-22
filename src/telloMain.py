from telloFunctions import *

frameWidth, frameHeight = 360, 240
# automatic scaling
# frameWidth = 1.5 * frameHeight

pidYaw = [0.5, 0.5, 0]
pidZ = [0.7, 0.6, 0]
pidX = [0.5, 0.2, 0]
pError = [0, 0, 0] # yaw, height, distance
startCounter = 0 # fly = 0, no fly = 1
quit = 0
# manualControl = True

drone = initializeTello()

# Flight
if startCounter == 0:
    drone.takeoff()
    startCounter = 1

while quit != 1:
    
    # Step 1
    img = telloGetFrame(drone, frameWidth, frameHeight)

    # Step 2
    img, info = findFace(img)

    # Step 3
    pError = trackFace(drone, info, frameWidth, frameHeight, pidYaw, pidX, pidZ, pError)

    # switch manual/tracking WIP

    # printing y center coordinate
    # print(info[0][1])

    img = rescale_frame(img, percent=200)
    cv2.imshow('The Window', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        quit = 1
        break
    # elif cv2.waitKey(1) & 0xFF == ord('m'):
    #     manualControl = !manualControl

drone.end()