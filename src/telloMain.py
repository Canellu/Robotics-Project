from telloFunctions import *


frameWidth, frameHeight = 640, 360

drone = initializeTello()

while True:
    
    # Step 1
    img = telloGetFrame(drone, frameWidth, frameHeight)
    cv2.imshow('The Window', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        break
