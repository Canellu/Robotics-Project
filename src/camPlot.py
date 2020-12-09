    # Create plot
    if plotOn:
        fig, ax = plt.subplots(ncols=2, nrows=2, constrained_layout=True, figsize=(8,8))
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        mngr.window.setGeometry(50,50,dx, dy)
        fig.show()


          # Plotting center coordinates
        if plotOn:
            if (loopCount % updateCycle) == 0:
                plotInfo, plotKalman = plot(frameWidth, frameHeight, fig, ax, info, X, loopCount, plotInfo, plotKalman, infoPID, pError, plotPID, plotError)
            loopCount += 1


            # Plotting variables, parameters
countArray = []
plotInfo = [[],[],[],[]] # [x,y,h,loopCount] Do not change value
plotKalman = [[],[],[]] # Do not change value
plotPID = [[],[],[]] # Do not change value
plotError = [[],[],[]] # Do not change value
loopCount = 0
updateCycle = 3

# FPS
counter = 0
FPS = 0
startTime = time.time()
pulse = True # For Red dot on OSD to pulse


 # qVal = readSlider('Q Value', 'Display') # For testing purposes
        # Q = np.array([[(qVal/50),0,0],[0,(qVal/50),0], [0,0,(qVal/50)]])

        # print(f"VARIANCE X: {np.var(plotInfo[0])} LEN: {len(plotInfo[0])}") # Measurement variance in X
    # print(f"VARIANCE Y: {np.var(plotInfo[1])} LEN: {len(plotInfo[1])}") # Measurement variance in Y
    # print(f"VARIANCE FB: {np.var(plotInfo[2])} LEN: {len(plotInfo[2])}") # Measurement variance in 


def qSlider(frame):
    maxVal = 200
    startVal = 100

    def nothing(var):
        pass
    sliderWindow = cv2.namedWindow(frame)
    cv2.createTrackbar("Q Value", frame, startVal, maxVal, nothing)


def plot(frameWidth, frameHeight, fig, ax, info, X, loop, plotInfo, plotKalman, infoPID, pError, plotPID, plotError):

    #plotInfo[0] = x
    #plotInfo[1] = y
    #plotInfo[2] = h
    #plotInfo[3] = loop

    #plotKalman[0] = x
    #plotKalman[1] = y
    #plotKalman[2] = h

    #plotPID[0] = speed left/right or yaw
    #plotPID[1] = speed for/back
    #plotPID[2] = speed up/down

    #pError[0] = error left/right or yaw (pixels)
    #pError[1] = error for/back (pixels)
    #pError[2] = error up/down (pixels)

    # defining axes
    x_axis = np.linspace(0, frameWidth, num=5)
    y_axis = np.linspace(frameHeight, 0, num=5)

    # limiting to 100 points in array
    if len(plotInfo[3]) > 100:
        for i in range(3):
            plotInfo[i].pop(0)
            plotKalman[i].pop(0)
            plotPID[i].pop(0)
            plotError[i].pop(0)

        plotInfo[3].pop(0)
        

    # appending new values
    if info[0] == 0: # x
        plotInfo[0].append(frameWidth//2)
        plotKalman[0].append(frameWidth//2)
    else:
        plotInfo[0].append(info[0])
        plotKalman[0].append(X[0])
    
    if info[1] == 0: # y
        plotInfo[1].append(frameHeight//2)
        plotKalman[1].append(frameHeight//2)
    else:
        plotInfo[1].append(info[1])
        plotKalman[1].append(X[1])

    if info[2] == 0: # h
        plotInfo[2].append(200)
        plotKalman[2].append(200)
    else:
        plotInfo[2].append(info[2])
        plotKalman[2].append(X[2])

    plotInfo[3].append(loop)

    for i in range(3):
        plotPID[i].append(infoPID[i])
        plotError[i].append(int(pError[i]))
    

    # Plotting
    fig.suptitle('Measurement vs Kalman', fontsize=16)
    

    # PID vs Error iteration
    ax[0].title.set_text('PID vs Error')
    ax[0].legend(('PID', 'Kalman'), loc='upper right', shadow=True)
    ax[0].set_xlabel('Kalman')
    ax[0].set_ylabel('Position')
    ax[0].set_title('Position vs Kalman')
    ax[0].cla()
    ax[0].plot(plotPID[0], plotInfo[3], color='b')
    ax[0].plot(plotError[0], plotInfo[3], color='r')
    ax[0].set_xticks(x_axis)
    ax[0].set_ylim(bottom=max(0, loop-100), top=loop+100)

    # x-axis vs loop iteration
    ax[1].title.set_text('Left - Right translation')
    ax[1].cla()
    ax[1].plot(plotInfo[0], plotInfo[3], color='r')
    ax[1].plot(plotKalman[0], plotInfo[3], color='b')
    ax[1].set_xticks(x_axis)
    ax[1].set_ylim(bottom=max(0, loop-100), top=loop+100)

    # y-axis vs loop iteration
    ax[2].title.set_text('Up - Down translation')
    ax[2].cla()
    ax[2].plot(plotInfo[3], plotInfo[1], color='r')
    ax[2].plot(plotInfo[3], plotKalman[1], color='b')
    ax[2].invert_yaxis()
    ax[2].set_xlim(left=max(0, loop-100), right=loop+100)
    ax[2].set_yticks(y_axis)

    # forwardback vs loop iteration   
    ax[3].title.set_text('Forward - Backward translation')
    ax[3].cla()
    ax[3].plot(plotInfo[3], plotInfo[2], color='r')
    ax[3].plot(plotInfo[3], plotKalman[2], color='b')
    ax[3].set_xlim(left=max(0, loop-100), right=loop+100)
    ax[3].set_yticks(y_axis)
    


    
    fig.canvas.draw()

    # Return updated x,y,t array
    return plotInfo, plotKalman