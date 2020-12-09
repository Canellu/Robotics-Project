from telloFunctions import *
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QWidget
from pyqtgraph.Qt import QtCore, QtGui
from random import randint


def update_plot_data():

    global cycle
    global plotInfo

    global line1
    global line2
    global line3

    ret, frame = cap.read()
    # My webcam yields frames in BGR format

    outputs = progYOLO(frame, net, whT)
    info = findObjectYOLO(outputs, frame, classNames, 0) # YOLO


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
    pix = QtGui.QPixmap.fromImage(img)
    camFrame.setPixmap(pix)

    if info[0] == 0: # x
        plotInfo[0].append(frame.shape[1]//2)
    else:
        plotInfo[0].append(info[0])
    
    if info[1] == 0: # y
        plotInfo[1].append(frame.shape[0]//2)
    else:
        plotInfo[1].append(info[1])
    
    if info[2] == 0: # h
        plotInfo[2].append(200)
    else:
        plotInfo[2].append(info[2])

    cycle.append(cycle[-1] + 1)  # Add a new value 1 higher than the last.

    if len(cycle) >= 100:
        line1.getViewBox().enableAutoRange(axis='y', enable=True)
        line2.getViewBox().enableAutoRange(axis='x', enable=True)
        line3.getViewBox().enableAutoRange(axis='x', enable=True)

    if len(cycle) > 100:
        for i in range(3):
            plotInfo[i].pop(0) 
        cycle.pop(0)


    line1.setData(plotInfo[0], cycle)  # Update the data.
    line2.setData(cycle, plotInfo[1])  # Update the data.
    line3.setData(cycle, plotInfo[2])  # Update the data.


classNames, net, whT = initYOLO()

cap = cv2.VideoCapture(0)

app = QApplication([])
win = QMainWindow()


# Data

# cycle = list(range(10))
cycle = [0]

plotInfo = [[],[],[]] # [x,y,h,loopCount] Do not change value

# plotInfo[0] = [randint(0,100) for _ in range(10)]
# plotInfo[1] = [randint(0,100) for _ in range(10)]
# plotInfo[2] = [randint(0,100) for _ in range(10)]

plotInfo[0].append(320)
plotInfo[1].append(240)
plotInfo[2].append(200)


# Creating Plot


# Left/Right
plot1Widget = pg.PlotWidget()
plot1Widget.setTitle("Left/Right", color='r', size='20pt')
plot1Widget.setLabel('left', "<span style=\"color:red;font-size:20px\">Pixels</span>")
plot1Widget.setLabel('bottom', "<span style=\"color:red;font-size:20px\">Cycle</span>")
plot1Widget.addLegend()
plot1Widget.setXRange(0, 640)
plot1Widget.setYRange(max(0, cycle[0]-100), (cycle[0]+100))

pen1 = pg.mkPen(color=(255, 0, 0), style=QtCore.Qt.DashLine)
line1 = plot1Widget.plot(cycle, plotInfo[0], name='temp', pen=pen1)



# Up/Down
plot2Widget = pg.PlotWidget()
plot2Widget.setTitle("Up/Down", color='r', size='20pt')
plot2Widget.setLabel('left', "<span style=\"color:red;font-size:20px\">Cycle</span>")
plot2Widget.setLabel('bottom', "<span style=\"color:red;font-size:20px\">Pixels</span>")
plot2Widget.addLegend()
plot2Widget.setXRange(max(0, cycle[0]-100), (cycle[0]+100))
plot2Widget.setYRange(0, 480)


pen2 = pg.mkPen(color=(255, 0, 0), style=QtCore.Qt.DashLine)
line2 = plot2Widget.plot(plotInfo[1], cycle, name='temp', pen=pen2)
line2.getViewBox().invertY(True)


# Forward/Back
plot3Widget = pg.PlotWidget()
plot3Widget.setTitle("Forward/Back", color='r', size='20pt')
plot3Widget.setLabel('left', "<span style=\"color:red;font-size:20px\">Pixels</span>")
plot3Widget.setLabel('bottom', "<span style=\"color:red;font-size:20px\">Cycle</span>")
plot3Widget.addLegend()
plot3Widget.setXRange(max(0, cycle[0]-100), (cycle[0]+100))
plot3Widget.setYRange(0, 480)


pen3 = pg.mkPen(color=(255, 0, 0), style=QtCore.Qt.DashLine)
line3 = plot3Widget.plot(cycle, plotInfo[2], name='temp', pen=pen3)

plot1Widget.setBackground('w')
plot2Widget.setBackground('w')
plot3Widget.setBackground('w')

# webcam

camFrame = QtGui.QLabel()

# update 

# graphs
timer = QtCore.QTimer()
timer.setInterval(50)
timer.timeout.connect(update_plot_data)
timer.start()

# # cam
# timer1 = QtCore.QTimer()
# timer1.setInterval(200)
# timer1.timeout.connect(getFrame)
# timer1.start()






# Creating window layout
central_widget = QWidget()

layout = QVBoxLayout(central_widget)

groupBox1 = QGroupBox()
groupBox2 = QGroupBox()

hBoxTop = QHBoxLayout()
hBoxTop.addWidget(camFrame)
hBoxTop.addWidget(plot1Widget)

groupBox1.setLayout(hBoxTop)

hBoxBot = QHBoxLayout()
hBoxBot.addWidget(plot2Widget)
hBoxBot.addWidget(plot3Widget)

groupBox2.setLayout(hBoxBot)

layout.addWidget(groupBox1)
layout.addWidget(groupBox2)



# Run program

win.setCentralWidget(central_widget)
win.show()
app.exit(app.exec_())
cap.release()