# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:13:45 2022

@author: murat
"""


import serial
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import pandas as pd
from datetime import datetime
import sys
from pynput import keyboard
import numpy as np
from collections import defaultdict

status= "0"

def on_press(key):
    global status
   
    status= str(key)      
    print(status)

ser=serial.Serial('COM5',9600)
ser.flushInput()
ser.close()
ser.open()
#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)
qt_keys = (
    (getattr(QtCore.Qt, attr), attr[4:])
    for attr in dir(QtCore.Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)
class KeyPressWindow(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)
        
def keymap(event):
    global status,keys_mapping
    status = keys_mapping[event.key()]
    print("status= " , status)
    

win = KeyPressWindow(show=True)
win.sigKeyPress.connect(keymap)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

p2 = win.addPlot(title="Multiple curves")
win.nextRow()
listem = np.zeros((100, 11),dtype="int32")
ptr = 0
def update():
    global curve, data, ptr, p6,listem, status
    inline = str(ser.readline().strip())
    inline=inline.replace("'","")
    inline=inline.replace("b","")
    #print(inline)c
    
    try:        
        info=inline.split(";")[:-1]
        now = datetime.now() # time object
        print("now =", now)
        a = np.array(info,dtype="int32").reshape((1,-1))
        listem = np.append(listem, a, axis=0)
        listem = np.delete(listem, (0), axis=0)

        p2.clear()
        p2.plot(listem[:,0], pen=(0,0,255), name="0 curve")
        p2.plot(listem[:,1], pen=(255,0,0), name="1 curve")
        p2.plot(listem[:,2], pen=(0,255,0), name="2 curve")
        p2.plot(listem[:,3], pen=(255,188,0), name="3 curve")
        p2.plot(listem[:,4], pen=(255,0,255), name="4 curve")
        p2.plot(listem[:,5], pen=(255,255,0), name="5 curve")
        p2.plot(listem[:,6], pen=(0,255,255), name="6 curve")
        p2.plot(listem[:,7], pen=(255,255,255), name="7 curve")
        p2.plot(listem[:,8], pen=(9,30,137), name="8 curve")
        p2.plot(listem[:,9], pen=(255,77,172), name="9 curve")
        p2.plot(listem[:,10], pen=(172,172,172), name="10 curve")
        with open("datatest_07.txt","a") as file:
            for i in info:
                file.write(i+' ')
            file.write(status+" "+str(now)+"\n")
            
    except:
        sys.exit(1)


timer = QtCore.QTimer()
timer.setInterval(10)
timer.timeout.connect(update)
timer.start(5)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

        

        
