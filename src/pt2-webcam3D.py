"""
This serve as our base openGL class.
"""

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys

import cv2
import time
import os

from estimator import TfPoseEstimator
from lifting.prob_model import Prob3dPose
from networks import get_graph_path, model_wh
import common

import math
import threading
import tkinter as tk
# from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

class Terrain(object):
    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        add some grids in the 3 point... that are just filling up the background
        """

        # setup the view window
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setFixedSize(1920, 1080)
        self.window.showMaximized()
        # self.window.setMaximumSize(1920, 1080)
        # self.win = pg.GraphicsWindow()
        # self.vb = self.win.addViewBox(col=0, row=0)
        # self.t = pg.TextItem("zeynep", (255, 255, 255), anchor=(0,0))
        # self.vb.addItem(self.t)
        self.window.setWindowTitle('Terrain')
        self.window.setGeometry(0, 0, 1920, 1080)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()
        # self.root = Tk()
        self.angle = ""
        # self.root = Tk()
        # self.root.geometry("100x50+%d+%d" % (0, 0))  # top left
        # self.root.overrideredirect(True)  # frameless tkinter window
        # self.root.resizable(False, False)
        # self.root.columnconfigure(0, weight=1)
        # self.root.rowconfigure(0, weight=1)

        # TEXT = ""
        # self.lbl = tk.Label(root, text=TEXT, bg="#e61c1c", font=("bold", 15), border=0, width=35)
        # self.lbl.grid(column=0, row=0, sticky="nsew")

        gx = gl.GLGridItem()
        gy = gl.GLGridItem()
        gz = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gy.rotate(90, 1, 0, 0)
        gx.translate(-10, 0, 0)
        gy.translate(0, -10, 0)  # we translate all of them
        gz.translate(0, 0, -10)  # so they're pushed out to the background
        self.window.addItem(gx)
        self.window.addItem(gy)  # add them all to the window
        self.window.addItem(gz)

        model = "mobilenet_thin_432x368"
        camera = 0

        self.lines = {}
        self.connections = [  # lines that we want to connect all those key points
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
            [15, 16]
        ]  # 5.35

        w, h = model_wh(model)
        # we are gonna create e objects but instead we're gonna call it
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        # we're basically just going the same thing(run.py line:37) instead of args.model
        # we just created our own object for model
        self.cam = cv2.VideoCapture(camera)
        ret_val, image = self.cam.read()
        self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')
        # we'll have this object to do our 3d pose translater? yukardaki
        keypoints = self.mesh(image)
        # rightPointList = keypoints[2:5]
        self.rightPointList = keypoints[11:14]
        self.leftPointList = keypoints[14:]

        self.a = (gl.GLScatterPlotItem(  # plot dot
            pos = keypoints[:11],
            color=pg.glColor((0,255,0)),
            size=15
        ))
        self.window.addItem(self.a)
        self.right = (gl.GLScatterPlotItem(  # plot dot
            pos = self.rightPointList,
            color=pg.glColor((255,0,0)),
            size=15
        ))
        self.window.addItem(self.right)
        self.left = (gl.GLScatterPlotItem(  # plot dot
            pos = self.leftPointList,
            color=pg.glColor((0,0,255)),
            size=15
        ))
        self.window.addItem(self.left)

        for n, pts in enumerate(self.connections):
            self.lines[n] = gl.GLLinePlotItem(  # lines dict with all of them
                pos=np.array([keypoints[p] for p in pts]),
                color=pg.glColor((0,0,255)),
                width=3,
                antialias=True
            )
            # add them to our window
            self.window.addItem(self.lines[n])


    def armAngle(self, rightArmPoint, leftArmPoint):
        ra = np.array([rightArmPoint[0][0], rightArmPoint[0][1], rightArmPoint[0][2]])
        rb = np.array([rightArmPoint[1][0], rightArmPoint[1][1], rightArmPoint[1][2]])
        rc = np.array([rightArmPoint[2][0], rightArmPoint[2][1], rightArmPoint[2][2]])
        rba = ra - rb
        rbc = rc - rb

        rcosine_angle = np.dot(rba, rbc) / (np.linalg.norm(rba) * np.linalg.norm(rbc))
        rangle = np.arccos(rcosine_angle)

        la = np.array([leftArmPoint[0][0], leftArmPoint[0][1], leftArmPoint[0][2]])
        lb = np.array([leftArmPoint[1][0], leftArmPoint[1][1], leftArmPoint[1][2]])
        lc = np.array([leftArmPoint[2][0], leftArmPoint[2][1], leftArmPoint[2][2]])
        lba = la - lb
        lbc = lc - lb

        lcosine_angle = np.dot(lba, lbc) / (np.linalg.norm(lba) * np.linalg.norm(lbc))
        langle = np.arccos(lcosine_angle)

        self.angle = "sağ kol açısı: %.2f\nsol kol açısı: %.2f" % (np.degrees(rangle), np.degrees(langle))

        # th1 = threading.Thread(self.addAngleToPlot(self.angle))
        # th1 = threading.Thread(addAngleToPlot(self.angle))
        # th1.start()
        print(self.angle)
        return self.angle

    def addAngleToPlot(self, txtAngle):
        # lbl = Label(root, text=angle, bg="#e61c1c", font=("bold", 15), border=0, width=35)
        # global TEXT 
        root = tk.Tk()
        root.geometry("250x100+%d+%d" % (0, 0))  # top left
        root.overrideredirect(True)  # frameless tkinter window
        root.resizable(False, False)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        TEXT = txtAngle
        lbl = tk.Label(root, text=TEXT, bg="#000000", fg="white",font=("bold", 15), border=0, width=35)
        # lbl.config(text=TEXT)
        lbl.grid(column=0, row=0, sticky="nsew")
        root.mainloop()

    def mesh(self, image):  # pass image and want it to return keypoints
        image_h, image_w = image.shape[:2]
        standard_w = 640
        standard_h = 480
        pose_2d_mpiis = []
        visibilities = []
        humans = self.e.inference(image, scales=[None])
        # get humans, get 2d and 3d points
        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            # print("\n\n")    
            # print(pose_2d_mpii, visibility)  
            pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])  # all 2d keypoints
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)  # convert numpy array from keypoints
        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)  # compute and get 3d keypoints
        # print("pose_3d", pose_3d)
        keypoints = pose_3d[0].transpose()  # önceki videodaki resimde de yapmıştık
        # print("keypoints", keypoints)
        return keypoints/90  # for keypoints show up the screen

    def update(self):  #  update self.points
        """
        update the mesh and shift the noise each time
        """
        ret_val, image = self.cam.read()
        try:
            keypoints = self.mesh(image)
        except AssertionError:  #  it means i couldn't find any human so that human list is empty
            print('body not in image')
        else:  # if we don't get the error
            self.rightPointList = keypoints[11:14]
            self.leftPointList = keypoints[14:]

            self.a.setData(pos=keypoints[:11])
            self.right.setData(pos=self.rightPointList)
            self.left.setData(pos=self.leftPointList)

            # self.points.setData(pos=keypoints)  # update points
            for n, pts in enumerate(self.connections):
                self.lines[n].setData(
                    pos=np.array([keypoints[p] for p in pts])
                )

            txtAngle = self.armAngle(self.rightPointList, self.leftPointList)
            th1 = threading.Thread(target=self.addAngleToPlot, args=(txtAngle, ), daemon=True)
            th1.start()

            # self.armAngle(self.rightPointList)
            # self.armAngle(self.leftPointList)

    def start(self):
        """
        get the graphics window open and setup
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def animation(self, frametime=10):
        """
        calls the update method to run in a loop
        """
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)

        timer.start(frametime)
        self.start()
"""
def addAngleToPlot(txtAngle=""):
    root = tk.Tk()
    root.geometry("250x100+%d+%d" % (0, 0))  # top left
    root.overrideredirect(True)  # frameless tkinter window
    root.resizable(False, False)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    TEXT = txtAngle
    lbl = tk.Label(root, text=TEXT, bg="#e61c1c", font=("bold", 15), border=0, width=35)
    # lbl.config(text=TEXT)
    lbl.grid(column=0, row=0, sticky="nsew")
    root.mainloop()
"""
if __name__ == '__main__':
    os.chdir('..')

    # root = tk.Tk()
    # root.geometry("250x100+%d+%d" % (0, 0))  # top left
    # root.overrideredirect(True)  # frameless tkinter window
    # root.resizable(False, False)
    # root.columnconfigure(0, weight=1)
    # root.rowconfigure(0, weight=1)

    # TEXT = ""
    # lbl = tk.Label(root, text=TEXT, bg="#e61c1c", font=("bold", 15), border=0, width=35)
    # lbl.grid(column=0, row=0, sticky="nsew")

    # lbl = Label(root, text="", bg="#e61c1c", font=("bold", 15), border=0, width=35)
    # lbl.grid(column=0, row=0, sticky="nsew")
    t = Terrain()
    # th0 = threading.Thread(target=t)
    # th0.start()
    
    # root = Tk()
    # greeting = tk.Label(text="Hello, Tkinter")
    t.animation()
    # root.mainloop()
    # root.mainloop()