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
from tkinter import *
# from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem

class Terrain(object):
    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        add some grids in the 3 point... that are just filling up the background
        """

        # setup the view window
        self.app = QtGui.QApplication(sys.argv)
        # self.window = gl.GLViewWidget()
        self.vb = pg.GraphicsWindow()
        self.vb.resize(920, 680)
        self.window = self.vb.addViewBox(col=0, row=0)
        self.windowText = self.vb.addViewBox(col=0, row=1)

        # setWindowTitle, setGeometry, setCameraPosition atladim
        self.vb.show()


        # self.win = pg.GraphicsWindow()
        # self.vb = self.win.addViewBox(col=0, row=0)
        # self.t = pg.TextItem("zeynep", (255, 255, 255), anchor=(0,0))
        # self.vb.addItem(self.t)



        # self.window.setWindowTitle('Terrain')
        # self.window.setGeometry(0, 110, 1920, 1080)
        # self.window.setCameraPosition(distance=30, elevation=12)
        # self.window.show()


        # self.root = Tk()

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

        self.a = (gl.GLScatterPlotItem(  # plot dots
            pos = keypoints[:11],
            color=pg.glColor((0,255,0)),
            size=15
        ))
        self.window.addItem(self.a)
        self.right = (gl.GLScatterPlotItem(  # plot dots
            pos = self.rightPointList,
            color=pg.glColor((255,0,0)),
            size=15
        ))
        self.window.addItem(self.right)
        self.left = (gl.GLScatterPlotItem(  # plot dots
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

    """
    def armAngle(self, armPoint):
        # print(armPoint)
        # print("args: ", armPoint)
        x1, y1, z1 = armPoint[0][0], armPoint[0][1], armPoint[0][2]
        x2, y2, z2 = armPoint[1][0], armPoint[1][1], armPoint[1][2]
        x3, y3, z3 = armPoint[2][0], armPoint[2][1], armPoint[2][2]

        # Find direction ratio of line AB
        ABx = x1 - x2
        ABy = y1 - y2
        ABz = z1 - z2

        # Find direction ratio of line BC
        BCx = x3 - x2
        BCy = y3 - y2
        BCz = z3 - z2

        # Find the dotProduct
        # of lines AB & BC
        dotProduct = (ABx * BCx +
                    ABy * BCy +
                    ABz * BCz)

        # Find magnitude of
        # line AB and BC
        magnitudeAB = (ABx * ABx +
                    ABy * ABy +
                    ABz * ABz)
        magnitudeBC = (BCx * BCx +
                    BCy * BCy +
                    BCz * BCz)

        # Find the cosine of
        # the angle formed
        # by line AB and BC
        angle = dotProduct
        angle /= math.sqrt(magnitudeAB *
                        magnitudeBC)

        # Find angle in radian
        angle = (angle * 180) / 3.14
        angle1 = round(abs(angle), 4)

        # Print angle
        print("acccciiii: ", angle1)
        self.addAngleToPlot(angle1)
    """

    def armAngle(self, armPoint):
        a = np.array([armPoint[0][0], armPoint[0][1], armPoint[0][2]])
        b = np.array([armPoint[1][0], armPoint[1][1], armPoint[1][2]])
        c = np.array([armPoint[2][0], armPoint[2][1], armPoint[2][2]])
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        print("aciiiiiiii: ", np.degrees(angle))
        # self.addAngleToPlot(angle1)


    def addAngleToPlot(self, angle):
        # self.window.paintGL(self, angle)
        # self.window.renderText(10,10,10, text=f"angle: {angle}")
        # gl.GLViewWidget.

        # t_up = pg.TextItem(str(angle), (0, 255, 0), anchor=(0, 0))
        # t_up.setPos(10,10)
        # self.window.addItem(t_up)

        self.t = pg.TextItem(str(angle), (255, 255, 255), anchor=(0,0))
        self.windowText.addItem(self.t)


        # label = Label( root, textvariable=angle, relief=RAISED )
        # label.pack()

        # t_up.setPos(i + 0.5, -j + 0.5)

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

            # self.armAngle(keypoints[11:14])
            # list1=(keypoints[11:14])
            # print(list1)
            # print("qqqqqqqqq")
            # self.armAngle(keypoints[14:])
            # print(list2)
            # print("aaaa")
            # print(type(list1))
            # bu kısmı ekleyince ketpoint ler e li falan oluyor, arguman tasmasi old.icin
            print(type(self.rightPointList))
            self.armAngle(self.rightPointList)
            self.armAngle(self.leftPointList)

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
        # t1 = threading.Thread(self.armAngle(self.rightPointList))
        # t2 = threading.Thread(self.armAngle(self.leftPointList))
        # t1.start()
        # t2.start()
        timer.start(frametime)
        self.start()


if __name__ == '__main__':
    os.chdir('..')
    # root = Tk()
    t = Terrain()
    # root = Tk()
    # greeting = tk.Label(text="Hello, Tkinter")
    t.animation()
    # root.mainloop()