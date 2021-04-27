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


class Terrain(object):
    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        add some grids in the 3 point... that are just filling up the background
        """

        # setup the view window
        self.app = QtGui.QApplication(sys.argv)
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('Terrain')
        self.window.setGeometry(0, 110, 1920, 1080)
        self.window.setCameraPosition(distance=30, elevation=12)
        self.window.show()


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

        self.points = gl.GLScatterPlotItem(  # plot dots
            pos = keypoints,
            color=pg.glColor((0,255,0)),
            size=15
        )
        self.window.addItem(self.points)

        for n, pts in enumerate(self.connections):
            self.lines[n] = gl.GLLinePlotItem(  # lines dict with all of them
                pos=np.array([keypoints[p] for p in pts]),
                color=pg.glColor((0,0,255)),
                width=3,
                antialias=True
            )
            # add them to our window
            self.window.addItem(self.lines[n])

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
            pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])  # all 2d keypoints
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)  # convert numpy array from keypoints
        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)  # compute and get 3d keypoints

        keypoints = pose_3d[0].transpose()  # önceki videodaki resimde de yapmıştık

        return keypoints/80  # for keypoints show up the screen

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
            self.points.setData(pos=keypoints)  # update points
            for n, pts in enumerate(self.connections):
                self.lines[n].setData(
                    pos=np.array([keypoints[p] for p in pts])
                )

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


if __name__ == '__main__':
    os.chdir('..')
    t = Terrain()
    t.animation()