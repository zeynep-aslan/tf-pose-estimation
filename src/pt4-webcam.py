import argparse
import logging
import time
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import os
import common
from lifting.prob_model import Prob3dPose
# import threading
# import tkinter as tk
import math

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
fps_time = 0
# strAngle = ""
# TEXT = ""

# def addAngleToPlot(txtAngle):

#     root = tk.Tk()
#     root.geometry("250x100+%d+%d" % (0, 0))  # top left
#     root.overrideredirect(True)  # frameless tkinter window
#     root.resizable(False, False)
#     root.columnconfigure(0, weight=1)
#     root.rowconfigure(0, weight=1)
#     TEXT = txtAngle
#     lbl = tk.Label(root, text=TEXT, bg="#000000", fg="white",font=("bold", 15), border=0, width=35)
#     # lbl.configure(text=TEXT)
#     lbl.grid(column=0, row=0, sticky="nsew")
#     root.mainloop()

def mesh(humans):
    standard_w = 640
    standard_h = 480
    pose_2d_mpiis = []
    visibilities = []
    for human in humans:
        pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
        # print(pose_2d_mpii, visibility)  
        pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])  # all 2d keypoints
        visibilities.append(visibility)

    pose_2d_mpiis = np.array(pose_2d_mpiis)  # convert numpy array from keypoints
    visibilities = np.array(visibilities)
    transformed_pose2d, weights = poseLifting.transform_joints(pose_2d_mpiis, visibilities)
    pose_3d = poseLifting.compute_3d(transformed_pose2d, weights)  # compute and get 3d keypoints
    # print("pose_3d", pose_3d)
    keypoints = pose_3d[0].transpose()  # onceki videodaki resimde de yapmıştık
    return keypoints/90  # for keypoints show up the screen

"""
def armAngle(rightArmPoint, leftArmPoint):
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

    # angle = "sag kol acisi: %.2f\nsol kol acisi: %.2f" % (np.degrees(rangle), np.degrees(langle))
    ranglee = "SAG KOL ACISI: %.2f" % (np.degrees(rangle))
    langlee = "SOL KOL ACISI: %.2f" % (np.degrees(langle))
    # print(angle)
    # return angle
    return ranglee, langlee


def calculateAngle(rightArmPoint, leftArmPoint):
						
	rABx = rightArmPoint[0][0] - rightArmPoint[1][0]
	rABy = rightArmPoint[0][1] - rightArmPoint[1][1]
	rABz = rightArmPoint[0][2] - rightArmPoint[1][2]

	rBCx = rightArmPoint[2][0] - rightArmPoint[1][0]
	rBCy = rightArmPoint[2][1] - rightArmPoint[1][1]
	rBCz = rightArmPoint[2][2] - rightArmPoint[1][2]

	rdotProduct = (rABx * rBCx +
				rABy * rBCy +
				rABz * rBCz)

	rmagnitudeAB = (rABx * rABx +
				rABy * rABy +
				rABz * rABz)
	rmagnitudeBC = (rBCx * rBCx +
				rBCy * rBCy +
				rBCz * rBCz)

	rangle = rdotProduct
	rangle /= math.sqrt(rmagnitudeAB *
					rmagnitudeBC)

	rangle = (rangle * 180) / 3.14


	ranglee = "SAG KOL ACISI: %.2f" % (round(abs(rangle), 4))

	# -----------------------------------------------

	lABx = leftArmPoint[0][0] - leftArmPoint[1][0]
	lABy = leftArmPoint[0][1] - leftArmPoint[1][1]
	lABz = leftArmPoint[0][2] - leftArmPoint[1][2]

	lBCx = leftArmPoint[2][0] - leftArmPoint[1][0]
	lBCy = leftArmPoint[2][1] - leftArmPoint[1][1]
	lBCz = leftArmPoint[2][2] - leftArmPoint[1][2]

	ldotProduct = (lABx * lBCx +
				lABy * lBCy +
				lABz * lBCz)

	lmagnitudeAB = (lABx * lABx +
				lABy * lABy +
				lABz * lABz)
	lmagnitudeBC = (lBCx * lBCx +
				lBCy * lBCy +
				lBCz * lBCz)

	langle = ldotProduct
	langle /= math.sqrt(lmagnitudeAB * lmagnitudeBC)

	langle = (langle * 180) / 3.14

	langlee = "SOL KOL ACISI: %.2f" % (round(abs(langle), 4))

	return ranglee, langlee

def armAngle(rightArmPoint, leftArmPoint):
    ra = np.array([rightArmPoint[0][0], rightArmPoint[0][1]])
    rb = np.array([rightArmPoint[1][0], rightArmPoint[1][1]])
    rc = np.array([rightArmPoint[2][0], rightArmPoint[2][1]])
    rba = ra - rb
    rbc = rc - rb

    rcosine_angle = np.dot(rba, rbc) / (np.linalg.norm(rba) * np.linalg.norm(rbc))
    rangle = np.arccos(rcosine_angle)

    la = np.array([leftArmPoint[0][0], leftArmPoint[0][1]])
    lb = np.array([leftArmPoint[1][0], leftArmPoint[1][1]])
    lc = np.array([leftArmPoint[2][0], leftArmPoint[2][1]])
    lba = la - lb
    lbc = lc - lb

    lcosine_angle = np.dot(lba, lbc) / (np.linalg.norm(lba) * np.linalg.norm(lbc))
    langle = np.arccos(lcosine_angle)

    # angle = "sag kol acisi: %.2f\nsol kol acisi: %.2f" % (np.degrees(rangle), np.degrees(langle))
    ranglee = "SAG KOL ACISI: %.2f" % (np.degrees(rangle))
    langlee = "SOL KOL ACISI: %.2f" % (np.degrees(langle))
    # print(angle)
    # return angle
    return ranglee, langlee
"""
def calculateArmAngle(rightArmDots, leftArmDots):
    try:
        # right arm angle
        ra = np.array([rightArmDots[0][0], rightArmDots[0][1]])
        rb = np.array([rightArmDots[1][0], rightArmDots[1][1]])
        rc = np.array([rightArmDots[2][0], rightArmDots[2][1]])

        rba = ra - rb
        rbc = rc - rb

        rcosine_angle = np.dot(rba, rbc) / (np.linalg.norm(rba) * np.linalg.norm(rbc))
        rangle = np.arccos(rcosine_angle)
        ranglee = str(np.degrees(rangle))
        print(ranglee)
    except:
        ranglee = "sagda eksik var"
    try:
        # left arm angle
        la = np.array([leftArmDots[0][0], leftArmDots[0][1]])
        lb = np.array([leftArmDots[1][0], leftArmDots[1][1]])
        lc = np.array([leftArmDots[2][0], leftArmDots[2][1]])

        lba = la - lb
        lbc = lc - lb

        lcosine_angle = np.dot(lba, lbc) / (np.linalg.norm(lba) * np.linalg.norm(lbc))
        langle = np.arccos(lcosine_angle)
        langlee = str(np.degrees(langle))
        print(langlee)
    except:
        langlee = "solda eksik var"

    return ranglee, langlee



if __name__ == '__main__':
    os.chdir('..')
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    # logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    w, h = model_wh(args.model)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    # logger.debug('cam read+')
        
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

    # th1 = threading.Thread(target=addAngleToPlot, args=(strAngle, ), daemon=True)
    # th1.start()

    while True:
        ret_val, image = cam.read()

        # logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        # logger.debug('image process+')
        humans = e.inference(image)
        try:
            keypoints = mesh(humans)
        except AssertionError:  #  it means i couldn't find any human so that human list is empty
            print('body not in image')
        else:  # if we don't get the error    
            rightArmPointList = keypoints[5:8]
            leftArmPointList = keypoints[2:5]
            # strAngle = armAngle(rightArmPointList, leftArmPointList)
            # rangle, langle = armAngle(rightArmPointList, leftArmPointList)
            # rangle, langle = calculateAngle(rightArmPointList, leftArmPointList)
            
            # th1 = threading.Thread(target=addAngleToPlot, args=(strAngle, ), daemon=True)
            # th1.start()

            # logger.debug('postprocess+')
            image, rightArmDots, leftArmDots = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            rangle, langle = calculateArmAngle(rightArmDots, leftArmDots)
            image[:70, :200] = (0,0,0)
            # logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 0, 255), 2)
            try:
                cv2.putText(image,
                            rangle,
                            (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            except:
                print("sağ listede eksik var")                
            try:
                cv2.putText(image,
                            langle,
                            (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            except:
                print("sol listede eksik var")                

            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
            # logger.debug('finished+')

    # root.mainloop()
    cv2.destroyAllWindows()
