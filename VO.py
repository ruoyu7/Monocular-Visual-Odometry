#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:19:50 2019

@author: liuruoyu
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time



class VisualOdometry():
 
    
    def __init__(self):
        
        self.image_cur = None
        self.image_ref = None
        self.camera_matrix = np.array([[718.8560, 0.0, 607.1928],
                                      [0.0, 718.8560, 185.2157],
                                      [0.0, 0.0, 1.0]])
        self.R_cur = None
        self.t_cur = None
        self.points3D_ref = None
        self.points3D_cur = None
        self.points2D_ref = None
        self.points2D_cur = None
        self.true_poses = None
        self.image_index = None
        self.scale = 1

    
    def FeaturesDetect(self, image):
        
        FAST = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
        keypoints = FAST.detect(image)
        
        #SIFT = cv2.xfeatures2d.SIFT_create()
        #keypoints = SIFT.detect(image, None)
        
        points = []
        for i in keypoints:
            points.append(i.pt)
        points = np.array(points, dtype=np.float32)
        points = points.reshape(-1,2)
        
        return points
 
    
    #Optical Flow Field
    #for the second frame       
    def OpticalFlowField_Second(self):
        
        lk_params = dict(winSize  = (21, 21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        points_ref = self.FeaturesDetect(self.image_ref)
        points_cur, status, error = cv2.calcOpticalFlowPyrLK(self.image_ref, self.image_cur, points_ref, None, **lk_params) 
        status = status.reshape(1, -1)[0]
        points_ref = points_ref[status == 1]
        points_cur = points_cur[status == 1]    
        self.points2D_ref = points_ref
        self.points2D_cur = points_cur
        
        E, mask = cv2.findEssentialMat(points_cur, points_ref, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, points_cur, points_ref, self.camera_matrix)
        
        return R, t


    #Optical Flow Field
    #for all the other frames except 1er and 2nd frame
    def OpticalFlowField(self):
        
        lk_params = dict(winSize  = (21, 21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        points_ref = self.points2D_cur
        points_cur, status, error = cv2.calcOpticalFlowPyrLK(self.image_ref, self.image_cur, points_ref, None, **lk_params) 
        status = status.reshape(1, -1)[0]
        self.points2D_ref = points_ref[status == 1]
        self.points2D_cur = points_cur[status == 1]    
        
        E, mask = cv2.findEssentialMat(self.points2D_cur, self.points2D_ref, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.points2D_cur, self.points2D_ref, self.camera_matrix)
        
        return R, t
    
    
    #2D->3D
    def Triangulation(self, R, t):
        
        pm1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        pm1 = self.camera_matrix.dot(pm1)
        pm2 = self.camera_matrix.dot(np.hstack((R, t)))

        return cv2.triangulatePoints(pm1, pm2, self.points2D_ref.reshape(2,-1), self.points2D_cur.reshape(2,-1))
    
    
    def RelativeScale(self):
        
        distance_ref = 0
        distance_cur = 0
        #for i in range(min(len(self.points3D_ref[0]), len(self.points3D_cur[0]))-1):
        for i in range(len(self.points3D_ref[0])-1):
            distance_ref = distance_ref\
             +((self.points3D_ref[0][i]-self.points3D_ref[0][i+1])**2\
             +(self.points3D_ref[1][i]-self.points3D_ref[1][i+1])**2\
             +(self.points3D_ref[2][i]-self.points3D_ref[2][i+1])**2)**(1/2)
        for i in range(len(self.points3D_cur[0])-1):
            distance_cur = distance_cur\
             +((self.points3D_cur[0][i]-self.points3D_cur[0][i+1])**2\
             +(self.points3D_cur[1][i]-self.points3D_cur[1][i+1])**2\
             +(self.points3D_cur[2][i]-self.points3D_cur[2][i+1])**2)**(1/2)
             
        distance_ref = distance_ref/(len(self.points3D_ref[0])-1)
        distance_cur = distance_cur/(len(self.points3D_cur[0])-1)
                    
        return distance_ref/distance_cur
    
    
    def AbsoluScale(self):
        
        x = self.true_poses[self.image_index, 3]
        y = self.true_poses[self.image_index, 7]
        z = self.true_poses[self.image_index, 11]
        x_ref = self.true_poses[self.image_index-1, 3]
        y_ref = self.true_poses[self.image_index-1, 7]
        z_ref = self.true_poses[self.image_index-1, 11]
        
        return ((x - x_ref)**2 + (y - y_ref)**2 + (z - z_ref)**2)**(1/2)

    
    def ProcessFrame(self):
        
        R, t = self.OpticalFlowField()    #features tracking
        #relative scale
        '''    
        self.points3D_ref = self.points3D_cur
        self.points3D_cur = self.Triangulation(R, t)
        self.scale = self.RelativeScale()
        '''  
        #absolu scale
        self.scale = self.AbsoluScale()
     
        if (t[2]>t[0] and t[2]>t[1]):    #if the car go ahead 
            self.t_cur = self.t_cur + self.scale * np.dot(self.R_cur, t)
            self.R_cur = np.dot(R, self.R_cur)
                    
        seuil_features = 1000    #threshold of the re-detection of features
        if len(self.points2D_cur) < seuil_features:
            self.points2D_cur = self.FeaturesDetect(self.image_cur)
        


traj = np.zeros((1000,1000,3), dtype=np.uint8)
vo = VisualOdometry()
poses = []
sequence = '00'

true_poses_path = 'dataset/true_poses/poses/{}.txt'.format(sequence)
f_true = open(true_poses_path, 'r')
true_poses = []
for line in f_true.readlines():
    for i in line.split():
        true_poses.append(float(i))
true_poses = np.array(true_poses).reshape(-1, 12)    #true ground pose
length_file = len(open(true_poses_path, 'r').readlines())
f_true.close()
vo.true_poses = true_poses

print("please wait...")
time1 = time.time()
for image in range(length_file):
    path = 'dataset/sequences/{}/image_0/00{}.png'\
    .format(sequence, str(image).zfill(4))
    vo.image_index = image
    if image == 0:
        vo.image_cur = cv2.imread(path)
    if image == 1:
        vo.image_ref = vo.image_cur
        vo.image_cur = cv2.imread(path)
        vo.R_cur, vo.t_cur = vo.OpticalFlowField_Second()    #track the features from 1er frame to 2nd frame
        vo.points3D_cur = vo.Triangulation(vo.R_cur, vo.t_cur)
    if image > 1:
        vo.image_ref = vo.image_cur
        vo.image_cur = cv2.imread(path)
        
        vo.ProcessFrame()
        '''
        cv2.circle(traj, (vo.t_cur[0]+400, vo.t_cur[2]+200), 1, color = (255,1,255))
        cv2.circle(traj, (int(true_poses[image, 3]+400), int(true_poses[image, 11])+200), 1, color = (255,255,255))
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)
        '''
        poses.append((vo.t_cur[0], vo.t_cur[2]))
time2 = time.time()


plt.plot(true_poses[:,3], -true_poses[:,11])
plt.show()
print('time:', time2-time1)

for pose in poses:
    cv2.circle(traj, (pose[0]+400, pose[1]+200), 1, color = (255,1,255))
 
for true_pose in true_poses:
    cv2.circle(traj, (int(true_pose[3]+400), int(true_pose[11])+200), 1, color = (255,255,255))

cv2.imshow('Trajectory', traj)
while cv2.waitKey(100) != 27:
    if cv2.getWindowProperty('Trajectory',cv2.WND_PROP_VISIBLE) <= 0:
        break
cv2.destroyWindow('Trajectory')

    
    