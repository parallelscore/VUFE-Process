#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Usage
#python face_detector.py --input dataset/TestVideo.mp4 --output output/test_output.avi --display 0

"""
Created on Wed Mar 10 11:21:37 2021

@author: ugot
"""

import numpy as np
import cv2
import argparse
import imutils

'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", type=str,
                help="path to output video")

args = vars(ap.parse_args())
'''
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


print("[INFO] processing video...")
stream = cv2.VideoCapture("dataset/TestVideo.mp4")
writer = None
results = []
while True:
    # grab the next frame
    (grabbed, frame) = stream.read()

    # if the frame was not grabbed, then we have reached the
    # end of the stream
    if not grabbed:
        break

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #r = frame.shape[1] / float(rgb.shape[1])
    faces = face_cascade.detectMultiScale(rgb, 1.3, 5)
    i = 0
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi = rgb[y:y+h, x:x+w]
        results.append(roi)
        path = 'output/img%d.jpg' % (i,)
        cv2.imwrite(path, roi)
        i += 1

stream.release()


array = np.array(results)

# Clustering algorithm
from sklearn.cluster import DBSCAN
clusters = DBSCAN(eps=3, min_samples=6).fit(array)
for i in array:
    img = clusters.predict()

