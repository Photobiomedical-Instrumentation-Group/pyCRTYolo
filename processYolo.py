import decord
from decord import cpu, gpu
import cv2
import numpy as np

import cv2
import pandas as pd
import torch
import numpy as np
import os
import random
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
import sys

model = torch.hub.load('ultralytics/yolov5', 'custom', path='finger.pt')

def detectFinger(image, confidenceThreshold):
    """
    Detect fingers in the image using the YOLOv5 model.
    
    Args:
        image: Input image.
        confidenceThreshold: Minimum confidence score for detections.
    
    Returns:
        YOLOv5 detection results.
    """
    results = model(image)
    detections = results.xyxy[0].numpy()
    detections = [det for det in detections if det[4] >= confidenceThreshold]
    results.xyxy[0] = torch.tensor(np.array(detections))
    return results




def hasSkinImage(videoPath):
    """
    Check if the video contains skin images using HSV and YCrCb color spaces.
    
    Args:
        videoPath: Path to the input video.
    
    Returns:
        Boolean indicating the presence of skin-like regions.
    """
    hasSkin = False
    cap = cv2.VideoCapture(videoPath)
  
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        hsvImage = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lowerHsv = np.array([0, 20, 80], dtype="uint8")
        upperHsv = np.array([255, 255, 255], dtype="uint8")
        hsvMask = cv2.inRange(hsvImage, lowerHsv, upperHsv)

        ycrcbImage = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lowerYcrcb = np.array([0, 136, 0], dtype="uint8")
        upperYcrcb = np.array([255, 173, 127], dtype="uint8")
        ycrcbMask = cv2.inRange(ycrcbImage, lowerYcrcb, upperYcrcb)

        combinedMask = cv2.bitwise_and(hsvMask, ycrcbMask)

        if np.count_nonzero(combinedMask) > 0:
            hasSkin = True
        else:
            hasSkin = False

    return hasSkin


def processDetectFinger(inputVideo, outputVideo, roiFile, confidenceThreshold):
    """
    Process video to detect fingers and save the output.
    
    Args:
        inputVideo: Path to the input video.
        outputVideo: Path to save the processed video.
        roiFile: Path to save the detected ROIs.
        confidenceThreshold: Minimum confidence score for detections.
    """
    if not hasSkinImage(inputVideo):
        print("No skin images found. Add another video")
        return

    vr = decord.VideoReader(inputVideo, ctx=cpu(0))
    writer = cv2.VideoWriter(
        outputVideo,
        cv2.VideoWriter_fourcc(*'mp4v'),
        vr.get_avg_fps(),
        (vr[0].shape[1], vr[0].shape[0]))
    
    roiValues = []

    for i in range(len(vr)):
        frame = vr[i].asnumpy()  # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Detecção YOLO
        results = detectFinger(frame_bgr, confidenceThreshold)  # Assume YOLO precisa de BGR
        detectedFrame = results.render()[0]
        
        # Processamento de ROIs (mesma lógica)
        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, detection[0:4])
                centerX = int((x1 + x2) / 2)
                centerY = int((y1 + y2) / 2)
                roiWidth = int((x2 - x1))
                roiHeight = int((y2 - y1))
                roiPcrt = (centerX - int(roiWidth / 2), centerY - int(roiHeight / 2), roiWidth, roiHeight)
                roiValues.append(roiPcrt) # possiveis ROIs candidatas 

        writer.write(detectedFrame)

    writer.release()
    del vr
    
    print(f"Processing complete. Video saved as {outputVideo}")

    roiDir = os.path.dirname(roiFile)
    if roiDir and not os.path.exists(roiDir):
        os.makedirs(roiDir)
        print(f"Directory {roiDir} created.")

    with open(roiFile, 'w') as f:
        for roi in roiValues:
            f.write(f"{roi}\n")
    print(f"ROI values saved in {roiFile}")