import os
import sys
import pandas as pd
import cv2
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from processYolo import processDetectFinger
from processLucasKanade import processLucasKanade
from validationROI import validateROI
from dataOperation import openTXT, ensure_directories_exist, transformVideoframe1010_ffmpeg, get_latest_video

# Load pyCRT locally
local_pyCRT_path = "C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT"
if not os.path.exists(local_pyCRT_path):
    from pyCRT import PCRT
else:
    sys.path.append(local_pyCRT_path)
    from src.pyCRT import PCRT

def calculatetimeprocess(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Processing time for {func.__name__}: {end_time - start_time:.2f} seconds")
    return result

# Step 0: Get the most recent video from the folder
try:
    video_folder = "Videos"
    inputVideo = get_latest_video(video_folder)
    if inputVideo is None:
        raise FileNotFoundError("No video found in the 'Videos' folder.")
    print(f"Processing the most recent video: {inputVideo}")
except Exception as e:
    print(f"Error loading video: {e}")
    sys.exit(1)

videoName = os.path.basename(inputVideo)
outputVideoframe = f"VideoFrame/{videoName}_frame1010.mp4"
outputVideo = f'VideosYoloDetect/{videoName}_Yolo.mp4'
roiFile = f'SaveRois/{videoName}.txt'
scale_factor = 0.5
numberFrames = 10
exclusionCriteria = 1

ensure_directories_exist()

# Step 1: Convert the video using FFmpeg (10 by 10 frames and rescale)
try:
    calculatetimeprocess(transformVideoframe1010_ffmpeg, inputVideo, outputVideoframe, numberFrames, scale_factor)
    if not os.path.exists(outputVideoframe):
        raise FileNotFoundError("FFmpeg conversion failed. Output video not created.")
except Exception as e:
    print(f"Error in video conversion: {e}")
    sys.exit(1)

# Step 2: Run YOLO to detect the finger in the video
try:
    confidence_threshold = 0.70
    calculatetimeprocess(processDetectFinger, outputVideoframe, outputVideo, roiFile, confidence_threshold)
    print("Finger detection completed.")
except Exception as e:
    print(f"Error during finger detection: {e}")
    sys.exit(1)

# Step 3: Read and filter detected ROIs
try:
    roi = openTXT(roiFile)
    roi_filtered = [r for r in roi if r[0] >= 5 and r[1] >= 5]  # Remove borders
    with open(f'SaveRois/{videoName}_filtered.txt', 'w') as f:
        for r in roi_filtered:
            f.write(f"{r}\n")
    print(f"Filtered ROIs: {roi_filtered}")
except Exception as e:
    print(f"Error filtering ROIs: {e}")
    sys.exit(1)

# Step 4: Use Lucas-Kanade to find the resting (significant) frame
try:
    significant_frame = calculatetimeprocess(processLucasKanade, outputVideoframe, roi_filtered, numberFrames,visualize=True)
    print(f"frame significativo:{significant_frame}")
    if significant_frame is None:
        raise ValueError("No significant frame was found.")
    print(f"Significant frame: {significant_frame}")
except Exception as e:
    print(f"Error detecting significant frame: {e}")
    sys.exit(1)

# Step 5: Validate the best ROI
try:
    roiCorrect = calculatetimeprocess(validateROI, videoName, inputVideo, roi_filtered, significant_frame, scale_factor)
    print(f"Validated ROI: {roiCorrect}")
    if roiCorrect is None:
        raise ValueError("ROI validation failed.")
except Exception as e:
    print(f"Error validating ROI: {e}")
    sys.exit(1)

# Resize the ROI to reduce width and height (for pyCRT input)
x, y, w, h = roiCorrect
new_w = int(w * (1 - scale_factor))
new_h = int(h * (1 - scale_factor))
delta_x = (w - new_w) // 2
delta_y = (h - new_h) // 2
adjusted_roi = (x + delta_x, y + delta_y, new_w, new_h)
print(f"Adjusted ROI: {adjusted_roi}")

# Step 6: Calculate the timestamp of the significant frame
try:
    def get_frame_time(video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0:
            raise ValueError("Could not retrieve FPS from video.")
        return frame_number / fps

    timestamp_seconds = get_frame_time(inputVideo, significant_frame)
except Exception as e:
    print(f"Error calculating frame timestamp: {e}")
    sys.exit(1)

# Step 7: Calculate pCRT using pyCRT
try:
    pcrt = calculatetimeprocess(
        PCRT.fromVideoFile,
        inputVideo,
        roi=roiCorrect,
        displayVideo=False,
        exclusionMethod='best fit',
        exclusionCriteria=exclusionCriteria,
        fromTime=timestamp_seconds
    )
    pcrt.showAvgIntensPlot()
    pcrt.showPCRTPlot()
    print(f"pCRT calculated: {pcrt.pCRT[0]:.2f} ± {pcrt.pCRT[1]:.2f}")
except Exception as e:
    print(f"Error calculating pCRT: {e}")
    sys.exit(1)

# Step 8: Save the results to Excel and TXT
try:
    results = [{
        "Folder": inputVideo,
        "Video": videoName,
        "pCRT": pcrt.pCRT[0],
        "uncert_pCRT": pcrt.pCRT[1],
        "CriticalTime": pcrt.criticalTime,
        "ROI": roiCorrect
    }]

    df = pd.DataFrame(results)
    output_file_excel = f"SaveExcelData{videoName}.xlsx"
    df.to_excel(output_file_excel, index=False)
    print(f"Results saved to: {output_file_excel}")

    output_file_txt = f"SaveData{videoName}.txt"
    with open(output_file_txt, 'w') as f:
        f.write('\t'.join(df.columns) + '\n')
        for index, row in df.iterrows():
            f.write('\t'.join(map(str, row)) + '\n')
except Exception as e:
    print(f"Error saving results: {e}")
    sys.exit(1)
