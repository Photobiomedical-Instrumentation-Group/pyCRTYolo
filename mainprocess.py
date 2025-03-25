import os
import sys
import pandas as pd
import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plotting
import matplotlib.pyplot as plt

from processYolo import processDetectFinger
from processLucasKanade import processLucasKanade
from validationROI import validateROI, filterROI
from dataOperation import openTXT, ensure_directories_exist

sys.path.append("C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT")
from src.pyCRT import PCRT

# Paths for input and output videos
videoName = "GCC5.mp4"
inputVideo = f"Videos/{videoName}"
outputVideo = f'VideosYoloDetect/{videoName}_Yolo.mp4'  # Save video as MP4 in the VideosYoloDetect folder
roiFile = f'SaveRois/{videoName}.txt'
ensure_directories_exist()

# Step 1: Process the video 
# Set confidence threshold - find Yolo finger image in the video
confidence_threshold = 0.80
#rocessDetectFinger(inputVideo, outputVideo, roiFile, confidence_threshold)
print("Video Processing...")

# Exclude irrelevant ROIs (Regions of Interest)
# Add a threshold to exclude ROIs that don't make sense
roi = openTXT(roiFile)
print(f"ROI: {roi}")
filtered_roi = filterROI(roi)

# Step 2 - Process the video and obtain the resting frame
significant_frame = processLucasKanade(inputVideo, filtered_roi)

if significant_frame is not None:
    print(f"Significant Frame: {significant_frame}")
else:
    print("No significant frame found.")

# Step 3: Validate the Best ROI
# Start the ROI validation process
roi = validateROI(videoName,inputVideo, filtered_roi, significant_frame)
print("Validated ROI:", roi)

# Step 4: Calculate pCRT by pyCRT
# Calculate pCRT (Capillary Refill Time) using the validated ROI
pcrt = PCRT.fromVideoFile(inputVideo, roi=roi, displayVideo=False, exclusionMethod='best fit', exclusionCriteria=9999)

# Check if pCRT could be calculated
if pcrt.pCRT[0] < 0.5:
    print("Unable to calculate pCRT. Please retake the test.")
else:
    # Display plots related to pCRT
    pcrt.showAvgIntensPlot()
    pcrt.showPCRTPlot()
    print("CRT Quantification:", pcrt)

    # Collect results
    results = []
    results.append({
        "Folder": inputVideo,
        "Video": videoName,
        "pCRT": pcrt.pCRT[0],
        "uncert_pCRT": pcrt.pCRT[1],
        "CriticalTime": pcrt.criticalTime,
        "ROI": roi
    })

    print(f"Processed: {videoName}")

    # Convert the results to a DataFrame and save to an Excel file
    df = pd.DataFrame(results)
    output_file = f"SaveExcelData{videoName}.xlsx"
    df.to_excel(output_file, index=False)

    print(f"All results are saved in {output_file}")
    
    # Save the results in a TXT file
    output_file_txt = f"SaveData{videoName}.txt"
    with open(output_file_txt, 'w') as f:
        # Write the header (column names)
        f.write('\t'.join(df.columns) + '\n')
        
        # Write the data (rows)
        for index, row in df.iterrows():
            f.write('\t'.join(map(str, row)) + '\n')
