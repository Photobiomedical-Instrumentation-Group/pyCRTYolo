import os
import sys
import pandas as pd
import cv2
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for interactive plotting
import matplotlib.pyplot as plt

from processYolo import processDetectFinger
from processLucasKanade import processLucasKanade
from validationROI import validateROI
from dataOperation import openTXT, ensure_directories_exist,transformVideoframe1010,get_latest_video,transformVideoframe1010_ffmpeg


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
    elapsed_time = end_time - start_time
    #print(f"Tempo de processamento de {func.__name__}: {elapsed_time:.6f} segundos")
    return result

# Paths for input and output videos
#videoName = "v1.mp4"
#inputVideo = f"Videos/{videoName}"


# Paths using video more recent
# Get the most recent video in the specified folder
video_folder ="Videos"
inputVideo = get_latest_video(video_folder)
print(f"Processando o vídeo mais recente: {inputVideo}")


videoName = os.path.basename(inputVideo)  
print(f"Processando: {videoName}")

outputVideoframe = f"VideoFrame/{videoName}_frame1010.mp4"
outputVideo = f'VideosYoloDetect/{videoName}_Yolo.mp4'  # Save video as MP4 in the VideosYoloDetect folder
roiFile = f'SaveRois/{videoName}.txt'
exclusionCriteria=0.4


ensure_directories_exist()



scale_factor=0.2  # Scale factor for resizing the video
# Step 0 : transform video em 10-10 frames# Início da etapa 1
numberFrames=10
calculatetimeprocess(transformVideoframe1010_ffmpeg, inputVideo, outputVideoframe,numberFrames,scale_factor)



# Step 1: Process the video 
# Set confidence threshold - find Yolo finger image in the video
confidence_threshold = 0.80
calculatetimeprocess(processDetectFinger,outputVideoframe, outputVideo, roiFile, confidence_threshold)
print("Video Processing...")



# Exclude irrelevant ROIs (Regions of Interest)
# Add a threshold to exclude ROIs that don't make sense
roi = openTXT(roiFile)
print(f"ROI: {roi}")
#filtered_roi = filterROI(roi)
#print(f"filtered_roi: {filtered_roi}")

# Step 2 - Process the video and obtain the resting frame
#significant_frame = processLucasKanade(inputVideo, roi)
significant_frame = calculatetimeprocess(processLucasKanade, outputVideoframe, roi,numberFrames)

if significant_frame is not None:
    print(f"Significant Frame: {significant_frame}")
else:
    print("No significant frame found.")
    
    

# Step 3: Validate the Best ROI
# Start the ROI validation process
roiCorrect = calculatetimeprocess(validateROI, videoName, inputVideo, roi, significant_frame,scale_factor)
print("Validated ROI:", roiCorrect)


# Check if the ROI is valid - for reduce widht and height of the ROI
if roiCorrect is not None:
    x, y, w, h = roiCorrect
    new_w = int(w * (1 - scale_factor))
    new_h = int(h * (1 - scale_factor))

    # Ajusta coordenadas para manter o centro
    delta_x = (w - new_w) // 2
    delta_y = (h - new_h) // 2
    adjusted_roi = (x + delta_x, y + delta_y, new_w, new_h)
    print(f"Adjusted ROI: {adjusted_roi}")


def get_frame_time(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Obtém o FPS do vídeo
    cap.release()
    
    if fps == 0:
        raise ValueError("Não foi possível obter o FPS do vídeo.")
    
    return frame_number / fps

tempo_segundos = get_frame_time(inputVideo, significant_frame)
print(tempo_segundos)

# Step 4: Calculate pCRT by pyCRT
# Calculate pCRT (Capillary Refill Time) using the validated ROI
#pcrt = PCRT.fromVideoFile(inputVideo, roi=roiCorrect, displayVideo=False, exclusionMethod='best fit', exclusionCriteria=exclusionCriteria)
pcrt = calculatetimeprocess(PCRT.fromVideoFile, inputVideo, roi=roiCorrect, displayVideo=False, 
                            exclusionMethod='best fit', exclusionCriteria=exclusionCriteria,
                            fromTime=tempo_segundos,livePlot=False)
# Check if pCRT could be calculated
#if pcrt.pCRT[0] < 0.5:
#    print("Unable to calculate pCRT. Please retake the test.")
#else:
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
    "ROI": roiCorrect
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
