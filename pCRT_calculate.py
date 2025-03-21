import cv2 as cv
import numpy as np
import os
import sys

sys.path.append("C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT")
from src.pyCRT import PCRT

video_name= "v6.mp4"
folder_path="Videos"
video_path = os.path.join(folder_path, video_name)
#output_video_path

roi=(301, 702, 269, 322)
pcrt = PCRT.fromVideoFile(video_path,roi=roi, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()
#outputFilePCRT = f"{video_name}.png"
#pcrt.savePCRTPlot(outputFilePCRT)

