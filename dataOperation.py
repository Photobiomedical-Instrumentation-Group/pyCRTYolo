import cv2

import matplotlib.pyplot as plt
import ast
import os 

def openTXT(filePath):
    """
    Read a text file and evaluate each line as a Python literal.
    
    Args:
        filePath: Path to the text file.
    
    Returns:
        list: A list of evaluated lines from the file.
    """
    with open(filePath, 'r') as f:
        lines = [ast.literal_eval(line.strip()) for line in f]
    return lines


def openTXTline(filePath, numberline):
    """
    Reads a specific line from a file.

    Parameters:
        file_path (str): Path to the file.
        numberline (int): Line number to be read (starting from 1).

    Returns:
        str: Content of the desired line.
    
    Raises:
        ValueError: If the file does not have the specified number of lines.
    """
    with open(filePath, 'r') as f:
        for i, line in enumerate(f, start=1):
            if i == numberline:
                return ast.literal_eval(line.strip())
    
    raise ValueError(f"The file does not have {numberline} lines.")

def showROI(videoPath, roi, significantFrame):
    """
    Show the Region of Interest (ROI) in the video at the given frame.
    
    Args:
        videoPath: Path to the video file.
        roi: The region of interest (ROI) defined by top-left corner and dimensions.
        significantFrame: Frame number where the ROI should be displayed.
    
    Returns:
        None
    """
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error opening the video!")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, significantFrame)
    ret, frame = cap.read()
    if not ret:
        print("Could not read the frame.")
        return
    
    # Crop the ROI from the frame
    x, y, width, height = roi
    frameRoi = frame[y:y + height, x:x + width]
    
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frameRoiRgb = cv2.cvtColor(frameRoi, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(frameRgb)
    plt.title(f"Frame {significantFrame} with ROI")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(frameRoiRgb)
    plt.title(f"ROI in Frame {significantFrame}")
    plt.axis('off')
    plt.show()
    
    cap.release()
    
    
# Função para garantir que as pastas necessárias existem
def ensure_directories_exist():
    if not os.path.exists('VideosYoloDetect'):
        os.makedirs('VideosYoloDetect')
   
    if not os.path.exists('SaveRois'):
        os.makedirs('SaveRois')