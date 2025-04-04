

import matplotlib.pyplot as plt
import ast
import os 
    
import cv2
import numpy as np
import decord

import os
import glob


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
        
    if not os.path.exists('VideoFrame'):
        os.makedirs('VideoFrame')
 
    


# Modifique a função transformVideoframe1010 em dataOperation.py
def transformVideoframe1010(inputName, outputName, numberFrames, scale_factor):
    vr = decord.VideoReader(inputName, ctx=decord.cpu(0))
    
    try:
        indices = np.arange(0, len(vr), numberFrames)
        frames = vr.get_batch(indices).asnumpy()
        
        # Redução de resolução
        resized_frames = [cv2.resize(frame, None, fx=scale_factor, fy=scale_factor) 
                         for frame in frames]
        
        # Configurar vídeo de saída
        if resized_frames:
            height, width = resized_frames[0].shape[:2]
            fps = int(vr.get_avg_fps() // numberFrames)
            
            writer = cv2.VideoWriter(
                outputName,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

            for frame in resized_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
    
    finally:
        del vr
        
        

def get_latest_video(folder_path, extensions=('mp4', 'avi', 'mov', 'mkv')):
    """
    Retorna o vídeo mais recente na pasta especificada.
    
    Parâmetros:
    - folder_path: Caminho da pasta onde os vídeos estão salvos
    - extensions: Extensões de arquivo consideradas (padrão: formatos comuns de vídeo)
    
    Retorna:
    - Caminho completo do vídeo mais recente
    """
    # Lista todos os arquivos com as extensões especificadas
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
    
    if not video_files:
        raise FileNotFoundError(f"Nenhum vídeo encontrado em {folder_path} com as extensões {extensions}")
    
    # Encontra o arquivo mais recente usando o timestamp de modificação
    latest_video = max(video_files, key=os.path.getmtime)
    
    return latest_video

