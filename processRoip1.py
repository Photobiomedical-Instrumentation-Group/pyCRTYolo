
import torch
import cv2
import numpy as np
import os

import ast
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt


# Verificar se a pasta VideosYoloDetect existe e criar se não
if not os.path.exists('VideosYoloDetect'):
    os.makedirs('VideosYoloDetect')
   
if not os.path.exists('SaveRois'):
    os.makedirs('SaveRois')


# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='finger.pt', force_reload=True, trust_repo=False)

# Function to detect finger in the image using YOLOv5
def detect_finger(image, confidence_threshold=0.5):
    results = model(image)
    detections = results.xyxy[0].numpy()
    
    # Filter detections by confidence threshold
    detections = [det for det in detections if det[4] >= confidence_threshold]
    results.xyxy[0] = torch.tensor(detections)
    
    return results

# Function to check if skin images exist in the video
def has_skin_image(video_path):
    cap = cv2.VideoCapture(video_path)
    has_skin = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and create a skin mask
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 20, 80], dtype="uint8")
        upper_hsv = np.array([255, 255, 255], dtype="uint8")
        hsv_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        # Convert to YCrCb and create a skin mask
        ycrcb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 136, 0], dtype="uint8")
        upper_ycrcb = np.array([255, 173, 127], dtype="uint8")
        ycrcb_mask = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)

        # Combine both skin masks
        combined_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

        if np.any(combined_mask):
            has_skin = True
            break

    cap.release()
    return has_skin

# Processar o vídeo e detectar os dedos
def process_video(input_video, output_video, roi_file, confidence_threshold):
    if not has_skin_image(input_video):
        print("No skin images found. Add another video")
        return  # Não prossegue se não encontrar imagem de pele

    # Inicializar captura de vídeo
    video_capture = cv2.VideoCapture(input_video)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Usar codec 'mp4v' para saída em MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30, (frame_width, frame_height))

    roi_values = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detectar dedos no frame
        results = detect_finger(frame, confidence_threshold)
        detected_frame = np.copy(results.render()[0])

        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                # Suponha que detection seja a detecção do YOLO
                x1, y1, x2, y2 = map(int, detection[0:4])  # Coordenadas da caixa delimitadora

                # Calcular o centro da caixa
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Definir o tamanho da ROI (largura e altura)
                roi_width = int((x2 - x1))  # Largura da caixa
                roi_height = int((y2 - y1))  # Altura da caixa

                # Definir a ROI centralizada
                xo1 = center_x - int(roi_width / 2)
                yo1 = center_y - int(roi_height / 2)
                xo2 = roi_width
                yo2 = roi_height

                # Criar a ROI final
                roi_pcrt = (xo1, yo1, xo2, yo2)
                #print(f"ROI calculada: {roi_pcrt}")
                roi_values.append(roi_pcrt)
                
                """
                x1, y1, x2, y2 = map(int, detection[0:4])
                xo1 = int(((x1 + x2) / 2) - 50)
                yo1 = int(((y1 + y2) / 2) - 50)
                xo2 = int((x2 - x1))
                yo2 = int((y2 - y1))
                roi_pcrt = (xo1, yo1, xo2, yo2)
                
                Keyword arguments:
                argument -- description
                Return: return_description
                """
                

        out.write(detected_frame)

    video_capture.release()
    out.release()

    print(f"Processing complete. Video saved as {output_video}")

    # Verificar e criar o diretório onde o arquivo será salvo
    roi_dir = os.path.dirname(roi_file)
    if roi_dir and not os.path.exists(roi_dir):
        os.makedirs(roi_dir)
        print(f"Diretório {roi_dir} criado.")

    # Salvar os valores de ROI em um arquivo
    with open(roi_file, 'w') as f:
        for roi in roi_values:
            f.write(f"{roi}\n")
    print(f"ROI values saved in {roi_file}")



# Paths for input and output videos
videoName= "v6.mp4"
inputVideo=f"Videos/{videoName}"
outputVideo = f'VideosYoloDetect/{videoName}_Yolo.mp4'  # Save video as MP4 in the VideosYoloDetect folder
roiFile = f'SaveRois/{videoName}.txt'



# Set confidence threshold
confidence_threshold = 0.88

# Process the video
process_video(inputVideo, outputVideo, roiFile, confidence_threshold)



# Encontrar o frame para mostrar a imagem 
#first_significant_movement_index


