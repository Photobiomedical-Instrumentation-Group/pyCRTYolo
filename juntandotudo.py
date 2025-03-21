import ast
import cv2
import pandas as pd
import torch
import numpy as np
import os

import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
import sys


# Verificar se a pasta VideosYoloDetect existe e criar se não
if not os.path.exists('VideosYoloDetect'):
    os.makedirs('VideosYoloDetect')
   
if not os.path.exists('SaveRois'):
    os.makedirs('SaveRois')


# Load the custom YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='finger.pt', force_reload=True, trust_repo=False)

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



#Lucas Kanade Optical Flow
def processar_video_com_roi(video_path, roi, movement_threshold=0.9):
    """
    Processa um vídeo usando uma ROI (Região de Interesse) e detecta movimento significativo.

    Parâmetros:
        video_path (str): Caminho do vídeo.
        roi (tuple): ROI no formato (x, y, width, height).
        movement_threshold (float): Limiar para considerar movimento significativo (0 a 1).

    Retorna:
        first_significant_movement_frame (np.array): Frame correspondente ao primeiro movimento significativo.
    """
    
    if not isinstance(roi, (tuple, list)) or len(roi) != 4:
        raise ValueError(f"A ROI deve ser uma tupla ou lista com 4 valores. Recebido: {roi}")

    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    # Função para calcular a magnitude do movimento
    def calculate_movement(old_points, new_points):
        movement = []
        for (new, old) in zip(new_points, old_points):
            a, b = new.ravel()
            c, d = old.ravel()
            mag = np.sqrt((a - c) ** 2 + (b - d) ** 2)
            movement.append(mag)
        return np.mean(movement)

    # Definir o detector de cantos Shi-Tomasi
    feature_params = dict(maxCorners=200, qualityLevel=0.5, minDistance=4, blockSize=7)

    # Definir os parâmetros para o cálculo do fluxo óptico Lucas-Kanade
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Ler o primeiro quadro
    ret, old_frame = cap.read()
    if not ret:
        raise ValueError("Não foi possível ler o primeiro quadro do vídeo.")

    # Converter o primeiro quadro para escala de cinza
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Verificar as dimensões da imagem
    print(f"Dimensões da imagem: {old_gray.shape}")

    # Desempacotar a ROI
    print(roi)
    x, y, width, height = roi
    print(f"ROI: x={x}, y={y}, width={width}, height={height}")

    # Verificar se a ROI está dentro dos limites da imagem
    if (x < 0 or y < 0 or x + width > old_gray.shape[1] or y + height > old_gray.shape[0]):
        raise ValueError(f"A ROI está fora dos limites da imagem. Dimensões da imagem: {old_gray.shape}, ROI: {roi}")

    # Criar uma máscara para a ROI
    mask = np.zeros_like(old_gray, dtype=np.uint8)  # Garantir que a máscara seja do tipo CV_8UC1
    mask[y:y+height, x:x+width] = 255

    # Detectar cantos na ROI
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)

    # Verificar se pontos foram detectados
    if p0 is None:
        raise ValueError("Nenhum ponto foi detectado na ROI.")

    # Listas para armazenar o movimento e a intensidade ao longo dos quadros
    movement_over_time = []
    frames = []  # Para armazenar os quadros lidos

    # Loop através dos quadros do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calcular o fluxo óptico
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Verificar se o fluxo óptico foi calculado corretamente
        if p1 is None or st is None:
            print("Erro: Fluxo óptico não pôde ser calculado. Reiniciando pontos.")
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
            continue

        # Selecionar os pontos bons
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Verificar se há pontos válidos
        if len(good_new) == 0:
            print("Nenhum ponto válido para rastrear. Reiniciando pontos.")
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
            continue

        # Calcular e armazenar o movimento
        movement = calculate_movement(good_old, good_new)
        movement_over_time.append(movement)

        # Armazenar o quadro atual
        frames.append(frame.copy())

        # Atualizar os frames e pontos antigos
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()

    # Normalizar os valores de movimento usando Min-Max normalization
    movement_over_time = np.array(movement_over_time)
    movement_min = np.min(movement_over_time)
    movement_max = np.max(movement_over_time)
    normalized_movement = (movement_over_time - movement_min) / (movement_max - movement_min)

    # Encontrar o índice do primeiro quadro que ultrapassa o limiar
    first_significant_movement_index = next((i for i, m in enumerate(normalized_movement)
                                             if m > movement_threshold and
                                             (i == 0 or normalized_movement[i-1] <= movement_threshold)), None)

    if first_significant_movement_index is not None:
        first_significant_movement_index = max(first_significant_movement_index - 50, 0)
        print(f'Frame estático: {first_significant_movement_index}')
        first_significant_movement_frame = frames[first_significant_movement_index]
        return first_significant_movement_index
    else:
        print("Nenhum movimento significativo foi encontrado.")
        return None


def openTXT(arquivo, linha_desejada):
    """
    Lê uma linha específica de um arquivo.

    Parâmetros:
        arquivo (str): Caminho do arquivo.
        linha_desejada (int): Número da linha a ser lida (começando em 1).

    Retorna:
        str: Conteúdo da linha desejada.
    """
    with open(arquivo, 'r') as f:
        for i, linha in enumerate(f, start=1):
            if i == linha_desejada:
                return ast.literal_eval(linha.strip())
    raise ValueError(f"O arquivo não tem {linha_desejada} linhas.")



# Paths for input and output videos
videoName= "v6.mp4"
inputVideo=f"Videos/{videoName}"
outputVideo = f'VideosYoloDetect/{videoName}_Yolo.mp4'  # Save video as MP4 in the VideosYoloDetect folder
roiFile = f'SaveRois/{videoName}.txt'


# Process the video - parte 1
# Set confidence threshold - find Yolo imagen finger in the video
confidence_threshold = 0.88
#process_video(inputVideo, outputVideo, roiFile, confidence_threshold)
print("Video processado...")

roi = openTXT(roiFile, 20)
print(f"ROI: {roi}")
#roi = (383, 809, 271, 322)  # ROI no formato (x, y, width, height)
movement_threshold = 0.9
# parte 2 - Processar o vídeo e obter o frame que esta em repouso
frame_significativo = processar_video_com_roi(inputVideo, roi, movement_threshold)
print(frame_significativo)

if frame_significativo is not None:
    # Salvar o frame significativo em um arquivo
    output_filename = 'quadro_antes_movimento_significativo.png'
    cv2.imwrite(output_filename, frame_significativo)
    print(f"Frame significativo salvo como {output_filename}")
else:
    print("Nenhum frame significativo foi encontrado.")
    
    

# Parte 3 validação

# Função para comparar se a diferença está abaixo do threshold
def compare_roi(roi1, roi2, threshold):
    x1, y1, width1, height1 = roi1
    x2, y2, width2, height2 = roi2
    return (
        abs(x1 - x2) <= threshold and
        abs(y1 - y2) <= threshold and
        abs(width1 - width2) <= threshold and
        abs(height1 - height2) <= threshold
    )

# Função para calcular o centro da ROI
def get_center(roi):
    x, y, width, height = roi
    return (x + width / 2, y + height / 2)

# Função para mostrar a imagem com a ROI
def mostrar_frame_com_roi(video_path, roi,frame_significativo):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        return
    
    # Definir o frame que queremos (frame frame_significativo)
    #target_frame = frame_significativo
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_significativo)
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível ler o frame.")
        return
    
    # Recortar a ROI do frame
    x, y, width, height = roi
    frame_roi = frame[y:y + height, x:x + width]
    
    # Converter de BGR para RGB para o Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
    
    # Exibir a imagem do frame original com a ROI e a imagem recortada com a ROI
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    plt.title(f"Frame {frame_significativo} com ROI")
    plt.axis('off')
    #plt.show()
    
    #plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 2)
    plt.imshow(frame_roi_rgb)
    plt.title(f"ROI em Frame 210")
    plt.axis('off')
    plt.show()
    
    cap.release()

# Função para perguntar ao usuário se a ROI está correta
def perguntar_validacao_roi():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal do Tkinter
    
    # Exibir uma caixa de mensagem perguntando se a ROI está correta
    resposta = messagebox.askyesno("Validação da ROI", "Essa ROI está correta?")
    
    if resposta:  # Se o usuário responder "sim"
        print("ROI salva com sucesso!")
        return True
    else:  # Se o usuário responder "não"
        print("Escolha uma nova ROI.")
        return False

# Função principal para rodar o processo
def validar_roi(video_path, rois, threshold,frame_significativo):
    while True:
        # Escolher uma ROI aleatória
        import random
        selected_roi = random.choice(rois)
        
        # Exibir o frame com a ROI
        mostrar_frame_com_roi(video_path, selected_roi,frame_significativo)
        
        # Perguntar ao usuário se a ROI está correta
        if perguntar_validacao_roi():
            # Salvar a ROI se o usuário validou como correta
            with open(f'SaveRois/{videoName}_validada.txt', 'w') as f:
                f.write(str(selected_roi))
            break  # Sai do loop se a ROI estiver correta
    return selected_roi


# Leitura do arquivo e preparação das ROIs
with open(f'SaveRois/{videoName}.txt', 'r') as roi_file:
    roi_lines = roi_file.readlines()

rois = [ast.literal_eval(line.strip()) for line in roi_lines]

# Definir o caminho do vídeo e o threshold
#video_path = 'Videos/v6.mp4'  # Substitua com o caminho do seu vídeo
threshold = 4

# Iniciar o processo de validação da ROI
roi=validar_roi(inputVideo, rois, threshold,frame_significativo)
print("roi validada",roi)


sys.path.append("C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT")
from src.pyCRT import PCRT

pcrt = PCRT.fromVideoFile(inputVideo,roi=roi, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()
print("CRT qunatification:",pcrt)

results=[]
results.append({
                "Pasta": inputVideo,
                "Video ": videoName,
                "pCRT ": pcrt.pCRT[0],
                "uncert_pCRT ": pcrt.pCRT[1],
                "CriticalTime ": pcrt.criticalTime,
                "roi": roi
            })

print(f"Processado: {videoName}")

    

# Converter os resultados para um DataFrame e salvar em um arquivo Excel
df = pd.DataFrame(results)
output_file = "SaveExcelData.xlsx"
df.to_excel(output_file, index=False)

print(f"All results are save {output_file}")