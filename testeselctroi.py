import cv2
import ast
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt


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
def mostrar_frame_com_roi(video_path, roi):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        return
    
    # Definir o frame que queremos (frame 210)
    target_frame = 100
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
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
    plt.title(f"Frame 210 com ROI")
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
def validar_roi(video_path, rois, threshold):
    while True:
        # Escolher uma ROI aleatória
        import random
        selected_roi = random.choice(rois)
        
        # Exibir o frame com a ROI
        mostrar_frame_com_roi(video_path, selected_roi)
        
        # Perguntar ao usuário se a ROI está correta
        if perguntar_validacao_roi():
            # Salvar a ROI se o usuário validou como correta
            with open(f'SaveRois/{videoName}_validada.txt', 'w') as f:
                f.write(str(selected_roi))
            break  # Sai do loop se a ROI estiver correta
    return selected_roi


videoName= "v6.mp4"
inputVideo=f"Videos/{videoName}"


# Leitura do arquivo e preparação das ROIs
with open(f'SaveRois/{videoName}.txt', 'r') as roi_file:
    roi_lines = roi_file.readlines()

rois = [ast.literal_eval(line.strip()) for line in roi_lines]

# Definir o caminho do vídeo e o threshold
#video_path = 'Videos/v6.mp4'  # Substitua com o caminho do seu vídeo
threshold = 4

# Iniciar o processo de validação da ROI
roi=validar_roi(inputVideo, rois, threshold)
print(roi)