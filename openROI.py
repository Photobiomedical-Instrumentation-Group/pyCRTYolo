# open number roi_file
roi_file = open('roi_values_v2.txt', 'r')
roi_file = roi_file.readlines() 
print(roi_file[1])
#print(roi_file[1])

import ast

# Abrir o arquivo
with open('roi_values_v2.txt', 'r') as roi_file:
    roi_lines = roi_file.readlines()

# Acessar a linha que contém a tupla como string e convertê-la para uma tupla
roi_tuple = ast.literal_eval(roi_lines[1].strip())  # Converte a string para uma tupla de números
print(roi_tuple)  # Exibe a tupla

# Se você quiser acessar os valores individuais dentro da tupla, pode fazer assim:
x, y, width, height = roi_tuple
print(f"x: {x}, y: {y}, width: {width}, height: {height}")


import ast

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

# Abrir o arquivo e ler as linhas
with open('roi_values_v2.txt', 'r') as roi_file:
    roi_lines = roi_file.readlines()

# Converte cada linha do arquivo para uma tupla de números
rois = [ast.literal_eval(line.strip()) for line in roi_lines]

# Comparar todas as ROIs entre si e agrupar as semelhantes
similar_rois = []

for i in range(len(rois)):
    for j in range(i + 1, len(rois)):  # Comparar uma ROI com as outras
        roi1 = rois[i]
        roi2 = rois[j]
        
        if compare_roi(roi1, roi2, threshold=4):  # Substitua o threshold conforme necessário
            similar_rois.append((roi1, roi2))

# Agora, para cada grupo de ROIs semelhantes, escolha a mais central
for group in similar_rois:
    # Para cada grupo, calculamos o centro médio
    centers = [get_center(roi) for roi in group]
    avg_center = (
        sum(center[0] for center in centers) / len(centers),
        sum(center[1] for center in centers) / len(centers)
    )
    
    # Encontrar a ROI que está mais próxima do centro médio
    best_roi = min(group, key=lambda roi: abs(get_center(roi)[0] - avg_center[0]) + abs(get_center(roi)[1] - avg_center[1]))

print(f"ROI selecionada: {best_roi}")


import cv2
import matplotlib.pyplot as plt

# Função para extrair e mostrar o frame com a ROI
def mostrar_frame_com_roi(video_path, roi):
    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        return

    # Definir o frame que queremos (frame 210)
    target_frame = 210
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    # Ler o frame
    ret, frame = cap.read()
    
    if not ret:
        print("Não foi possível ler o frame.")
        return
    
    # ROI: (x, y, width, height)
    x, y, width, height = roi
    
    # Recortar a ROI do frame
    frame_roi = frame[y:y + height, x:x + width]
    
    # Converter de BGR para RGB para o Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
    
    # Mostrar o frame original com a ROI
    plt.figure(figsize=(10, 10))
    plt.imshow(frame_rgb)
    plt.title(f"Frame 210 com ROI")
    plt.axis('off')  # Ocultar eixos
    plt.show()
    
    # Mostrar o frame recortado da ROI
    plt.figure(figsize=(5, 5))
    plt.imshow(frame_roi_rgb)
    plt.title(f"ROI em Frame 210")
    plt.axis('off')  # Ocultar eixos
    plt.show()
    
    # Fechar o vídeo
    cap.release()


# Defina o caminho do vídeo e a ROI
video_path = 'v2_yoloTr0.8.mp4'  # Substitua com o caminho do seu vídeo
#roi = (39, 365, 170, 168)  # Exemplo de ROI, substitua com a ROI final selecionada

# Mostrar o frame com a ROI
mostrar_frame_com_roi(video_path, best_roi)