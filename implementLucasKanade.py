import cv2
import numpy as np
import matplotlib.pyplot as plt
import ast

# Carregar o vídeo
videoName = "v6.mp4"
inputVideo = f"Videos/{videoName}"

# Ler a ROI validada do arquivo
with open(f'SaveRois/{videoName}_validada.txt', 'r') as roi_file:
    roi_lines = roi_file.readlines()
roi_validada = ast.literal_eval(roi_lines[0].strip())   
print(f"ROI validada: {roi_validada}")

# Carregar o vídeo
cap = cv2.VideoCapture(inputVideo)

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
x1, y1, x2, y2 = roi_validada
x = x1
y = y1
width = x2 
height = y2 
roiYolo= (x, y, width, height)
print(roiYolo)

print(f"ROI: x={x}, y={y}, width={width}, height={height}")

# Verificar se a ROI está dentro dos limites da imagem
if (x < 0 or y < 0 or x + width > old_gray.shape[1] or y + height > old_gray.shape[0]):
    raise ValueError(f"A ROI está fora dos limites da imagem. Dimensões da imagem: {old_gray.shape}, ROI: {roi_validada}")

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
intensities = []
frames = []  # Para armazenar os quadros lidos

# Loop através dos quadros do vídeo
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calcular a intensidade média da ROI
    roi_intensity = np.mean(frame_gray[y:y+height, x:x+width])
    intensities.append(roi_intensity)
    
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

# Definir um limiar para considerar um movimento significativo usando os valores normalizados
movement_threshold = 0.9

# Encontrar o índice do primeiro quadro que ultrapassa o limiar
first_significant_movement_index = next((i for i, m in enumerate(normalized_movement) 
                                         if m > movement_threshold and 
                                         (i == 0 or normalized_movement[i-1] <= movement_threshold)), None)

if first_significant_movement_index is not None:
    first_significant_movement_index = max(first_significant_movement_index - 50, 0)
    print(f'Frame estatico {first_significant_movement_index}')
    first_significant_movement_frame = frames[first_significant_movement_index]

    # Exibir o quadro com o primeiro movimento significativo
    output_filename = 'quadro_antes_movimento_significativo.png'
    cv2.imwrite(output_filename, first_significant_movement_frame)
else:
    print("Nenhum movimento significativo foi encontrado.")

"""
# Plotar os gráficos sobrepostos
fig, ax1 = plt.subplots(figsize=(10, 6))

# Gráfico de intensidade
ax1.plot(intensities, label='Intensidade Média da ROI', color='g')
ax1.set_xlabel('Número de Frames')
ax1.set_ylabel('Intensidade Média', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Criar um segundo eixo y para o gráfico de movimento
ax2 = ax1.twinx()
ax2.plot(normalized_movement, label='Magnitude do Movimento (Normalizada)', color='black')
ax2.set_ylabel('Magnitude do Movimento (Normalizada)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Adicionar uma bolinha no ponto de movimento significativo
if first_significant_movement_index is not None:
    ax2.plot(first_significant_movement_index, normalized_movement[first_significant_movement_index], 'ro', label='Primeiro Movimento Significativo')

# Adicionar título e legendas
#fig.tight_layout() 
plt.title('Intensidade Média da ROI e Movimento Normalizado ao Longo do Vídeo')
plt.legend(loc='upper left')
plt.show()


Keyword arguments:
argument -- description
Return: return_description
"""



