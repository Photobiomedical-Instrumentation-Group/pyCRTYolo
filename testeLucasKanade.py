import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar o vídeo
input_video = 'Videos/v2.mp4'
cap = cv2.VideoCapture(input_video)

def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized_frame

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
feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)

# Definir os parâmetros para o cálculo do fluxo óptico Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Ler o primeiro quadro
ret, old_frame = cap.read()
if not ret:
    raise ValueError("Não foi possível ler o primeiro quadro do vídeo.")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

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
    
    # Calcular a intensidade média do quadro inteiro
    mean_intensity = np.mean(frame_gray)
    intensities.append(mean_intensity)
    
    # Calcular o fluxo óptico
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Selecionar os pontos bons
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
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
    first_significant_movement_index = max(first_significant_movement_index - 10, 0)
    print(f'O primeiro quadro com movimento significativo é o quadro número {first_significant_movement_index}')
    first_significant_movement_frame = frames[first_significant_movement_index]

    # Exibir o quadro com o primeiro movimento significativo
    output_filename = 'quadro_antes_movimento_significativo.png'
    cv2.imwrite(output_filename, first_significant_movement_frame)
else:
    print("Nenhum movimento significativo foi encontrado.")

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
fig.tight_layout()  # Para melhor alinhamento dos eixos
plt.title('Intensidade Média da ROI e Movimento Normalizado ao Longo do Vídeo')
plt.legend(loc='upper left')
plt.show()
