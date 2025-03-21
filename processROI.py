import torch
import cv2
import numpy as np

# Carregar o modelo YOLOv5 customizado
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='finger.pt', force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='finger.pt', force_reload=True, trust_repo=False)


def detect_finger(image, confidence_threshold=0.5):
    results = model(image)
    detections = results.xyxy[0].numpy()
    detections = [det for det in detections if det[4] >= confidence_threshold]
    results.xyxy[0] = torch.tensor(detections)
    return results

def verifica_imagens_de_pele(video_path):
    cap = cv2.VideoCapture(video_path)
    tem_imagem_de_pele = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        imagem_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 20, 80], dtype="uint8")
        upper_hsv = np.array([255, 255, 255], dtype="uint8")
        mascara_hsv = cv2.inRange(imagem_hsv, lower_hsv, upper_hsv)

        imagem_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_ycrcb = np.array([0, 136, 0], dtype="uint8")
        upper_ycrcb = np.array([255, 173, 127], dtype="uint8")
        mascara_ycrcb = cv2.inRange(imagem_ycrcb, lower_ycrcb, upper_ycrcb)

        mascara_combinada = cv2.bitwise_and(mascara_hsv, mascara_ycrcb)

        if np.any(mascara_combinada):
            tem_imagem_de_pele = True
            break

    cap.release()
    return tem_imagem_de_pele

def processa_video(input_video, output_video, roi_file, confidence_threshold):
    if not verifica_imagens_de_pele(input_video):
        print("Imagens de pele não foram encontradas.")
        return

    video_capture = cv2.VideoCapture(input_video)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Usando o codec 'mp4v' para salvar em formato MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30, (frame_width, frame_height))

    roi_values = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        results = detect_finger(frame, confidence_threshold)
        detected_frame = np.copy(results.render()[0])

        if len(results.xyxy[0]) > 0:
            for detection in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, detection[0:4])
                xo1 = int(((x1 + x2) / 2) - 50)
                yo1 = int(((y1 + y2) / 2) - 50)
                xo2 = int((x2 - x1))
                yo2 = int((y2 - y1))
                roi_pcrt = (xo1, yo1, xo2, yo2)
                roi_values.append(roi_pcrt)
                #cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out.write(detected_frame)

    video_capture.release()
    out.release()
    print(f"Processamento concluído. Vídeo salvo em {output_video}")

    with open(roi_file, 'w') as f:
        for roi in roi_values:
            f.write(f"{roi}\n")
    print(f"Valores da ROI salvos em {roi_file}")

# Caminho dos vídeos de entrada e saída


#imput video name process:
input_video = 'Videos/v2.mp4'

output_video = f'VideosYoloDetect/{input_video}_Yolo.mp4'  # Salvando o vídeo como MP4
roi_file = f'SaveRois{input_video}.txt'

# Processar o vídeo
confidence_threshold=0.88
processa_video(input_video, output_video, roi_file, confidence_threshold)
