from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import requests
import time
from io import BytesIO

app = Flask(__name__)

# Dirección del ESP32-CAM
_URL = 'http://192.168.1.124'
_PORT = '81'
_ST = '/stream'
stream_url = f"{_URL}:{_PORT}{_ST}"

# Variables globales
fgbg = cv2.createBackgroundSubtractorMOG2()
prev_time = time.time()
fps = 0.0  
modo_actual = 'normal'  
media = 0
desviacion = 10
varianza = 0.01
kernel_size = 3
bitwise_op = 'and'  
ruido_para_suavizado = 'gaussiano'  
ruido_para_bordes = 'gaussiano'
suavizado_para_bordes = 'gaussiano'



def get_frame():
    """
    Captura de frames desde el stream de la ESP32-CAM
    """
    try:
        res = requests.get(stream_url, stream=True, timeout=5)
        for chunk in res.iter_content(chunk_size=100000):
            if len(chunk) > 100:
                img_data = BytesIO(chunk)
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                if frame is not None:
                    yield frame
    except requests.exceptions.RequestException as e:
        print(f"rror de conexión con la cámara: {e}")

def procesar_frame(frame):
    """
    Procesa el frame según el modo actual seleccionado
    """
    global prev_time, modo_actual, fps

# Imagn original sin alteraciones
    if modo_actual == 'normal':
        return frame  

    elif modo_actual == 'motion':
    # Detección de movimiento con MOG2 
        current_time = time.time()
        delta = current_time - prev_time
        new_fps = 1.0 / delta if delta > 0 else 0.0
        fps = 0.9 * fps + 0.1 * new_fps 
        ms_per_frame = delta * 1000
        prev_time = current_time

    # Aplicar sustracción de fondo y morfología
        fg_mask = fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        motion_view = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    # Copias de frame para anotaciones
        frame_copy = frame.copy()
        motion_copy = motion_view.copy()

    # Anotar la cámara original con FPS y tiempo entre frames
        cv2.putText(frame_copy, f"FPS: {fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
        cv2.putText(frame_copy, f"{ms_per_frame:.1f} ms/frame", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Anotar la detección de movimiento
        cv2.putText(motion_copy, "Deteccion de Movimiento", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Concatenar las vistas
        combined = np.hstack((frame_copy, motion_copy))
        return combined

    
    


    elif modo_actual == 'filters':
        # Conversión a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Filtro 1: Ecualización de histograma
        hist_eq = cv2.equalizeHist(gray)
        hist_eq_bgr = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
        cv2.putText(hist_eq_bgr, "Ecualizacion", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2)

        # Filtro 2: CLAHE con pre-suavizado
        blurred_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0)
        clahe_img = clahe.apply(blurred_gray)
        clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
        cv2.putText(clahe_bgr, "CLAHE ", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 204, 102), 2)

        # Filtro 3: Corrección gamma
        gamma = 1.5
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        gamma_img = cv2.LUT(frame, table)
        cv2.putText(gamma_img, "Gamma 1.5", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (102, 255, 178), 2)

        # Imagen original anotada
        original = frame.copy()
        cv2.putText(original, "Camara Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # imagen original vs cada filtro
        fila1 = np.hstack((original, hist_eq_bgr))
        fila2 = np.hstack((original, clahe_bgr))
        fila3 = np.hstack((original, gamma_img))
        combined = np.vstack((fila1, fila2, fila3))
        return combined
    
    elif modo_actual == 'bordes':
        # 1️ Agregar ruido
        gauss = np.random.normal(media, desviacion, frame.shape).astype(np.int16)
        noisy = np.clip(frame.astype(np.int16) + gauss, 0, 255).astype(np.uint8)
        noisy_gray = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)

        # 2️ Suavizado
        smooth = cv2.GaussianBlur(noisy_gray, (kernel_size, kernel_size), 0)

        # 3️ Bordes sin suavizado
        canny_raw = cv2.Canny(noisy_gray, 50, 150)
        sobel_raw = cv2.Sobel(noisy_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_raw = cv2.convertScaleAbs(sobel_raw)

        # 4️ Bordes con suavizado
        canny_smooth = cv2.Canny(smooth, 50, 150)
        sobel_smooth = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        sobel_smooth = cv2.convertScaleAbs(sobel_smooth)

        # 5️Convertir a BGR para concatenar
        canny_raw = cv2.cvtColor(canny_raw, cv2.COLOR_GRAY2BGR)
        sobel_raw = cv2.cvtColor(sobel_raw, cv2.COLOR_GRAY2BGR)
        canny_smooth = cv2.cvtColor(canny_smooth, cv2.COLOR_GRAY2BGR)
        sobel_smooth = cv2.cvtColor(sobel_smooth, cv2.COLOR_GRAY2BGR)

        # 6️ Etiquetas
        cv2.putText(canny_raw, "Canny sin suavizado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(sobel_raw, "Sobel sin suavizado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canny_smooth, "Canny con suavizado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(sobel_smooth, "Sobel con suavizado", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        fila1 = np.hstack((canny_raw, sobel_raw))
        fila2 = np.hstack((canny_smooth, sobel_smooth))
        return np.vstack((fila1, fila2))



    elif modo_actual == 'suavizados':
        # 1 Agregar ruido según selección
        if ruido_para_suavizado == 'gaussiano':
            ruido = np.random.normal(media, desviacion, frame.shape).astype(np.int16)
            noisy = np.clip(frame.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
            label = "Ruido Gaussiano"
        else:
            base = frame.astype(np.float32)
            speckle = base + base * np.random.normal(0, varianza, frame.shape)
            noisy = np.clip(speckle, 0, 255).astype(np.uint8)
            label = "Ruido Speckle"

        # 2 Aplicar filtros sobre la imagen ruidosa
        blurred = cv2.blur(noisy, (kernel_size, kernel_size))
        median = cv2.medianBlur(noisy, kernel_size)
        gaussian = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), 0)

        # 3️Etiquetas
        cv2.putText(noisy, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(blurred, "Blur", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(median, "Mediana", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(gaussian, "Gaussiano", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        fila1 = np.hstack((noisy, blurred))
        fila2 = np.hstack((median, gaussian))
        return np.vstack((fila1, fila2))

    elif modo_actual == 'ruido':
        original = frame.copy()

        # Ruido Gaussiano
        gauss_noise = np.random.normal(media, desviacion, original.shape).astype(np.int16)
        gauss_img = np.clip(original.astype(np.int16) + gauss_noise, 0, 255).astype(np.uint8)

        speckle_noise = original.astype(np.float32) * np.random.normal(0, varianza * 5, original.shape)
        speckle_img = np.clip(original + speckle_noise, 0, 255).astype(np.uint8)


        # Etiquetas
        cv2.putText(original, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(gauss_img, "Ruido Gaussiano", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(speckle_img, "Ruido Speckle", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return np.hstack((original, gauss_img, speckle_img))
    
    elif modo_actual == 'bitwise':
    # Obtener máscara de movimiento con MOG2 + morfología
        fg_mask = fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Binarizar la máscara para asegurar valores 0 y 255
        _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

    if bitwise_op == 'and':
        result = cv2.bitwise_and(frame, frame, mask=fg_mask)
        label = "Bitwise AND"
        color = (0, 255, 0)
    elif bitwise_op == 'or':
        result = cv2.bitwise_or(frame, np.zeros_like(frame), mask=fg_mask)
        label = "Bitwise OR"
        color = (255, 128, 0)
    elif bitwise_op == 'xor':
        result = cv2.bitwise_xor(frame, np.full_like(frame, 255), mask=fg_mask)
        label = "Bitwise XOR"
        color = (128, 0, 255)
    else:
        result = np.zeros_like(frame)
        label = "Operación no válida"
        color = (0, 0, 255)

    # Anotar imágenes
    annotated_frame = frame.copy()
    mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    annotated_mask = mask_bgr.copy()
    annotated_result = result.copy()

    cv2.putText(annotated_frame, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_mask, "Máscara", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(annotated_result, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return np.hstack((annotated_frame, annotated_mask, annotated_result))




    return frame   # Convertir la máscara a color para aplicar operaciones bitwise entre dos imágenes del mismo tipo
   

def video_generator():
    """
    Generador de frames JPEG codificados para Flask
    """
    for frame in get_frame():
        if frame is None:
            continue
        final = procesar_frame(frame)
        _, jpeg = cv2.imencode('.jpg', final)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(video_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_noise_params', methods=['POST'])
def set_noise_params():
    global media, desviacion, varianza, kernel_size
    data = request.get_json()
    media = float(data.get('media', 0))
    desviacion = float(data.get('desviacion', 10))
    varianza = float(data.get('varianza', 0.01))
    kernel_size = int(data.get('kernel_size', 3))
    print(f"Parámetros actualizados: media={media}, desviación={desviacion}, varianza={varianza}, kernel={kernel_size}")
    return ('', 204)

@app.route("/set_mode/<modo>")
def set_mode(modo):
    global modo_actual
    if modo in ['normal', 'motion', 'filters', 'bitwise', 'ruido', 'suavizados', 'bordes']:
        print(f"Modo cambiado a: {modo}")
        modo_actual = modo
    return ('', 204)
@app.route("/set_bitwise_op", methods=['POST'])
def set_bitwise_op():
    global bitwise_op
    data = request.get_json()
    bitwise_op = data.get('operation', 'and')
    print(f"Operación Bitwise seleccionada: {bitwise_op}")
    return ('', 204)

@app.route('/set_suavizado_ruido', methods=['POST'])
def set_suavizado_ruido():
    global ruido_para_suavizado
    data = request.get_json()
    ruido_para_suavizado = data.get('ruido', 'gaussiano')
    print(f" Ruido para suavizado seleccionado: {ruido_para_suavizado}")
    return ('', 204)

if __name__ == "__main__":
    print(" Servidor Flask iniciado en http://0.0.0.0:5050")
    app.run(host="0.0.0.0", port=5050, debug=True)