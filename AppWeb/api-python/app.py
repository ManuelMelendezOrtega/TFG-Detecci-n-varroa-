# ============================================================================
# API REST de Inferencia para Detección de Varroa Destructor
# Proyecto: Trabajo de Fin de Grado
# Tecnologías: Flask, PyTorch, OpenCV
# ============================================================================

import os
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Inicialización del microservicio web
app = Flask(__name__)

# Configuración de rutas estáticas para almacenamiento temporal
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Detección automática de aceleración por hardware (NVIDIA GPU / CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    """
    Carga la arquitectura Faster R-CNN, ajusta la cabeza del clasificador
    para 2 clases (Fondo y Varroa) y carga los pesos entrenados mediante Transfer Learning.
    """
    # 1. Cargar arquitectura base
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    
    # 2. Modificar la capa final para nuestro dominio específico
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    # 3. Cargar los pesos del Fine-Tuning
    path_pesos = os.path.join(os.path.dirname(BASE_DIR), "weights_varroa_best.pth")
    if os.path.exists(path_pesos):
        model.load_state_dict(torch.load(path_pesos, map_location=device))
        print(">>> [OK] IA lista para trabajar en:", device)
    
    # Poner el modelo en modo inferencia (evaluación)
    return model.to(device).eval()

# Cargar el modelo en memoria RAM/VRAM al arrancar el servidor
model = get_model()

@app.route("/api/analizar", methods=["POST"])
def analizar():
    """
    Endpoint principal. Recibe una imagen por POST, la procesa mediante la IA,
    dibuja las Bounding Boxes en las detecciones y devuelve los resultados en JSON.
    """
    # Verificación de la recepción del archivo
    if "foto" not in request.files: 
        return jsonify({"error": "No hay foto"}), 400
    
    file = request.files["foto"]
    nombre_seguro = file.filename 
    ruta_input = os.path.join(UPLOAD_FOLDER, nombre_seguro)
    file.save(ruta_input)

    # Lectura robusta de la imagen (Evita errores de codificación con tildes/ñ en Windows)
    with open(ruta_input, "rb") as f:
        bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 500

    # Preprocesamiento de la imagen para PyTorch (BGR -> RGB, Normalización 0-1, Tensor a GPU)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float().to(device)
    
    # Inferencia del modelo sin calcular gradientes (ahorro de memoria)
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    conteo = 0
    # Post-procesamiento: Filtrado por umbral de confianza y dibujo de cajas
    for score, box in zip(prediction["scores"].cpu().numpy(), prediction["boxes"].cpu().numpy()):
        if score > 0.5: # Umbral del 50% de confianza
            conteo += 1
            b = box.astype(int)
            # Dibujar caja verde (BGR: 0, 255, 0) de grosor 2
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    # Guardado de la imagen resultante
    nombre_resultado = "det_" + nombre_seguro
    ruta_output = os.path.join(RESULT_FOLDER, nombre_resultado)
    
    # Codificación de vuelta a JPG robusta
    _, img_encoded = cv2.imencode(".jpg", img)
    with open(ruta_output, "wb") as f:
        f.write(img_encoded)
    
    print(f">>> [IA] Detectadas {conteo} varroas en {nombre_seguro}")
    
    # Respuesta RESTful para el Backend de Java
    return jsonify({
        "conteo": conteo, 
        "rutaMarcada": nombre_resultado, 
        "nombreArchivo": nombre_seguro
    })

if __name__ == "__main__":
    # Arrancar el servidor en el puerto 5000
    app.run(port=5000, debug=True)