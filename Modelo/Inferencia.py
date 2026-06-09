import os
import torch
import cv2
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ============================================================================
# MÓDULO DE INFERENCIA Y RENDERIZADO VISUAL
# Objetivo: Desplegar el modelo entrenado para predecir la ubicación de las 
# varroas en imágenes nuevas y generar una representación gráfica de los resultados.
# ============================================================================

MODEL_PATH = "../AppWeb/weights_varroa_best.pth"
IMAGE_DIR = "./dataset_split/test"
OUT_DIR = "./detecciones_test"

# Umbral de confianza operativo para producción (Solo dibuja si la seguridad es >= 50%)
CONF_THRESHOLD = 0.5
NUM_CLASSES = 2

os.makedirs(OUT_DIR, exist_ok=True)

# Detección de aceleración por hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# INICIALIZACIÓN Y CARGA DEL MODELO
# ============================================================================
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)

# Adaptación de la arquitectura a nuestro dominio (2 clases)
in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)

# Carga de los pesos entrenados almacenados en el disco
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.to(device)

# Modo Inferencia: Optimiza el modelo congelando capas de entrenamiento (ej. Dropout)
model.eval()

print(f"Modelo cargado correctamente desde {MODEL_PATH} (Ejecutando en {device})")


def detectar_varroas(image_path, output_path):
    """
    Procesa una imagen individual, ejecuta la red neuronal y dibuja las 
    cajas delimitadoras sobre las predicciones que superen el umbral de confianza.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error de lectura en la imagen: {image_path}")
        return
        
    # Preprocesamiento: OpenCV lee en formato BGR, pero PyTorch espera RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Conversión a Tensor, reorganización de canales (H,W,C -> C,H,W) y normalización [0, 1]
    tensor = (
        torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    )
    # Expansión de dimensiones para simular un batch de tamaño 1 (requerido por el modelo)
    tensor = tensor.unsqueeze(0).to(device)

    # Inferencia: Se desactiva el motor de gradientes para ahorrar memoria RAM/VRAM
    with torch.no_grad():
        preds = model(tensor)[0]

    # Transferencia de los resultados de la GPU a la CPU para su procesado con NumPy
    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    detecciones = 0
    
    # Filtrado y renderizado geométrico de los resultados
    for (box, score) in zip(boxes, scores):
        if score < CONF_THRESHOLD:
            continue
            
        detecciones += 1
        (x1, y1, x2, y2) = box.astype(int)
        
        # Dibuja la caja delimitadora (Bounding Box) en color verde
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Superpone una etiqueta de texto con el porcentaje de confianza
        cv2.putText(
            img,
            f"Varroa {score:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Guarda físicamente la imagen resultante en el disco
    cv2.imwrite(output_path, img)
    print(f"{os.path.basename(image_path)} -> {detecciones} detecciones (guardada en {OUT_DIR})")


# ============================================================================
# BUCLE DE PROCESAMIENTO (BATCH INFERENCE)
# ============================================================================
imagenes = [
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))
]

if not imagenes:
    print(f"No hay imágenes disponibles para procesar en {IMAGE_DIR}.")
else:
    for img_name in imagenes:
        in_path = os.path.join(IMAGE_DIR, img_name)
        out_path = os.path.join(OUT_DIR, img_name)
        detectar_varroas(in_path, out_path)

print("Proceso de inferencia y renderizado visual finalizado con éxito.")