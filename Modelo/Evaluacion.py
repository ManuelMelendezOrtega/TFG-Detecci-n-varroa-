import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import torch
import cv2
import numpy as np
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ============================================================================
# MÓDULO DE EVALUACIÓN Y CÁLCULO DE MÉTRICAS (TESTING)
# Objetivo: Evaluar el rendimiento del modelo sobre el conjunto de Test aislado,
# calculando las métricas estándar de detección de objetos: mAP, Precisión y Recall.
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "AppWeb", "weights_varroa_best.pth")
TEST_DIR = os.path.join(BASE_DIR, "dataset_split", "test")

# Hiperparámetros de Evaluación
CONF_THRESH = 0.05  # Umbral de confianza inicial bajo para construir la curva PR completa
IOU_THRESH = 0.5    # Nivel de solapamiento mínimo (50%) para considerar una detección como válida
NUM_CLASSES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_voc(xml_path):
    """ Extrae y valida las coordenadas reales (Ground Truth) del dataset de Test. """
    root = ET.parse(xml_path).getroot()
    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)
    gts = []
    
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if not name.startswith("varr"):
            continue
            
        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        
        xmin, xmax = max(0, min(W - 1, xmin)), max(0, min(W - 1, xmax))
        ymin, ymax = max(0, min(H - 1, ymin)), max(0, min(H - 1, ymax))
        
        if xmax > xmin and ymax > ymin:
            gts.append([xmin, ymin, xmax, ymax])
            
    return np.array(gts, dtype=np.float32)

def iou(boxA, boxB):
    """
    Calcula la métrica 'Intersection over Union' (IoU).
    Mide matemáticamente el área de solapamiento entre la caja predicha por la IA 
    y la caja real etiquetada por el humano.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter = max(0, xB - xA) * max(0, yB - yA)
    a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    union = a + b - inter + 1e-9  # Se añade 1e-9 para evitar divisiones por cero
    return inter / union

# ============================================================================
# CARGA DEL MODELO Y PESOS ENTRENADOS
# ============================================================================

weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)

# Restauración del estado óptimo alcanzado durante el entrenamiento
state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)

# Modo Evaluación: Desactiva el cálculo de gradientes y fija capas como Dropout o BatchNorm
model.to(device).eval()

img_names = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png"))]
img_names.sort()
assert img_names, f"Error: Conjunto de Test vacío en {TEST_DIR}"

detections = []
gt_boxes_per_img = {}
gt_count = 0

# ============================================================================
# FASE DE INFERENCIA
# ============================================================================
for name in img_names:
    stem, _ = os.path.splitext(name)
    img_path = os.path.join(TEST_DIR, name)
    xml_path = os.path.join(TEST_DIR, stem + ".xml")
    
    if not os.path.exists(xml_path):
        continue

    # Carga de la verdad base (Ground Truth)
    gts = parse_voc(xml_path)
    gt_boxes_per_img[name] = gts
    gt_count += len(gts)

    img = cv2.imread(img_path)
    if img is None:
        continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalización del tensor de entrada (formato [C, H, W] esperado por PyTorch)
    tensor = (torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0)
    tensor = tensor.unsqueeze(0).to(device)

    # Inferencia sin cálculo de gradientes para optimizar la memoria
    with torch.no_grad():
        pred = model(tensor)[0]
        
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    # Filtrado de detecciones por el umbral mínimo de confianza
    for b, s in zip(boxes, scores):
        if s >= CONF_THRESH:
            x1, y1, x2, y2 = b.astype(float)
            detections.append((name, float(s), x1, y1, x2, y2))

# Ordenación descendente por nivel de confianza (Requisito para la Curva PR)
detections.sort(key=lambda x: x[1], reverse=True)

# ============================================================================
# CÁLCULO DE VERDADEROS POSITIVOS (TP) Y FALSOS POSITIVOS (FP)
# ============================================================================
tp = []
fp = []

# Diccionario para controlar qué cajas reales ya han sido descubiertas por la IA
matched = {img: np.zeros(len(gt_boxes_per_img[img]), dtype=bool) for img in gt_boxes_per_img}

for (img_id, score, x1, y1, x2, y2) in detections:
    # Penalización si la IA predice en una imagen que no tiene Ground Truth
    if img_id not in gt_boxes_per_img or len(gt_boxes_per_img[img_id]) == 0:
        fp.append(1)
        tp.append(0)
        continue
        
    gts = gt_boxes_per_img[img_id]
    det = np.array([x1, y1, x2, y2], dtype=np.float32)

    # Búsqueda de la caja real (GT) con mayor grado de solapamiento (IoU)
    ious = np.array([iou(det, gt) for gt in gts])
    j = int(np.argmax(ious))
    iou_max = ious[j]
    
    # Evaluación: Si supera el 50% de solapamiento y esa varroa no había sido contada antes
    if iou_max >= IOU_THRESH and not matched[img_id][j]:
        tp.append(1)
        fp.append(0)
        matched[img_id][j] = True  # Marca la varroa como 'Detectada'
    else:
        # Falso positivo: La caja no acierta a la varroa o señala una que ya estaba descubierta
        fp.append(1)
        tp.append(0)

# ============================================================================
# INTEGRACIÓN DE MÉTRICAS (Precisión, Recall y mAP)
# ============================================================================
tp = np.array(tp)
fp = np.array(fp)
cum_tp = np.cumsum(tp)  # Suma acumulada de Verdaderos Positivos
cum_fp = np.cumsum(fp)  # Suma acumulada de Falsos Positivos

# Precisión: De todas las cajas que dibuja la IA, ¿qué porcentaje son varroas reales?
prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

# Recall (Sensibilidad): De todas las varroas que hay en la colmena, ¿qué porcentaje detectó la IA?
rec = cum_tp / max(gt_count, 1e-9)

def average_precision(prec, rec):
    """
    Calcula el Área Bajo la Curva Precisión-Recall (mAP).
    Aplica una interpolación continua para suavizar la curva métrica.
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    
    # Se garantiza que la precisión sea monótonamente decreciente
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
        
    # Integración numérica del área bajo la curva
    inds = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[inds + 1] - mrec[inds]) * mpre[inds + 1])
    return ap

AP = average_precision(prec, rec)
mAP = AP

# F1-Score: Media armónica entre Precisión y Recall (El punto de equilibrio perfecto)
f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
best_idx = int(np.argmax(f1)) if len(f1) else 0

best_P = prec[best_idx] if len(prec) else 0.0
best_R = rec[best_idx] if len(rec) else 0.0
best_F1 = f1[best_idx] if len(f1) else 0.0

# REPORTE DE RESULTADOS DE INVESTIGACIÓN
print("\n===== RESULTADOS TEST =====")
print(f"Ground Truth total (varroas reales): {gt_count}")
print(f"Detecciones evaluadas: {len(detections)} (confianza >= {CONF_THRESH})")
print(f"mAP@0.5: {mAP:.4f}")
print(f"Punto Óptimo (PR) -> Precisión: {best_P:.4f} | Recall: {best_R:.4f} | F1-Score: {best_F1:.4f}")