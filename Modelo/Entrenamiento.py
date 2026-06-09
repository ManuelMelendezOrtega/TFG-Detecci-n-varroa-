import os
import glob
import xml.etree.ElementTree as ET
from collections import deque
import time
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ============================================================================
# MÓDULO DE ENTRENAMIENTO DEL MODELO DE VISIÓN ARTIFICIAL
# Arquitectura: Faster R-CNN con backbone ResNet-50-FPN
# Estrategia: Transfer Learning y Precisión Mixta Automática (AMP)
# ============================================================================

DATA_ROOT = "./dataset_split"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NUM_CLASSES = 2       # Obligatorio: Clase 0 (Fondo/Background) + Clase 1 (Varroa)
BATCH_SIZE = 2        # Tamaño del lote ajustado a la memoria VRAM disponible
NUM_WORKERS = 0       # Hilos de CPU dedicados a la carga de datos
SEED = 42

torch.manual_seed(SEED)


def read_voc_boxes(xml_path):
    """
    Extrae y normaliza las coordenadas (Bounding Boxes) de los archivos XML.
    Aplica restricciones de límites para evitar desbordamientos de los tensores.
    """
    root = ET.parse(xml_path).getroot()
    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)
    
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        
        # Unificación de nomenclatura hacia la clase objetivo
        if name.startswith("varr"):
            label = 1
        else:
            continue
            
        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        
        # Corrección de fronteras (Clipping)
        xmin = max(0, min(W - 1, xmin))
        xmax = max(0, min(W - 1, xmax))
        ymin = max(0, min(H - 1, ymin))
        ymax = max(0, min(H - 1, ymax))
        
        # Se asegura de que la caja posea un área matemática real
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            
    return boxes, labels


class VOCDataset(Dataset):
    """
    Clase adaptadora que hereda de torch.utils.data.Dataset.
    Estructura las imágenes y etiquetas en el formato exacto de diccionarios 
    que requiere la arquitectura Faster R-CNN de PyTorch.
    """
    def __init__(self, img_dir):
        self.imgs = sorted(
            glob.glob(os.path.join(img_dir, "*.jpg"))
            + glob.glob(os.path.join(img_dir, "*.png"))
        )
        self.samples = []
        for ip in self.imgs:
            xp = os.path.splitext(ip)[0] + ".xml"
            if os.path.exists(xp):
                self.samples.append((ip, xp))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Bucle de seguridad: Si una imagen no tiene cajas válidas tras la lectura, 
        # salta automáticamente a la siguiente para no interrumpir el DataLoader
        idx = idx % len(self.samples)
        start_idx = idx
        while True:
            img_path, xml_path = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
            boxes, labels = read_voc_boxes(xml_path)
            
            if len(boxes) > 0:
                break
                
            idx = (idx + 1) % len(self.samples)
            if idx == start_idx:
                raise RuntimeError("Excepción Crítica: Ausencia total de anotaciones válidas en el dataset.")

        # Conversión de listas nativas a Tensores de PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # Diccionario 'Target' requerido por la API de torchvision
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        img = F.to_tensor(img)
        return img, target


def collate_fn(batch):
    """ Función de empaquetado personalizado para listas de tensores de distinto tamaño. """
    return tuple(zip(*batch))


if __name__ == "__main__":
    train_ds = VOCDataset(TRAIN_DIR)
    val_ds = VOCDataset(VAL_DIR)

    # Inicialización de los pipelines de carga de datos (DataLoaders)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )

    print(f"Muestras mapeadas -> Entrenamiento: {len(train_ds)} | Validación: {len(val_ds)}")
    
    # Detección de aceleración por hardware (NVIDIA CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Aplicación de Transfer Learning (Pesos Preentrenados COCO)
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # 2. Reemplazo de la cabeza clasificadora (Fine-Tuning)
    # Se cambian las 91 clases originales por nuestras 2 clases (Fondo + Varroa)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)
    model.to(device)

    # Configuración de optimizadores y algoritmos de descenso de gradiente
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Decaimiento del Learning Rate (Ajusta la tasa de aprendizaje durante el proceso)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Scaler para Entrenamiento de Precisión Mixta (AMP) para optimizar memoria en GPU
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    NUM_EPOCHS = 20
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print(f"Iniciando entrenamiento de {NUM_EPOCHS} épocas en {device}...")

    # ========================================================================
    # BUCLE PRINCIPAL DE ENTRENAMIENTO Y VALIDACIÓN
    # ========================================================================
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_hist = deque(maxlen=50)
        t0 = time.time()

        # FASE DE ENTRENAMIENTO (Forward y Backward Pass)
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad(set_to_none=True)

            # Uso de Precisión Mixta Automática (AMP - float16) si hay GPU
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()

            loss_hist.append(float(loss.item()))

        scheduler.step()

        # FASE DE VALIDACIÓN (Evaluación del error con datos no vistos)
        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
            # Nota técnica: Se mantiene model.train() temporalmente en validación 
            # para forzar a Faster R-CNN a devolver métricas de pérdida (loss) 
            # en lugar de predicciones finales (bounding boxes).
            model.train() 
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                val_loss += float(sum(loss_dict.values()).item())
                n_val_batches += 1
                
        val_loss = val_loss / max(1, n_val_batches)
        epoch_time = time.time() - t0
        tr_loss = sum(loss_hist) / len(loss_hist)
        
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(f"[Época {epoch:02d}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  tiempo={epoch_time:.1f}s")

        # Checkpoint: Guardado automático de los pesos si el modelo mejora
        if val_loss < best_val:
            best_val = val_loss
            ruta_guardado = os.path.join("..", "AppWeb", "weights_varroa_best.pth")
            torch.save(model.state_dict(), ruta_guardado)
            print(f"  -> Mejor modelo actualizado y exportado a: {ruta_guardado}")

    # ========================================================================
    # RENDERIZADO DE GRÁFICAS DE MÉTRICAS
    # ========================================================================
    print("\nGenerando métricas visuales del entrenamiento...")
    plt.figure(figsize=(10, 6))
    epocas = range(1, NUM_EPOCHS + 1)
    
    plt.plot(epocas, history["train_loss"], label='Pérdida de Entrenamiento', color='#2980B9', linewidth=2, marker='o')
    plt.plot(epocas, history["val_loss"], label='Pérdida de Validación', color='#E67E22', linewidth=2, marker='s')
    
    plt.title('Evolución de las métricas de error (Loss) durante el entrenamiento', fontsize=14)
    plt.xlabel('Épocas (Epochs)', fontsize=12)
    plt.ylabel('Pérdida (Loss)', fontsize=12)
    plt.xticks(epocas)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    nombre_archivo = "grafica_loss.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Exportación completada: '{nombre_archivo}'.")