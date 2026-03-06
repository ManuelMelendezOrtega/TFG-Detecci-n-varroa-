import os

import torch
import cv2
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_PATH = "weights_varroa_best.pth"
IMAGE_DIR = "./dataset_split/test"
OUT_DIR = "./detecciones_test"
CONF_THRESHOLD = 0.5
NUM_CLASSES = 2

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.to(device)
model.eval()

print(f"Modelo cargado desde {MODEL_PATH} (usando {device})")


def detectar_varroas(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)
        / 255.0
    )
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(tensor)[0]

    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()

    detecciones = 0
    for (box, score) in zip(boxes, scores):
        if score < CONF_THRESHOLD:
            continue
        detecciones += 1
        (x1, y1, x2, y2) = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
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

    cv2.imwrite(output_path, img)
    print(
        f"{os.path.basename(image_path)} -> {detecciones} detecciones (guardada en {OUT_DIR})"
    )


imagenes = [
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))
]
if not imagenes:
    print(f"No hay imágenes en {IMAGE_DIR}.")
else:
    for img_name in imagenes:
        in_path = os.path.join(IMAGE_DIR, img_name)
        out_path = os.path.join(OUT_DIR, img_name)
        detectar_varroas(in_path, out_path)

print("Detección finalizada.")
