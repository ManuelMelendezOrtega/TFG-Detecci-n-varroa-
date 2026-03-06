import os
import xml.etree.ElementTree as ET
import math
import csv
from collections import defaultdict

import torch
import cv2
import numpy as np
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_PATH = "weights_varroa_best.pth"
TEST_DIR = "./dataset_split/test"
CONF_THRESH = 0.05
IOU_THRESH = 0.5
NUM_CLASSES = 2
SAVE_DETS_CSV = "detecciones_test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_voc(xml_path):
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
        xmin = max(0, min(W - 1, xmin))
        xmax = max(0, min(W - 1, xmax))
        ymin = max(0, min(H - 1, ymin))
        ymax = max(0, min(H - 1, ymax))
        if xmax > xmin and ymax > ymin:
            gts.append([xmin, ymin, xmax, ymax])
    return np.array(gts, dtype=np.float32)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = a + b - inter + 1e-9
    return inter / union


weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)
state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.to(device).eval()

img_names = [
    f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png"))
]
img_names.sort()
assert img_names, f"No hay imágenes en {TEST_DIR}"

detections = []
gt_boxes_per_img = {}
gt_count = 0

for name in img_names:
    stem, _ = os.path.splitext(name)
    img_path = os.path.join(TEST_DIR, name)
    xml_path = os.path.join(TEST_DIR, stem + ".xml")
    if not os.path.exists(xml_path):
        continue

    gts = parse_voc(xml_path)
    gt_boxes_per_img[name] = gts
    gt_count += len(gts)

    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.as_tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    )
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)[0]
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    for b, s in zip(boxes, scores):
        if s >= CONF_THRESH:
            x1, y1, x2, y2 = b.astype(float)
            detections.append((name, float(s), x1, y1, x2, y2))

if SAVE_DETS_CSV:
    with open(SAVE_DETS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "score", "x1", "y1", "x2", "y2"])
        for row in detections:
            w.writerow(row)

detections.sort(key=lambda x: x[1], reverse=True)

tp = []
fp = []
matched = {
    img: np.zeros(len(gt_boxes_per_img[img]), dtype=bool)
    for img in gt_boxes_per_img
}

for (img_id, score, x1, y1, x2, y2) in detections:
    if img_id not in gt_boxes_per_img:
        fp.append(1)
        tp.append(0)
        continue
    gts = gt_boxes_per_img[img_id]
    if len(gts) == 0:
        fp.append(1)
        tp.append(0)
        continue
    det = np.array([x1, y1, x2, y2], dtype=np.float32)

    ious = np.array([iou(det, gt) for gt in gts])
    j = int(np.argmax(ious))
    iou_max = ious[j]
    if iou_max >= IOU_THRESH and not matched[img_id][j]:
        tp.append(1)
        fp.append(0)
        matched[img_id][j] = True
    else:
        fp.append(1)
        tp.append(0)

tp = np.array(tp)
fp = np.array(fp)
cum_tp = np.cumsum(tp)
cum_fp = np.cumsum(fp)

prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
rec = cum_tp / max(gt_count, 1e-9)


def average_precision(prec, rec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    inds = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[inds + 1] - mrec[inds]) * mpre[inds + 1])
    return ap


AP = average_precision(prec, rec)
mAP = AP

f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
best_idx = int(np.argmax(f1)) if len(f1) else 0
best_P, best_R, best_F1 = (
    prec[best_idx] if len(prec) else 0.0,
    rec[best_idx] if len(rec) else 0.0,
    f1[best_idx] if len(f1) else 0.0,
)

print("\n===== RESULTADOS TEST (VOC IoU=0.5) =====")
print(f"GT total (varroas): {gt_count}")
print(f"Detecciones evaluadas: {len(detections)} (conf ≥ {CONF_THRESH})")
print(f"mAP@0.5: {mAP:.4f}")
print(
    f"Mejor punto PR -> Precisión: {best_P:.4f} | Recall: {best_R:.4f} | F1: {best_F1:.4f}"
)
print(f"(CSV con detecciones: {SAVE_DETS_CSV})")
