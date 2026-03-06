import os
from datetime import datetime

import cv2
import torch
from flask import Flask, render_template, request, url_for
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, "weights_varroa_best.pth")
NUM_CLASSES = 2

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)

    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


model = load_model()
print(f"Modelo cargado desde {MODEL_PATH} en {device}")


def detectar_varroa(image_path, output_path, conf_thresh=0.5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return 0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.as_tensor(img_rgb, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )
    tensor = tensor.to(device)

    with torch.no_grad():
        pred = model(tensor)[0]

    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()

    count = 0
    for box, score in zip(boxes, scores):
        if score < conf_thresh:
            continue
        count += 1
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Varroa {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, img)
    return count


@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        files = request.files.getlist("images")
        for f in files:
            if not f or f.filename == "":
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_name = f"{timestamp}_{f.filename}"
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            f.save(upload_path)

            result_name = "det_" + safe_name
            result_path = os.path.join(app.config["RESULT_FOLDER"], result_name)

            count = detectar_varroa(upload_path, result_path, conf_thresh=0.5)

            results.append(
                {
                    "orig": url_for("static", filename=f"uploads/{safe_name}"),
                    "det": url_for("static", filename=f"results/{result_name}"),
                    "count": count,
                    "filename": f.filename,
                }
            )

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
