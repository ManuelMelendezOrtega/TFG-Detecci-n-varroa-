import os
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datetime import datetime

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    path_pesos = os.path.join(os.path.dirname(BASE_DIR), "weights_varroa_best.pth")
    if os.path.exists(path_pesos):
        model.load_state_dict(torch.load(path_pesos, map_location=device))
        print(">>> [OK] IA lista para trabajar.")
    return model.to(device).eval()

model = get_model()

@app.route("/api/analizar", methods=["POST"])
def analizar():
    if "foto" not in request.files: return jsonify({"error": "No hay foto"}), 400
    
    file = request.files["foto"]
    
    nombre_seguro = file.filename 
    
    ruta_input = os.path.join(UPLOAD_FOLDER, nombre_seguro)
    file.save(ruta_input)

    with open(ruta_input, "rb") as f:
        bytes_data = bytearray(f.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "No se pudo decodificar la imagen"}), 500

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb / 255.0).permute(2, 0, 1).float().to(device)
    
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    conteo = 0
    for score, box in zip(prediction["scores"].cpu().numpy(), prediction["boxes"].cpu().numpy()):
        if score > 0.5:
            conteo += 1
            b = box.astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    nombre_resultado = "det_" + nombre_seguro
    ruta_output = os.path.join(RESULT_FOLDER, nombre_resultado)
    
    _, img_encoded = cv2.imencode(".jpg", img)
    with open(ruta_output, "wb") as f:
        f.write(img_encoded)
    
    print(f">>> [IA] Detectadas {conteo} varroas en {nombre_seguro}")
    return jsonify({
        "conteo": conteo, 
        "rutaMarcada": nombre_resultado, 
        "nombreArchivo": nombre_seguro
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)