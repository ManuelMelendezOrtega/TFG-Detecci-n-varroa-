import os
import glob
import random
import xml.etree.ElementTree as ET
import cv2

# ============================================================================
# Módulo de Validación Visual (Visual Debugging)
# Objetivo: Renderizar las cajas delimitadoras (Bounding Boxes) sobre una 
# muestra aleatoria del dataset para verificar la integridad espacial de los 
# datos tras el proceso de Data Augmentation.
# ============================================================================

IN_DIR = "./augmented"     # Directorio origen con el dataset aumentado
OUT_DIR = "./debug_bb"     # Directorio destino para las imágenes de comprobación
N_SAMPLES = 1000           # Tamaño de la muestra aleatoria a evaluar
os.makedirs(OUT_DIR, exist_ok=True)


def read_boxes(xml_path):
    """
    Parsea el archivo XML asociado a la imagen y extrae las coordenadas
    de las cajas delimitadoras, aplicando correcciones de bordes (clipping).
    """
    root = ET.parse(xml_path).getroot()
    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)
    
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        
        # Corrección de bordes: Garantiza que las coordenadas no excedan 
        # las dimensiones reales de la imagen.
        xmin = max(0, min(W - 1, xmin))
        xmax = max(0, min(W - 1, xmax))
        ymin = max(0, min(H - 1, ymin))
        ymax = max(0, min(H - 1, ymax))
        
        # Filtro de validación: Solo añade cajas con área matemática positiva
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
            
    return boxes

# ============================================================================
# BUCLE DE MUESTREO Y RENDERIZADO
# ============================================================================

# Recopilación de todas las imágenes generadas en el directorio
imgs = sorted(
    glob.glob(os.path.join(IN_DIR, "*.jpg"))
    + glob.glob(os.path.join(IN_DIR, "*.png"))
)

# Fijación de semilla y selección aleatoria de la muestra de validación
random.seed(42)
samples = random.sample(imgs, min(N_SAMPLES, len(imgs)))

for ip in samples:
    xp = os.path.splitext(ip)[0] + ".xml"
    
    # Omite las imágenes que no posean anotaciones (XML)
    if not os.path.exists(xp):
        continue
        
    im = cv2.imread(ip)
    if im is None:
        continue
        
    # Renderizado iterativo de las cajas delimitadoras sobre la imagen
    for (xmin, ymin, xmax, ymax) in read_boxes(xp):
        # Dibuja un rectángulo verde (BGR: 0, 255, 0) de grosor 2
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
    # Persistencia de la imagen marcada en el directorio de depuración
    out = os.path.join(OUT_DIR, os.path.basename(ip))
    cv2.imwrite(out, im)

print(f"Validación finalizada. Revisa el directorio {OUT_DIR} con {len(samples)} imágenes de muestra.")