import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# ============================================================================
# Procesamiento de Data Augmentation (Aumento de Datos)
# Objetivo: Aumentar el tamaño del dataset original para que la red neuronal
# aprenda mejor y no memorice las imágenes (evitar el overfitting).
# ============================================================================

INPUT_DIR = "./DatasetOriginal"  
OUTPUT_DIR = "./augmented"       
AUGS_PER_IMAGE = 3               # Número de variaciones generadas por cada imagen original
SEED = 42                        # Semilla aleatoria para que el proceso sea reproducible

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- PIPELINE DE TRANSFORMACIONES (Configuración de filtros) ---
# Se define qué cambios se aplicarán de forma aleatoria a las imágenes
transform = A.Compose(
    [
        # 1. Transformaciones de posición: Volteos y rotaciones de 90 grados
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
        ], p=0.7),
        
        # 2. Modificaciones afines: Pequeños desplazamientos, escala y rotaciones libres
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.7,
        ),
        
        # 3. Ajustes de iluminación y contraste de la imagen
        A.OneOf([
            A.RandomBrightnessContrast(0.15, 0.15, p=0.7),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ], p=0.8),
        
        # 4. Filtros de desenfoque e introducción de ruido de cámara
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ], p=0.4),
    ],
    # Recalcula automáticamente las coordenadas de las cajas delimitiadoras (Bounding Boxes)
    # para que sigan encajando con los objetos tras mover o rotar la imagen.
    bbox_params=A.BboxParams(
        format="pascal_voc", 
        label_fields=["labels"], 
        min_area=8,          # Descarta la caja si el objeto queda con un tamaño menor a 8 píxeles
        min_visibility=0.3,  # Descarta la caja si más del 70% del objeto se queda fuera de la imagen
    ),
)


def read_voc(xml_path):
    """
    Lee los archivos XML en formato PASCAL VOC.
    Devuelve la estructura del XML, las dimensiones de la imagen y las coordenadas de las cajas.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()

        # Corrección de nombres para asegurar que todas las etiquetas sean "varroa"
        if name.startswith("varr"):
            name = "varroa"

        bb = obj.find("bndbox")
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))

        # Restringe las coordenadas para que no se salgan de los límites de la imagen
        xmin, xmax = clamp(xmin, 0, W - 1), clamp(xmax, 0, W - 1)
        ymin, ymax = clamp(ymin, 0, H - 1), clamp(ymax, 0, H - 1)

        # Corrige la orientación si las coordenadas de las esquinas están invertidas
        if xmax < xmin: xmin, xmax = xmax, xmin
        if ymax < ymin: ymin, ymax = ymax, ymin

        # Valida que la caja tenga un tamaño mínimo antes de añadirla
        if (xmax - xmin) >= 2 and (ymax - ymin) >= 2:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)

    return tree, (W, H), boxes, labels


def write_voc(template_tree, img_name, out_xml_path, bboxes, labels, width, height):
    """
    Crea un nuevo archivo XML para la imagen modificada, guardando
    las nuevas dimensiones y las coordenadas recalculadas de los objetos.
    """
    root = template_tree.getroot()
    root.find("filename").text = img_name
    
    path_el = root.find("path")
    if path_el is not None:
        path_el.text = img_name

    size = root.find("size")
    size.find("width").text = str(width)
    size.find("height").text = str(height)
    size.find("depth").text = "3"

    # Elimina las cajas antiguas del XML original
    for obj in root.findall("object"):
        root.remove(obj)

    # Añade las nuevas cajas recalculadas al archivo XML
    for (xmin, ymin, xmax, ymax), label in zip(bboxes, labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "occluded").text = "0"

        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(max(0, xmin)))
        ET.SubElement(bnd, "ymin").text = str(int(max(0, ymin)))
        ET.SubElement(bnd, "xmax").text = str(int(min(width - 1, xmax)))
        ET.SubElement(bnd, "ymax").text = str(int(min(height - 1, ymax)))

    # Guarda el XML formateado con saltos de línea e indentaciones
    xml_str = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="\t")
    with open(out_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty)


def safe_name(base, idx):
    """ Genera el nombre de archivo para las nuevas imágenes (Ej: imagen_aug1.jpg) """
    stem, ext = os.path.splitext(base)
    return f"{stem}_aug{idx}{ext}"


# ============================================================================
# BUCLE PRINCIPAL DE PROCESAMIENTO POR LOTES
# ============================================================================

images = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    + glob.glob(os.path.join(INPUT_DIR, "*.png"))
)
assert images, f"Error: No se encontraron imágenes en el directorio {INPUT_DIR}"


for img_path in tqdm(images, desc="Aplicando aumento de datos"):
    xml_path = os.path.splitext(img_path)[0] + ".xml"

    # Si la imagen no tiene archivo XML asociado, simplemente se copia al destino
    if not os.path.exists(xml_path):
        shutil.copy2(img_path, os.path.join(OUTPUT_DIR, os.path.basename(img_path)))
        continue

    tree, (W, H), boxes, labels = read_voc(xml_path)
    img = cv2.imread(img_path)
    if img is None:
        continue

    base = os.path.basename(img_path)

    # Guarda una copia exacta de la imagen y XML originales en la carpeta de destino
    out_img0 = os.path.join(OUTPUT_DIR, base)
    cv2.imwrite(out_img0, img)
    out_xml0 = os.path.join(OUTPUT_DIR, os.path.splitext(base)[0] + ".xml")
    write_voc(tree, os.path.basename(out_img0), out_xml0, boxes, labels, W, H)
    
    # Genera las imágenes modificadas según el número indicado
    for i in range(1, AUGS_PER_IMAGE + 1):
        tried = 0
        created = False

        # Sistema de reintentos por si los filtros aplicados eliminan todos los objetos de la imagen
        while tried < 6 and not created:
            tried += 1

            transformed = transform(image=img, bboxes=boxes, labels=labels)
            aug_img = transformed["image"]
            h2, w2 = aug_img.shape[:2]

            tb, tl = [], []
            for (xmin, ymin, xmax, ymax), lab in zip(
                transformed["bboxes"], transformed["labels"]
            ):
                xmin = max(0, min(w2 - 1, int(round(xmin))))
                xmax = max(0, min(w2 - 1, int(round(xmax))))
                ymin = max(0, min(h2 - 1, int(round(ymin))))
                ymax = max(0, min(h2 - 1, int(round(ymax))))

                if (xmax - xmin) >= 2 and (ymax - ymin) >= 2:
                    tb.append([xmin, ymin, xmax, ymax])
                    tl.append(lab)
            
            # Condición de rechazo: Si la transformación eliminó todas las cajas, vuelve a intentarlo
            if not tb and len(boxes) > 0:
                continue

            # Guarda en disco la imagen modificada y su nuevo archivo XML
            out_name = safe_name(base, i)
            out_img_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_img_path, aug_img)

            out_xml_path = os.path.join(
                OUTPUT_DIR, os.path.splitext(out_name)[0] + ".xml"
            )
            write_voc(tree, os.path.basename(out_img_path), out_xml_path, tb, tl, w2, h2)
            created = True

        if not created:
            print(f"[WARN] No se pudo generar una variación válida para {base} (Variación {i})")