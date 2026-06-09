import os
import glob
import random
import shutil
from tqdm import tqdm

# ============================================================================
# Módulo de Particionado del Dataset (Dataset Split)
# Objetivo: Segregar de forma aleatoria y controlada el dataset en subconjuntos 
# independientes (Entrenamiento, Validación y Test) para garantizar una 
# evaluación objetiva y evitar el sobreajuste (overfitting).
# ============================================================================

INPUT_DIR = "./augmented"
OUTPUT_DIR = "./dataset_split"

# Proporciones de distribución estadística (Suman 1.0 / 100%)
R_TRAIN, R_VAL, R_TEST = 0.7, 0.2, 0.1
SEED = 42
random.seed(SEED)  # Garantiza la reproducibilidad de la partición aleatoria

# Creación de la estructura jerárquica de directorios destino
for sub in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# Recopilación e indexación de las imágenes disponibles
images = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    + glob.glob(os.path.join(INPUT_DIR, "*.png"))
)

# Emparejamiento estricto: Asegura que cada imagen tiene su anotación XML asociada
pairs = []
for img in images:
    xml = os.path.splitext(img)[0] + ".xml"
    if os.path.exists(xml):
        pairs.append((img, xml))

# Mezcla aleatoria de los pares para eliminar sesgos de ordenación en la captura
random.shuffle(pairs)

# Cálculo de los índices de corte basados en los porcentajes definidos
n = len(pairs)
n_train = int(n * R_TRAIN)
n_val = int(n * R_VAL)

# Segmentación de los subconjuntos mediante slicing de listas
train_pairs = pairs[:n_train]
val_pairs = pairs[n_train : n_train + n_val]
test_pairs = pairs[n_train + n_val :]


def copy_pairs(pairs, sub):
    """
    Persiste físicamente los pares de archivos (Imagen + XML) 
    dentro del subdirectorio correspondiente.
    """
    dest = os.path.join(OUTPUT_DIR, sub)
    for img, xml in tqdm(pairs, desc=f"Copiando conjunto de {sub}"):
        shutil.copy2(img, os.path.join(dest, os.path.basename(img)))
        shutil.copy2(xml, os.path.join(dest, os.path.basename(xml)))


# Ejecución de la transferencia física de los datos segmentados
copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")
copy_pairs(test_pairs, "test")

print(f"\nParticionado completado de forma exitosa.")
print(f"Total pares válidos: {n} | Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")