import os
import glob
import random
import shutil

from tqdm import tqdm

INPUT_DIR = "./augmented"
OUTPUT_DIR = "./dataset_split"
R_TRAIN, R_VAL, R_TEST = 0.7, 0.2, 0.1
SEED = 42
random.seed(SEED)

for sub in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

images = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    + glob.glob(os.path.join(INPUT_DIR, "*.png"))
)
pairs = []
for img in images:
    xml = os.path.splitext(img)[0] + ".xml"
    if os.path.exists(xml):
        pairs.append((img, xml))

random.shuffle(pairs)
n = len(pairs)
n_train = int(n * R_TRAIN)
n_val = int(n * R_VAL)
train_pairs = pairs[:n_train]
val_pairs = pairs[n_train : n_train + n_val]
test_pairs = pairs[n_train + n_val :]


def copy_pairs(pairs, sub):
    dest = os.path.join(OUTPUT_DIR, sub)
    for img, xml in tqdm(pairs, desc=f"Copiando {sub}"):
        shutil.copy2(img, os.path.join(dest, os.path.basename(img)))
        shutil.copy2(xml, os.path.join(dest, os.path.basename(xml)))


copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")
copy_pairs(test_pairs, "test")

print(
    f"Total pares: {n} | train: {len(train_pairs)} | val: {len(val_pairs)} | test: {len(test_pairs)}"
)
