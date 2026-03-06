import os
import glob
import xml.etree.ElementTree as ET

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

DATA_ROOT = "./dataset_split"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NUM_CLASSES = 2
BATCH_SIZE = 2
NUM_WORKERS = 0
SEED = 42
torch.manual_seed(SEED)


def read_voc_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    W = int(root.find("size/width").text)
    H = int(root.find("size/height").text)
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name.startswith("varr"):
            label = 1
        else:
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
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
    return boxes, labels


class VOCDataset(Dataset):
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
                raise RuntimeError("No hay cajas válidas en el dataset.")

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

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
    return tuple(zip(*batch))


if __name__ == "__main__":
    from collections import deque
    import time

    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    train_ds = VOCDataset(TRAIN_DIR)
    val_ds = VOCDataset(VAL_DIR)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Samples -> train: {len(train_ds)} | val: {len(val_ds)}")
    print("Preparado el dataset. Se crea el modelo y se empieza a entrenar.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    NUM_EPOCHS = 20
    print(f"Entrenando {NUM_EPOCHS} épocas en {device}...")

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_hist = deque(maxlen=50)
        t0 = time.time()

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)

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

        val_loss, n_val_batches = 0.0, 0
        with torch.no_grad():
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

        print(
            f"[Época {epoch:02d}] "
            f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
            f"tiempo={epoch_time:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "weights_varroa_best.pth")
            print("  -> Mejor modelo actualizado: weights_varroa_best.pth")

    torch.save(model.state_dict(), "weights_varroa_last.pth")
    print("Entrenamiento finalizado. Pesos guardados en weights_varroa_last.pth")
