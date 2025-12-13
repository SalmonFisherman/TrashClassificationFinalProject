import os
import cv2
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "garbage"
OUTPUT_DIR = BASE_DIR / "data" / "dataset"
IMG_SIZE = 224

OVERSAMPLE_RATIO = 0.7     # minor class → 70% target
UNDERSAMPLE_RATIO = 1.3   # major class → max 1.3x median

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resize
def resize_with_padding(img, size=224):
    h, w, _ = img.shape
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    img = cv2.resize(img, (nw, nh))

    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left

    return cv2.copyMakeBorder(
        img,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

# Change format to RGB
def load_and_process(path):
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img = resize_with_padding(img)
    return img


class_files = {}
for cls in sorted(os.listdir(INPUT_DIR)):
    cls_path = os.path.join(INPUT_DIR, cls)
    class_files[cls] = [
        os.path.join(cls_path, f)
        for f in os.listdir(cls_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

counts = {cls: len(files) for cls, files in class_files.items()}
median_count = int(np.median(list(counts.values())))

print("Original distribution:", counts)
print("Median:", median_count)

final_labels = []

for cls, files in class_files.items():
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    target_min = int(median_count * OVERSAMPLE_RATIO)
    target_max = int(median_count * UNDERSAMPLE_RATIO)

    # Undersampling
    if len(files) > target_max:
        files = random.sample(files, target_max)

    # Oversampling (duplication)
    while len(files) < target_min:
        files.append(random.choice(files))

    for idx, path in enumerate(tqdm(files, desc=f"Processing {cls}")):
        img = load_and_process(path)

        out_name = f"{cls}_{idx}.jpg"
        out_path = os.path.join(OUTPUT_DIR, cls, out_name)

        cv2.imwrite(
            out_path,
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        final_labels.append(cls)

label_to_id = {cls: i for i, cls in enumerate(sorted(class_files))}
y_encoded = [label_to_id[label] for label in final_labels]