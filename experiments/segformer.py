from pathlib import Path

import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

IMAGE_PATH = Path("data/Raw_Data/vidd_1/00001.jpg")
OUTPUT_DIR = Path("outputs/segformer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

original_width, original_height = image.size

image_small = image.copy()
image_small.thumbnail((256, 256))

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

height, width = image.height, image.width

logits = torch.nn.functional.interpolate(
    outputs.logits,
    size=(height, width),
    mode="bilinear",
    align_corners=False,
)

prediction = logits.argmax(dim=1)[0].cpu().numpy()

for class_id, name in model.config.id2label.items():
    if any(word in name.lower() for word in ["road", "floor", "earth"]):
        print(class_id, name)

road_ids = [
    int(class_id)
    for class_id, name in model.config.id2label.items()
    if name.lower().strip() == "road"
]

if not road_ids:
    raise RuntimeError("No road class found in model labels")

road_mask = (
    np.isin(prediction, road_ids).astype(np.uint8) * 255
)

cv2.imwrite(
    str(OUTPUT_DIR / "pavement_mask.png"),
    road_mask
)

original = cv2.imread(str(IMAGE_PATH))

overlay = original.copy()
overlay[road_mask > 0] = (0, 255, 0)

visualization = cv2.addWeighted(
    original, 0.65, overlay, 0.35, 0
)

cv2.imwrite(
    str(OUTPUT_DIR / "segmentation_overlay.jpg"),
    visualization
)

print("Pavement pixels:", cv2.countNonZero(road_mask))
print("Saved SegFormer segmentation results.")

