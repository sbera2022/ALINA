from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

IMAGE_PATH = Path("data/Raw_Data/vidd_1/01060.jpg")
OUTPUT_DIR = Path("outputs/hybrid_segformer_hsv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(IMAGE_PATH)

original_height, original_width = img.shape[:2]
small = cv2.resize(img, (256, 256))

torch.set_num_threads(2)
device = "cpu"

print("Loading SegFormer...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
).to(device)
model.eval()

rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb)

inputs = processor(
    images=pil_image,
    return_tensors="pt",
    do_resize=False
)

with torch.no_grad():
    outputs = model(**inputs)

small_height, small_width = small.shape[:2]

logits = torch.nn.functional.interpolate(
    outputs.logits,
    size=(small_height, small_width),
    mode="bilinear",
    align_corners=False
)

prediction = logits.argmax(dim=1)[0].cpu().numpy()

pavement_ids = [
    int(class_id)
    for class_id, label in model.config.id2label.items()
    if label.lower().strip() in ("road", "runway")
]

pavement_mask = (
    np.isin(prediction, pavement_ids).astype(np.uint8) * 255
)

hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

yellow_mask = cv2.inRange(
    hsv,
    np.array([15, 80, 100], dtype=np.uint8),
    np.array([45, 255, 255], dtype=np.uint8)
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (21, 21)
)

expanded_pavement = cv2.dilate(
    pavement_mask, kernel, iterations=1
)

height, width = yellow_mask.shape[:2]

search_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

left_bound = int(width * 0.25)
right_bound = int(width * 0.75)

top_bound = int(height * 0.40)
bottom_bound = int(height * 0.85)

search_mask[
    top_bound:bottom_bound,
    left_bound:right_bound
] = 255

expanded_pavement = cv2.resize(
    expanded_pavement,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)

hybrid_mask = cv2.bitwise_and(
    yellow_mask,
    expanded_pavement
)

hybrid_mask = cv2.bitwise_and(
    hybrid_mask,
    search_mask
)

print("Hybrid mask:", hybrid_mask.shape)
print("Hybrid white pixels:", cv2.countNonZero(hybrid_mask))

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    hybrid_mask, connectivity=8
)

cleaned = np.zeros_like(hybrid_mask)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= 20:
        cleaned[labels == i] = 255

cv2.imwrite(str(OUTPUT_DIR / "pavement_mask.png"), pavement_mask)
cv2.imwrite(str(OUTPUT_DIR / "yellow_mask.png"), yellow_mask)
cv2.imwrite(str(OUTPUT_DIR / "hybrid_mask.png"), cleaned)

ys, xs = np.where(cleaned > 0)

if len(xs) < 30:
    raise RuntimeError(
        "Too few hybrid pixels to generate ROI. "
        "Inspect the saved masks."
    )

x_min, x_max = np.percentile(xs, [5, 95])
y_min, y_max = np.percentile(ys, [5, 95])

margin_x = 30
margin_y = 20

left = max(0, int(x_min - margin_x))
right = min(small_width - 1, int(x_max + margin_x))
top = max(0, int(y_min - margin_y))
bottom = min(small_height - 1, int(y_max + margin_y))

if right <= left or bottom <= top:
    raise RuntimeError("Invalid ROI bounds")

center_x = (left + right) / 2
top_width = (right - left) * 0.55

roi_small = np.array([
    [left, bottom],
    [center_x - top_width / 2, top],
    [center_x + top_width / 2, top],
    [right, bottom]
], dtype=np.float32)

roi_original = roi_small.copy()

roi_original[:, 0] *= original_width / small_width
roi_original[:, 1] *= original_height / small_height

roi_original = np.rint(roi_original).astype(np.int32)

bl, tl, tr, br = roi_original

alina_roi = np.array([[
    bl, tl, tl, tr, tr, br
]], dtype=np.int32)

np.save(str(OUTPUT_DIR / "alina_roi.npy"), alina_roi)

overlay = small.copy()
cv2.polylines(
    overlay,
    [roi_small.astype(np.int32)],
    True,
    (0, 0, 255),
    2
)

cv2.imwrite(str(OUTPUT_DIR / "roi_overlay.jpg"), overlay)

print("Hybrid mask pixels:", cv2.countNonZero(cleaned))
print("ALINA ROI:", alina_roi)
print("Saved outputs to:", OUTPUT_DIR.resolve())
