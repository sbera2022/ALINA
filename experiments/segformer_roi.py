from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


ROOT = Path(__file__).resolve().parents[1]

image_path = ROOT / "data/Raw_Data/vidd_1/00001.jpg"
output_dir = ROOT / "outputs/segformer"
output_dir.mkdir(parents=True, exist_ok=True)

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

np.random.seed(42)
torch.manual_seed(42)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

image_small = cv2.resize(image, (256, 256))
image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

pil_image = Image.fromarray(image_rgb)

print("Loading SegFormer model...")

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_ID
)
model.eval()

print("Running semantic segmentation...")

inputs = processor(
    images=pil_image,
    return_tensors="pt"
)

with torch.inference_mode():
    outputs = model(**inputs)

segmentation = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=[(360, 640)]
)[0].cpu().numpy()

np.save(
    output_dir / "segmentation_labels.npy",
    segmentation
)

counts = Counter(segmentation.flatten().tolist())

print("\nPredicted classes:")

for class_id, count in counts.most_common(15):
    class_name = model.config.id2label[int(class_id)]

    percentage = 100 * count / segmentation.size

    print(
        f"{class_name}: {percentage:.2f}% "
        f"({count} pixels)"
    )

rng = np.random.default_rng(42)

num_classes = model.config.num_labels

colors = rng.integers(
    0, 256,
    size=(num_classes, 3),
    dtype=np.uint8
)

segmentation_color = colors[segmentation]

overlay = cv2.addWeighted(
    image_small,
    0.55,
    cv2.cvtColor(
        segmentation_color,
        cv2.COLOR_RGB2BGR
    ),
    0.45,
    0
)

cv2.imwrite(
    str(output_dir / "original.jpg"),
    image_small
)

cv2.imwrite(
    str(output_dir / "segmentation.jpg"),
    cv2.cvtColor(
        segmentation_color,
        cv2.COLOR_RGB2BGR
    )
)

cv2.imwrite(
    str(output_dir / "segmentation_overlay.jpg"),
    overlay
)

print("\nSaved outputs to:", output_dir)

road_ids = [
    int(class_id)
    for class_id, name in model.config.id2label.items()
    if name.lower() == "road"
]

if not road_ids:
    raise RuntimeError("Model has no 'road' class.")

road_mask = np.isin(
    segmentation, road_ids
).astype(np.uint8) * 255

h, w = road_mask.shape

search = np.zeros((h, w), dtype=np.uint8)
search[
    int(h * 0.50):int(h * 0.85),
    int(w * 0.45):int(w * 0.85)
] = 255

road_mask = cv2.bitwise_and(road_mask, search)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

road_mask = cv2.morphologyEx(
    road_mask, cv2.MORPH_OPEN, kernel
)

cv2.imwrite(
    str(output_dir / "road_mask.jpg"),
    road_mask
)

top_y = int(h * 0.55)
bottom_y = int(h * 0.78)

def road_center(y, band=12):
    y1 = max(0, y - band)
    y2 = min(h, y + band + 1)

    ys, xs = np.where(road_mask[y1:y2] > 0)

    if len(xs) < 30:
        raise RuntimeError(
            f"Insufficient road pixels near y={y}. "
            "Inspect road_mask.jpg."
        )

    return int(np.median(xs))

top_center = road_center(top_y)
bottom_center = road_center(bottom_y)

half_width = int(w * 0.10)

roi_points = np.array([
    [bottom_center - half_width, bottom_y],
    [top_center - half_width, top_y],
    [top_center + half_width, top_y],
    [bottom_center + half_width, bottom_y]
], dtype=np.int32)

if (
    np.any(roi_points[:, 0] < 0)
    or np.any(roi_points[:, 0] >= w)
):
    raise RuntimeError("Generated ROI exceeds image bounds.")

roi_visualization = image_small.copy()

cv2.polylines(
    roi_visualization,
    [roi_points],
    True,
    (0, 255, 0),
    2
)

for i, (x, y) in enumerate(roi_points):
    cv2.circle(
        roi_visualization,
        (int(x), int(y)),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        roi_visualization,
        str(i + 1),
        (int(x) + 7, int(y) - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

cv2.imwrite(
    str(output_dir / "automatic_roi.jpg"),
    roi_visualization
)

original_h, original_w = image.shape[:2]

roi_original = np.rint(
    roi_points.astype(np.float64) *
    [original_w / w, original_h / h]
).astype(np.int32)

bl, tl, tr, br = roi_original

alina_roi = np.array([[
    bl,
    tl,
    tl,
    tr,
    tr,
    br
]], dtype=np.int32)

np.save(
    output_dir / "alina_roi.npy",
    alina_roi
)

print("\nSegFormer ROI points:")
print(roi_points)

print("\nALINA ROI:")
print(alina_roi)

print("\nALINA ROI shape:", alina_roi.shape)