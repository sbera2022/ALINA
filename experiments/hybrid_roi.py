from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

image_path = ROOT / "data/Raw_Data/vidd_1/00001.jpg"
segmentation_path = (
    ROOT / "outputs/segformer/segmentation_labels.npy"
)

output_dir = ROOT / "outputs/hybrid"
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

if not segmentation_path.exists():
    raise FileNotFoundError(segmentation_path)

image_small = cv2.resize(image, (640, 360))
h, w = image_small.shape[:2]

segmentation = np.load(segmentation_path)

if segmentation.shape != (h, w):
    raise ValueError(
        f"Unexpected segmentation shape: {segmentation.shape}"
    )

pavement_ids = [6, 13]

pavement_mask = np.isin(
    segmentation, pavement_ids
).astype(np.uint8) * 255

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (21, 21)
)

pavement_mask = cv2.dilate(
    pavement_mask, kernel, iterations=1
)

hsv = cv2.cvtColor(
    image_small, cv2.COLOR_BGR2HSV
)

lower_yellow = np.array([15, 60, 70], dtype=np.uint8)
upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

yellow_mask = cv2.inRange(
    hsv, lower_yellow, upper_yellow
)

search_mask = np.zeros((h, w), dtype=np.uint8)

search_mask[
    int(h * 0.45):int(h * 0.90),
    int(w * 0.40):int(w * 0.85)
] = 255

yellow_mask = cv2.bitwise_and(
    yellow_mask, search_mask
)

hybrid_mask = cv2.bitwise_and(
    yellow_mask, pavement_mask
)

small_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

hybrid_mask = cv2.morphologyEx(
    hybrid_mask,
    cv2.MORPH_OPEN,
    small_kernel
)

cv2.imwrite(
    str(output_dir / "pavement_mask.jpg"),
    pavement_mask
)

cv2.imwrite(
    str(output_dir / "yellow_mask.jpg"),
    yellow_mask
)

cv2.imwrite(
    str(output_dir / "hybrid_mask.jpg"),
    hybrid_mask
)

num_labels, labels, stats, _ = (
    cv2.connectedComponentsWithStats(
        hybrid_mask, connectivity=8
    )
)

candidates = []

for i in range(1, num_labels):
    x, y, width, height, area = stats[i]

    if area < 15 or height < 20:
        continue

    if height / max(width, 1) < 0.6:
        continue

    candidates.append((i, area))

print("Hybrid candidates:", len(candidates))

if not candidates:
    raise RuntimeError(
        "No suitable hybrid component found. "
        "Inspect hybrid_mask.jpg. "
        "Do not generate an ROI from an empty mask."
    )

component_id, component_area = max(
    candidates, key=lambda item: item[1]
)

component_mask = np.where(
    labels == component_id, 255, 0
).astype(np.uint8)

cv2.imwrite(
    str(output_dir / "selected_component.jpg"),
    component_mask
)

ys, xs = np.where(component_mask > 0)

if len(xs) < 2 or np.unique(ys).size < 2:
    raise RuntimeError(
        "Insufficient points for line fitting."
    )

a, b = np.polyfit(ys, xs, 1)

top_y = int(h * 0.55)
bottom_y = int(h * 0.78)

top_x = int(round(a * top_y + b))
bottom_x = int(round(a * bottom_y + b))

def semantic_margin(y, line_x, maximum_distance=100):
    y1 = max(0, y - 8)
    y2 = min(h, y + 9)

    ys, xs = np.where(pavement_mask[y1:y2] > 0)

    if len(xs) < 20:
        return int(w * 0.08), int(w * 0.08)

    left_pixels = xs[xs < line_x]
    right_pixels = xs[xs > line_x]

    left_distance = (
        line_x - np.percentile(left_pixels, 75)
        if len(left_pixels) else 40
    )

    right_distance = (
        np.percentile(right_pixels, 25) - line_x
        if len(right_pixels) else 40
    )

    left_margin = int(np.clip(
        left_distance + 15, 20, maximum_distance
    ))

    right_margin = int(np.clip(
        right_distance + 15, 20, maximum_distance
    ))

    return left_margin, right_margin


top_left_margin, top_right_margin = semantic_margin(
    top_y, top_x
)

bottom_left_margin, bottom_right_margin = semantic_margin(
    bottom_y, bottom_x
)

roi_points = np.array([
    [bottom_x - bottom_left_margin, bottom_y],
    [top_x - top_left_margin, top_y],
    [top_x + top_right_margin, top_y],
    [bottom_x + bottom_right_margin, bottom_y]
], dtype=np.int32)

if (
    np.any(roi_points[:, 0] < 0)
    or np.any(roi_points[:, 0] >= w)
):
    raise RuntimeError(
        f"ROI extends outside image: {roi_points.tolist()}"
    )

visualization = image_small.copy()

cv2.polylines(
    visualization,
    [roi_points],
    True,
    (0, 255, 0),
    2
)

for i, (x, y) in enumerate(roi_points):
    cv2.circle(
        visualization,
        (int(x), int(y)),
        5, (0, 0, 255), -1
    )

    cv2.putText(
        visualization,
        str(i + 1),
        (int(x) + 7, int(y) - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

cv2.imwrite(
    str(output_dir / "automatic_roi.jpg"),
    visualization
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

print("\nSelected component area:", component_area)
print("\nROI points (640x360):")
print(roi_points)
print("\nALINA ROI:")
print(alina_roi)
print("\nALINA ROI shape:", alina_roi.shape)
print("\nSaved results to:", output_dir)

hsv_roi = np.load(
    ROOT / "outputs/hsv/alina_roi.npy"
)

hybrid_roi = np.load(
    ROOT / "outputs/hybrid/alina_roi.npy"
)

print(
    "Hybrid ROI equals HSV ROI:",
    np.array_equal(hsv_roi, hybrid_roi)
)