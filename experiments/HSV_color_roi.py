from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

image_path = ROOT / "data/Raw_Data/vidd_1/00001.jpg"
output_dir = ROOT / "outputs/hsv"
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)

image_small = cv2.resize(image, (640, 360))
height, width = image_small.shape[:2]

hsv = cv2.cvtColor(image_small, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([15, 60, 70], dtype=np.uint8)
upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

search_mask = np.zeros((height, width), dtype=np.uint8)

x_min = int(width * 0.40)
x_max = int(width * 0.85)
y_min = int(height * 0.45)
y_max = int(height * 0.90)

search_mask[y_min:y_max, x_min:x_max] = 255

yellow_mask = cv2.bitwise_and(yellow_mask, search_mask)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

cleaned_mask = cv2.morphologyEx(
    yellow_mask, cv2.MORPH_OPEN, kernel
)

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        cleaned_mask, connectivity=8
    )
)

candidates = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]

    if area < 15 or h < 20:
        continue

    if h / max(w, 1) < 0.6:
        continue

    candidates.append((i, area, x, y, w, h))

print("Yellow component candidates:", len(candidates))

if not candidates:
    raise RuntimeError(
        "No suitable yellow component found. "
        "Inspect yellow_mask.jpg and adjust HSV thresholds."
    )

best = max(candidates, key=lambda item: item[1])
component_id = best[0]

component_mask = np.where(
    labels == component_id, 255, 0
).astype(np.uint8)

ys, xs = np.where(component_mask > 0)

if len(xs) < 2:
    raise RuntimeError("Not enough pixels to fit a line.")

a, b = np.polyfit(ys, xs, 1)

top_y = int(height * 0.55)
bottom_y = int(height * 0.78)

top_x = int(round(a * top_y + b))
bottom_x = int(round(a * bottom_y + b))

left_margin = int(width * 0.02)
right_margin = int(width * 0.16)

roi_points = np.array([
    [bottom_x - left_margin, bottom_y],
    [top_x - left_margin, top_y],
    [top_x + right_margin, top_y],
    [bottom_x + right_margin, bottom_y]
], dtype=np.int32)

if (
    np.any(roi_points[:, 0] < 0)
    or np.any(roi_points[:, 0] >= width)
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

for index, (x, y) in enumerate(roi_points):
    cv2.circle(
        visualization, (int(x), int(y)),
        5, (0, 0, 255), -1
    )
    cv2.putText(
        visualization, str(index + 1),
        (int(x) + 7, int(y) - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 255), 2
    )

cv2.imwrite(
    str(output_dir / "yellow_mask.jpg"), yellow_mask
)
cv2.imwrite(
    str(output_dir / "cleaned_mask.jpg"), cleaned_mask
)
cv2.imwrite(
    str(output_dir / "component_mask.jpg"), component_mask
)
cv2.imwrite(
    str(output_dir / "automatic_roi.jpg"), visualization
)

original_height, original_width = image.shape[:2]

scale_x = original_width / width
scale_y = original_height / height

roi_original = np.rint(
    roi_points.astype(np.float64) * [scale_x, scale_y]
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

np.save(output_dir / "alina_roi.npy", alina_roi)

print("Selected component area:", best[1])
print("ROI points (640x360):")
print(roi_points)
print("ALINA ROI shape:", alina_roi.shape)
print("Saved results to:", output_dir)
