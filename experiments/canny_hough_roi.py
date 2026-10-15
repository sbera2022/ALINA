from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

image_path = ROOT / "data/Raw_Data/vidd_1/00001.jpg"
output_dir = ROOT / "outputs/canny_hough"
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)

image_small = cv2.resize(image, (640, 360))
height, width = image_small.shape[:2]

gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150)

search_mask = np.zeros_like(edges)

x_min = int(width * 0.50)
x_max = int(width * 0.85)
y_min = int(height * 0.40)
y_max = int(height * 0.85)

search_mask[y_min:y_max, x_min:x_max] = 255
masked_edges = cv2.bitwise_and(edges, search_mask)

lines = cv2.HoughLinesP(
    masked_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=25,
    minLineLength=25,
    maxLineGap=20
)

line_visualization = image_small.copy()
candidates = []

if lines is not None:
    print("Hough lines shape:", lines.shape)

    # Normalize the result into rows of four coordinates
    lines = np.asarray(lines).reshape(-1, 4)

    for line in lines:
        x1, y1, x2, y2 = map(int, line)

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(dx, dy)

        if abs(dx) < 5 or abs(dy) < 5:
            continue

        slope = dy / dx

        if 0.25 <= abs(slope) <= 5.0 and length >= 25:
            candidates.append((x1, y1, x2, y2, length))

            cv2.line(
                line_visualization,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

cv2.imwrite(str(output_dir / "canny_edges.jpg"), edges)
cv2.imwrite(str(output_dir / "masked_edges.jpg"), masked_edges)
cv2.imwrite(str(output_dir / "hough_lines.jpg"), line_visualization)

print("Number of candidate line segments:", len(candidates))

if not candidates:
    raise RuntimeError(
        "No suitable Hough lines found. Inspect hough_lines.jpg "
        "and adjust the thresholds."
    )

best_line = max(candidates, key=lambda item: item[4])

x1, y1, x2, y2, _ = best_line

print("Selected Hough line:", (x1, y1, x2, y2))

if y1 == y2:
    raise RuntimeError("Selected line is horizontal.")

a = (x2 - x1) / (y2 - y1)
b = x1 - a * y1

top_y = int(height * 0.55)
bottom_y = int(height * 0.78)

top_left_x = int(round(a * top_y + b))
bottom_left_x = int(round(a * bottom_y + b))

roi_width = int(width * 0.18)

top_right_x = top_left_x + roi_width
bottom_right_x = bottom_left_x + roi_width

roi_points = np.array([
    [bottom_left_x, bottom_y],
    [top_left_x, top_y],
    [top_right_x, top_y],
    [bottom_right_x, bottom_y]
], dtype=np.int32)

if np.any(roi_points[:, 0] < 0) or np.any(roi_points[:, 0] >= width):
    raise RuntimeError(
        f"Generated ROI extends outside image: {roi_points.tolist()}"
    )

roi_visualization = image_small.copy()

cv2.polylines(
    roi_visualization,
    [roi_points],
    True,
    (0, 255, 0),
    2
)

for index, (x, y) in enumerate(roi_points):
    cv2.circle(roi_visualization, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.putText(
        roi_visualization,
        str(index + 1),
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

print("ALINA ROI shape:", alina_roi.shape)
print("Saved outputs to:", output_dir)