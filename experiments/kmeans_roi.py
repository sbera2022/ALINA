from pathlib import Path
from alina.roi import roi_points_to_quad
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

image_path = ROOT / "data/Raw_Data/vidd_1/00001.jpg"
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"File not found: {image_path}")

image_small = cv2.resize(image, (640, 360))

lab = cv2.cvtColor(image_small, cv2.COLOR_BGR2LAB)
pixels = lab.reshape(-1, 3).astype(np.float32)

cv2.setRNGSeed(42)

K = 4
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)

_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

centers = np.uint8(centers)
segmented_lab = centers[labels.flatten()].reshape(lab.shape)
segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)

output_dir = ROOT / "output/kmeans"
output_dir.mkdir(parents=True, exist_ok=True)

label_map = labels.reshape(image_small.shape[:2])

height, width = label_map.shape
sample_region = label_map[int(height * 0.55):int(height * 0.75), int(width * 0.60):int(width * 0.95)]

counts = np.bincount(sample_region.flatten(), minlength=K)
pavement_cluster = int(np.argmax(counts))

print("Pavement cluster: ", pavement_cluster)

pavement_mask = np.where(
    label_map == pavement_cluster, 255, 0
).astype(np.uint8)

cv2.imwrite(str(output_dir / "pavement_mask.jpg"), pavement_mask)

cv2.imwrite(str(output_dir / "original.jpg"), image_small)
cv2.imwrite(str(output_dir / "segmented.jpg"), segmented)

kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

clean_mask = cv2.morphologyEx(pavement_mask, cv2.MORPH_CLOSE, kernal)
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernal)

cv2.imwrite(str(output_dir / "clean_pavement_mask.jpg"), clean_mask)

height, width = clean_mask.shape

top_y = int(height * 0.55)
bottom_y = int(height * 0.78)

def get_pavement_bound(mask, y):
    row = mask[y, :]

    start_x = int(width * 0.50)
    end_x = int(width * 0.78)

    xs = np.where(row[start_x:end_x] > 0)[0]

    if len(xs) == 0:
        return None

    return (
        int(xs.min() + start_x),
        int(xs.max() + start_x)
    )

top_bounds = get_pavement_bound(clean_mask, top_y)
bottom_bounds = get_pavement_bound(clean_mask, bottom_y)

if top_bounds is None or bottom_bounds is None:
    raise RuntimeError("Unable to prevent boundaries")

roi_points = np.array([
    [bottom_bounds[0], bottom_y],
    [top_bounds[0], top_y],
    [top_bounds[1], top_y],
    [bottom_bounds[1], bottom_y]
], dtype=np.int32)

original_height, original_width = image.shape[:2]
small_height, small_width = image_small.shape[:2]

scale_x = original_width / small_width
scale_y = original_height / small_height

roi_original = roi_points.astype(np.float64).copy()

roi_original[:, 0] *= scale_x
roi_original[:, 1] *= scale_y

roi_original = np.rint(roi_original).astype(np.int32)

bl, tl, tr, br = roi_original

alina_roi = np.array([[
    bl,
    tl,
    tl,
    tr,
    tr,
    br
]], dtype=np.int32)

roi_save_path = ROOT / "outputs/kmeans/alina_roi.npy"

roi_save_path.parent.mkdir(parents=True, exist_ok=True)

np.save(roi_save_path, alina_roi)

print("ROI saved successfully!")
print("Saved location:", roi_save_path)
print("ALINA-compatible ROI:")
print(alina_roi)
print("ROI shape:", alina_roi.shape)
print("Automatically generated ROI:")
print(roi_points)

roi_visualization = image_small.copy()

cv2.polylines(
    roi_visualization,
    [roi_points],
    isClosed=True,
    color=(0, 255, 0),
    thickness=3
)

for i, (x, y) in enumerate(roi_points):
    cv2.circle(roi_visualization, (int(x), int(y)), 6, (0, 0, 255), -1)

    cv2.putText(roi_visualization, str(i + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imwrite(str(output_dir / "automatic_roi.jpg"), roi_visualization)
cv2.imshow("Original Image", image_small)
cv2.imshow("K-Means Segmentation", segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Automatic ROI", roi_visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()

quad = roi_points_to_quad(alina_roi)
print(quad)

np.save(
    str(output_dir / "alina_roi.npy"),
    alina_roi
)

print("Saved automatic ROI coordinates.")