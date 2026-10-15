from pathlib import Path
import cv2
import numpy as np

from alina.roi import roi_points_to_quad

ROOT = Path(__file__).resolve().parents[1]

reference_path = (
    ROOT / "outputs/evaluation/manual_roi.npy"
)

methods = {
    "K-Means": "outputs/kmeans/alina_roi.npy",
    "Canny + Hough": "outputs/canny_hough/alina_roi.npy",
    "HSV": "outputs/hsv/alina_roi.npy",
    "SegFormer": "outputs/segformer/alina_roi.npy",
    "Hybrid": "outputs/hybrid/alina_roi.npy"
}

reference = np.load(reference_path)

image_path = ROOT / "data/Raw_Data/vidd_1/00001.jpg"
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

height, width = image.shape[:2]

def polygon_mask(roi_points):
    quad = roi_points_to_quad(roi_points)

    mask = np.zeros(
        (height, width), dtype=np.uint8
    )

    cv2.fillPoly(mask, [quad[0]], 1)

    return mask

reference_mask = polygon_mask(reference)

print("\nROI IoU Evaluation")
print("-" * 40)

for method, path in methods.items():
    automatic_roi = np.load(ROOT / path)

    automatic_mask = polygon_mask(automatic_roi)

    intersection = np.logical_and(
        reference_mask, automatic_mask
    ).sum()

    union = np.logical_or(
        reference_mask, automatic_mask
    ).sum()

    iou = intersection / union if union else 0.0

    print(f"{method}: IoU = {iou:.4f}")