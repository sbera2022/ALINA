from pathlib import Path
import cv2
import numpy as np

from alina.roi import select_roi, roi_points_to_quad

ROOT = Path(__file__).resolve().parents[1]

image_path = ROOT / "data/Raw_Data/vidd_1/00001.jpg"
output_dir = ROOT / "outputs/evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

roi_points = select_roi(str(image_path))

np.save(
    output_dir / "manual_roi.npy",
    roi_points
)

quad = roi_points_to_quad(roi_points)

image = cv2.imread(str(image_path))

cv2.polylines(
    image,
    [quad[0]],
    True,
    (0, 255, 0),
    8
)

cv2.imwrite(
    str(output_dir / "manual_roi.jpg"),
    image
)

print("\nFour corners used for evaluation:")
quad = roi_points_to_quad(roi_points)

print(quad)
print("\nALINA ROI shape:", roi_points.shape)

