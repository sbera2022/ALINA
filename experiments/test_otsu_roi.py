from pathlib import Path

import cv2
import numpy as np

from alina.auto_threshold import otsu_color_mask
from alina.config import ROIConfig
from alina.roi import roi_points_to_quad

image_path = Path("data/Raw_Data/vidd_1/00001.jpg")
roi_path = Path("outputs/evaluation/manual_roi.npy")
output_dir = Path("outputs/threshold_otsu")

output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

roi_points = np.load(roi_path)
roi_quad = roi_points_to_quad(roi_points, dtype=np.float32)

roi_config = ROIConfig()

dst = np.array([
    roi_config.dst_bottom_left,
    roi_config.dst_top_left,
    roi_config.dst_top_right,
    roi_config.dst_bottom_right,
], dtype=np.float32)

M = cv2.getPerspectiveTransform(roi_quad, dst)
warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

mask = otsu_color_mask(warped)

x1, y1 = roi_config.dst_top_left
x2, y2 = roi_config.dst_bottom_right

roi_only = np.zeros_like(mask)
roi_only[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

cv2.imwrite(str(output_dir / "warped_00001.png"), warped)

cv2.imwrite(str(output_dir / "otsu_roi_mask_00001.png"), roi_only)

print("ROI pixels: ", roi_only[y1:y2, x1:x2].size)
print("Candidate pixels: ", np.count_nonzero(roi_only))
print("ROI coverage: ", round(100 * np.count_nonzero(roi_only) / roi_only[y1:y2, x1:x2].size, 2), "%")
