from pathlib import Path

import cv2
import numpy as np
from alina.auto_threshold import otsu_color_mask

image_path = Path("data/Raw_Data/vidd_1/00001.jpg")
output_dir = Path("outputs/threshold_otsu")
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

mask = otsu_color_mask(image)

print("Mask shape:", mask.shape)
print("Mask values:", np.unique(mask))
print("Candidate pixels:", np.count_nonzero(mask))
