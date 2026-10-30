from pathlib import Path

import cv2
import numpy as np

from alina.config import PipelineConfig
from alina.pipeline import process_image

roi_points = np.load("outputs/segformer/alina_roi.npy")

config = PipelineConfig(
    input_dir=Path("data/Raw_Data/vidd_1"),
    output_images_dir=Path("outputs/threshold_segformer_alina/annotated"),
    output_coords_dir=Path("outputs/threshold_segformer_alina/coords"),
    threshold_method="manual",
    peak_pixel_threshold=50,
    min_white_pixels=200,
    circular_threshold=15,
    mask_ignore_left_columns=300,
)

image_path = Path("data/Raw_Data/vidd_1/01060.jpg")

img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(image_path)

annotated, coords, had_lines = process_image(
    img,
    roi_points,
    config
)

output_dir = Path("outputs/segformer_alina")
output_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(
    str(output_dir / "test_01060.jpg"),
    annotated
)

print("Detected lines:", had_lines)
print("Detected pixels:", len(coords))