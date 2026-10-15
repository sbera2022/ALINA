from pathlib import Path
import numpy as np

from alina.config import PipelineConfig
from alina.pipeline import run_batch
from alina.roi import roi_points_to_quad

ROOT = Path(__file__).resolve().parents[1]

roi_path = ROOT / "outputs/hybrid/alina_roi.npy"

if not roi_path.exists():
    raise FileNotFoundError(
       "Hybrid ROI not found. Run hybrid_roi.py first."
    )

roi_points = np.load(roi_path)

assert roi_points.shape == (1, 6, 2), (
    f"Unexpected ROI shape: {roi_points.shape}"
)

print("Hybrid ROI:")
print(roi_points)

quad = roi_points_to_quad(roi_points)

print("\nPerspective transformation corners:")
print(quad)

config = PipelineConfig(
    input_dir=ROOT / "data/Raw_Data/vidd_1",
    output_images_dir=ROOT / "outputs/hybrid_alina/annotated",
    output_coords_dir=ROOT / "outputs/hybrid_alina/coords",
    log_file=ROOT / "outputs/hybrid_alina/timing.log",
    peak_pixel_threshold=50,
    min_white_pixels=200,
    circular_threshold=15,
    yellow_lower=(0, 70, 170),
    yellow_upper=(255, 255, 255),
    mask_ignore_left_columns=300,
)

print("\nStarting ALINA with automatic Hybrid ROI...")

run_batch(config, roi_points)

print("\nHybrid ALINA experiment completed!")