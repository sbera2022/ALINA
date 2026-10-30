from pathlib import Path
import numpy as np

from alina.config import PipelineConfig
from alina.pipeline import run_batch


def main():
    roi_path = Path("outputs/segformer/alina_roi.npy")

    if not roi_path.exists():
        raise FileNotFoundError(roi_path)

    roi_points = np.load(roi_path)

    print("Loaded SegFormer ROI:")
    print(roi_points)

    config = PipelineConfig(
        input_dir=Path("data/Raw_Data/vidd_1"),

        output_images_dir=Path(
            "outputs/segformer_alina/annotated"
        ),

        output_coords_dir=Path(
            "outputs/segformer_alina/coords"
        ),

        log_file=Path(
            "outputs/segformer_alina/timing.log"
        ),

        threshold_method="manual",

        peak_pixel_threshold=50,
        min_white_pixels=200,
        circular_threshold=15,

        mask_ignore_left_columns=300,
    )

    config.output_images_dir.mkdir(
        parents=True, exist_ok=True
    )

    config.output_coords_dir.mkdir(
        parents=True, exist_ok=True
    )

    results = run_batch(config, roi_points)

    labeled = sum(result.had_lines for result in results)

    print(
        f"SegFormer ROI labeled: {labeled}/{len(results)}"
    )

main()