from pathlib import Path
import cv2
import numpy as np

image_path = Path("outputs/kmeans/test_warped_01060.jpg")
output_dir = Path("outputs/kmeans")
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

pixels = lab.reshape((-1, 3)).astype(np.float32)

cv2.setRNGSeed(42)

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

_, labels, centers = cv2.kmeans(
    pixels,
    4,
    None,
    criteria,
    3,
    cv2.KMEANS_PP_CENTERS
)

labels = labels.reshape(image.shape[:2])

centers_bgr = cv2.cvtColor(
    np.uint8(centers.reshape(1, -1, 3)),
    cv2.COLOR_LAB2BGR
).reshape(-1, 3)

print("Cluster centers (BGR):", centers_bgr)

for i in range(4):
    mask = np.uint8(labels == i) * 255
    cv2.imwrite(
        str(output_dir / f"warped_cluster_{i}.png"),
        mask
    )
segmented = centers_bgr[labels]

cv2.imwrite(
    str(output_dir / "warped_kmeans_segmented.png"),
    segmented
)

print("Saved all K-Means cluster images.")

yellow_cluster = 1
yellow_mask = np.uint8(labels == yellow_cluster) * 255

cv2.imwrite(
    "outputs/kmeans/warped_yellow_mask.png",
    yellow_mask
)