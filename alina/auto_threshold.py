import cv2
import numpy as np

def otsu_color_mask(warped_image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    valid = v > 0
    values = v[valid]

    if values.size == 0:
        return np.zeros(v.shape, dtype=np.uint8)

    otsu_threshold,_ = cv2.threshold(values.reshape(-1, 1),0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bright = (v >= otsu_threshold) & valid

    yellow = (
        (h >= 15) &
        (h <= 40) &
        (s >= 60)
    )
    white = s <= 60

    combined = bright & (yellow | white)

    mask = (combined.astype(np.uint8) * 255)

    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

    return mask

def adaptive_color_mask(warped_image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(warped_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    valid = v > 0

    if not np.any(valid):
        return np.zeros(v.shape, dtype=np.uint8)

    adaptive = cv2.adaptiveThreshold(
        v,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        2
    )

    yellow = (
            (h >= 15) &
            (h <= 40) &
            (s >= 60)
    )

    white = (
            (s <= 60) &
            (v >= 150)
    )

    combined = (
            (adaptive > 0) &
            valid &
            (yellow | white)
    )

    mask = combined.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def kmeans_color_mask(warped_image: np.ndarray) -> np.ndarray:
    height, width = warped_image.shape[:2]

    small = cv2.resize(
        warped_image,
        (640, 360),
        interpolation=cv2.INTER_AREA
    )

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    valid = np.any(small > 0, axis=2)

    if not np.any(valid):
        return np.zeros((height, width), dtype=np.uint8)

    pixels = lab[valid].astype(np.float32)
    cv2.setRNGSeed(42)

    K = 6

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.2
    )

    _, labels, centers = cv2.kmeans(pixels, K,None, criteria,3, cv2.KMEANS_PP_CENTERS)

    label_image = np.full(
        valid.shape,
        -1,
        dtype=np.int32
    )
    label_image[valid] = labels.flatten()
    centers_lab = np.clip(
        centers,
        0,
        255
    ).astype(np.uint8).reshape(1, K, 3)

    centers_bgr = cv2.cvtColor(
        centers_lab,
        cv2.COLOR_LAB2BGR
    )

    centers_hsv = cv2.cvtColor(
        centers_bgr,
        cv2.COLOR_BGR2HSV
    ).reshape(K, 3)


    selected_clusters = []

    for i, (h, s, v) in enumerate(centers_hsv):
        yellow = (
                15 <= int(h) <= 40
                and int(s) >= 60
                and int(v) >= 100
        )

        white = (
                int(s) <= 60
                and int(v) >= 180
        )

        if yellow or white:
            selected_clusters.append(i)


    h, s, v = cv2.split(hsv)

    yellow_pixels = (
            (h >= 15) &
            (h <= 45) &
            (s >= 35) &
            (v >= 80)
    )

    white_pixels = (
            (s <= 45) &
            (v >= 200)
    )

    small_mask = np.zeros(valid.shape, dtype=np.uint8)
    small_mask[valid & yellow_pixels] = 255

    for cluster_id in selected_clusters:
        cluster_pixels = label_image == cluster_id
        small_mask[cluster_pixels & white_pixels & valid] = 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    small_mask = cv2.morphologyEx(
        small_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.resize(
        small_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    hsv_only = ((yellow_pixels & valid).astype(np.uint8) * 255)

    cv2.imwrite(
        "outputs/threshold_kmeans/hsv_only_debug.png",
        hsv_only
    )

    return mask

