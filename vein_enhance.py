import cv2
import numpy as np

def automatic_gamma_correction(image, gamma=1.0, is_auto_mode=True):
    """
    Automatically adjusts gamma value based on image brightness.
    """
    correction_factor = 1.0
    intensity_scale = 1.0 / 255.0

    if is_auto_mode:
        mean_val = np.mean(image) / 255.0
        if mean_val > 0:
            correction_factor = np.log10(0.5) / np.log10(mean_val)
        else:
            correction_factor = 1.0

    gamma *= correction_factor

    # Create LUT for gamma correction
    lut = np.array([
        np.clip((i * intensity_scale) ** gamma * 255.0, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)

    corrected_image = cv2.LUT(image, lut)
    return corrected_image

# def enhance_vein(image, gamma_value=1.0, laplacian_delta=0.0):
#     """
#     Enhances vein visibility using gamma correction, CLAHE, Gaussian and Laplacian filters.
#     """
#     if image is None or image.size == 0:
#         return image

#     # Step 1: Automatic Gamma Correction
#     agc_image = automatic_gamma_correction(image, gamma=gamma_value, is_auto_mode=True)

#     # Step 2: CLAHE
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
#     clahe_image = clahe.apply(agc_image)

#     # Step 3: Gaussian blur (low-pass filtering)
#     normalized_image = clahe_image.astype(np.float64) / 255.0
#     blurred_image = cv2.GaussianBlur(normalized_image, (0, 0), sigmaX=4)

#     # Step 4: Laplacian (high-pass filtering)
#     laplacian_image = cv2.Laplacian(
#         blurred_image,
#         ddepth=cv2.CV_64F,
#         ksize=1,
#         scale=1,
#         delta=laplacian_delta,
#         borderType=cv2.BORDER_DEFAULT
#     )
#     laplacian_image = np.maximum(laplacian_image, 0.0)

#     # Normalize output to 0~255
#     lap_min, lap_max, _, _ = cv2.minMaxLoc(laplacian_image)
#     scale = 255.0 / max(-lap_min, lap_max) if max(-lap_min, lap_max) != 0 else 1.0
#     final_image = laplacian_image * scale

#     return final_image
