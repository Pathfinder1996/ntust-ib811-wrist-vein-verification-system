import cv2
import numpy as np

def automatic_gamma_correction(src, gamma, isAutoMode=True):

    if src is None:
        return src

    c = 1.0
    d = 1.0 / 255.0

    if isAutoMode:
        mean_val = np.mean(src) / 255.0
        if mean_val > 0:
            c = np.log10(0.5) / np.log10(mean_val)
        else:
            c = 1.0

    gamma *= c

    # build LUT
    lut = np.zeros((256,), dtype=np.uint8)
    for i in range(256):
        v = (i * d) ** gamma * 255.0
        v = np.clip(v, 0.0, 255.0)
        lut[i] = int(v)

    return cv2.LUT(src, lut)
