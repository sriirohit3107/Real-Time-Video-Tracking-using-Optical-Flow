import numpy as np
import cv2
from LucasKanade import LucasKanade

def LucasKanadePyramid(It, It1, rect, levels=3, scale=0.5,
                       gauss_ksize=(5, 5), gauss_sigma=1.0):
    """
    Translation-only pyramidal Lucas–Kanade with explicit Gaussian anti-aliasing.

    Inputs:
      It, It1       – H×W grayscale template and current images (2D arrays).
      rect          – [x1, y1, x2, y2] window in It.
      levels        – number of pyramid levels (default 3).
      scale         – down-sampling factor per level (default 0.5).
      gauss_ksize   – kernel size for Gaussian blur (default (5,5)).
      gauss_sigma   – sigma for Gaussian blur (default 1.0).

    Returns:
      p – (dx, dy) total translation in original-pixel units.
    """
    # 1) Build Gaussian pyramid with explicit blur + resize
    pyr_It, pyr_It1 = [], []
    # start at level 0
    pyr_It.append(It.astype(np.float64))
    pyr_It1.append(It1.astype(np.float64))
    # build coarser levels
    for lvl in range(1, levels):
        prev_I  = pyr_It[-1]
        prev_I1 = pyr_It1[-1]
        # Gaussian anti‐aliasing
        blur_I  = cv2.GaussianBlur(prev_I,  gauss_ksize, gauss_sigma)
        blur_I1 = cv2.GaussianBlur(prev_I1, gauss_ksize, gauss_sigma)
        # down‐sample by scale factor
        down_I  = cv2.resize(blur_I,  (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        down_I1 = cv2.resize(blur_I1, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pyr_It .append(down_I)
        pyr_It1.append(down_I1)

    # 2) accumulate translation at full resolution
    p = np.zeros(2, dtype=np.float64)

    # 3) process from coarsest → finest
    for lvl in range(levels - 1, -1, -1):
        I   = pyr_It [lvl]
        I1  = pyr_It1[lvl]
        s   = scale ** lvl

        # a) scale window to this level
        rect_lvl = np.array(rect, dtype=np.float64) * s
        # b) shift by current accumulated p (down‐scaled)
        p_lvl    = p * s
        rect_lvl += [p_lvl[0], p_lvl[1],
                     p_lvl[0], p_lvl[1]]

        # c) run translation-only LK at this level
        delta_p = LucasKanade(I, I1, rect_lvl)

        # d) lift update back to full resolution
        p += delta_p / s

    return tuple(p)
