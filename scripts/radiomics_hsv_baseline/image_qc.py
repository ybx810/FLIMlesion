import cv2
import numpy as np
from PIL import Image
from .roi_utils import make_roi

def read_rgb(path):
    # Reuse the existing reader's RGB8/bit-depth/ICC/EXIF validation.
    from scripts.down_sample.downsample_images import inspect_image
    from pathlib import Path
    info = {}
    im, _ = inspect_image(Path(path), info)
    try:
        if int(im.getexif().get(274, 1)) != 1: raise ValueError('Unexpected EXIF orientation; regenerate aligned inputs')
        if getattr(im, 'n_frames', 1) != 1: raise ValueError('Multi-frame image is unsupported')
        rgb = np.array(im)
    finally: im.close()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8: raise ValueError('Expected RGB uint8')
    return rgb

def inspect(row, cfg):
    rgb = read_rgb(row.path)
    roi, mode = make_roi(rgb, cfg['roi'], row.mask_path)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blank = ((hsv[..., 2] < cfg['roi']['min_value']) |
             ((hsv[..., 2] >= cfg['roi']['white_min_value']) & (hsv[..., 1] <= cfg['roi']['white_max_saturation'])))
    warnings = []
    if max(rgb.shape[:2]) != row.resolution: warnings.append('unexpected_long_edge')
    if blank.mean() > cfg['roi']['warning_background_ratio']: warnings.append('high_blank_fraction')
    return rgb, roi, dict(width=rgb.shape[1], height=rgb.shape[0], channels=3,
                         dtype=str(rgb.dtype), min_pixel=int(rgb.min()), max_pixel=int(rgb.max()),
                         blank_fraction=float(blank.mean()), roi_fraction=float(roi.mean()),
                         roi_pixels=int(roi.sum()), roi_mode=mode, warning=';'.join(warnings))
