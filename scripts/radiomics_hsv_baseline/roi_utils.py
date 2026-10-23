import cv2
import numpy as np
from PIL import Image

def validate_roi(rgb, mask):
    if mask.ndim != 2 or mask.shape != rgb.shape[:2]: raise ValueError('ROI/image dimension mismatch')
    if not mask.any(): raise ValueError('ROI is empty')

def make_roi(rgb, cfg, mask_path=''):
    if mask_path:
        with Image.open(mask_path) as im:
            a = np.asarray(im)
        if a.ndim != 2 or not set(np.unique(a)).issubset({0, 1, 255}):
            raise ValueError('Mask must be binary 0/1 or 0/255 in matching pixel coordinates')
        mask, mode = a > 0, 'provided_mask'
    elif cfg['background_filter']:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = hsv[..., 2] >= cfg['min_value']
        if cfg['exclude_white']:
            mask &= ~((hsv[..., 2] >= cfg['white_min_value']) & (hsv[..., 1] <= cfg['white_max_saturation']))
        mode = 'configured_background_filter'
    else:
        mask, mode = np.ones(rgb.shape[:2], dtype=bool), 'full_image'
    validate_roi(rgb, mask)
    if mask.sum() < cfg['min_pixels']: raise ValueError('Too few ROI pixels')
    return mask, mode
