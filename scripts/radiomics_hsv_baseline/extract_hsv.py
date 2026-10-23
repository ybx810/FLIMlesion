import cv2
import numpy as np
from .roi_utils import validate_roi

def color_masks(hsv, roi, cfg):
    h, s, v = cv2.split(hsv)
    result = {}
    for name, ranges in cfg['colors'].items():
        mask = np.zeros(roi.shape, bool)
        for lo, hi in ranges['h_ranges']:
            if not 0 <= lo <= hi <= 179: raise ValueError('Hue ranges must lie in 0..179; wrap requires two ranges')
            mask |= (h >= lo) & (h <= hi)
        result[name] = mask & roi & (s >= ranges['s'][0]) & (s <= ranges['s'][1]) & (v >= ranges['v'][0]) & (v <= ranges['v'][1])
    return result

def extract(rgb, roi, cfg):
    validate_roi(rgb, roi)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hvalid = roi & (hsv[..., 1] >= cfg['hue_min_saturation'])
    features = {}
    for i, (name, bins, upper) in enumerate([('H', cfg['hue_bins'], 180), ('S', cfg['saturation_bins'], 256), ('V', cfg['value_bins'], 256)]):
        a = hsv[..., i][hvalid if i == 0 else roi].astype(float)
        vals = [a.mean(), a.std(), *np.percentile(a, [50, 10, 25, 75, 90])] if len(a) else [np.nan] * 7
        features.update({f'HSV_{name}_{key}': float(value) for key, value in zip(['Mean', 'Std', 'Median', 'P10', 'P25', 'P75', 'P90'], vals)})
        hist = np.histogram(a, bins=bins, range=(0, upper))[0] / max(len(a), 1)
        features.update({f'HSV_{name}_Hist_{j+1:02d}': float(value) for j, value in enumerate(hist)})
    angles = hsv[..., 0][hvalid].astype(float) * (2 * np.pi / 180)
    z = np.exp(1j * angles).mean() if len(angles) else complex(np.nan, np.nan)
    features['HSV_H_CircularMean'] = float((np.angle(z) % (2*np.pi)) * 180/(2*np.pi)) if abs(z) > 1e-12 else np.nan
    features['HSV_H_CircularDispersion'] = float(1 - abs(z))
    features['HSV_H_ValidFraction'] = float(hvalid.sum() / roi.sum())
    masks = color_masks(hsv, roi, cfg)
    for name, mask in masks.items():
        n = int(mask.sum())
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        areas = areas[areas >= cfg['min_component_area']]
        values = dict(AreaRatio=n / int(roi.sum()), PixelCount=n,
                      MeanS=float(hsv[..., 1][mask].mean()) if n else np.nan,
                      MeanV=float(hsv[..., 2][mask].mean()) if n else np.nan,
                      StdV=float(hsv[..., 2][mask].std()) if n else np.nan,
                      ComponentCount=len(areas), MeanComponentArea=float(areas.mean()) if len(areas) else 0,
                      LargestComponentAreaRatio=float(areas.max()/roi.sum()) if len(areas) else 0)
        features.update({f'COLOR_{name}_{key}': val for key, val in values.items()})
    return features, masks
