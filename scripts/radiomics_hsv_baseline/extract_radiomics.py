import cv2
import numpy as np
import SimpleITK as sitk
import logging
import radiomics
from radiomics import featureextractor, getFeatureClasses
from .roi_utils import validate_roi

CLASSES = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']

def feature_names():
    return sorted(f'RAD_{cls}_{name}' for cls in CLASSES
                  for name, deprecated in getFeatureClasses()[cls].getFeatureNames().items() if not deprecated)

def create_extractor(cfg):
    radiomics.setVerbosity(logging.ERROR)
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    ex = featureextractor.RadiomicsFeatureExtractor(
        binWidth=cfg['binWidth'], distances=cfg['distances'], force2D=True, force2Ddimension=0,
        normalize=False, resampledPixelSpacing=None, correctMask=False, minimumROIDimensions=2,
        additionalInfo=False)
    ex.disableAllImageTypes()
    ex.enableImageTypeByName('Original')
    ex.disableAllFeatures()
    for name in CLASSES: ex.enableFeatureClassByName(name)
    return ex

def extract(rgb, roi, extractor):
    validate_roi(rgb, roi)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return extract_channel(gray, roi, extractor)


def extract_channel(gray, roi, extractor, prefix='RAD_'):
    """Shared Original 2D extraction; one channel is materialized at a time."""
    validate_roi(gray, roi)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('Expected a uint8 2D display channel')
    # One-slice 3D container makes the excluded force2D axis unambiguous.
    # Zero border supplies background for a full-image ROI; no original pixel is excluded.
    image = sitk.GetImageFromArray(np.pad(gray, 1)[None, ...])
    mask = sitk.GetImageFromArray(np.pad(roi.astype(np.uint8), 1)[None, ...])
    mask.CopyInformation(image)
    for getter in ['GetSize', 'GetSpacing', 'GetDirection', 'GetOrigin']:
        if getattr(image, getter)() != getattr(mask, getter)(): raise ValueError('SimpleITK geometry mismatch')
    result = extractor.execute(image, mask, label=1)
    return {prefix + key.removeprefix('original_'): float(value)
            for key, value in result.items() if key.startswith('original_')}
