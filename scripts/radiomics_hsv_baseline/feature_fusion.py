import pandas as pd
from .common import META

def fuse(radiomics, hsv):
    for d in [radiomics, hsv]:
        if d.sample_id.duplicated().any(): raise ValueError('Duplicate sample ID')
    if set(radiomics.sample_id) != set(hsv.sample_id): raise ValueError('Feature sample mismatch')
    if (set(radiomics) & set(hsv)) - set(META): raise ValueError('Overlapping feature names')
    d = radiomics.merge(hsv, on=META, validate='one_to_one')
    if len(d) != len(radiomics): raise ValueError('Feature metadata mismatch')
    for c in set(d) - set(META): d[c] = pd.to_numeric(d[c], errors='raise')
    return d
