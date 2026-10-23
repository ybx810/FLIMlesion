from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
META = ['sample_id', 'patient_id', 'label']

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, Path): return str(x)
        return x
    path.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')

def versions():
    result = {'python': platform.python_version()}
    for name in ['numpy', 'pandas', 'opencv-python-headless', 'pyradiomics', 'SimpleITK', 'scikit-learn', 'xgboost', 'matplotlib', 'PyYAML', 'PyWavelets', 'Pillow', 'scipy']:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = 'missing'
    return result

def code_hash():
    files = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'scripts/down_sample/downsample_images.py']
    return digest({str(p.relative_to(ROOT)): sha256(p) for p in files})
