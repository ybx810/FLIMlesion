"""Exact metadata matching using the existing downsampling provenance log."""
from pathlib import Path, PureWindowsPath
import logging
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .common import META, save_csv, save_json

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

def read_metadata(cfg):
    path = Path(cfg['metadata'])
    d = pd.read_excel(path, dtype=str) if path.suffix.lower() == '.xlsx' else pd.read_csv(path, dtype=str)
    d = d.rename(columns={v: k for k, v in cfg['metadata_columns'].items()})
    if not set(META).issubset(d): raise ValueError('Metadata requires sample_id, patient_id, label')
    d = d[META].copy()
    if d.isna().any().any(): raise ValueError('Missing metadata; patient IDs cannot be inferred')
    for col in META: d[col] = d[col].str.strip()
    if (d == '').any().any() or d.sample_id.duplicated().any(): raise ValueError('Empty or duplicate metadata sample IDs')
    if d.label.nunique() != 8: raise ValueError('Expected exactly eight metadata classes')
    return d.sort_values('sample_id').reset_index(drop=True)

def discover(cfg, write=True):
    root, out = Path(cfg['data_root']), Path(cfg['output_root']) / 'manifests'
    meta = read_metadata(cfg)
    log = pd.read_csv(root / 'processing_log.csv', dtype=str, keep_default_na=False)
    # Ignore stale success/failure state: actual file existence is checked below.
    provenance = {}
    for row in log.itertuples():
        key = PureWindowsPath(row.output_path).name
        original = PureWindowsPath(row.relative_path)
        value = (original.stem, original.parts[0], str(original))
        if key in provenance and provenance[key] != value: raise ValueError('Ambiguous provenance: ' + key)
        provenance[key] = value
    records, errors, configs = [], [], []
    indexed = meta.set_index('sample_id')
    for method in cfg['methods']:
        actual = method
        if method in ('gaussain_pyramid', 'gaussian_pyramid'):
            candidates = [m for m in ('gaussain_pyramid', 'gaussian_pyramid') if (root / m).is_dir()]
            if not candidates: raise FileNotFoundError('No Gaussian pyramid input directory')
            actual = candidates[0]
            if actual != method: logging.warning('Requested %s; using existing %s', method, actual)
        for resolution in cfg['resolutions']:
            directory = root / actual / str(resolution)
            if not directory.is_dir(): raise FileNotFoundError(directory)
            key = f'{actual}_{resolution}'
            files = sorted(p for p in directory.rglob('*') if p.suffix.lower() in EXTENSIONS)
            if not files: raise ValueError(f'Empty input: {directory}')
            configs.append(dict(config=key, downsampling=actual, resolution=resolution, n_files=len(files)))
            for p in files:
                if p.name not in provenance:
                    errors.append(dict(path=str(p), reason='No downsampling provenance')); continue
                sid, source_label, source_path = provenance[p.name]
                if sid not in indexed.index:
                    errors.append(dict(path=str(p), reason='No exact metadata sample ID', sample_id=sid)); continue
                md = indexed.loc[sid]
                folder = p.relative_to(directory).parts[0]
                if source_label != md.label or folder != md.label:
                    errors.append(dict(path=str(p), reason='Metadata/folder label mismatch', sample_id=sid)); continue
                mask = ''
                if cfg['roi']['mask_root']:
                    mask = str(Path(cfg['roi']['mask_root']) / actual / str(resolution) / p.relative_to(directory))
                    if not Path(mask).is_file(): raise FileNotFoundError(f'Configured mask missing: {mask}')
                records.append(dict(sample_id=sid, patient_id=md.patient_id, label=md.label,
                                    config=key, downsampling=actual, resolution=resolution,
                                    path=str(p), mask_path=mask, source_relative_path=source_path))
    df = pd.DataFrame(records)
    if write:
        save_csv(pd.DataFrame(errors, columns=['path', 'reason', 'sample_id']), out / 'matching_errors.csv')
        save_csv(pd.DataFrame(configs), out / 'configurations.csv')
    if errors: raise ValueError(f'{len(errors)} unmatched/mismatched inputs; inspect matching_errors.csv')
    if df.duplicated(['config', 'sample_id']).any(): raise ValueError('Multiple images for one sample/config')
    present = df.groupby('sample_id').config.nunique()
    common = set(present[present == len(configs)].index)
    missing = [dict(sample_id=sid, config=c['config']) for sid in meta.sample_id for c in configs
               if not ((df.sample_id == sid) & (df.config == c['config'])).any()]
    cohort = meta[meta.sample_id.isin(common)].reset_index(drop=True)
    if cohort.empty: raise ValueError('No common samples across all configurations')
    if write:
        save_csv(df, out / 'all_images.csv')
        save_csv(pd.DataFrame(missing, columns=['sample_id', 'config']), out / 'missing_versions.csv')
        save_csv(meta[~meta.sample_id.isin(common)], out / 'excluded_samples.csv')
        save_csv(cohort, out / 'cohort.csv')
        save_csv(meta.groupby('label').agg(n_samples=('sample_id', 'size'), n_patients=('patient_id', 'nunique')).reset_index(), out / 'class_counts.csv')
        multi = meta.groupby('patient_id').label.nunique()
        save_csv(meta[meta.patient_id.isin(multi[multi > 1].index)], out / 'patients_multiple_labels.csv')
        save_csv(meta[meta.sample_id.str.contains('副本|copy', case=False, regex=True)], out / 'possible_copies.csv')
    return df[df.sample_id.isin(common)].reset_index(drop=True), cohort

def validate_splits(cohort, splits, n_splits=5):
    if splits.sample_id.duplicated().any() or set(splits.sample_id) != set(cohort.sample_id):
        raise ValueError('Split sample IDs must exactly equal cohort')
    d = cohort.merge(splits, on='sample_id', validate='one_to_one', suffixes=('', '_saved'))
    for col in ['patient_id', 'label']:
        if col + '_saved' in d and not (d[col] == d[col + '_saved']).all(): raise ValueError('Stale split metadata')
    if d.fold.isna().any() or set(d.fold) != set(range(n_splits)): raise ValueError('Invalid fold IDs')
    if d.groupby('patient_id').fold.nunique().max() != 1: raise ValueError('Patient leakage')
    for fold in range(n_splits):
        if set(d[d.fold != fold].label) != set(cohort.label): raise ValueError('Training fold missing a class')
    return d[META + ['fold']]

def get_splits(cohort, cfg):
    path = Path(cfg['output_root']) / 'cv_splits' / 'cv_splits.csv'
    source = Path(cfg['cv']['existing_splits']) if cfg['cv']['existing_splits'] else path
    if source.exists():
        splits = pd.read_csv(source, dtype={k: str for k in META})
    else:
        if cfg['cv']['existing_splits']: raise FileNotFoundError(source)
        splits = cohort.copy()
        splits['fold'] = -1
        shuffle = cfg['cv']['shuffle']
        cv = StratifiedGroupKFold(n_splits=cfg['cv']['n_splits'], shuffle=shuffle,
                                 random_state=cfg['seed'] if shuffle else None)
        for fold, (_, test) in enumerate(cv.split(cohort, cohort.label, cohort.patient_id)):
            splits.loc[test, 'fold'] = fold
    result = validate_splits(cohort, splits, cfg['cv']['n_splits'])
    save_csv(result, path)
    save_csv(result.groupby(['fold', 'label']).size().rename('n_samples').reset_index(), path.parent / 'fold_counts.csv')
    save_json(path.parent / 'policy.json', dict(strategy='StratifiedGroupKFold', **cfg['cv'], seed=cfg['seed']))
    return result
