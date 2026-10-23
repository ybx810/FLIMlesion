import json
from pathlib import Path
import pandas as pd
from scripts.radiomics_hsv_baseline.common import META, ROOT, save_csv, save_json, sha256
from scripts.radiomics_hsv_baseline.dataset import discover, read_metadata, validate_splits
from scripts.radiomics_hsv_baseline.evaluation import validate_oof

FEATURE_SETS = ('gray_radiomics', 'rgb_radiomics', 'gray_radiomics_hsv')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_csv(path):
    return pd.read_csv(path, dtype={c: str for c in META})

def radiomics_experiment(cfg, key, feature_set, classifier):
    root = Path(cfg['radiomics_root']) / 'experiments' / key
    folder = root / feature_set / classifier
    if feature_set == 'gray_radiomics_hsv' and not (folder / 'metrics.json').exists():
        folder = root / classifier
    return folder


def reference(cfg):
    from scripts.radiomics_hsv_baseline.run import load_config
    rad = load_config(cfg['radiomics_config'])
    if rad['cv']['n_splits'] != 5:
        raise ValueError('Current CNN baselines require the retained patient-level 5-fold split')
    if Path(rad['output_root']).resolve() != Path(cfg['radiomics_root']).resolve():
        raise ValueError('Radiomics config/output root mismatch')
    return rad


def prepare(cfg, method, resolution):
    rad = reference(cfg)
    root = Path(cfg['radiomics_root'])
    cohort = read_csv(root / 'manifests/cohort.csv').sort_values('sample_id').reset_index(drop=True)
    metadata = read_metadata(rad).set_index('sample_id').loc[cohort.sample_id].reset_index()
    pd.testing.assert_frame_equal(cohort[META], metadata[META])
    source = Path(cfg['split_source'])
    saved = read_csv(source)
    if 'true_label' in saved: raise ValueError('Use the retained five-fold split CSV, not historical OOF predictions')
    splits = validate_splits(cohort, saved[META + ['fold']], rad['cv']['n_splits'])
    classes = sorted(cohort.label.unique())
    mapping = {c: i for i, c in enumerate(classes)}
    audit = []
    selected = dict(rad, methods=[method], resolutions=[resolution], output_root=cfg['output_root'])
    images, _ = discover(selected, write=False)
    images = images[images.sample_id.isin(cohort.sample_id)].sort_values('sample_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(images[META], cohort[META])
    key = images.iloc[0]['config']
    # The retained split file is authoritative; all available results must agree.
    for fs in FEATURE_SETS:
        for clf in ('rf', 'xgboost'):
            folder = radiomics_experiment(cfg, key, fs, clf)
            if not (folder / 'metrics.json').exists():
                audit.append(dict(feature_set=fs, classifier=clf, status='not run'))
                continue
            oof = read_csv(folder / 'oof_predictions.csv')
            validate_oof(oof, cohort, classes)
            checked = validate_splits(cohort, oof.rename(columns={'true_label':'label'})[META+['fold']], rad['cv']['n_splits'])
            pd.testing.assert_frame_equal(splits.sort_values('sample_id').reset_index(drop=True), checked.sort_values('sample_id').reset_index(drop=True))
            if read_json(folder/'class_mapping.json') != mapping: raise ValueError('Class encoding mismatch')
            provenance = read_json(folder/'run_provenance.json')
            contract = provenance.get('contract', dict(parameters=provenance.get('config', {}).get(clf), seed=provenance.get('config', {}).get('seed')))
            if contract['parameters'] != rad[clf] or contract['seed'] != rad['seed']:
                raise ValueError('Saved classifier settings differ from reference results')
            audit.append(dict(feature_set=fs, classifier=clf, source=str(folder), status='identical cohort/classes/folds'))
    out = Path(cfg['output_root'])
    save_csv(splits, out/'cv_splits/reused_5fold.csv')
    save_json(out/'qc/split_provenance.json', dict(source=str(source), sha256=sha256(source), audit=audit, n_splits=5, regenerated=False))
    save_csv(images, out/'qc'/f'{method}_{resolution}_images.csv')
    save_json(out/'qc'/f'{method}_{resolution}_inspection.json', dict(n_samples=len(cohort), n_patients=cohort.patient_id.nunique(), classes=mapping, counts=cohort.label.value_counts().to_dict(), folds=splits.groupby('fold').size().to_dict(), actual_method=images.downsampling.iloc[0], roi=rad['roi'], preprocessing_difference='Radiomics uses full image ROI; CNN uses official resize and center crop, with no ROI mask or extra downsampling. Peripheral tissue can be lost.'))
    return images, splits, rad
