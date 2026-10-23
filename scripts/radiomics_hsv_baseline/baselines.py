"""Three controlled feature representations; immutable legacy results and resumable caches."""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .common import META, ROOT, digest, sha256, save_json, save_csv, versions
from .dataset import validate_splits
from .evaluation import validate_oof
from .extract_radiomics import create_extractor, feature_names, extract_channel
from .image_qc import inspect

FEATURE_SETS = ('gray_radiomics', 'rgb_radiomics', 'gray_radiomics_hsv')
EXTRACTION_FILES = ('extract_radiomics.py', 'extract_hsv.py', 'roi_utils.py', 'image_qc.py', 'feature_fusion.py')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def source_hashes():
    files = [Path(__file__).with_name(n) for n in EXTRACTION_FILES]
    files.append(ROOT / 'scripts/down_sample/downsample_images.py')
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}

def extraction_base(cfg):
    return dict(settings={k: cfg[k] for k in ('radiomics', 'roi', 'hsv')},
                sources=source_hashes(), versions=versions())

def validate_table(table, cohort, feature_set):
    if table.sample_id.duplicated().any() or set(table.sample_id) != set(cohort.sample_id):
        raise ValueError('Feature sample IDs must exactly match the common cohort')
    a = table.set_index('sample_id').loc[cohort.sample_id]
    b = cohort.set_index('sample_id')
    if not a[['patient_id','label']].equals(b[['patient_id','label']]):
        raise ValueError('Feature metadata differs from current cohort')
    cols = [c for c in table if c not in META]
    gray = set(feature_names())
    expected = {f'RGB_{c}_{n[4:]}' for c in 'RGB' for n in gray}
    if feature_set == 'gray_radiomics' and set(cols) != gray:
        raise ValueError('Gray schema must contain only the 93 Original Radiomics features')
    if feature_set == 'rgb_radiomics' and set(cols) != expected:
        raise ValueError('RGB schema must contain only three channel Radiomics blocks')
    if feature_set == 'gray_radiomics_hsv' and (set(c for c in cols if c.startswith('RAD_')) != gray or
            any(not c.startswith(('RAD_', 'HSV_', 'COLOR_')) for c in cols)):
        raise ValueError('Fusion schema is not Gray Radiomics + HSV')
    return table

def signatures(images):
    from .run import input_signature
    return [input_signature(row) for row in images.itertuples(index=False)]

def legacy_tables(images, cfg, inputs):
    """Validate legacy bytes + per-image hashes before explicit cohort projection."""
    folder = Path(cfg['output_root']) / 'features' / images.iloc[0]['config']
    path = folder / 'provenance.json'
    if not path.exists(): return None
    p = read_json(path)
    contract = read_json(Path(__file__).with_name('legacy_cache_contract.json'))
    if p['base']['code'] not in contract['legacy_codes']:
        # Newly extracted by the unchanged legacy orchestration is also supported.
        from .common import code_hash
        if p['base']['code'] != code_hash(): raise ValueError('Unreviewed legacy extractor version')
    elif source_hashes() != contract['approved_extraction_sources']:
        raise ValueError('Extraction code changed since legacy compatibility review')
    if p['status'] != 'complete' or p['base']['versions'] != versions():
        raise ValueError('Legacy extraction failed or environment changed; use a new output root')
    for k in ('radiomics', 'roi', 'hsv'):
        if p['base']['config'][k] != cfg[k]: raise ValueError('Legacy extraction settings changed: ' + k)
    old = {r['sample_id']: r for r in p['inputs']}
    if any(old.get(r['sample_id']) != r for r in inputs):
        raise ValueError('Image, ROI or metadata changed; legacy cache cannot be reused')
    tables = {}
    for name in ('radiomics','hsv','fused'):
        if sha256(folder / f'{name}.csv') != p['tables'][name]: raise ValueError('Legacy feature CSV modified')
        t = pd.read_csv(folder / f'{name}.csv', dtype={c:str for c in META})
        if t.sample_id.duplicated().any() or set(t.sample_id) != set(old): raise ValueError('Legacy table IDs invalid')
        tables[name] = t.set_index('sample_id').loc[images.sample_id].reset_index()
    r, f = tables['radiomics'], tables['fused']
    pd.testing.assert_frame_equal(r.set_index('sample_id'), f.set_index('sample_id')[r.columns[1:]])
    removed = sorted(set(old) - set(images.sample_id))
    save_json(folder / 'legacy_reuse_audit.json', dict(source_provenance_sha256=sha256(path),
        reused_tables=p['tables'], excluded_historical_ids=removed,
        reason='Explicit projection onto current metadata/common image cohort; no extraction failures dropped'))
    if removed: logging.warning('Legacy cache projected onto current cohort: %d historical IDs excluded (see legacy_reuse_audit.json)', len(removed))
    return tables

def prepare_features(df, cfg, feature_sets, extract=True):
    from .run import process_images
    out = Path(cfg['output_root'])
    for key, images in df.groupby('config', sort=False):
        folder = out / 'features' / key
        inputs = signatures(images)
        base = extraction_base(cfg)
        token = digest(dict(base=base, inputs=inputs))
        legacy = None
        for name in feature_sets:
            path, provenance = folder / f'{name}.csv', folder / f'{name}.json'
            if path.exists() and provenance.exists():
                p = read_json(provenance)
                if p['token'] != token or p['sha256'] != sha256(path):
                    raise ValueError(f'Stale or modified {path}; choose a new output root, existing results protected')
                validate_table(pd.read_csv(path, dtype={c:str for c in META}), images[META], name)
                logging.info('Reusing validated %s / %s', key, name)
                continue
            if not extract: raise FileNotFoundError(f'Extract first: {path}')
            if name != 'rgb_radiomics':
                if legacy is None:
                    if not (folder / 'provenance.json').exists(): process_images(images, cfg, extract=True)
                    legacy = legacy_tables(images, cfg, inputs)
                table = legacy['radiomics' if name == 'gray_radiomics' else 'fused'].copy()
            else:
                ex = create_extractor(cfg['radiomics'])
                rows, errors = [], []
                for i, (row, signature) in enumerate(zip(images.itertuples(index=False), inputs), 1):
                    cache = folder / 'rgb_cache' / (digest(row.sample_id) + '.json')
                    ct = digest(dict(base=base, input=signature))
                    cached = read_json(cache) if cache.exists() else None
                    try:
                        if cached and cached['token'] == ct:
                            values = cached['features']
                        else:
                            rgb, roi, _ = inspect(row, cfg)
                            values = {}
                            try:
                                for channel, index in zip('RGB', range(3)):
                                    try:
                                        pixels = rgb[..., index][roi]
                                        if pixels.min() == pixels.max(): logging.warning('%s channel %s is constant', row.sample_id, channel)
                                        del pixels
                                        v = extract_channel(rgb[..., index], roi, ex, f'RGB_{channel}_')
                                        bad = [k for k,x in v.items() if not np.isfinite(x)]
                                        if bad: raise ValueError('NaN/Inf features: ' + ','.join(bad))
                                        values.update(v)
                                    except Exception:
                                        logging.exception('%s channel %s extraction failed', row.sample_id, channel)
                                        raise
                            finally:
                                del rgb, roi
                            save_json(cache, dict(token=ct, features=values))
                        if not all(v is not None and np.isfinite(v) for v in values.values()): raise ValueError('Nonfinite cached RGB features')
                        rows.append({**{c:getattr(row,c) for c in META}, **values})
                    except Exception as e:
                        logging.exception('RGB failed: %s', row.sample_id)
                        errors.append(dict(sample_id=row.sample_id, error=str(e)))
                    filled = int(20 * i / len(images))
                    logging.info('RGB %s [%s%s] %d/%d (%.1f%%)', key, '#' * filled,
                                 '-' * (20 - filled), i, len(images), 100 * i / len(images))
                save_csv(pd.DataFrame(errors, columns=['sample_id','error']), folder / 'rgb_extraction_errors.csv')
                if errors: raise RuntimeError(f'{len(errors)} RGB failures; no incomplete cohort can train')
                table = pd.DataFrame(rows)
            validate_table(table, images[META], name)
            save_csv(table, path)
            save_json(provenance, dict(token=token, base=base, inputs=inputs, sha256=sha256(path),
                                      n_samples=len(table), n_features=len(table.columns)-3))

def fixed_splits(cohort, cfg):
    out = Path(cfg['output_root']) / 'cv_splits'
    source = Path(cfg['cv']['existing_splits']) if cfg['cv']['existing_splits'] else out / 'cv_splits.csv'
    if not source.exists(): raise FileNotFoundError(f'Existing patient splits required: {source}; no new splits generated')
    saved = pd.read_csv(source, dtype={c:str for c in META})
    if saved.sample_id.duplicated().any() or not set(cohort.sample_id).issubset(set(saved.sample_id)):
        raise ValueError('Existing split cannot cover current cohort')
    selected = saved[saved.sample_id.isin(cohort.sample_id)].copy()
    result = validate_splits(cohort, selected, cfg['cv']['n_splits'])
    path = out / 'comparison_cv_splits.csv'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path, dtype={c:str for c in META}), result.reset_index(drop=True))
    else: save_csv(result, path)
    save_json(out / 'comparison_split_provenance.json', dict(source=str(source), source_sha256=sha256(source),
        removed_historical_ids=sorted(set(saved.sample_id)-set(cohort.sample_id)),
        strategy='Reuse exact saved fold assignments; no random repartition', n_splits=cfg['cv']['n_splits']))
    return result, path

def check_oof(path, table, splits):
    oof = pd.read_csv(path / 'oof_predictions.csv', dtype={'sample_id':str,'patient_id':str,'true_label':str})
    validate_oof(oof, table, sorted(table.label.unique()))
    j = splits[['sample_id','fold']].merge(oof[['sample_id','fold']], on='sample_id', validate='one_to_one', suffixes=('_expected',''))
    if not (j.fold_expected == j.fold).all(): raise ValueError('Saved OOF fold assignments differ')
    return oof

def reuse_legacy_experiment(out, key, model, table, splits, cfg, destination):
    """Only copy completed fusion outputs when cohort, folds and training agree."""
    import shutil
    old = out / 'experiments' / key / model
    provenance = old / 'run_provenance.json'
    if not provenance.exists(): return False
    p = read_json(provenance)
    if p['versions'] != versions(): return False
    if any(p['config'][k] != cfg[k] for k in ('seed', 'preprocessing', model)): return False
    contract = read_json(Path(__file__).with_name('legacy_cache_contract.json'))
    for name in ('train_classifiers.py', 'evaluation.py'):
        path = Path(__file__).with_name(name)
        if sha256(path) != contract['source_hashes'][str(path.relative_to(ROOT))]: return False
    source = out / 'features' / key / 'fused.csv'
    if p['features_sha256'] != sha256(source): return False
    original = pd.read_csv(source, dtype={c:str for c in META})
    try:
        pd.testing.assert_frame_equal(original.set_index('sample_id').sort_index(), table.set_index('sample_id').sort_index())
        check_oof(old, table, splits)
    except (AssertionError, ValueError):
        return False
    required = ['metrics.json', 'classification_report.csv', 'oof_predictions.csv',
                'confusion_matrix.csv', 'confusion_matrix.png', 'confusion_matrix_normalized.png',
                'feature_importance.csv', 'feature_importance.png']
    if not all((old / n).exists() for n in required): return False
    destination.mkdir(parents=True, exist_ok=True)
    for path in old.iterdir():
        if path.is_file() and path.name != 'run_provenance.json': shutil.copy2(path, destination / path.name)
    save_json(destination / 'legacy_result_reuse.json', dict(source=str(old), provenance_sha256=sha256(provenance)))
    logging.info('Reused exact completed fusion experiment: %s', old)
    return True

def train_all(df, cohort, cfg, feature_sets, classifiers, n_jobs):
    from .train_classifiers import train
    from . import visualize
    out = Path(cfg['output_root'])
    splits, split_path = fixed_splits(cohort, cfg)
    results, reports = [], []
    for key, images in df.groupby('config', sort=False):
        for name in feature_sets:
            feature_path = out / 'features' / key / f'{name}.csv'
            table = validate_table(pd.read_csv(feature_path, dtype={c:str for c in META}), cohort, name)
            for model in classifiers:
                experiment = out / 'experiments' / key / name / model
                expected = dict(features_sha256=sha256(feature_path), splits_sha256=sha256(split_path),
                    model=model, parameters=cfg[model], seed=cfg['seed'], preprocessing=cfg['preprocessing'], versions=versions(),
                    training_sources={n:sha256(Path(__file__).with_name(n)) for n in ('train_classifiers.py','evaluation.py')})
                manifest = experiment / 'run_provenance.json'
                if manifest.exists():
                    p = read_json(manifest)
                    if p['contract'] != expected: raise ValueError(f'Completed experiment differs: {experiment}; not overwritten')
                    check_oof(experiment, table, splits)
                    for filename, value in p['artifacts'].items():
                        if sha256(experiment / filename) != value: raise ValueError('Completed artifact modified: ' + filename)
                    metrics = read_json(experiment / 'metrics.json')
                    logging.info('Reusing completed %s %s %s', key, name, model)
                else:
                    if (experiment / 'metrics.json').exists(): raise ValueError('Untracked completed experiment; refusing overwrite')
                    logging.info('Training %s %s %s', key, name, model)
                    reused = name == 'gray_radiomics_hsv' and reuse_legacy_experiment(out, key, model, table, splits, cfg, experiment)
                    metrics = read_json(experiment / 'metrics.json') if reused else train(table, splits, cfg, model, experiment, n_jobs)
                    check_oof(experiment, table, splits)
                    save_json(manifest, dict(contract=expected, artifacts={p.name:sha256(p) for p in experiment.iterdir() if p.is_file()}))
                result = dict(downsampling=images.iloc[0].downsampling, resolution=int(images.iloc[0].resolution),
                    feature_set=name, classifier=model, n_samples=len(table), n_features=len(table.columns)-3, **metrics)
                results.append(result)
                report = pd.read_csv(experiment / 'classification_report.csv')
                report = report[report.label.isin(sorted(cohort.label.unique()))].copy()
                for k in ('downsampling','resolution','feature_set','classifier'): report[k] = result[k]
                reports.append(report)
    comparison_dir = out / 'comparisons'
    # Merge only previous rows tied to the identical comparison split, never historical 188-sample results.
    csv = comparison_dir / 'baseline_comparison.csv'
    cohort_token = sha256(split_path)
    stamp = comparison_dir / 'comparison_contract.json'
    comparison = pd.DataFrame(results)
    per_class = pd.concat(reports, ignore_index=True)
    keys = ['downsampling','resolution','feature_set','classifier']
    if csv.exists() and stamp.exists() and read_json(stamp)['splits_sha256'] == cohort_token:
        comparison = pd.concat([pd.read_csv(csv), comparison]).drop_duplicates(keys, keep='last')
        old_report = comparison_dir / 'baseline_per_class.csv'
        if old_report.exists(): per_class = pd.concat([pd.read_csv(old_report),per_class]).drop_duplicates(keys+['label'],keep='last')
    save_csv(comparison.sort_values(keys), csv)
    save_csv(per_class, comparison_dir / 'baseline_per_class.csv')
    save_json(stamp, dict(splits_sha256=cohort_token, classes=sorted(cohort.label.unique())))
    visualize.baseline_comparisons(comparison, per_class, sorted(cohort.label.unique()), comparison_dir)
