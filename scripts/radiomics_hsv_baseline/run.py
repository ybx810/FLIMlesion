"""Run from project root: python -m scripts.radiomics_hsv_baseline.run --stage all."""
import argparse
import json
import logging
from pathlib import Path
import sys
import traceback
import cv2
import numpy as np
import pandas as pd
import yaml
from .common import ROOT, META, save_csv, save_json, sha256, digest, versions, code_hash
from .dataset import discover, get_splits
from .image_qc import inspect
from .extract_hsv import extract as extract_hsv
from .feature_fusion import fuse
from . import visualize

def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for key in ['data_root', 'metadata', 'output_root']:
        cfg[key] = str((ROOT / cfg[key]).resolve())
    for section, key in [('roi', 'mask_root'), ('cv', 'existing_splits')]:
        if cfg[section][key]: cfg[section][key] = str((ROOT / cfg[section][key]).resolve())
    return cfg

def input_signature(row):
    return dict(sample_id=row.sample_id, patient_id=row.patient_id, label=row.label,
                image=sha256(row.path), mask=sha256(row.mask_path) if row.mask_path else None)

def process_images(df, cfg, extract=False, force=False):
    out = Path(cfg['output_root'])
    ex = None
    if extract:
        from .extract_radiomics import create_extractor, feature_names, extract as extract_rad
        ex = create_extractor(cfg['radiomics'])
        names = feature_names()
    failures = []
    for key, images in df.groupby('config', sort=False):
        qc, rad_rows, hsv_rows, signatures = [], [], [], []
        # Preview selection is fixed and independent of model performance.
        ids = images.sample(n=min(cfg['preview_count'], len(images)), random_state=cfg['seed']).sample_id.tolist()
        base_signature = dict(config=cfg, code=code_hash(), versions=versions())
        for i, row in enumerate(images.itertuples(index=False)):
            md = {c: getattr(row, c) for c in META}
            signature = input_signature(row)
            signatures.append(signature)
            cache = out / 'features' / key / 'cache' / (digest(row.sample_id) + '.json')
            token = digest(dict(base=base_signature, image=signature))
            cached = json.loads(cache.read_text(encoding='utf-8')) if cache.exists() and not force and extract else None
            if cached and cached['token'] == token and cached.get('success'):
                rad, hsv, info = cached['radiomics'], cached['hsv'], cached['qc']
            else:
                info = dict(**md, path=row.path, status='ok', error='')
                try:
                    rgb, roi, details = inspect(row, cfg)
                    info.update(details)
                    hsv, masks = extract_hsv(rgb, roi, cfg['hsv'])
                    info['absent_colors'] = ';'.join(name for name, mask in masks.items() if not mask.any())
                    if row.sample_id in ids:
                        preview_path = out / 'qc' / key / 'previews' / digest(row.sample_id)[:16]
                        info['preview_file'] = str(preview_path.with_suffix('.png'))
                        visualize.preview(rgb, roi, masks, preview_path)
                    rad = {}
                    if extract:
                        try:
                            rad = extract_rad(rgb, roi, ex)
                            if set(rad) != set(names): raise ValueError('Radiomics schema changed')
                        except Exception as e:
                            rad = dict.fromkeys(names, np.nan)
                            info.update(status='radiomics_error', error=str(e))
                    del rgb, roi, masks
                except Exception as e:
                    info.update(status='image_error', error=str(e))
                    qc.append(info)
                    failures.append(dict(config=key, **info))
                    logging.exception('Image/ROI failed: %s', row.path)
                    continue
                if extract:
                    save_json(cache, dict(token=token, success=info['status'] == 'ok', radiomics=rad, hsv=hsv, qc=info))
            qc.append(info)
            info['nonfinite_features'] = ';'.join(k for k, v in {**rad, **hsv}.items() if v is None or not np.isfinite(v))
            if info['status'] != 'ok': failures.append(dict(config=key, **info))
            hsv_rows.append({**md, **hsv})
            if extract: rad_rows.append({**md, **rad})
            if (i+1) % 10 == 0 or i == 0: logging.info('%s %s %d/%d', 'extract' if extract else 'qc', key, i+1, len(images))
        save_csv(pd.DataFrame(qc), out / 'qc' / key / 'image_qc.csv')
        save_csv(pd.DataFrame(hsv_rows), out / 'qc' / key / 'color_distribution.csv')
        save_csv(pd.DataFrame(signatures), out / 'qc' / key / 'input_hashes.csv')
        hashes = pd.DataFrame(signatures)
        save_csv(hashes[hashes.image.duplicated(keep=False)], out / 'qc' / key / 'identical_images.csv')
        if extract and len(rad_rows) == len(images):
            r, h = pd.DataFrame(rad_rows), pd.DataFrame(hsv_rows)
            f = fuse(r, h)
            folder = out / 'features' / key
            for name, table in [('radiomics', r), ('hsv', h), ('fused', f)]: save_csv(table, folder / f'{name}.csv')
            numeric = f.drop(columns=META).apply(pd.to_numeric).to_numpy(float)
            save_csv(pd.DataFrame({'feature': list(f.columns[3:]), 'nonfinite_count': (~np.isfinite(numeric)).sum(axis=0)}),
                     folder / 'feature_missingness.csv')
            save_json(folder / 'provenance.json', dict(token=digest(dict(base=base_signature, inputs=signatures)),
                inputs=signatures, base=base_signature, n_samples=len(f), n_radiomics=len(r.columns)-3,
                n_hsv=len(h.columns)-3, n_fused=len(f.columns)-3, nonfinite=int((~np.isfinite(numeric)).sum()),
                features=list(f.columns[3:]), tables={name: sha256(folder / f'{name}.csv') for name in ['radiomics', 'hsv', 'fused']},
                status='complete' if all(q['status']=='ok' for q in qc) else 'errors',
                radiomics_settings=ex.settings, enabled_feature_classes=list(ex.enabledFeatures),
                spacing='1 pixel; no physical calibration', gray='OpenCV COLOR_RGB2GRAY'))
    save_csv(pd.DataFrame(failures, columns=['config', *META, 'path', 'status', 'error']), out / 'logs' / 'extraction_errors.csv')
    if failures: raise RuntimeError(f'{len(failures)} image/feature failures. Outputs retained; training blocked pending review.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--stage', choices=['qc', 'extract', 'train', 'all'], default='all')
    parser.add_argument('--method', choices=['area', 'lanczos', 'gaussain_pyramid', 'gaussian_pyramid'])
    parser.add_argument('--resolution', type=int, choices=[1024, 2048, 4096])
    parser.add_argument('--feature-set', choices=['gray_radiomics', 'rgb_radiomics', 'gray_radiomics_hsv', 'fused', 'radiomics', 'all'], default='gray_radiomics_hsv')
    parser.add_argument('--classifier', choices=['rf', 'xgboost', 'all'], default='all')
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit', type=int, help='Extraction/QC smoke test only, writes to separate smoke output')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.n_jobs < 1: parser.error('--n-jobs must be positive')
    if args.limit is not None and (args.limit < 1 or args.stage in ['train', 'all']): parser.error('--limit only for qc/extract and must be positive')
    if args.limit: cfg['output_root'] += f'_smoke_{args.limit}'
    out = Path(cfg['output_root'])
    if not args.dry_run:
        (out / 'logs').mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(out / 'logs' / 'run.log', encoding='utf-8')])
    else: logging.basicConfig(level=logging.INFO)
    logging.getLogger('radiomics').setLevel(logging.ERROR)
    cv2.setNumThreads(1)
    from threadpoolctl import threadpool_limits
    blas_limits = threadpool_limits(limits=1, user_api='blas')
    try:
        df, cohort = discover(cfg, write=not args.dry_run)
        logging.info('Common cohort: %d images, %d patients, %d classes', len(cohort), cohort.patient_id.nunique(), cohort.label.nunique())
        if args.method:
            method = args.method
            if method in ('gaussain_pyramid', 'gaussian_pyramid'):
                method = next(m for m in df.downsampling.unique() if m in ('gaussain_pyramid', 'gaussian_pyramid'))
            df = df[df.downsampling == method]
        if args.resolution: df = df[df.resolution == args.resolution]
        if args.limit:
            ids = cohort.sample(n=min(args.limit, len(cohort)), random_state=cfg['seed']).sample_id
            df = df[df.sample_id.isin(ids)]
        if df.empty: raise ValueError('No selected configurations')
        if args.dry_run:
            print(df.groupby(['downsampling', 'resolution']).size().to_string()); return 0
        save_json(out / 'logs' / 'environment.json', versions())
        save_json(out / 'logs' / 'resolved_config.json', cfg)
        save_json(out / 'logs' / 'last_command.json', vars(args))
        if args.stage == 'qc': process_images(df, cfg, force=args.force)
        from .baselines import FEATURE_SETS, prepare_features, train_all, fixed_splits
        selected = list(FEATURE_SETS) if args.feature_set == 'all' else [{'fused':'gray_radiomics_hsv', 'radiomics':'gray_radiomics'}.get(args.feature_set, args.feature_set)]
        if args.force: raise ValueError('--force is disabled for protected baseline caches; choose a new output_root')
        if args.stage in ['train', 'all']: fixed_splits(cohort, cfg)
        if args.stage in ['extract', 'train', 'all']:
            prepare_features(df, cfg, selected, extract=args.stage != 'train')
        if args.stage in ['train', 'all']:
            train_all(df, cohort, cfg, selected,
                      ['rf', 'xgboost'] if args.classifier=='all' else [args.classifier], args.n_jobs)
        return 0
    except Exception:
        logging.exception('Pipeline stopped; no samples silently discarded')
        if not args.dry_run: (out / 'logs' / 'last_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return 1

if __name__ == '__main__': sys.exit(main())
