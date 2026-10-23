"""Audit saved baseline OOFs against the retained split, without training."""
import json
from pathlib import Path
import pandas as pd
from scripts.radiomics_hsv_baseline.common import META, ROOT, sha256, save_json, save_csv
from scripts.radiomics_hsv_baseline.dataset import validate_splits
from scripts.radiomics_hsv_baseline.evaluation import validate_oof


def main():
    source = ROOT / 'outputs/radiomics_hsv_baseline_5fold/cv_splits/cv_splits.csv'
    expected = pd.read_csv(source, dtype={c: str for c in META})
    validate_splits(expected[META], expected)
    rows = []
    for path in sorted((ROOT / 'outputs').glob('*baseline*/experiments/**/oof_predictions.csv')):
        oof = pd.read_csv(path, dtype={'sample_id': str, 'patient_id': str, 'true_label': str})
        active = path.relative_to(ROOT / 'outputs').parts[0].endswith('_5fold')
        row = dict(path=str(path.relative_to(ROOT)), current=active, samples=len(oof),
                   folds=','.join(map(str, sorted(oof.fold.unique()))), matches_retained=False)
        try:
            validate_oof(oof, expected[META], sorted(expected.label.unique()))
            checked = validate_splits(expected[META], oof.rename(columns={'true_label': 'label'})[META + ['fold']])
            pd.testing.assert_frame_equal(checked.sort_values('sample_id').reset_index(drop=True),
                                          expected[META + ['fold']].sort_values('sample_id').reset_index(drop=True))
            assert all((path.parent / f'fold_{i}.joblib').exists() for i in range(5)), 'Missing fold model'
            row['matches_retained'] = True
        except (ValueError, AssertionError) as error:
            row['reason'] = str(error)
        rows.append(row)
    table = pd.DataFrame(rows)
    save_csv(table, ROOT / 'outputs/five_fold_audit.csv')
    required = []
    for feature in ('gray_radiomics', 'rgb_radiomics', 'gray_radiomics_hsv'):
        for classifier in ('rf', 'xgboost'):
            required.append(f'outputs/radiomics_hsv_baseline_5fold/experiments/area_1024/{feature}/{classifier}/oof_predictions.csv')
    for cnn in ('efficientnet_b0', 'resnet50', 'densenet121'):
        for classifier in ('rf', 'xgboost'):
            required.append(f'outputs/cnn_pca_baseline_5fold/experiments/area_1024/{cnn}/{classifier}/oof_predictions.csv')
    valid = {Path(r['path']).as_posix() for r in rows if r['matches_retained']}
    missing = sorted(set(required) - valid)
    summary = dict(source=str(source), source_sha256=sha256(source), samples=len(expected),
                   patients=expected.patient_id.nunique(), fold_counts=expected.groupby('fold').size().to_dict(),
                   current_oof_count=int(table.current.sum()), historical_oof_count=int((~table.current).sum()),
                   required_baselines=12, missing_or_invalid=missing,
                   other_image_configurations='Not trained previously; not included in the migration rerun',
                   passed=not missing and all(r['matches_retained'] for r in rows if r['current']))
    save_json(ROOT / 'outputs/five_fold_audit.json', summary)
    print(json.dumps(summary, indent=2))
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
