import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from .common import META
from .dataset import discover, get_splits, validate_splits
from .feature_fusion import fuse
from .roi_utils import validate_roi
from .extract_hsv import extract
from .evaluation import validate_oof
from .train_classifiers import FoldPreprocessor, train
from .run import load_config

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).with_name('config.yaml'))
        # The synthetic legacy fixture has three patients per class.
        self.cfg['cv']['n_splits'] = 3

    def cohort(self):
        return pd.DataFrame([dict(sample_id=f's{c}_{p}_{i}', patient_id=f'p{c}_{p}', label=f'c{c}')
                             for c in range(8) for p in range(3) for i in range(2)])

    def test_fusion_by_id_and_metadata(self):
        a = self.cohort().iloc[:3].copy(); a['RAD_test'] = [1., 2., 3.]
        b = a[META].iloc[::-1].copy(); b['HSV_test'] = [30., 20., 10.]
        self.assertEqual(fuse(a, b).HSV_test.tolist(), [10., 20., 30.])
        b.loc[b.index[0], 'label'] = 'wrong'
        with self.assertRaises(ValueError): fuse(a, b)
        with self.assertRaises(ValueError): fuse(a, pd.concat([b, b]))

    def test_exact_matching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = copy.deepcopy(self.cfg)
            cfg.update(data_root=str(root), metadata=str(root/'metadata.csv'), methods=['area'], resolutions=[1024])
            cfg['metadata_columns'] = {c: c for c in META}
            cohort = self.cohort().groupby('label').head(1)
            cohort.to_csv(cfg['metadata'], index=False)
            rows = []
            for r in cohort.itertuples():
                p = root/'area'/'1024'/r.label/(r.sample_id+'.png'); p.parent.mkdir(parents=True, exist_ok=True); p.touch()
                rows.append(dict(output_path=str(p), relative_path=f'{r.label}\\{r.sample_id}.jpg'))
            pd.DataFrame(rows).to_csv(root/'processing_log.csv', index=False)
            images, common = discover(cfg, write=False)
            self.assertEqual(len(common), 8)
            self.assertEqual(set(images.patient_id), set(cohort.patient_id))
            # An unknown image must stop discovery, not silently disappear.
            (p.parent/'unknown.png').touch()
            with self.assertRaises(ValueError): discover(cfg, write=False)

    def test_roi_dimensions_and_empty(self):
        rgb = np.zeros((10, 11, 3), np.uint8)
        with self.assertRaises(ValueError): validate_roi(rgb, np.ones((11, 10), bool))
        with self.assertRaises(ValueError): validate_roi(rgb, np.zeros((10, 11), bool))

    def test_hsv_ranges_empty_and_histograms(self):
        rgb = np.zeros((12, 12, 3), np.uint8); rgb[:] = [255, 255, 0]
        f, masks = extract(rgb, np.ones((12, 12), bool), self.cfg['hsv'])
        self.assertEqual(f['COLOR_Yellow_AreaRatio'], 1)
        self.assertEqual(f['COLOR_Green_ComponentCount'], 0)
        self.assertTrue(np.isnan(f['COLOR_Green_MeanV']))
        self.assertAlmostEqual(f['HSV_H_CircularMean'], 30)
        for channel in 'HSV': self.assertAlmostEqual(sum(v for k,v in f.items() if k.startswith(f'HSV_{channel}_Hist_')), 1)
        rgb[:] = 100
        gray, _ = extract(rgb, np.ones((12, 12), bool), self.cfg['hsv'])
        self.assertTrue(np.isnan(gray['HSV_H_Mean']))
        self.assertEqual(gray['HSV_H_ValidFraction'], 0)

    def test_patient_splits_stable_and_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(self.cfg); cfg['output_root'] = temp
            c = self.cohort(); s = get_splits(c, cfg)
            self.assertEqual(s.groupby('patient_id').fold.nunique().max(), 1)
            pd.testing.assert_frame_equal(s, get_splits(c, cfg))
            s.loc[0, 'fold'] = (s.loc[0, 'fold']+1) % 3
            with self.assertRaises(ValueError): validate_splits(c, s, n_splits=3)

    def test_training_only_missing_and_filters(self):
        x = pd.DataFrame({'allmissing': [np.nan]*3, 'constant':[1]*3, 'signal':[1., np.nan, 3.], 'copy':[1., 2., 3.]})
        prep = FoldPreprocessor(correlation_threshold=.95).fit(x)
        self.assertEqual(len(prep.feature_names_out_), 1)
        test = pd.DataFrame({'allmissing':[999.], 'constant':[5.], 'signal':[np.nan], 'copy':[800.]})
        self.assertEqual(prep.transform(test)[0, 0], 2.)

    def test_five_fold_patient_split(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(self.cfg)
            cfg['output_root'] = temp
            cfg['cv']['n_splits'] = 5
            cohort = pd.DataFrame([dict(sample_id=f's{c}_{p}_{i}', patient_id=f'p{c}_{p}', label=f'c{c}')
                                   for c in range(8) for p in range(5) for i in range(2)])
            splits = get_splits(cohort, cfg)
            validate_splits(cohort, splits, n_splits=5)
            self.assertEqual(set(splits.fold), set(range(5)))
            self.assertEqual(splits.groupby('patient_id').fold.nunique().max(), 1)
            self.assertTrue((splits.groupby('fold').label.nunique() == 8).all())
            pd.testing.assert_frame_equal(splits, get_splits(cohort, cfg))

    def test_default_validation_rejects_historical_three_fold(self):
        cohort = self.cohort()
        splits = cohort.copy()
        splits['fold'] = np.tile(np.repeat(np.arange(3), 2), 8)
        validate_splits(cohort, splits, n_splits=3)
        with self.assertRaisesRegex(ValueError, 'Invalid fold IDs'):
            validate_splits(cohort, splits)

    def test_oof_completeness_guard(self):
        c = self.cohort(); oof = c.rename(columns={'label': 'true_label'}).copy()
        oof['fold'] = 0; oof['predicted_label'] = 'c0'
        for i in range(8): oof[f'probability_c{i}'] = 1/8
        validate_oof(oof, c, [f'c{i}' for i in range(8)])
        with self.assertRaises(ValueError): validate_oof(oof.iloc[:-1], c, [f'c{i}' for i in range(8)])
        oof.loc[0, 'probability_c0'] = np.nan
        with self.assertRaises(ValueError): validate_oof(oof, c, [f'c{i}' for i in range(8)])

    def test_radiomics_full_roi(self):
        from .extract_radiomics import create_extractor, extract, feature_names
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        f = extract(rgb, np.ones(rgb.shape[:2], bool), create_extractor(self.cfg['radiomics']))
        self.assertEqual(set(f), set(feature_names()))
        self.assertTrue(np.isfinite(list(f.values())).all())

    def test_extraction_cache_content_invalidation(self):
        import json
        from PIL import Image
        from .run import process_images
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(self.cfg); cfg['output_root'] = temp; cfg['preview_count'] = 0
            path = Path(temp)/'input.png'
            rgb = np.random.default_rng(42).integers(0, 256, (24, 32, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(path)
            rows = pd.DataFrame([dict(sample_id='s1', patient_id='p1', label='c1', config='area_32',
                                      path=str(path), mask_path='', resolution=32)])
            process_images(rows, cfg, extract=True)
            cache = next((Path(temp)/'features/area_32/cache').glob('*.json'))
            original = cache.read_bytes(); mtime = cache.stat().st_mtime_ns
            process_images(rows, cfg, extract=True)
            self.assertEqual(mtime, cache.stat().st_mtime_ns)
            rgb[0,0] = [0,0,0]; Image.fromarray(rgb).save(path)
            process_images(rows, cfg, extract=True)
            self.assertNotEqual(json.loads(original)['token'], json.loads(cache.read_bytes())['token'])

    def test_rf_xgboost_oof_and_reload(self):
        import joblib
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(self.cfg); cfg['output_root'] = temp
            cfg['rf']['n_estimators'] = 8; cfg['xgboost']['n_estimators'] = 8
            d = self.cohort(); splits = get_splits(d, cfg)
            d['RAD_signal'] = [int(v[1:]) for v in d.label]
            d['HSV_noise'] = np.random.default_rng(42).normal(size=len(d))
            d['HSV_missing'] = np.nan
            for model in ['rf', 'xgboost']:
                out = Path(temp)/model
                train(d, splits, cfg, model, out, 1)
                oof = pd.read_csv(out/'oof_predictions.csv')
                self.assertEqual(len(oof), len(d))
                cm = pd.read_csv(out/'confusion_matrix.csv').drop(columns='true_label')
                self.assertEqual(cm.to_numpy().sum(), len(d))
                bundle = joblib.load(out/'fold_0.joblib')
                test = d[d.sample_id.isin(splits[splits.fold==0].sample_id)]
                expected = oof.set_index('sample_id').loc[test.sample_id, [f'probability_c{i}' for i in range(8)]].to_numpy()
                np.testing.assert_allclose(bundle['pipeline'].predict_proba(test[bundle['input_features']]), expected, atol=1e-6)
                train(d, splits, cfg, model, out, 1)
                repeated = pd.read_csv(out/'oof_predictions.csv')
                pd.testing.assert_frame_equal(oof, repeated)

if __name__ == '__main__': unittest.main()
