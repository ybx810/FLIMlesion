"""Regression guards for the incremental three-baseline extension."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from .common import META
from .run import load_config
from .baselines import FEATURE_SETS, prepare_features, fixed_splits, validate_table, check_oof
from .extract_radiomics import extract, extract_channel, create_extractor, feature_names

class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).with_name('config_5fold.yaml'))
        self.cfg['preview_count'] = 0

    def test_gray_refactor_and_rgb_channel_identity(self):
        rgb = np.random.default_rng(3).integers(0,256,(25,27,3),dtype=np.uint8)
        roi = np.ones(rgb.shape[:2],bool); roi[:2] = False
        ex = create_extractor(self.cfg['radiomics'])
        a = extract(rgb,roi,ex)
        b = extract_channel(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),roi,ex)
        self.assertEqual(a,b)
        for i,c in enumerate('RGB'):
            values = extract_channel(rgb[...,i],roi,ex,f'RGB_{c}_')
            self.assertAlmostEqual(values[f'RGB_{c}_firstorder_Mean'],rgb[...,i][roi].mean())
            self.assertEqual(len(values),93)
        self.assertEqual(len(FEATURE_SETS),3)
        self.assertNotIn('hsv',FEATURE_SETS)

    def fixture(self, temp):
        cfg=copy.deepcopy(self.cfg); cfg['output_root']=temp
        path=Path(temp)/'image.png'
        Image.fromarray(np.random.default_rng(10).integers(0,256,(24,32,3),dtype=np.uint8)).save(path)
        rows=pd.DataFrame([dict(sample_id='s',patient_id='p',label='c',config='area_32',path=str(path),mask_path='',resolution=32)])
        return cfg,path,rows

    def test_rgb_cache_reuse_and_image_roi_parameter_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg,path,rows=self.fixture(temp)
            prepare_features(rows,cfg,['rgb_radiomics'])
            cached=next((Path(temp)/'features/area_32/rgb_cache').glob('*.json'))
            before=cached.stat().st_mtime_ns
            with patch('scripts.radiomics_hsv_baseline.baselines.inspect',side_effect=AssertionError('cache should avoid pixel decoding')):
                prepare_features(rows,cfg,['rgb_radiomics'])
            self.assertEqual(before,cached.stat().st_mtime_ns)
            cfg['radiomics']['binWidth']=8
            with self.assertRaisesRegex(ValueError,'Stale'): prepare_features(rows,cfg,['rgb_radiomics'])
            cfg['radiomics']['binWidth']=16; cfg['roi']['min_pixels']=65
            with self.assertRaisesRegex(ValueError,'Stale'): prepare_features(rows,cfg,['rgb_radiomics'])
            cfg['roi']['min_pixels']=64
            Image.fromarray(np.zeros((24,32,3),np.uint8)).save(path)
            with self.assertRaisesRegex(ValueError,'Stale'): prepare_features(rows,cfg,['rgb_radiomics'])

    def test_failed_channel_is_logged_and_blocks_whole_table(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg,_,rows=self.fixture(temp)
            with patch('scripts.radiomics_hsv_baseline.baselines.extract_channel',return_value={'RGB_R_firstorder_Mean':np.inf}):
                with self.assertRaisesRegex(RuntimeError,'RGB failures'): prepare_features(rows,cfg,['rgb_radiomics'])
            folder=Path(temp)/'features/area_32'
            self.assertFalse((folder/'rgb_radiomics.csv').exists())
            self.assertIn('NaN/Inf',(folder/'rgb_extraction_errors.csv').read_text(encoding='utf-8-sig'))

    def test_split_projection_retains_assignments_and_original_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=copy.deepcopy(self.cfg);cfg['output_root']=temp
            saved=pd.DataFrame([dict(sample_id=f's{c}_{p}',patient_id=f'p{c}_{p}',label=f'c{c}',fold=p) for c in range(8) for p in range(5)])
            source=Path(temp)/'cv_splits/cv_splits.csv';source.parent.mkdir();saved.to_csv(source,index=False)
            before=source.read_bytes()
            cohort=saved.iloc[1:][META]
            actual,_=fixed_splits(cohort,cfg)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(actual.set_index('sample_id').fold.to_dict(),saved.iloc[1:].set_index('sample_id').fold.to_dict())
            # Reordering features is harmless; membership/metadata corruption is rejected.
            gray=cohort.copy()
            for n in feature_names(): gray[n]=1.
            validate_table(gray.iloc[::-1],cohort,'gray_radiomics')
            gray['HSV_fake']=1.
            with self.assertRaisesRegex(ValueError,'Gray schema'): validate_table(gray,cohort,'gray_radiomics')
            with self.assertRaises(ValueError): fixed_splits(pd.concat([cohort,cohort.iloc[:1]]),cfg)

if __name__=='__main__': unittest.main()
