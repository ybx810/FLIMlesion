import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.radiomics_hsv_baseline.common import META
from scripts.radiomics_hsv_baseline.dataset import validate_splits
from .train_pca_classifiers import FoldPCA, train_embedding
from .extract_embeddings import validate_table

class Tests(unittest.TestCase):
    def test_cache_content_and_transform_invalidation(self):
        from PIL import Image
        from .extract_embeddings import contract, cached
        from scripts.radiomics_hsv_baseline.common import save_csv, save_json, digest, sha256
        from . import extract_embeddings as mod
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/'image.png'
            Image.new('RGB',(20,20),'red').save(image)
            images=pd.DataFrame([dict(sample_id='a',patient_id='p',label='c',path=str(image),config='area_1024')])
            cfg=dict(output_root=temp,weight_cache=str(root/'weights'))
            table=pd.concat([images[META],pd.DataFrame(np.zeros((1,1024)),columns=[f'EMB_{i:04d}' for i in range(1024)])],axis=1)
            path=root/'embeddings/area_1024/densenet121.csv'
            save_csv(table,path)
            save_json(path.with_suffix('.json'),dict(token=digest(contract(images,'densenet121',cfg)),csv_sha256=sha256(path)))
            pd.testing.assert_frame_equal(cached(images,'densenet121',cfg),table)
            original=mod.specification
            def changed(*args):
                weights,transform,checkpoint,spec=original(*args)
                spec['transform']='changed official preprocessing'
                return weights,transform,checkpoint,spec
            with patch.object(mod,'specification',side_effect=changed):
                with self.assertRaises(ValueError): cached(images,'densenet121',cfg)
            Image.new('RGB',(20,20),'blue').save(image)
            with self.assertRaises(ValueError): cached(images,'densenet121',cfg)

    def test_pca_train_only_and_rank(self):
        rng=np.random.default_rng(12)
        x=pd.DataFrame(rng.normal(size=(100,80)))
        model=FoldPCA().fit(x)
        before=model.pipeline_.named_steps['scaler'].mean_.copy()
        self.assertEqual(model.transform(x+10000).shape,(100,64))
        np.testing.assert_array_equal(before,model.pipeline_.named_steps['scaler'].mean_)
        np.testing.assert_allclose(before,x.mean())
        with self.assertRaises(ValueError): FoldPCA().fit(x.iloc[:63])
        with self.assertRaises(ValueError): FoldPCA().fit(pd.DataFrame(np.ones((100,80))))
        with self.assertRaises(ValueError): FoldPCA(32).fit(x)

    def test_embedding_key_join(self):
        meta=pd.DataFrame(dict(sample_id=['a','b'],patient_id=['p','q'],label=['x','y']))
        features=pd.DataFrame(np.zeros((2,1024)),columns=[f'EMB_{i:04d}' for i in range(1024)])
        table=pd.concat([meta,features],axis=1)
        got=validate_table(table.iloc[::-1],meta,'densenet121')
        self.assertEqual(got.sample_id.tolist(),['a','b'])
        with self.assertRaises(ValueError): validate_table(pd.concat([table,table]),meta,'densenet121')
        table.loc[0,'EMB_0000']=np.nan
        with self.assertRaises(ValueError): validate_table(table,meta,'densenet121')

    def test_shared_trainer_fold_fit_and_oof(self):
        rng=np.random.default_rng(42)
        meta=pd.DataFrame([dict(sample_id=f's{c}_{f}_{i}',patient_id=f'p{c}_{f}',label=f'c{c}',fold=f) for c in range(8) for f in range(5) for i in range(5)])
        table=pd.concat([meta[META],pd.DataFrame(rng.normal(size=(200,80)),columns=[f'E{i}' for i in range(80)])],axis=1)
        cfg=dict(cv={'n_splits':5},seed=42,rf={'n_estimators':3},preprocessing={})
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            with patch('scripts.radiomics_hsv_baseline.visualize.save'),patch('scripts.cnn_pca_baseline.train_pca_classifiers.variance_plot'):
                train_embedding(table,meta,cfg,'rf',out,1)
            import joblib
            for f in range(5):
                scaler=joblib.load(out/f'scaler_fold_{f}.joblib')
                np.testing.assert_allclose(scaler.mean_,table.loc[meta.fold!=f].drop(columns=META).mean(),atol=1e-12)
                self.assertEqual(scaler.n_samples_seen_,160)
            oof=pd.read_csv(out/'oof_predictions.csv')
            self.assertEqual(len(oof),200); self.assertFalse(oof.sample_id.duplicated().any())
            self.assertEqual(pd.read_csv(out/'feature_importance.csv').shape[0],64)

if __name__=='__main__': unittest.main()
