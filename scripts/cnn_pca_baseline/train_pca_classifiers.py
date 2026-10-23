"""Exact PCA fitted only by the shared trainer on each training partition."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from scripts.radiomics_hsv_baseline.train_classifiers import train
from scripts.radiomics_hsv_baseline.common import save_json, save_csv, sha256
from .visualization import variance_plot

class FoldPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=64, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.n_components != 64: raise ValueError('This baseline requires exactly 64 components')
        if min(X.shape) < 64 or not np.isfinite(X).all().all():
            raise ValueError('Insufficient training samples/features or nonfinite embeddings')
        self.pipeline_ = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=64, svd_solver='full', random_state=self.random_state))])
        self.pipeline_.fit(X)
        pca = self.pipeline_.named_steps['pca']
        tolerance = np.finfo(float).eps * max(X.shape) * pca.singular_values_[0]
        if pca.singular_values_[-1] <= tolerance:
            raise ValueError('Training matrix numerical rank is below 64; refusing to reduce dimension')
        if not np.isfinite(pca.explained_variance_ratio_).all(): raise ValueError('Invalid PCA variance')
        self.feature_names_out_ = [f'PC_{i+1:02d}' for i in range(64)]
        return self

    def transform(self, X):
        result = self.pipeline_.transform(X)
        if result.shape != (len(X),64) or not np.isfinite(result).all(): raise ValueError('Invalid PCA output')
        return result

def train_embedding(table, splits, rad, classifier, out, threads=4):
    out = Path(out)
    rows = []
    def callback(fold, pipeline, X_train, X_test):
        pre = pipeline.named_steps['preprocess']
        scaler = pre.pipeline_.named_steps['scaler']
        pca = pre.pipeline_.named_steps['pca']
        assert int(scaler.n_samples_seen_) == len(X_train)
        assert pca.n_samples_ == len(X_train)
        pre.transform(X_test)
        joblib.dump(scaler, out/f'scaler_fold_{fold}.joblib')
        joblib.dump(pca, out/f'pca_fold_{fold}.joblib')
        save_json(out/f'pca_fold_{fold}_audit.json', dict(fold=int(fold), n_train=len(X_train), n_validation=len(X_test), fit_samples=int(pca.n_samples_), pca_dim=64, cumulative_variance=float(pca.explained_variance_ratio_.sum()), fit_scope='training fold only'))
        rows.extend(dict(fold=int(fold), component=i+1, explained_variance_ratio=float(v), cumulative_variance=float(c)) for i,(v,c) in enumerate(zip(pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_))))
    with threadpool_limits(limits=threads):
        metrics = train(table, splits, rad, classifier, out, n_jobs=threads, preprocessor_factory=lambda: FoldPCA(64,rad['seed']), fold_callback=callback)
    variance = pd.DataFrame(rows)
    save_csv(variance,out/'pca_explained_variance.csv')
    variance_plot(variance,out/'pca_explained_variance')
    save_json(out/'feature_importance_policy.json',dict(limitation='Importance refers to independently fitted PCA components, not original texture/color or biomarkers. Component indices are fold-specific; their mean importance is only exploratory. PCA variance is not classification performance.'))
    return metrics
