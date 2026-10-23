from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from .common import META, save_csv, save_json
from .dataset import validate_splits
from .evaluation import evaluate, validate_oof
from . import visualize

class FoldPreprocessor(BaseEstimator, TransformerMixin):
    """Persist the training-only feature decisions, including all-missing columns."""
    def __init__(self, correlation_threshold=None, standardize=False):
        self.correlation_threshold = correlation_threshold
        self.standardize = standardize

    def fit(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        self.input_features_ = list(X.columns)
        self.nonempty_ = list(X.columns[X.notna().any()])
        if not self.nonempty_: raise ValueError('No non-missing training features')
        self.imputer_ = SimpleImputer(strategy='median')
        a = self.imputer_.fit_transform(X[self.nonempty_])
        self.variance_ = VarianceThreshold().fit(a)
        a = self.variance_.transform(a)
        names = np.array(self.nonempty_)[self.variance_.get_support()]
        self.keep_ = np.arange(a.shape[1])
        if self.correlation_threshold is not None:
            if not 0 < self.correlation_threshold <= 1: raise ValueError('Invalid correlation threshold')
            corr = np.atleast_2d(np.abs(np.corrcoef(a, rowvar=False)))
            chosen = []
            for j in range(a.shape[1]):
                if not any(corr[j, k] > self.correlation_threshold for k in chosen): chosen.append(j)
            self.keep_ = np.array(chosen)
        self.feature_names_out_ = names[self.keep_]
        self.scaler_ = StandardScaler().fit(a[:, self.keep_]) if self.standardize else None
        return self

    def transform(self, X):
        a = self.imputer_.transform(X[self.nonempty_].replace([np.inf, -np.inf], np.nan))
        a = self.variance_.transform(a)[:, self.keep_]
        return self.scaler_.transform(a) if self.scaler_ else a

def train(features, splits, cfg, model_name, out, n_jobs=4, preprocessor_factory=None, fold_callback=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    validate_splits(features[META], splits, cfg['cv']['n_splits'])
    d = features.merge(splits[['sample_id', 'fold']], on='sample_id', validate='one_to_one')
    classes = sorted(d.label.unique())
    if len(classes) != 8: raise ValueError('Expected eight classes')
    mapping = {label: i for i, label in enumerate(classes)}
    save_json(out / 'class_mapping.json', mapping)
    cols = [c for c in features if c not in META]
    X, y = d[cols], d.label.map(mapping).to_numpy()
    probabilities = np.full((len(d), len(classes)), np.nan)
    visits = np.zeros(len(d), int)
    importances, fold_info = [], []
    for fold in sorted(d.fold.unique()):
        tr, te = np.flatnonzero(d.fold != fold), np.flatnonzero(d.fold == fold)
        if model_name == 'rf':
            model = RandomForestClassifier(**cfg['rf'], random_state=cfg['seed'], n_jobs=n_jobs)
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier(**cfg['xgboost'], objective='multi:softprob', num_class=8,
                                  random_state=cfg['seed'], n_jobs=n_jobs)
        preprocess = preprocessor_factory() if preprocessor_factory else FoldPreprocessor(**cfg['preprocessing'])
        pipeline = Pipeline([('preprocess', preprocess), ('model', model)])
        pipeline.fit(X.iloc[tr], y[tr])
        probabilities[np.ix_(te, model.classes_.astype(int))] = pipeline.predict_proba(X.iloc[te])
        visits[te] += 1
        selected = pipeline.named_steps['preprocess'].feature_names_out_
        if fold_callback is not None:
            fold_callback(fold, pipeline, X.iloc[tr], X.iloc[te])
        joblib.dump(dict(pipeline=pipeline, classes=classes, input_features=cols, fold=int(fold)), out / f'fold_{fold}.joblib')
        importance_cols = list(selected) if preprocessor_factory else cols
        imp = pd.Series(model.feature_importances_, index=selected).reindex(importance_cols, fill_value=0)
        importances.append(imp)
        fold_info.append(dict(fold=int(fold), n_train=len(tr), n_test=len(te), n_features=len(selected), selected_features=list(selected)))
    if not (visits == 1).all(): raise ValueError('Each sample must be predicted exactly once')
    oof = d[['sample_id', 'patient_id', 'label', 'fold']].rename(columns={'label': 'true_label'})
    oof['predicted_label'] = np.array(classes)[probabilities.argmax(axis=1)]
    for j, c in enumerate(classes): oof[f'probability_{c}'] = probabilities[:, j]
    validate_oof(oof, features, classes)
    save_csv(oof, out / 'oof_predictions.csv')
    save_json(out / 'folds.json', fold_info)
    importance = pd.DataFrame({'feature': importance_cols, 'importance': np.mean(importances, axis=0), 'std_across_folds': np.std(importances, axis=0)})
    save_csv(importance.sort_values('importance', ascending=False), out / 'feature_importance.csv')
    save_json(out / 'feature_importance_policy.json', dict(
        source='mean decrease in impurity' if model_name == 'rf' else 'XGBoost gain',
        aggregation='unweighted mean across training folds; filtered columns receive zero',
        limitation='exploratory built-in importance; not evidence of stable biomarkers'))
    visualize.importance(importance, out / 'feature_importance')
    return evaluate(oof, classes, out)
