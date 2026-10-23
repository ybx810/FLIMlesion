import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, roc_auc_score)
from .common import save_json, save_csv
from . import visualize

def validate_oof(oof, cohort, classes):
    if len(oof) != len(cohort) or oof.sample_id.duplicated().any() or set(oof.sample_id) != set(cohort.sample_id):
        raise ValueError('Incomplete or repeated OOF predictions')
    joined = cohort[['sample_id', 'patient_id', 'label']].merge(oof, on='sample_id', validate='one_to_one', suffixes=('_expected', ''))
    if not ((joined.patient_id_expected == joined.patient_id) & (joined.label == joined.true_label)).all():
        raise ValueError('OOF metadata differs from cohort')
    if oof.fold.isna().any() or oof.groupby('patient_id').fold.nunique().max() != 1:
        raise ValueError('OOF patient/fold leakage')
    p = oof[[f'probability_{c}' for c in classes]].to_numpy()
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Invalid OOF probabilities')
    if not (oof.predicted_label.to_numpy() == np.array(classes)[p.argmax(axis=1)]).all():
        raise ValueError('OOF predicted labels do not match probabilities')

def evaluate(oof, classes, out):
    y, pred = oof.true_label, oof.predicted_label
    metrics = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred))
    for avg in ['macro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=classes, average=avg, zero_division=0)
        metrics.update({f'{avg}_precision': p, f'{avg}_recall': r, f'{avg}_f1': f})
    try:
        metrics['roc_auc_ovr_macro'] = roc_auc_score(y, oof[[f'probability_{c}' for c in classes]], labels=classes, multi_class='ovr', average='macro')
    except ValueError as e:
        metrics.update(roc_auc_ovr_macro=None, roc_auc_reason=str(e))
    report = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=classes)
    if cm.sum() != len(oof): raise ValueError('Confusion matrix count mismatch')
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    save_csv(pd.DataFrame(report).T.rename_axis('label').reset_index(), out / 'classification_report.csv')
    for name, values, norm in [('confusion_matrix', cm, False), ('confusion_matrix_normalized', normalized, True)]:
        save_csv(pd.DataFrame(values, columns=classes).assign(true_label=classes), out / f'{name}.csv')
        visualize.matrix(values, classes, out / name, norm)
    visualize.class_scores(report, classes, out / 'per_class_scores')
    save_json(out / 'metrics.json', metrics)
    return metrics
