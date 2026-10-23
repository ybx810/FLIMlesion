from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ['png', 'pdf']: fig.savefig(path.with_suffix('.' + suffix), dpi=150, bbox_inches='tight')
    plt.close(fig)

def matrix(values, names, path, normalized=False):
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else None)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
           yticklabels=names, xlabel='Predicted label', ylabel='True label')
    plt.setp(ax.get_xticklabels(), rotation=50, ha='right')
    for (i, j), v in np.ndenumerate(values):
        ax.text(j, i, f'{v:.2f}' if normalized else str(int(v)), ha='center', va='center',
                color='white' if v > (0.5 if normalized else np.max(values)/2) else 'black')
    save(fig, path)

def importance(table, path):
    top = table.sort_values('importance').tail(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.feature, top.importance)
    ax.set_title('Mean built-in importance across training folds')
    ax.set_xlabel('Baseline interpretation only; not validated biomarkers')
    save(fig, path)

def class_scores(report, classes, path):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(classes))
    for i, metric in enumerate(['precision', 'recall', 'f1-score']):
        ax.bar(x+(i-1)*0.25, [report[name][metric] for name in classes], width=0.25, label=metric)
    ax.set(xticks=x, xticklabels=classes, ylim=(0, 1), ylabel='OOF score')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    ax.legend()
    save(fig, path)

def preview(rgb, roi, masks, path):
    step = max(1, int(np.ceil(max(rgb.shape[:2]) / 600)))
    rgb, roi = rgb[::step, ::step], roi[::step, ::step]
    masks = {name: mask[::step, ::step] for name, mask in masks.items()}
    fig, axes = plt.subplots(1, len(masks)+2, figsize=(18, 4))
    axes[0].imshow(rgb); axes[0].set_title('Original RGB')
    for ax, (name, mask) in zip(axes[1:], [('ROI', roi), *masks.items()]):
        ax.imshow(rgb)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = [0, 0.6, 1, 0.45]
        ax.imshow(overlay); ax.set_title(name)
    for ax in axes: ax.axis('off')
    save(fig, path)

def comparisons(table, out):
    for (model, feature_set), d in table.groupby(['classifier', 'feature_set']):
        for metric in ['macro_f1', 'balanced_accuracy']:
            pivot = d.pivot(index='downsampling', columns='resolution', values=metric)
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap='viridis')
            ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
                   yticks=range(len(pivot)), yticklabels=pivot.index, title=f'{model} / {feature_set} / {metric}')
            for (i, j), v in np.ndenumerate(pivot.to_numpy()): ax.text(j, i, f'{v:.3f}', ha='center', color='white')
            fig.colorbar(im, ax=ax)
            save(fig, Path(out) / f'{model}_{feature_set}_{metric}')

def baseline_comparisons(table, per_class, classes, out):
    """Fixed scales/order, with unrun configurations explicitly left blank."""
    from .baselines import FEATURE_SETS
    methods = ['area', 'lanczos', 'gaussain_pyramid']
    display = ['Area', 'Lanczos', 'Gaussian Pyramid']
    t = table.copy()
    t['downsampling'] = t.downsampling.replace({'gaussian_pyramid':'gaussain_pyramid'})
    for name in FEATURE_SETS:
        for classifier in ('rf','xgboost'):
            d = t[(t.feature_set == name) & (t.classifier == classifier)]
            p = d.pivot(index='downsampling', columns='resolution', values='macro_f1').reindex(index=methods,columns=[1024,2048,4096])
            fig, ax = plt.subplots(figsize=(7,4))
            im = ax.imshow(p.to_numpy(), vmin=0,vmax=1,cmap='viridis')
            ax.set(xticks=range(3),xticklabels=p.columns,yticks=range(3),yticklabels=display,title=f'{name} / {classifier} / OOF Macro-F1')
            for (i,j),v in np.ndenumerate(p.to_numpy()): ax.text(j,i,f'{v:.3f}' if np.isfinite(v) else 'not run',ha='center',color='white' if np.isfinite(v) else 'black')
            fig.colorbar(im,ax=ax)
            save(fig, Path(out)/f'heatmap_{name}_{classifier}')
    for (method,resolution), d in t.groupby(['downsampling','resolution']):
        p = d.pivot(index='feature_set',columns='classifier',values='macro_f1').reindex(FEATURE_SETS)
        ax = p.plot.bar(figsize=(10,5),ylim=(0,1),ylabel='OOF Macro-F1',title=f'{method} / {resolution}')
        ax.tick_params(axis='x',rotation=15)
        save(ax.figure,Path(out)/f'baselines_{method}_{resolution}')
    for name,d in t.groupby('feature_set'):
        d = d.copy(); d['configuration'] = d.downsampling + '_' + d.resolution.astype(str)
        p = d.pivot(index='configuration',columns='classifier',values='macro_f1').reindex(columns=['rf','xgboost'])
        ax = p.plot.bar(figsize=(12,5),ylim=(0,1),ylabel='OOF Macro-F1',title=name)
        save(ax.figure,Path(out)/f'classifiers_{name}')
    for (method,resolution,model),d in per_class.groupby(['downsampling','resolution','classifier']):
        p = d.pivot(index='label',columns='feature_set',values='f1-score').reindex(index=classes,columns=FEATURE_SETS)
        ax = p.plot.bar(figsize=(13,6),ylim=(0,1),ylabel='OOF class F1',title=f'{method} / {resolution} / {model}')
        plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
        save(ax.figure,Path(out)/f'class_f1_{method}_{resolution}_{model}')
