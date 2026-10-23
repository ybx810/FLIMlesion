from pathlib import Path
import numpy as np
import pandas as pd
from scripts.radiomics_hsv_baseline import visualize as shared
plt = shared.plt

def variance_plot(table, path):
    fig, ax = plt.subplots(figsize=(8,5))
    for fold,d in table.groupby('fold'):
        ax.plot(d.component,d.cumulative_variance,label=f'Training fold {fold}')
    ax.set(xlabel='PCA component',ylabel='Cumulative explained variance',title='PCA=64 (not a performance score)',ylim=(0,1))
    ax.legend(); shared.save(fig,path)

def comparisons(table, out):
    if table.empty: return
    for (method,res),d in table.groupby(['downsampling','resolution']):
        ax=d.pivot(index='cnn_model',columns='classifier',values='macro_f1').plot.bar(ylim=(0,1),ylabel='OOF Macro-F1',title=f'{method}/{res}: PCA=64',rot=15,figsize=(10,5))
        shared.save(ax.figure,Path(out)/f'cnn_macro_f1_{method}_{res}')
    if len(table[['downsampling','resolution']].drop_duplicates()) == 9:
        shared.comparisons(table.assign(feature_set=table.cnn_model),out)
