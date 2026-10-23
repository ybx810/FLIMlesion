import argparse
import logging
import sys
import time
from pathlib import Path
import yaml
import pandas as pd
from scripts.radiomics_hsv_baseline.common import ROOT, META, save_json, save_csv, digest, sha256, versions
from scripts.radiomics_hsv_baseline.evaluation import validate_oof
from .dataset import prepare, read_json, read_csv, FEATURE_SETS, radiomics_experiment, reference
from .extract_embeddings import DIMS, extract, cached
from .train_pca_classifiers import train_embedding
from . import visualization as viz

def compare(cfg):
    root=Path(cfg['output_root']); out=root/'comparisons'; rows=[]; radrows=[]; audit=[]
    from scripts.radiomics_hsv_baseline.dataset import validate_splits
    rad = reference(cfg)
    expected = read_csv(cfg['split_source'])
    validate_splits(expected[META], expected, rad['cv']['n_splits'])
    for path in sorted((root/'experiments').glob('*/*/*/run.json')):
        info=read_json(path)
        if info.get('status')!='complete': continue
        folder=path.parent
        cnn=info['cnn_model']; classifier=info['classifier']; key=folder.parents[1].name
        oof=read_csv(folder/'oof_predictions.csv')
        mapping=read_json(folder/'class_mapping.json')
        classes=sorted(mapping,key=mapping.get)
        validate_oof(oof,expected[META],classes)
        checked = validate_splits(expected[META], oof.rename(columns={'true_label':'label'})[META+['fold']], 5)
        pd.testing.assert_frame_equal(checked.sort_values('sample_id').reset_index(drop=True), expected[META+['fold']].sort_values('sample_id').reset_index(drop=True))
        rows.append({**info['summary'],**read_json(folder/'metrics.json')})
        for fs in FEATURE_SETS:
            other=radiomics_experiment(cfg,key,fs,classifier)
            try:
                r=read_csv(other/'oof_predictions.csv')
                validate_oof(r,oof.rename(columns={'true_label':'label'})[META],classes)
                columns=['sample_id','patient_id','true_label','fold']
                pd.testing.assert_frame_equal(oof[columns].sort_values('sample_id').reset_index(drop=True),r[columns].sort_values('sample_id').reset_index(drop=True))
                if read_json(other/'class_mapping.json')!=mapping: raise ValueError('Class mapping mismatch')
                radrows.append(dict(downsampling=info['summary']['downsampling'],resolution=info['summary']['resolution'],feature_set=fs,classifier=classifier,**read_json(other/'metrics.json')))
                audit.append(dict(config=key,cnn=cnn,feature_set=fs,classifier=classifier,comparable=True))
            except Exception as e:
                audit.append(dict(config=key,cnn=cnn,feature_set=fs,classifier=classifier,comparable=False,reason=str(e)))
    columns=['downsampling','resolution','cnn_model','embedding_dim','pca_dim','classifier','n_samples','accuracy','balanced_accuracy','macro_f1','weighted_f1','macro_precision','macro_recall']
    table=pd.DataFrame(rows)
    save_csv(table.reindex(columns=columns),out/'cnn_pca_comparison.csv')
    save_json(out/'radiomics_comparability.json',audit)
    viz.comparisons(table,out)
    if not table.empty:
        combined=pd.concat([table.assign(feature_set=table.cnn_model+' + PCA64'),pd.DataFrame(radrows)]).drop_duplicates(['downsampling','resolution','feature_set','classifier'])
        save_csv(combined,out/'radiomics_comparison.csv')
        for (method,res,clf),d in combined.groupby(['downsampling','resolution','classifier']):
            ax=d.set_index('feature_set').macro_f1.plot.bar(ylim=(0,1),ylabel='OOF Macro-F1',title=f'{method}/{res}/{clf}',figsize=(12,6),rot=25)
            viz.shared.save(ax.figure,out/f'radiomics_{method}_{res}_{clf}')
        for key,group in table.groupby(['downsampling','resolution']):
            method,res=key
            for clf in ('rf','xgboost'):
                fig,axes=viz.plt.subplots(1,3,figsize=(22,7))
                for ax,cnn in zip(axes,DIMS):
                    folder=root/'experiments'/f'{method}_{res}'/cnn/clf
                    if not (folder/'run.json').exists(): ax.set_visible(False); continue
                    cm=read_csv(folder/'confusion_matrix.csv').drop(columns='true_label')
                    ax.imshow(cm.to_numpy(),cmap='Blues'); ax.set(title=cnn,xticks=range(8),yticks=range(8),xticklabels=cm.columns,yticklabels=cm.columns)
                    ax.tick_params(axis='x',rotation=90)
                viz.shared.save(fig,out/f'confusion_comparison_{method}_{res}_{clf}')
            fig,axes=viz.plt.subplots(1,3,figsize=(18,5))
            for ax,cnn in zip(axes,DIMS):
                paths=list((root/'experiments'/f'{method}_{res}'/cnn).glob('*/pca_explained_variance.csv'))
                if not paths: ax.set_visible(False); continue
                for fold,d in read_csv(paths[0]).groupby('fold'): ax.plot(d.component,d.cumulative_variance,label=f'fold {fold}')
                ax.set(title=f'{cnn} PCA=64',ylim=(0,1),xlabel='Components',ylabel='Cumulative variance'); ax.legend()
            viz.shared.save(fig,out/f'pca_comparison_{method}_{res}')
    return table

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='scripts/cnn_pca_baseline/config.yaml')
    p.add_argument('--stage',choices=['inspect','extract','train','compare','all'],default='inspect')
    p.add_argument('--method',choices=['area','lanczos','gaussain_pyramid'],default='area')
    p.add_argument('--resolution',type=int,choices=[1024,2048,4096],default=1024)
    p.add_argument('--cnn',choices=['all',*DIMS],default='all')
    p.add_argument('--classifier',choices=['all','rf','xgboost'],default='all')
    p.add_argument('--pca-components',type=int,choices=[64],default=64)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['cpu'],default='cpu')
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--force',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    args=p.parse_args()
    if args.batch_size<1 or args.threads<1 or args.num_workers<0: p.error('Invalid resource settings')
    cfg=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'))
    for k in ('radiomics_root','radiomics_config','output_root','split_source','weight_cache'): cfg[k]=str((ROOT/cfg[k]).resolve())
    out=Path(cfg['output_root']); (out/'logs').mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'logs/run.log',encoding='utf-8'),logging.StreamHandler()])
    save_json(out/'logs/environment.json',versions())
    start=time.perf_counter()
    try:
        if args.stage=='compare': compare(cfg); return
        images,splits,rad=prepare(cfg,args.method,args.resolution)
        print(f'Validated {len(images)} samples, {images.patient_id.nunique()} patients, folds {sorted(splits.fold.unique())}',flush=True)
        if args.stage=='inspect' or args.dry_run: return
        import torch
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(rad['seed'])
        names=list(DIMS) if args.cnn=='all' else [args.cnn]
        classifiers=['rf','xgboost'] if args.classifier=='all' else [args.classifier]
        for name in names:
            if args.stage in ('extract','all'):
                extract(images.head(2) if args.smoke_test else images,name,cfg,args.batch_size,args.num_workers,args.force,args.smoke_test)
            if args.stage in ('train','all') and not args.smoke_test:
                table=cached(images,name,cfg)
                key=images.iloc[0]['config']
                for clf in classifiers:
                    folder=out/'experiments'/key/name/clf
                    token=digest(dict(embedding=sha256(out/'embeddings'/key/f'{name}.csv'),splits=splits.to_dict('records'),params=rad[clf],seed=rad['seed'],pca=64,code={str(f.name):sha256(f) for f in Path(__file__).parent.glob('*.py')}))
                    if (folder/'run.json').exists() and not args.force:
                        previous=read_json(folder/'run.json')
                        if previous.get('token')!=token: raise ValueError('Stale experiment; use --force')
                        if previous.get('status')=='complete': continue
                    save_json(folder/'run.json',dict(status='running',token=token))
                    metrics=train_embedding(table,splits,rad,clf,folder,args.threads)
                    summary=dict(downsampling=images.downsampling.iloc[0],resolution=args.resolution,cnn_model=name,embedding_dim=DIMS[name],pca_dim=64,classifier=clf,n_samples=len(table),n_splits=rad['cv']['n_splits'])
                    save_json(folder/'run.json',dict(status='complete',token=token,cnn_model=name,classifier=clf,summary=summary,parameters=rad[clf],seed=rad['seed'],n_splits=rad['cv']['n_splits'],split_source=cfg['split_source'],splits_sha256=sha256(cfg['split_source'])))
                    print(name,clf,metrics,flush=True)
        if args.stage in ('train','all') and not args.smoke_test: compare(cfg)
        save_json(out/'logs/last_run.json',dict(status='complete',arguments=vars(args),seconds=time.perf_counter()-start))
    except Exception as e:
        save_json(out/'logs/last_run.json',dict(status='failed',arguments=vars(args),error=str(e),seconds=time.perf_counter()-start))
        logging.exception('CNN baseline stopped; no random-weight fallback'); raise

if __name__=='__main__': main()
