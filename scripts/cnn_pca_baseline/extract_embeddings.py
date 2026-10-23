"""Sequential CPU extraction; strict content-addressed, model-specific caches."""
import gc
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scripts.radiomics_hsv_baseline.common import META, save_csv, save_json, sha256, digest
from .dataset import read_json, read_csv

DIMS = {'efficientnet_b0':1280, 'resnet50':2048, 'densenet121':1024}

def specification(name, cfg):
    import torch
    import torchvision
    from torchvision import models
    enums = {'efficientnet_b0':models.EfficientNet_B0_Weights, 'resnet50':models.ResNet50_Weights, 'densenet121':models.DenseNet121_Weights}
    weights = enums[name].DEFAULT
    transform = weights.transforms()
    torch.hub.set_dir(str(Path(cfg['weight_cache']).resolve()))
    checkpoint = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[1]
    if checkpoint.exists() and not sha256(checkpoint).startswith(checkpoint.stem.rsplit('-',1)[1]):
        raise ValueError(f'Checkpoint does not match official weight hash: {checkpoint}')
    return weights, transform, checkpoint, dict(model=name, weights=str(weights), weight_url=weights.url, torch=torch.__version__, torchvision=torchvision.__version__, transform=repr(transform), input_crop=transform.crop_size, resize=transform.resize_size, dim=DIMS[name], extraction_version=1)

def contract(images,name,cfg):
    weights, transform, checkpoint, spec = specification(name,cfg)
    spec['checkpoint_sha256'] = sha256(checkpoint) if checkpoint.exists() else None
    spec['source_sha256'] = sha256(__file__)
    spec['inputs'] = [dict(sample_id=r.sample_id, patient_id=r.patient_id, label=r.label, path=str(Path(r.path).resolve()), sha256=sha256(r.path)) for r in images.itertuples()]
    spec['configuration'] = str(images.iloc[0]['config'])
    return spec

def validate_table(table,images,name):
    cols = [f'EMB_{i:04d}' for i in range(DIMS[name])]
    if list(table.columns) != META+cols or table.sample_id.duplicated().any() or set(table.sample_id)!=set(images.sample_id):
        raise ValueError('Embedding schema or sample set mismatch')
    table = table.set_index('sample_id').loc[images.sample_id].reset_index()
    pd.testing.assert_frame_equal(table[META].reset_index(drop=True),images[META].reset_index(drop=True))
    if not np.isfinite(table[cols].to_numpy()).all(): raise ValueError('Nonfinite embedding')
    return table

def cached(images,name,cfg):
    folder = Path(cfg['output_root'])/'embeddings'/images.iloc[0]['config']
    path = folder/f'{name}.csv'
    meta = read_json(path.with_suffix('.json'))
    if meta['token'] != digest(contract(images,name,cfg)) or meta['csv_sha256'] != sha256(path):
        raise ValueError(f'Stale or modified embedding cache: {path}; extract --force to rebuild')
    return validate_table(read_csv(path),images,name)

class Images:
    def __init__(self,images,transform): self.images,self.transform=images,transform
    def __len__(self): return len(self.images)
    def __getitem__(self,index):
        row=self.images.iloc[index]
        try:
            with Image.open(row.path) as im: return self.transform(im.convert('RGB'))
        except Exception as e: raise RuntimeError(f'Failed sample {row.sample_id}: {row.path}: {e}') from e

def extract(images,name,cfg,batch_size=4,num_workers=0,force=False,smoke_only=False):
    import torch
    from torchvision import models
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    start=time.perf_counter()
    folder=Path(cfg['output_root'])/'embeddings'/images.iloc[0]['config']
    path=folder/f'{name}.csv'
    qc=Path(cfg['output_root'])/'qc'/images.iloc[0]['config']/name
    if path.exists() and not force and not smoke_only:
        table=cached(images,name,cfg)
        print(f'Reused verified {path}',flush=True)
        return table
    weights,transform,checkpoint,spec=specification(name,cfg)
    # weights argument always official; torchvision uses a local official checkpoint if present.
    model=getattr(models,name)(weights=weights).cpu()
    if not sha256(checkpoint).startswith(checkpoint.stem.rsplit('-',1)[1]):
        raise ValueError('Downloaded checkpoint does not match official weight hash')
    if name=='resnet50': model.fc=torch.nn.Identity()
    else: model.classifier=torch.nn.Identity()
    model.eval()
    for param in model.parameters(): param.requires_grad=False
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    dataset=Images(images,transform)
    x=dataset[0].unsqueeze(0)
    with torch.inference_mode():
        a,b=model(x),model(x)
    if a.shape!=(1,DIMS[name]) or not torch.isfinite(a).all(): raise ValueError('Incorrect embedding dimension or nonfinite output')
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    save_json(qc/'smoke.json',dict(**spec, actual_dimension=a.shape[1], actual_input=list(x.shape), frozen=True, repeat_max_abs=float((a-b).abs().max()), pretrained=True))
    # Show the retained center-crop rectangle on full input, without changing inference input.
    crop_records=[]
    for index,row in enumerate(images.itertuples()):
        with Image.open(row.path) as im:
            w,h=im.size
            scale=transform.resize_size[0]/min(w,h)
            cw=transform.crop_size[-1]/scale; ch=transform.crop_size[0]/scale
            box=((w-cw)/2,(h-ch)/2,(w+cw)/2,(h+ch)/2)
            crop_records.append(dict(sample_id=row.sample_id,width=w,height=h,retained_area_fraction=min(cw,w)*min(ch,h)/(w*h),potential_peripheral_tissue_loss=True))
            if index<3:
                preview=im.convert('RGB'); ImageDraw.Draw(preview).rectangle(box,outline='red',width=max(2,w//150)); preview.thumbnail((700,700)); qc.mkdir(parents=True,exist_ok=True); preview.save(qc/f'crop_preview_{index}.png')
    save_csv(pd.DataFrame(crop_records),qc/'crop_geometry.csv')
    if smoke_only:
        del model; gc.collect(); return None
    try:
        values=[]
        with torch.inference_mode():
            for batch in tqdm(DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False),desc=name):
                z=model(batch.cpu()).numpy()
                if z.shape[1]!=DIMS[name] or not np.isfinite(z).all(): raise ValueError('Invalid embedding batch')
                values.append(z)
        table=pd.concat([images[META].reset_index(drop=True),pd.DataFrame(np.concatenate(values),columns=[f'EMB_{i:04d}' for i in range(DIMS[name])])],axis=1)
        validate_table(table,images,name)
        spec=contract(images,name,cfg)
        save_csv(table,path)
        save_json(path.with_suffix('.json'),dict(token=digest(spec),contract=spec,csv_sha256=sha256(path),n_samples=len(table),embedding_dim=DIMS[name],created_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start))
        save_json(qc/'extraction.json',dict(status='complete',failed_samples=[],seconds=time.perf_counter()-start))
        return table
    except Exception as e:
        save_json(qc/'extraction.json',dict(status='failed',error=str(e),seconds=time.perf_counter()-start)); raise
    finally:
        del model; gc.collect()

