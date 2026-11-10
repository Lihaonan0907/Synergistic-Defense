"""Validate user-provided source-separated manifests; never invent annotations."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from datasets.manifest import ManifestDataset,load_jsonl
from utils.io import sha256,write_json

def prepare(source,target,stage='both'):
    source=Path(source).resolve();records=load_jsonl(source);splits={k:[] for k in ['train','validation','test']};known={}
    if len({r['id'] for r in records})!=len(records):raise ValueError('Duplicate sample id')
    for r in records:
        split=r.get('split');sid=r.get('source_id')
        if split not in splits or not sid:raise ValueError('Each record requires an explicit split and original source_id')
        if sid in known and known[sid]!=split:raise ValueError('Original source image crosses dataset splits')
        known[sid]=split
        if stage in ['both','stage1'] and not (r.get('patch_labels') or 'patch_boxes_xyxy' in r):raise ValueError('Stage1 requires original patch annotations; no generated fallback')
        if stage in ['both','stage2'] and not (r.get('person_labels') or 'person_boxes_xyxy' in r):raise ValueError('Stage2 requires complete original person annotations; no generated fallback')
        row=dict(r)
        for key in ['image','clean','patch_labels','person_labels']:
            if row.get(key):
                path=Path(row[key]);path=path if path.is_absolute() else source.parent/path
                if not path.is_file():raise FileNotFoundError(path)
                row[key]=os.path.relpath(path,Path(target).resolve())
        if stage in ['both','stage2'] and split in ['train','validation'] and not row.get('clean'):raise ValueError('Stage2 training/validation requires paired clean images')
        splits[split].append(row)
    target=Path(target)
    target.mkdir(parents=True,exist_ok=True)
    # Separate Stage1/2 corpora can share a manifest folder, but may not put an
    # original source into different splits or overwrite an earlier manifest.
    for existing in target.glob('stage*_*.jsonl'):
        for row in load_jsonl(existing):
            if row['source_id'] in known and known[row['source_id']]!=row['split']:raise ValueError('Original source crosses splits between stages')
    stages=['stage1','stage2'] if stage=='both' else [stage]
    pending=[target/f'{st}_{split}.jsonl' for st in stages for split,rows in splits.items() if rows]
    if any(p.exists() for p in pending):raise FileExistsError('Preserve existing manifest; use a new directory or stage')
    for split,rows in splits.items():
        if not rows:continue
        for st in stages:
            path=target/f'{st}_{split}.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            dataset=ManifestDataset(path,416,st=='stage2' and split!='test','patch' if st=='stage1' else 'person')
            for i in range(len(dataset)):dataset[i]
    write_json(target/('preparation.json' if stage=='both' else 'preparation_'+stage+'.json'),{'input_manifest_sha256':sha256(source),'counts':{k:len(v) for k,v in splits.items()},
        'source_split_overlap':False,'scope':'Validates supplied identity and annotations; cannot certify unknown original sources or license permissions'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--stage',choices=['both','stage1','stage2'],default='both');a=p.parse_args();prepare(a.input,a.output_dir,a.stage)
if __name__=='__main__':main()
