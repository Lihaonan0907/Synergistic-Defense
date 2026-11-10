"""Generate independent white-box evidence and evaluate BOTH systems on shared input."""
import argparse,json,shutil,uuid,traceback
from pathlib import Path
import torch
from datasets.manifest import ManifestDataset
from training.runtime import ROOT,write_json,sha256,code_fingerprint,seed_all
from evaluation.run import load_trained_model,metric_bundle
from attacks.whitebox import optimize_image_patch

def run_attack(cfg):
    if cfg.get('blocked_reason'):raise RuntimeError(cfg['blocked_reason'])
    out=ROOT/'results'/cfg['run_id']
    seed_all(cfg.get('seed',2026))
    resume=bool(cfg.get('resume',False))
    if out.exists() and not resume:raise ValueError('Preserve existing attack evidence; use explicit verified resume or new run ID')
    out.mkdir(parents=True,exist_ok=resume)
    binding={'config':{k:v for k,v in cfg.items() if k!='resume'},'checkpoint_sha256':sha256(ROOT/cfg['checkpoint']),
             'manifest_sha256':sha256(ROOT/cfg['manifest']),'code_sha256':code_fingerprint()}
    if resume:
        previous=json.loads((out/'start.json').read_text())
        if previous!=binding:raise ValueError('Attack resume code/data/model/config differs from initial attempt')
    else:
        write_json(out/'start.json',binding)
        for directory in ['models','losses','training','datasets','attacks','evaluation']:
            destination=out/'source'/directory;destination.mkdir(parents=True,exist_ok=True)
            for path in (ROOT/directory).glob('*.py'):shutil.copyfile(path,destination/path.name)
    device=torch.device(cfg.get('device','cuda:0'))
    system,ck=load_trained_model(ROOT/cfg['checkpoint'],device)
    if ck['stage']!='stage2':raise ValueError('White-box attack requires a complete defense checkpoint')
    ds=ManifestDataset(ROOT/cfg['manifest'],cfg.get('size',416),True,'person')
    preds={'no_defense':[],'defense':[]};gt=[];source_ids=set()
    if resume and (out/'progress.json').exists():
        progress=json.loads((out/'progress.json').read_text());preds=progress['predictions'];gt=progress['ground_truth']
        expected=[r['id'] for r in ds.records[:len(gt)]]
        if [r['image_id'] for r in gt]!=expected or any([r['image_id'] for r in values]!=expected for values in preds.values()):
            raise ValueError('Attack progress sample identities differ')
        for i in range(len(gt)):
            sample=json.loads((out/f'{i:06d}/completed.json').read_text())
            if sample['attack_tensors_sha256']!=sha256(out/f'{i:06d}/attack_tensors.pt'):raise ValueError('Corrupt completed optimized patch')
    completed=len(gt)
    for index in range(len(ds)):
        batch=ds[index];r=batch['record']
        if r['source_id'] in source_ids:raise ValueError('White-box manifest repeats source image; prepare explicit unique-source manifest')
        source_ids.add(r['source_id'])
        if index<completed:continue
        image=batch['clean'][None].to(device);target=batch['target']
        boxes=target['boxes']
        if not len(boxes):raise ValueError('Attack placement needs an explicit policy for empty-person image; no silent image deletion')
        area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]);box=boxes[area.argmax()]
        center=[float((box[0]+box[2])/2),float((box[1]+box[3])/2)]
        acfg=dict(cfg.get('attack',{}));acfg['seed']=cfg.get('seed',2026)+index
        result=optimize_image_patch(system,image,center,acfg)
        sample_dir=out/f'{index:06d}'
        if sample_dir.exists():
            archived=out/'partial_attempts'/f'{index:06d}_{uuid.uuid4().hex[:12]}'
            archived.parent.mkdir(exist_ok=True);sample_dir.rename(archived)
        sample_dir.mkdir()
        torch.save({'attacked':result['image'].cpu(),'texture':result['texture'].cpu(),
                    'patch_mask':result['patch_mask'].cpu()},sample_dir/'attack_tensors.pt')
        write_json(sample_dir/'attack.json',{k:v for k,v in result.items() if not isinstance(v,torch.Tensor)})
        with torch.no_grad():
            same_input=result['image']
            no_defense=system.person.predict(same_input)[0]
            defended=system(same_input)['person'][0]
        for key,pred in [('no_defense',no_defense),('defense',defended)]:
            preds[key].append({'image_id':r['id'],'boxes':pred['boxes'].cpu().tolist(),'scores':pred['scores'].cpu().tolist()})
        gt.append({'image_id':r['id'],'boxes':boxes.tolist(),'iscrowd':target['iscrowd'].tolist()})
        write_json(sample_dir/'completed.json',{'image_id':r['id'],'attack_tensors_sha256':sha256(sample_dir/'attack_tensors.pt')})
        write_json(out/'predictions.json',preds);write_json(out/'ground_truth.json',gt)
        write_json(out/'progress.json',{'predictions':preds,'ground_truth':gt,'completed_images':len(gt)})
        write_json(out/'status.json',{'status':'running','completed_images':index+1,'total_images':len(ds)})
    write_json(out/'metrics.json',{k:metric_bundle(v,gt) for k,v in preds.items()})
    write_json(out/'run.json',{'config':cfg,'code_sha256':code_fingerprint(),'checkpoint_sha256':sha256(ROOT/cfg['checkpoint']),
            'manifest_sha256':sha256(ROOT/cfg['manifest']),'same_adversarial_input':True,
            'same_person_detector_weights':True,'image_specific_patches':True,'cross_dataset_texture_transfer':False})
    write_json(out/'status.json',{'status':'complete','completed_images':len(ds),'paper_comparable':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    try:run_attack(cfg)
    except BaseException as exc:
        out=ROOT/'results'/cfg['run_id']
        # Preserve prior successful runs on accidental duplicate invocation.
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if status.get('status')!='complete':
            write_json(out/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc()})
            write_json(out/'status.json',{'status':'failed','error':repr(exc)})
        raise
if __name__=='__main__':main()
