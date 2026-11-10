"""Save raw predictions/GT before calculating any independently verifiable metric."""
import argparse,json,time,shutil
from pathlib import Path
import torch
from datasets.manifest import ManifestDataset,collate,read_yolo
from models.defense import boxes_to_mask
from training.run import build_localizer,build_system
from training.runtime import ROOT,load_checkpoint,sha256,write_json,seed_all,code_fingerprint,utcnow
from evaluation.metrics import evaluate,coco_reference
from evaluation.masks import perturb_mask
from evaluation.quality import image_quality


def replace_frozen_person_detector(system,person):
    """Fixed-detector repair control: preserve localization/repair and input path."""
    from models.defense import DefenseSystem
    if not isinstance(system,DefenseSystem):raise ValueError('Person override requires a complete trained defense')
    for parameter in person.parameters():parameter.requires_grad_(False)
    person.eval();system.person=person
    return system


def load_trained_model(path,device,allow_smoke=False):
    # Published inference bundles need no optimizer/RNG or initializer file.
    # Native checkpoints remain supported for training-history evaluations.
    if Path(path).resolve().is_relative_to(ROOT/'weights'):
        from inference.api import load_model
        model,bundle=load_model(path,device)
        return model,{**bundle,'run_id':bundle['metadata']['run_id']}
    checkpoint=load_checkpoint(path,allow_smoke)
    if checkpoint['stage']=='stage1':model=build_localizer(checkpoint['config'])
    elif checkpoint['stage']=='stage2':model=build_system(checkpoint['config'])
    else:
        from models.detectors import build_detector
        cfg=dict(checkpoint['config'].get('person',{}))
        if cfg.get('public_weights'):cfg['public_weights']=str(ROOT/cfg['public_weights'])
        model=build_detector(**cfg)
    model.load_state_dict(checkpoint['model'],strict=True);model.to(device).eval()
    return model,checkpoint


def metric_bundle(pred,gt):
    result=evaluate(pred,gt)
    reference=coco_reference(pred,gt)
    if reference is not None and abs(reference-result['AP50_coco101'])>1e-8:
        raise AssertionError('Independent AP differs from COCO reference; do not publish result')
    return {**result,'pycocotools_AP50':reference}


def evaluate_config(cfg):
    if cfg.get('blocked_reason'):raise RuntimeError(cfg['blocked_reason'])
    seed_all(cfg.get('seed',2026));device=torch.device(cfg.get('device','cuda:0'))
    out=ROOT/'results'/cfg['run_id']
    if (out/'predictions.json').exists():raise ValueError('Evaluation exists; preserve results and use a new ID')
    out.mkdir(parents=True,exist_ok=True)
    source_sha=code_fingerprint()
    for folder in ['models','losses','training','datasets','attacks','evaluation']:
        dest=out/'source'/folder;dest.mkdir(parents=True,exist_ok=True)
        for src in (ROOT/folder).glob('*.py'):shutil.copyfile(src,dest/src.name)
    write_json(out/'status.json',{'status':'running','phase':'loading_model','started_utc':utcnow()})
    write_json(out/'run.json',{'config':cfg,'category':cfg.get('category','independent_candidate_evaluation'),
        'code_sha256':source_sha,'source_snapshot':'source/','manifest_sha256':sha256(ROOT/cfg['manifest']),
        'status':'running','started_utc':utcnow(),'paper_protocol_verified':False})
    model,ck=load_trained_model(ROOT/cfg['checkpoint'],device,cfg.get('category')=='smoke')
    detector_override=None
    if cfg.get('person_override_checkpoint'):
        person,person_ck=load_trained_model(ROOT/cfg['person_override_checkpoint'],device,cfg.get('category')=='smoke')
        if person_ck['stage'] not in ['at_only','clean_only']:
            raise ValueError('Fixed detector must be an independently trained person model, not a localization/defense checkpoint')
        if ck['stage']!='stage2':raise ValueError('Fixed-detector repair control requires Stage2 repair weights')
        model=replace_frozen_person_detector(model,person)
        detector_override={'checkpoint':cfg['person_override_checkpoint'],
            'sha256':sha256(ROOT/cfg['person_override_checkpoint']),'training_run':person_ck['run_id'],
            'stage':person_ck['stage'],'same_detector_for_unrepaired_and_repaired':True}
    ds=ManifestDataset(ROOT/cfg['manifest'],cfg.get('size',416),False,'patch' if ck['stage']=='stage1' else 'person')
    if cfg.get('category')=='paper_comparable' and any(r.get('person_gt_provenance')!='verified_original' for r in ds.records):
        raise ValueError('Unverified dataset cannot be called paper-comparable')
    start=time.monotonic();gt=[];preds={};quality=[];latency=[]
    loader=torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,collate_fn=collate,num_workers=cfg.get('workers',2))
    def append(condition,p,r):
        preds.setdefault(condition,[]).append({'image_id':r['id'],'boxes':p['boxes'].cpu().tolist(),
                       'scores':p['scores'].cpu().tolist(),'labels':p['labels'].cpu().tolist()})
    try:
        with torch.no_grad():
            for index,batch in enumerate(loader):
                x=batch['image'].to(device);r=batch['record'][0];target=batch['target'][0]
                gt.append({'image_id':r['id'],'boxes':target['boxes'].tolist(),'iscrowd':target['iscrowd'].tolist(),
                           'attack':r.get('attack','unspecified'),'dataset':r.get('dataset','unspecified'),
                           'original_annotation':r.get('patch_labels') if ck['stage']=='stage1' else r.get('person_labels',r.get('annotations_source'))})
                if device.type=='cuda':torch.cuda.synchronize()
                tick=time.perf_counter()
                if ck['stage']=='stage1':append('localization',model.predict(x)[0],r)
                elif ck['stage'] in ['at_only','clean_only']:append('unrepaired',model.predict(x)[0],r)
                else:
                    predicted,_=model.mask(x)
                    gt_mask=None
                    conditions=cfg.get('mask_conditions',['predicted'])
                    if 'gt_box_proxy' in conditions:
                        boxes=read_yolo(r['patch_labels'],(r['width'],r['height']),1).to(device)
                        boxes*=boxes.new_tensor([x.shape[-1]/r['width'],x.shape[-2]/r['height']]*2)
                        gt_mask=boxes_to_mask([{'boxes':boxes,'scores':torch.ones(len(boxes),device=device)}],x.shape[-2:])
                    append('unrepaired',model.person.predict(x)[0],r)
                    for condition in conditions:
                        mask=perturb_mask(predicted,condition,gt_mask)
                        repaired=model.repair(x,mask)
                        append('repaired/'+condition,model.person.predict(repaired)[0],r)
                        if cfg.get('quality') and batch['clean'] is not None:
                            quality.append({'image_id':r['id'],'mask_condition':condition,
                                **image_quality(repaired[0],batch['clean'][0].to(device))})
                    if cfg.get('evaluate_clean') and batch['clean'] is not None:
                        append('clean',model.person.predict(batch['clean'].to(device))[0],r)
                if device.type=='cuda':torch.cuda.synchronize()
                latency.append(time.perf_counter()-tick)
                if (index+1)%25==0 or index+1==len(loader):
                    write_json(out/'status.json',{'status':'running','phase':'inference','images_processed':index+1,
                        'images':len(ds),'elapsed_seconds':time.monotonic()-start})
        # Evidence is written before metrics; AP can always be recalculated offline.
        write_json(out/'predictions.json',preds);write_json(out/'ground_truth.json',gt)
        write_json(out/'repair_quality.json',quality)
        summaries={condition:metric_bundle(p,gt) for condition,p in preds.items()}
        write_json(out/'metrics.json',summaries)
        attacks=sorted(set(g['attack'] for g in gt))
        if len(attacks)>1:
            groups={}
            for condition,p in preds.items():
                by_attack={}
                for attack in attacks:
                    subset=[g for g in gt if g['attack']==attack];ids={g['image_id'] for g in subset}
                    by_attack[attack]=metric_bundle([r for r in p if r['image_id'] in ids],subset)
                values=[v['AP50_coco101'] for v in by_attack.values() if v['AP50_coco101'] is not None]
                groups[condition]={'per_attack':by_attack,'unweighted_mean_AP50_coco101':sum(values)/len(values),
                    'pooled_AP50_coco101':summaries[condition]['AP50_coco101'],
                    'protocol':'Pooled image AP and mean attack AP are distinct; neither is selected to match paper numbers'}
            write_json(out/'attack_breakdown.json',groups)
        write_json(out/'run.json',{'config':cfg,'code_sha256':source_sha,'source_snapshot':'source/','checkpoint_sha256':sha256(ROOT/cfg['checkpoint']),
            'manifest_sha256':sha256(ROOT/cfg['manifest']),'model_training_run':ck['run_id'],
            'fixed_person_override':detector_override,
            'elapsed_seconds':time.monotonic()-start,'per_sample_seconds':latency,
            'timing_scope':'all requested conditions together; NOT Table XIII single-forward latency',
            'category':cfg.get('category','independent_candidate_evaluation'),'paper_protocol_verified':False})
        write_json(out/'status.json',{'status':'evaluation_complete','images':len(ds),
            'elapsed_seconds':time.monotonic()-start,'paper_protocol_verified':False})
        print(json.dumps({k:{m:v for m,v in r.items() if not m.endswith('_curve')} for k,r in summaries.items()},indent=2))
    except BaseException as exc:
        write_json(out/'failure.json',{'error':repr(exc),'elapsed_seconds':time.monotonic()-start})
        write_json(out/'status.json',{'status':'failed','error':repr(exc),'elapsed_seconds':time.monotonic()-start});raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--recalculate',type=Path)
    args=p.parse_args()
    if args.recalculate:
        pred=json.loads((args.recalculate/'predictions.json').read_text());gt=json.loads((args.recalculate/'ground_truth.json').read_text())
        write_json(args.recalculate/'metrics_recalculated.json',{k:metric_bundle(v,gt) for k,v in pred.items()})
    elif args.config:
        cfg=json.loads(Path(args.config).read_text())
        evaluate_config(cfg)
    else:p.error('config or recalculate is required')
if __name__=='__main__':main()
