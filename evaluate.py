"""Fixed COCO101 AP50 evaluation and offline raw prediction recalculation."""
import argparse,json
from pathlib import Path
import torch
from datasets.manifest import ManifestDataset
from inference.api import DefensePredictor,load_model,load_settings,ROOT
from utils.io import sha256,write_json
from utils.metrics import metric_bundle,compact

def record(pred,id):return {'image_id':id,'boxes':pred['boxes'].cpu().tolist(),'scores':pred['scores'].cpu().tolist()}

def recalculate(source,output=None):
    source=Path(source);pred=json.loads((source/'predictions.json').read_text());gt=json.loads((source/'ground_truth.json').read_text())
    metrics={key:metric_bundle(rows,gt) for key,rows in pred.items()}
    target=Path(output) if output else source
    if output:target.mkdir(parents=True,exist_ok=False)
    write_json(target/'metrics_recalculated.json',metrics)
    # Per-attack evaluation retains fixed order, including a separate macro.
    groups={}
    attacks=sorted({g.get('attack','unspecified') for g in gt})
    if len(attacks)>1:
        for key,rows in pred.items():
            per={}
            for attack in attacks:
                subset=[g for g in gt if g.get('attack','unspecified')==attack];ids={g['image_id'] for g in subset}
                per[attack]=compact(metric_bundle([p for p in rows if p['image_id'] in ids],subset))
            groups[key]={'per_attack':per,'macro_AP50':sum(m['AP50_coco101'] for m in per.values())/len(per),'pooled_AP50':metrics[key]['AP50_coco101']}
        write_json(target/'attack_breakdown_recalculated.json',groups)
    return metrics,groups

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');p.add_argument('--checkpoint')
    p.add_argument('--stage1-weights');p.add_argument('--stage2-weights');p.add_argument('--task',choices=['patch','person'],default='person')
    p.add_argument('--output');p.add_argument('--recalculate');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    if a.recalculate:
        metrics,_=recalculate(a.recalculate,a.output);print(json.dumps({k:compact(v) for k,v in metrics.items()}));return
    if not a.manifest or not a.output:p.error('--manifest and --output required')
    settings=load_settings(a.config)
    if a.task=='patch':
        path=Path(a.checkpoint or a.stage1_weights or ROOT/settings['stage1_weights']);model,bundle=load_model(path,a.device,'stage1');size=bundle['config']['size']
    else:
        predictor=DefensePredictor(a.checkpoint or a.stage2_weights,a.device,a.stage1_weights,a.config);model=predictor.model;path=predictor.stage2_path;size=predictor.size
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);ds=ManifestDataset(a.manifest,size,False,a.task);predictions={};gt=[]
    with torch.inference_mode():
        for i,batch in enumerate(ds):
            x=batch['image'][None].to(a.device);r=batch['record'];target=batch['target']
            if a.task=='patch':outputs={'localization':model.predict(x)[0]}
            else:
                repaired=predictor(x);outputs={'attacked':model.person.predict(x)[0],'repaired':repaired['person'][0]}
                if batch['clean'] is not None:outputs['clean']=model.person.predict(batch['clean'][None].to(a.device))[0]
            for key,value in outputs.items():predictions.setdefault(key,[]).append(record(value,r['id']))
            gt.append({'image_id':r['id'],'boxes':target['boxes'].tolist(),'iscrowd':target['iscrowd'].tolist(),'attack':r.get('attack','unspecified')})
            if (i+1)%50==0:write_json(output/'status.json',{'status':'running','images':i+1,'total':len(ds)})
    write_json(output/'predictions.json',predictions);write_json(output/'ground_truth.json',gt)
    metrics,groups=recalculate(output);write_json(output/'metrics.json',metrics)
    write_json(output/'protocol.json',{'checkpoint_sha256':sha256(path),'manifest_sha256':sha256(a.manifest),'images':len(ds),'task':a.task,
        'mask':'predicted only','same_person_weights_and_GT':True,'GT_or_clean_reference_used_in_defense':False,
        'metric':'COCO101 AP50; maxDets100; fixed score.001/NMS.6; mask.25; macro and pooled separately'})
    write_json(output/'status.json',{'status':'evaluation_complete','images':len(ds)})
    print(json.dumps({k:compact(v) for k,v in metrics.items()}))

if __name__=='__main__':main()
