"""A separate original-GT diagnostic, never represented as the paper baseline."""
import argparse,json,time
from pathlib import Path
import torch
from datasets.manifest import ManifestDataset,collate
from models.detectors import build_detector
from training.runtime import ROOT,write_json,seed_all,sha256,code_fingerprint
from evaluation.run import metric_bundle
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    seed_all(2026);torch.set_num_threads(4);device=torch.device(args.device)
    out=ROOT/'results'/args.run_id
    if out.exists():raise ValueError('Preserve prior result; choose new run ID')
    out.mkdir(parents=True)
    weight=ROOT/'checkpoints/public/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
    model=build_detector(size=416,initialization='public_coco',public_weights=weight).to(device).eval()
    ds=ManifestDataset(ROOT/args.manifest,size=416,task='person')
    loader=torch.utils.data.DataLoader(ds,batch_size=2,shuffle=False,num_workers=4,collate_fn=collate)
    pred=[];gt=[];start=time.monotonic()
    try:
        with torch.no_grad():
            for batch in loader:
                x=batch['image'].to(device);outputs=model.predict(x)
                for r,t,det in zip(batch['record'],batch['target'],outputs):
                    pred.append({'image_id':r['id'],'boxes':det['boxes'].cpu().tolist(),'scores':det['scores'].cpu().tolist()})
                    gt.append({'image_id':r['id'],'boxes':t['boxes'].tolist(),'iscrowd':t['iscrowd'].tolist(),
                          'annotation_ids':r.get('original_annotation_ids'),'annotation_source':r['annotations_source']})
                if len(pred)%100==0:write_json(out/'status.json',{'status':'running','images':len(pred),'total':len(ds),'elapsed_seconds':time.monotonic()-start})
        write_json(out/'predictions.json',{'public_pretrained_clean':pred});write_json(out/'ground_truth.json',gt)
        metrics=metric_bundle(pred,gt);write_json(out/'metrics.json',{'public_pretrained_clean':metrics})
        write_json(out/'run.json',{'category':'native_original_GT_diagnostic_not_paper_table','code_sha256':code_fingerprint(),
             'manifest':args.manifest,'manifest_sha256':sha256(ROOT/args.manifest),'public_weights_sha256':sha256(weight),
             'parameters':'R50FPN416; confidence.001; NMS.6; maxDets100; person class1; all original images/GT including negatives and crowd',
             'elapsed_seconds':time.monotonic()-start,'optimization_performed':False,'paper_comparable':False})
        write_json(out/'status.json',{'status':'evaluation_complete','images':len(ds),'elapsed_seconds':time.monotonic()-start,'paper_comparable':False})
    except BaseException as exc:
        write_json(out/'status.json',{'status':'failed','error':repr(exc)});raise
if __name__=='__main__':main()
