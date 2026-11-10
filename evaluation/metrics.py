"""Independent single-class AP50 + counts; optional COCO reference validation."""
import contextlib,io
import numpy as np


def overlap(a,b,crowd=False):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa if crowd else aa+bb-inter,1e-12)


def evaluate(predictions,ground_truth,iou=.5,count_confidence=.25,max_dets=100):
    gt_by={str(g['image_id']):g for g in ground_truth}
    pred_by={str(p['image_id']):p for p in predictions}
    if len(gt_by)!=len(ground_truth) or len(pred_by)!=len(predictions):raise ValueError('Duplicate image IDs')
    if set(pred_by)-set(gt_by):raise ValueError('Prediction image absent from GT manifest')
    npos=0;events=[]
    for image_id,g in gt_by.items():
        boxes=g.get('boxes',[]);crowd=g.get('iscrowd',[0]*len(boxes));ignore=g.get('ignore',crowd)
        order_gt=sorted(range(len(boxes)),key=lambda j:bool(ignore[j] or crowd[j]))
        ignored=[bool(ignore[j] or crowd[j]) for j in range(len(boxes))]
        npos+=sum(not v for v in ignored)
        used=set()
        p=pred_by.get(image_id,{'boxes':[],'scores':[]})
        if len(p['boxes'])!=len(p['scores']):raise ValueError('boxes/scores length mismatch')
        order=np.argsort(-np.asarray(p['scores']),kind='stable')[:max_dets]
        for di in order:
            best=None;best_iou=iou-1e-12
            for gi in order_gt:
                if gi in used and not crowd[gi]:continue
                if best is not None and not ignored[best] and ignored[gi]:break
                ov=overlap(p['boxes'][di],boxes[gi],bool(crowd[gi]))
                if ov>=best_iou:best,best_iou=gi,ov
            if best is None:events.append((float(p['scores'][di]),0,1))
            else:
                used.add(best)
                if not ignored[best]:events.append((float(p['scores'][di]),1,0))
    events.sort(key=lambda x:-x[0])
    scores=np.asarray([x[0] for x in events]);tp=np.asarray([x[1] for x in events]);fp=np.asarray([x[2] for x in events])
    ctp=np.cumsum(tp);cfp=np.cumsum(fp)
    recall=ctp/max(npos,1);precision=ctp/np.maximum(ctp+cfp,1)
    envelope=np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    ap101=float(np.mean([np.max(precision[recall>=r],initial=0) for r in np.linspace(0,1,101)])) if npos else None
    rr=np.concatenate([[0],recall,[1]]);pp=np.concatenate([[0],envelope,[0]])
    ap_cont=float(np.sum((rr[1:]-rr[:-1])*pp[1:])) if npos else None
    keep=scores>=count_confidence
    ntp,nfp=int(tp[keep].sum()),int(fp[keep].sum());nfn=npos-ntp
    pr=ntp/max(ntp+nfp,1);re=ntp/max(npos,1)
    return {'AP50_coco101':ap101,'AP50_continuous':ap_cont,'TP':ntp,'FP':nfp,'FN':nfn,
            'precision':pr,'recall':re,'F1':2*pr*re/max(pr+re,1e-12),'num_gt':npos,
            'num_images':len(gt_by),'iou_threshold':iou,'count_confidence':count_confidence,
            'max_dets':max_dets,'crowd_policy':'COCO crowd IoA; ignored matches excluded',
            'recall_curve':recall.tolist(),'precision_curve':precision.tolist()}


def coco_reference(predictions,ground_truth,iou=.5):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ids={str(g['image_id']):i+1 for i,g in enumerate(ground_truth)}
    ds={'images':[{'id':v} for v in ids.values()],'categories':[{'id':1,'name':'target'}],'annotations':[],'info':{}}
    def xywh(b):return [b[0],b[1],b[2]-b[0],b[3]-b[1]]
    for g in ground_truth:
        for j,b in enumerate(g['boxes']):
            box=xywh(b);ds['annotations'].append({'id':len(ds['annotations'])+1,'image_id':ids[str(g['image_id'])],
                'category_id':1,'bbox':box,'area':box[2]*box[3],'iscrowd':g.get('iscrowd',[0]*len(g['boxes']))[j]})
    detections=[{'image_id':ids[str(p['image_id'])],'category_id':1,'bbox':xywh(b),'score':s}
                for p in predictions for b,s in zip(p['boxes'],p['scores'])]
    if not detections:return 0. if ds['annotations'] else None
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO();coco.dataset=ds;coco.createIndex();dt=coco.loadRes(detections)
        ev=COCOeval(coco,dt,'bbox');ev.params.iouThrs=np.asarray([iou]);ev.evaluate();ev.accumulate()
    p=ev.eval['precision'][0,:,0,0,-1]
    return float(p[p>=0].mean()) if (p>=0).any() else None
