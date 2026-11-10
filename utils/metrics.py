from evaluation.metrics import evaluate,coco_reference

def metric_bundle(predictions,ground_truth):
    if {p['image_id'] for p in predictions}!={g['image_id'] for g in ground_truth}:raise ValueError('Prediction and complete GT image identities differ')
    value=evaluate(predictions,ground_truth);reference=coco_reference(predictions,ground_truth)
    if reference is None or abs(value['AP50_coco101']-reference)>1e-8:raise ValueError('Independent metric and pycocotools disagree')
    return {**value,'pycocotools_AP50':reference}

def compact(value):return {k:v for k,v in value.items() if not k.endswith('_curve')}
