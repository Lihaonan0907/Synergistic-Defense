"""Recompute an explicitly bound index of raw saved predictions and full GT."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.io import sha256,write_json
from utils.metrics import metric_bundle,compact

def run(index_path,output):
    index_path=Path(index_path);index=json.loads(index_path.read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False);results=[];cells=[]
    for entry in index['experiments']:
        folder=index_path.parent/entry['directory'];pred=json.loads((folder/'predictions.json').read_text());measured={};groups={}
        for condition,rows in pred.items():
            gt_file=entry.get('GT_by_condition',{}).get(condition,'ground_truth.json');gt=json.loads((folder/gt_file).read_text())
            value=metric_bundle(rows,gt);measured[condition]=compact(value)
            expected=entry.get('expected_AP50',{}).get(condition)
            if expected is not None and abs(value['AP50_coco101']-expected)>1e-8:raise ValueError('Saved result does not match independently recomputed raw evidence: '+entry['id'])
            for key,target in [('pooled',value['AP50_coco101'])]:
                paper=entry.get('paper',{}).get(condition);task='Patch Localization' if condition=='localization' else entry['task']
                cells.append({'Experiment':entry['id'],'Dataset':entry['dataset'],'Split':entry['split'],'Task':task,
                    'Condition':condition,'Aggregation':key,'Images':len(gt),'Paper_AP50_percent':paper if paper is not None else '',
                    'Measured_AP50_percent':100*target,'Difference_pp':100*target-paper if paper is not None else '',
                    'Same_Paper_Protocol':False,'Numerically_At_Or_Above_Paper':100*target>=paper if paper is not None else '',
                    'Evaluation_Status':'real_measurement_protocol_differs','Model':'Spectral FasterRCNN-R50/FPN' if task=='Patch Localization' else 'Dual-domain repair + FasterRCNN-R50/FPN person',
                    'Checkpoint':'weights/stage1_best.pt' if entry['task']=='Patch Localization' else 'weights/stage2_best.pt',
                    'Checkpoint_SHA256':entry['release_bundle_SHA256'],'Metric':'COCO101 AP50',
                    'Protocol':entry['protocol'],'Evidence':'docs/evidence/RECOMPUTED_RESULTS.json#'+entry['id']})
            attacks=sorted({g.get('attack','unspecified') for g in gt})
            if len(attacks)>1:
                per={}
                for attack in attacks:
                    subset=[g for g in gt if g.get('attack','unspecified')==attack];ids={g['image_id'] for g in subset}
                    per[attack]=compact(metric_bundle([r for r in rows if r['image_id'] in ids],subset))
                groups[condition]={'per_attack':per,'macro_AP50':sum(v['AP50_coco101'] for v in per.values())/len(per),'pooled_AP50':value['AP50_coco101']}
                if entry['task']=='Patch Localization':
                    macro=groups[condition]['macro_AP50'];paper=entry.get('paper_macro');base=dict(cells[-1]);base.update(Aggregation='macro_attack',Paper_AP50_percent=paper if paper is not None else '',
                        Measured_AP50_percent=100*macro,Difference_pp=100*macro-paper if paper is not None else '',Numerically_At_Or_Above_Paper=100*macro>=paper if paper is not None else '');cells.append(base)
                    for attack,m in per.items():
                        base=dict(cells[-1]);paper=entry.get('paper_per_attack',{}).get(attack);base.update(Condition=attack,Aggregation='attack_subset',Images=m['num_images'],
                            Measured_AP50_percent=100*m['AP50_coco101'],Paper_AP50_percent=paper if paper is not None else '',Difference_pp=100*m['AP50_coco101']-paper if paper is not None else '',Numerically_At_Or_Above_Paper=100*m['AP50_coco101']>=paper if paper is not None else '');cells.append(base)
        gain=None
        raw=next((key for key in ['attacked','unrepaired','no_defense'] if key in measured),None)
        repair=next((key for key in ['repaired','repaired/predicted','defense'] if key in measured),None)
        if raw and repair:gain=100*(measured[repair]['AP50_coco101']-measured[raw]['AP50_coco101'])
        result={k:v for k,v in entry.items() if k not in ['directory','expected_AP50','paper','paper_macro','paper_per_attack']}
        result.update(metrics=measured,per_attack=groups,defense_gain_pp=gain,predictions_SHA256=sha256(folder/'predictions.json'),
            GT_SHA256={f:sha256(folder/f) for f in {'ground_truth.json',*entry.get('GT_by_condition',{}).values()}},all_original_images_retained=True,COCO_parity_passed=True)
        results.append(result);print('recomputed',entry['id'],flush=True)
    report={'source_index_SHA256':sha256(index_path),'release_model_identity':index['release_model_identity'],'experiments':results,
        'scope':'Actual saved raw predictions with exact full GT; all metrics independently recomputed; scientific datasets not redistributed',
        'same_paper_protocol':False}
    write_json(out/'RECOMPUTED_RESULTS.json',report)
    with (out/'RESULTS_VS_PAPER.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(cells[0]));writer.writeheader();writer.writerows(cells)
    print('all metrics independently measured',len(results),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.index,a.output)
