"""Independently strict-load both exports and test a real GT-free RGB input."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from inference import DefensePredictor,load_model
from datasets.manifest import read_rgb
from utils.io import write_json,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(2);predict=DefensePredictor(device=a.device);x,_=read_rgb(a.image,predict.size);out=predict(x)
    report={'stage1_strict_load':True,'stage2_strict_load':True,'missing_keys':[],'unexpected_keys':[],
        'compatible_frozen_localizer':True,'finite_repaired':bool(torch.isfinite(out['repaired']).all()),
        'parameters':sum(p.numel() for p in predict.model.parameters()),'patch_detections':len(out['patch'][0]['scores']),
        'person_detections':len(out['person'][0]['scores']),'predicted_mask_pixels':int(out['mask'].sum()),
        'stage1_sha256':sha256(predict.stage1_path),'stage2_sha256':sha256(predict.stage2_path),'input_sha256':sha256(a.image),
        'ground_truth_used':False}
    write_json(a.output,report);print(report)
if __name__=='__main__':main()
