"""GT-free complete deployment entry. The only inputs are image and model weights."""
import argparse,json
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
import torch
from datasets.manifest import read_rgb
from evaluation.run import load_trained_model
from training.runtime import ROOT,write_json,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--allow-smoke',action='store_true')
    args=p.parse_args();device=torch.device(args.device)
    out=(ROOT/args.output).resolve()
    if not out.is_relative_to(ROOT):raise ValueError('Output must be in new project')
    out.mkdir(parents=True,exist_ok=True)
    system,ck=load_trained_model(ROOT/args.checkpoint,device,args.allow_smoke)
    if ck['stage']!='stage2':raise ValueError('Complete inference requires Stage2 system')
    x,original=read_rgb(args.image,416 if ck['config'].get('size',416)==416 else ck['config']['size'])
    with torch.no_grad():result=system(x[None].to(device))
    repaired=result['repaired'][0].cpu();mask=result['mask'][0,0].cpu()
    torch.save({'repaired_unclamped':repaired,'mask':mask},out/'raw_tensors.pt')
    Image.fromarray((mask.numpy()*255).astype(np.uint8)).save(out/'predicted_mask.png')
    rgb=(repaired.clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8)
    image=Image.fromarray(rgb);image.save(out/'repaired_export.png')
    draw=ImageDraw.Draw(image);pred=result['person'][0]
    for box,score in zip(pred['boxes'].cpu().tolist(),pred['scores'].cpu().tolist()):
        if score>=.25:draw.rectangle(box,outline='lime',width=2)
    image.save(out/'person_overlay.png')
    write_json(out/'prediction.json',{'boxes':pred['boxes'],'scores':pred['scores'],'labels':pred['labels'],
        'coordinate_grid':list(x.shape[-2:]),'original_size':list(original),'input_sha256':sha256(args.image),
        'checkpoint_sha256':sha256(ROOT/args.checkpoint),'clean_reference_used':False,'model_updates':False,
        'image_clipping':'PNG export only; raw repaired tensor and detector input are unclamped'})
if __name__=='__main__':main()
