import argparse
from pathlib import Path
import torch,numpy as np
from PIL import Image,ImageDraw
from datasets.manifest import read_rgb
from inference import DefensePredictor
from utils.io import sha256,write_json

def serialize(pred,scale):
    boxes=pred['boxes'].detach().cpu()*torch.tensor(scale)
    return {'boxes':boxes.tolist(),'scores':pred['scores'].cpu().tolist(),'labels':pred['labels'].cpu().tolist()}

def main():
    p=argparse.ArgumentParser(description='RGB to predicted patch/mask, dual-domain repair and person detections')
    p.add_argument('--input',required=True);p.add_argument('--stage1-weights');p.add_argument('--stage2-weights')
    p.add_argument('--config');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads);predict=DefensePredictor(a.stage2_weights,a.device,a.stage1_weights,a.config)
    source=Path(a.input);files=sorted(x for x in source.rglob('*') if x.is_file() and x.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']) if source.is_dir() else [source]
    if not files or any(not x.is_file() for x in files):raise ValueError('Input contains no readable images')
    output=Path(a.output)
    if source.is_dir() and output.resolve().is_relative_to(source.resolve()):raise ValueError('Output must be outside input directory')
    output.mkdir(parents=True,exist_ok=False)
    for path in files:
        relative=path.relative_to(source) if source.is_dir() else Path(path.name);folder=output/relative.parent/(relative.name.replace('.','_'));folder.mkdir(parents=True,exist_ok=False)
        rgb,original=read_rgb(path,predict.size);r=predict(rgb);width,height=original;scale=[width/predict.size,height/predict.size]*2
        patches=serialize(r['patch'][0],scale);persons=serialize(r['person'][0],scale)
        original_rgb=Image.open(path).convert('RGB');original_rgb.save(folder/'original.png')
        mask=r['mask'][0,0].cpu();repaired=r['repaired'][0].cpu()
        torch.save({'predicted_mask':mask,'repaired_unclamped':repaired},folder/'raw_tensors.pt')
        Image.fromarray((mask.numpy()*255).astype('uint8')).resize(original,Image.Resampling.NEAREST).save(folder/'predicted_mask.png')
        image=Image.fromarray((repaired.clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')).resize(original,Image.Resampling.BILINEAR);image.save(folder/'repaired.png')
        person_canvas=image.copy();patch_canvas=original_rgb.copy()
        for canvas,detections,color in [(person_canvas,persons,'lime'),(patch_canvas,patches,'red')]:
            draw=ImageDraw.Draw(canvas)
            for box,score in zip(detections['boxes'],detections['scores']):
                if score>=.25:draw.rectangle(box,outline=color,width=2);draw.text((box[0],box[1]),f'{score:.3f}',fill=color)
        person_canvas.save(folder/'person_detections.png');patch_canvas.save(folder/'patch_localization.png')
        write_json(folder/'results.json',{'input_name':path.name,'input_sha256':sha256(path),'original_size_wh':original,'grid_hw':[predict.size]*2,
            'coordinates':'original image xyxy','patch':patches,'person':persons,'mask_source':'predicted only','GT_or_clean_reference_used':False,
            'stage1_sha256':sha256(predict.stage1_path),'stage2_sha256':sha256(predict.stage2_path),'display_threshold':.25,
            'export_only':'PNG clipping/resizing; detector uses unclamped grid tensor'})
        print(folder,flush=True)

if __name__=='__main__':main()
