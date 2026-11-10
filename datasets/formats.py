"""Original annotation parsers. They preserve GT; no predictions/pseudo labels become GT."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def coco_records(annotation_file,image_root,include_empty=True):
    annotation_file=Path(annotation_file);data=json.loads(annotation_file.read_text())
    anns={int(i['id']):[] for i in data['images']}
    for a in data['annotations']:
        if int(a['category_id'])==1:anns[int(a['image_id'])].append(a)
    records=[]
    for im in data['images']:
        targets=anns[int(im['id'])]
        if not targets and not include_empty:continue
        path=Path(image_root)/im['file_name']
        if not path.is_file():raise FileNotFoundError(path)
        boxes=[]
        for a in targets:
            x,y,w,h=a['bbox']
            if w<=0 or h<=0:raise ValueError(f'Invalid original annotation {a["id"]}; do not silently remove it')
            boxes.append([x,y,x+w,y+h])
        records.append({'id':f'COCO_original/{im["id"]}','image':str(path),'clean':str(path),
            'dataset':'COCO_original','attack':'clean','split':'test_candidate','raw_split':'official_annotation_file',
            'width':im['width'],'height':im['height'],'person_boxes_xyxy':boxes,
            'person_iscrowd':[int(a.get('iscrowd',0)) for a in targets],
            'original_annotation_ids':[a['id'] for a in targets],'annotations_source':str(annotation_file),
            'person_gt_available':True,'person_gt_provenance':'original_COCO_annotations',
            'source_id':'COCO/'+str(im['id']),'include_negative_images':include_empty})
    return records


def inria_annotation(path,coordinate_origin=0):
    """Preserve all listed persons. Origin convention is explicit, not guessed."""
    # Original INRIA files contain Latin-1 text (e.g. "Rhône-Alpes").
    # Box coordinates are ASCII; Latin-1 preserves the original header as well.
    text=Path(path).read_text(encoding='latin-1')
    boxes=[]
    pattern=r'Bounding box for object\s+\d+[^\n]*:\s*\(([-\d]+),\s*([-\d]+)\)\s*-\s*\(([-\d]+),\s*([-\d]+)\)'
    for match in re.finditer(pattern,text):
        x1,y1,x2,y2=map(int,match.groups())
        boxes.append([x1-coordinate_origin,y1-coordinate_origin,x2,y2])
    count=re.search(r'Objects with ground truth\s*:\s*(\d+)',text)
    if count and int(count.group(1))!=len(boxes):raise ValueError('INRIA person-count parse mismatch')
    return boxes


def citypersons_records(annotation_file,image_root,person_class=1,ignore_classes=()):
    """Native full boxes; class/ignore policy must be declared by caller."""
    data=loadmat(annotation_file,simplify_cells=True)
    key=next(k for k in data if k.startswith('anno_') and not k.startswith('__'))
    entries=data[key]
    if isinstance(entries,dict):entries=[entries]
    result=[]
    for entry in entries:
        path=Path(image_root)/entry['cityname']/entry['im_name']
        if not path.is_file():raise FileNotFoundError(path)
        bbs=np.asarray(entry['bbs']).reshape(-1,10)
        selected=[b for b in bbs if int(b[0])==person_class or int(b[0]) in ignore_classes]
        boxes=[[float(b[1]),float(b[2]),float(b[1]+int(b[3])),float(b[2]+int(b[4]))] for b in selected]
        result.append({'id':'CityPersons_original/'+entry['im_name'],'image':str(path),'clean':str(path),
            'dataset':'CityPersons_original','attack':'clean','split':'test_candidate','raw_split':key,
            'person_boxes_xyxy':boxes,'person_iscrowd':[int(int(b[0]) in ignore_classes) for b in selected],
            'native_classes':[int(b[0]) for b in selected],'person_gt_available':True,
            'person_gt_provenance':'original_CityPersons_annotations','source_id':entry['im_name'],
            'annotations_source':str(annotation_file),'class_policy':{'person_class':person_class,'ignore_classes':list(ignore_classes)}})
    return result
