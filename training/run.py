"""Stage1, Stage2 and AT-only training. Formal configs never use tiny networks."""
import argparse,json,math,sys,time,traceback,shutil,uuid
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.manifest import ManifestDataset,collate
from models.localizer import SpectralPatchLocalizer
from models.repair import DualDomainRepair
from models.detectors import build_detector
from models.defense import DefenseSystem
from training.objectives import stage1_objective,stage2_objective
from training.runtime import ROOT,seed_all,sha256,code_fingerprint,write_json,append_jsonl,utcnow,save_checkpoint,load_checkpoint,example_seed,EpochPermutationSampler
from evaluation.metrics import evaluate
from training.gradients import stage1_optimizer


def build_localizer(cfg):return SpectralPatchLocalizer(**cfg.get('localizer',{}))

def wait_for_training(run_id):
    status_path=ROOT/'results'/run_id/'status.json'
    while True:
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status')=='training_complete':break
        if status.get('status') in ['failed','stopped_superseded']:
            raise RuntimeError('Required training unavailable: '+run_id+': '+str(status.get('error',status['status'])))
        time.sleep(20)


def build_system(cfg):
    for key in ['wait_stage1_run','wait_person_run','wait_resource_run']:
        if cfg.get(key):wait_for_training(cfg[key])
    localizer=build_localizer(cfg)
    checkpoint=load_checkpoint(ROOT/cfg['stage1_checkpoint'],allow_smoke=cfg.get('category')=='smoke')
    localizer.load_state_dict(checkpoint['model'],strict=True)
    for p in localizer.parameters():p.requires_grad_(False)
    repair=DualDomainRepair(**cfg.get('repair',{}))
    person_cfg=dict(cfg.get('person',{}))
    if person_cfg.get('public_weights'):person_cfg['public_weights']=str(ROOT/person_cfg['public_weights'])
    person=build_detector(**person_cfg)
    if cfg.get('person_checkpoint'):
        pc=load_checkpoint(ROOT/cfg['person_checkpoint'],allow_smoke=cfg.get('category')=='smoke')
        if pc['stage'] not in ['at_only','clean_only']:raise ValueError('Expected independent person-detector checkpoint')
        person.load_state_dict(pc['model'],strict=True)
    return DefenseSystem(localizer,repair,person,cfg.get('mask_threshold',.25))


def phase_at(epoch,cfg):
    if not cfg.get('progressive',True):return 'joint',1.
    first,transition=cfg.get('phase_epochs',[20,20,40])[:2]
    if epoch<first:return 'initial',0.
    if epoch<first+transition:return 'transition',.01
    return 'joint',.05


def validate(model,loader,device,stage):
    predictions=[];gt=[];losses=[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x=batch['image'].to(device);targets=[{k:v.to(device) for k,v in t.items()} for t in batch['target']]
            if stage=='stage1':
                p=model.predict(x)
                b=model.coefficients(x,enhance=model.mode=='shared_enhancement');loss,_=model.band_loss(b,x.shape[-2:],targets)
            elif stage in ['at_only','clean_only']:
                if stage=='clean_only':
                    if batch['clean'] is None:raise ValueError('Clean-only validation requires actual clean images')
                    x=batch['clean'].to(device)
                p=model.predict(x);loss,_=model.loss(x,targets)
            else:
                outputs=model(x);p=outputs['person'];loss,_=model.person.loss(outputs['repaired'],targets)
            losses.append(float(loss))
            for record,pred,target in zip(batch['record'],p,targets):
                predictions.append({'image_id':record['id'],'boxes':pred['boxes'].cpu().tolist(),'scores':pred['scores'].cpu().tolist()})
                gt.append({'image_id':record['id'],'boxes':target['boxes'].cpu().tolist(),'iscrowd':target['iscrowd'].cpu().tolist()})
    metrics=evaluate(predictions,gt)
    return {'loss':sum(losses)/max(len(losses),1),'metrics':metrics},predictions,gt


def train(cfg,config_path):
    seed_all(cfg['seed'])
    device=torch.device(cfg.get('device','cuda:0'))
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no silent CPU formal training')
    stage=cfg['stage'];run_id=cfg['run_id'];run_dir=ROOT/'results'/run_id
    if (run_dir/'status.json').exists() and not cfg.get('resume'):
        raise ValueError(f'Run exists: {run_id}; choose new ID or resume, never overwrite evidence')
    run_dir.mkdir(parents=True,exist_ok=True)
    attempt_id='resume_'+uuid.uuid4().hex[:12] if cfg.get('resume') else 'initial'
    attempt_dir=run_dir/'resumes'/attempt_id if cfg.get('resume') else run_dir
    attempt_dir.mkdir(parents=True,exist_ok=True)
    cfg['_attempt_id']=attempt_id
    cfg['_code_sha256']=code_fingerprint()
    for folder in ['models','losses','training','datasets','attacks','evaluation']:
        dest=attempt_dir/'source'/folder;dest.mkdir(parents=True,exist_ok=True)
        for src in (ROOT/folder).glob('*.py'):shutil.copyfile(src,dest/src.name)
    ds=ManifestDataset(ROOT/cfg['train_manifest'],cfg.get('size',416),stage in ['stage2','clean_only'],'patch' if stage=='stage1' else 'person')
    vd=ManifestDataset(ROOT/cfg['validation_manifest'],cfg.get('size',416),False,'patch' if stage=='stage1' else 'person')
    if any(r.get('split')!='train' for r in ds.records):raise ValueError('Train manifest contains validation/test records')
    if any(r.get('split')!='validation' for r in vd.records):raise ValueError('Checkpoint selection must use reserved training-origin validation')
    train_sources={r.get('source_id') for r in ds.records};val_sources={r.get('source_id') for r in vd.records}
    if train_sources&val_sources:raise ValueError('Training/validation source leakage')
    if cfg.get('category')=='paper_comparable' and any(r.get('person_gt_provenance')!='verified_original' for r in ds.records+vd.records):
        raise ValueError('Paper-comparable flag requires verified original complete GT and exact paper split')
    batch_size=cfg.get('microbatch',2);accumulate=cfg.get('effective_batch',16)//batch_size
    if accumulate<1 or accumulate*batch_size!=cfg.get('effective_batch',16):raise ValueError('effective_batch must divide microbatch')
    generator=torch.Generator().manual_seed(cfg['seed'])
    sampler=EpochPermutationSampler(ds,cfg['seed']) if cfg.get('example_rng_policy')=='world_independent' else None
    loader=DataLoader(ds,batch_size=batch_size,shuffle=sampler is None,sampler=sampler,num_workers=cfg.get('workers',2),collate_fn=collate,generator=generator,pin_memory=device.type=='cuda')
    vloader=DataLoader(vd,batch_size=cfg.get('validation_batch',2),shuffle=False,num_workers=cfg.get('workers',2),collate_fn=collate)
    if stage=='stage1':model=build_localizer(cfg)
    elif stage=='stage2':model=build_system(cfg)
    elif stage in ['at_only','clean_only']:
        pc=dict(cfg.get('person',{}))
        if pc.get('public_weights'):pc['public_weights']=str(ROOT/pc['public_weights'])
        model=build_detector(**pc)
    else:raise ValueError(stage)
    model.to(device)
    if stage=='stage1':optimizer=stage1_optimizer(model,cfg)
    elif stage=='stage2':
        optimizer=torch.optim.AdamW([{'params':model.repair.parameters(),'lr':cfg.get('repair_lr',1e-4),'name':'repair'},
                     {'params':model.person.parameters(),'lr':cfg.get('person_lr',5e-4),'name':'person'}],weight_decay=1e-4)
    else:optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get('person_lr',5e-4),weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=cfg.get('amp',True) and device.type=='cuda',init_scale=cfg.get('amp_initial_scale',128.))
    best=-math.inf;start_epoch=0
    if cfg.get('resume'):
        ck=load_checkpoint(ROOT/cfg['resume'],allow_smoke=cfg.get('category')=='smoke')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch']+1;best=ck['best_validation_metric']
        torch.set_rng_state(ck['rng']['torch'])
        import random,numpy as np
        random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy'])
        if device.type=='cuda' and ck['rng']['cuda']:torch.cuda.set_rng_state_all(ck['rng']['cuda'])
        # DataLoader generator is reseeded by epoch below, avoiding hidden resume state.
    snapshot={'config':cfg,'command':sys.argv,'config_sha256':sha256(config_path),'code_sha256':code_fingerprint(),
          'train_manifest_sha256':sha256(ROOT/cfg['train_manifest']),'validation_manifest_sha256':sha256(ROOT/cfg['validation_manifest']),
          'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'device':str(device),
          'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
          'parameters':sum(p.numel() for p in model.parameters()),'started_utc':utcnow()}
    snapshot['attempt_id']=attempt_id
    if cfg.get('resume'):
        snapshot['resume_checkpoint_sha256']=sha256(ROOT/cfg['resume'])
        snapshot['resumed_epoch_zero_based']=start_epoch
    write_json(attempt_dir/'run.json',snapshot)
    start=time.monotonic();global_step=start_epoch*math.ceil(len(loader)/accumulate)
    try:
        for epoch in range(start_epoch,cfg['epochs']):
            generator.manual_seed(cfg['seed']+epoch)
            if sampler is not None:sampler.set_epoch(epoch)
            model.train();phase,scale=('stage1',1.) if stage=='stage1' else phase_at(epoch,cfg)
            if stage=='stage2':
                model.localizer.eval()
                for p in model.person.parameters():p.requires_grad_(scale>0)
                optimizer.param_groups[1]['lr']=cfg.get('person_lr',5e-4)*scale
            epoch_start=time.monotonic();running=[];optimizer.zero_grad(set_to_none=True)
            for index,batch in enumerate(loader):
                if cfg.get('example_rng_policy')=='world_independent':
                    if batch_size!=1:raise ValueError('World-independent example RNG requires microbatch1')
                    seed_all(example_seed(cfg['seed'],epoch,index))
                x=batch['image'].to(device,non_blocking=True)
                targets=[{k:v.to(device) for k,v in t.items()} for t in batch['target']]
                group_start=(index//accumulate)*accumulate
                group_examples=min(cfg['effective_batch'],len(ds)-group_start*batch_size)
                factor=len(x)/group_examples
                amp_dtype=cfg.get('amp_dtype','bfloat16' if device.type=='cpu' else 'float16')
                with torch.autocast(device_type=device.type,enabled=scaler.is_enabled(),dtype=getattr(torch,amp_dtype)):
                    if stage=='stage1':loss,terms=stage1_objective(model,x,targets,cfg.get('afal',{}))
                    elif stage in ['at_only','clean_only']:
                        if stage=='clean_only':x=batch['clean'].to(device)
                        loss,components=model.loss(x,targets);terms={'person':loss,**components}
                    else:
                        clean=batch['clean'].to(device)
                        mask_override=None
                        if cfg.get('self_supervised_random_mask',False):
                            # A17 unpaired control: only clean input, generated training
                            # corruption and known random masks. Never read paired attack pixels.
                            mask_override=torch.zeros_like(clean[:,:1]);x=clean.clone()
                            for bi in range(len(x)):
                                side=int(torch.randint(16,65,()).item())
                                top=int(torch.randint(0,max(1,x.shape[-2]-side+1),()).item())
                                left=int(torch.randint(0,max(1,x.shape[-1]-side+1),()).item())
                                mask_override[bi,:,top:top+side,left:left+side]=1
                            x=clean*(1-mask_override)+torch.rand_like(clean)*mask_override
                        loss,terms=stage2_objective(model,x,clean,targets,cfg.get('loss',{}),phase,mask_override)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss*factor).backward();running.append(float(loss.detach()))
                if (index+1)%accumulate==0 or index+1==len(loader):
                    scaler.unscale_(optimizer)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.get('clip_norm',1.),error_if_nonfinite=not scaler.is_enabled())
                    finite=bool(torch.isfinite(norm))
                    groups={}
                    for name,module in ([('localizer',model)] if stage=='stage1' else [('repair',model.repair),('person',model.person)] if stage=='stage2' else [('person',model)]):
                        groups[name]=sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None) if finite else None
                    parameter_gradients={}
                    if stage=='stage1':
                        parameter_gradients={name:float(p.grad.float().norm()) if finite else None for name,p in model.named_parameters()
                            if p.grad is not None and (name=='enhancement.kernel' or name=='fusion.weight' or name.endswith('backbone.body.conv1.weight'))}
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);global_step+=1
                    append_jsonl(run_dir/'train.jsonl',{'epoch':epoch+1,'step':global_step,'batch_index':index+1,'phase':phase,'attempt_id':attempt_id,
                        'loss':loss,'terms':terms,'preclip_grad_norm':norm if finite else None,'postclip_group_norm_sums':groups,
                        'optimizer_step_executed':finite,'amp_scale':scaler.get_scale(),
                        'selected_parameter_grad_norms':parameter_gradients,
                        'lr':[g['lr'] for g in optimizer.param_groups],'elapsed_seconds':time.monotonic()-start})
                    write_json(run_dir/'status.json',{'status':'running','epoch':epoch+1,'epochs':cfg['epochs'],'step':global_step,
                        'batch_index':index+1,'batches':len(loader),'elapsed_seconds':time.monotonic()-start,'last_loss':loss,'phase':phase})
                    if cfg.get('max_steps') and global_step>=cfg['max_steps']:break
            val,pred,gt=validate(model,vloader,device,stage)
            write_json(run_dir/f'validation_epoch_{epoch+1:03d}.json',{'summary':val,'predictions':pred,'ground_truth':gt})
            score=val['metrics']['AP50_coco101']
            if score is None:raise ValueError('Validation has no GT; no valid model-selection criterion')
            improved=score>best
            if improved:best=score
            ckdir=ROOT/'checkpoints'/run_id
            complete=epoch+1==cfg['epochs'] and not cfg.get('max_steps')
            save_checkpoint(ckdir/'last.pth',model,optimizer,scaler,epoch,cfg,stage,run_id,best,complete)
            if improved:save_checkpoint(ckdir/'best.pth',model,optimizer,scaler,epoch,cfg,stage,run_id,best,complete)
            append_jsonl(run_dir/'epochs.jsonl',{'epoch':epoch+1,'attempt_id':attempt_id,'train_loss_mean':sum(running)/len(running),'validation':val,
                'best_validation_AP50':best,'selected':improved,'epoch_seconds':time.monotonic()-epoch_start})
            print(f'{run_id} epoch={epoch+1} loss={sum(running)/len(running):.5f} valAP={score:.5f} seconds={time.monotonic()-epoch_start:.1f}',flush=True)
            if cfg.get('max_steps') and global_step>=cfg['max_steps']:
                write_json(run_dir/'status.json',{'status':'smoke_complete','epoch':epoch+1,'steps':global_step,'formal_training_complete':False,'elapsed_seconds':time.monotonic()-start})
                return
        write_json(run_dir/'status.json',{'status':'training_complete','epochs':cfg['epochs'],'best_validation_AP50':best,
            'elapsed_seconds':time.monotonic()-start,'paper_protocol_verified':False})
    except BaseException as exc:
        write_json(run_dir/'status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start})
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();p=Path(args.config)
    cfg=json.loads(p.read_text());train(cfg,p)

if __name__=='__main__':main()
