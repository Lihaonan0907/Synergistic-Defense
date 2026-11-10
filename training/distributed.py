"""Literal higher-order Stage1 with manually summed replica gradients.

DDP's backward hooks are deliberately avoided: vulnerability uses autograd.grad
and second derivatives. Each example contributes loss/global_group_count; sum
replica gradients BEFORE unscale, global norm clipping and one optimizer step.
No training sample is padded, duplicated or dropped at an epoch boundary.
"""
import argparse,json,math,os,shutil,sys,time,traceback,uuid
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
from datasets.manifest import ManifestDataset,collate
from training.run import build_localizer,validate
from training.objectives import stage1_objective
from training.runtime import ROOT,seed_all,sha256,code_fingerprint,write_json,append_jsonl,save_checkpoint,load_checkpoint,utcnow,example_seed
from evaluation.metrics import evaluate
from training.gradients import clip_gradients,stage1_optimizer


@torch.no_grad()
def sum_replica_gradients(parameters):
    parameters=[p for p in parameters if p.requires_grad]
    if not parameters:return
    flat=torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])
    if dist.is_initialized():dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    offset=0
    for parameter in parameters:
        value=flat[offset:offset+parameter.numel()].view_as(parameter)
        if parameter.grad is None:parameter.grad=value.clone()
        else:parameter.grad.copy_(value)
        offset+=parameter.numel()


@torch.no_grad()
def replica_parameter_difference(parameters):
    flat=torch.cat([p.detach().reshape(-1) for p in parameters])
    reference=flat.clone()
    if dist.is_initialized():dist.broadcast(reference,src=0)
    delta=(flat-reference).abs().max()
    if dist.is_initialized():dist.all_reduce(delta,op=dist.ReduceOp.MAX)
    return float(delta)


def source_snapshot(destination):
    for folder in ['models','losses','training','datasets','attacks','evaluation']:
        target=destination/folder;target.mkdir(parents=True,exist_ok=True)
        for path in (ROOT/folder).glob('*.py'):shutil.copyfile(path,target/path.name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    path=Path(args.config);cfg=json.loads(path.read_text())
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    backend=cfg.get('distributed_backend','nccl')
    device=torch.device('cpu' if backend=='gloo' else f'cuda:{local}')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group(backend=backend)
    seed_all(cfg['seed'])
    if cfg['stage']!='stage1' or cfg.get('microbatch',1)!=1:raise ValueError('This executor supports Stage1 microbatch1 only')
    effective=int(cfg['effective_batch'])
    if effective%world:raise ValueError('Effective batch must be divisible by replicas')
    accumulation=effective//world
    run=cfg['run_id'];output=ROOT/'results'/run;checkpoint_dir=ROOT/'checkpoints'/run
    resuming=bool(cfg.get('resume'))
    attempt_name=('resume_'+uuid.uuid4().hex[:12]) if rank==0 and resuming else 'initial'
    if world>1:
        names=[attempt_name];dist.broadcast_object_list(names,src=0);attempt_name=names[0]
    attempt_dir=output/'resumes'/attempt_name if resuming else output
    if rank==0:
        if (output/'run.json').exists() and not resuming:raise ValueError('Run exists; use a new ID or explicit resume')
        if resuming and not (output/'run.json').exists():raise ValueError('Resume requires preserved original run metadata')
        attempt_dir.mkdir(parents=True,exist_ok=True);source_snapshot(attempt_dir/'source')
    if world>1:dist.barrier()
    ds=ManifestDataset(ROOT/cfg['train_manifest'],cfg.get('size',416),False,'patch')
    validation=ManifestDataset(ROOT/cfg['validation_manifest'],cfg.get('size',416),False,'patch')
    if any(r['split']!='train' for r in ds.records) or any(r['split']!='validation' for r in validation.records):raise ValueError('Training-origin validation only')
    if {r['source_id'] for r in ds.records}&{r['source_id'] for r in validation.records}:raise ValueError('Source leakage')
    model=build_localizer(cfg).to(device)
    optimizer=stage1_optimizer(model,cfg)
    scaler=torch.cuda.amp.GradScaler(enabled=cfg.get('amp',True) and device.type=='cuda',init_scale=cfg.get('amp_initial_scale',128.))
    # A short final group can leave a replica without examples. Initialize AMP
    # on every replica before summed gradients arrive from the other replicas.
    scaler.scale(torch.zeros((),device=device))
    start_epoch=0;best=-math.inf;parent=None
    parent_path=cfg.get('resume') or cfg.get('initialization_checkpoint')
    if parent_path:
        parent=load_checkpoint(ROOT/parent_path,allow_smoke=cfg.get('category')=='smoke')
        if parent['stage']!='stage1':raise ValueError('Expected independently trained Stage1 parent')
        if resuming and parent['run_id']!=run:raise ValueError('Resume must use the same independently trained run')
        for key in ['size','seed','effective_batch','train_manifest','validation_manifest','lr','localizer','afal']:
            if parent['config'].get(key)!=cfg.get(key):raise ValueError('Parent protocol differs at '+key)
        for key,default in [('enhancement_lr_scale',1.),('clip_policy','global'),('clip_norm',1.)]:
            if parent['config'].get(key,default)!=cfg.get(key,default):raise ValueError('Parent optimization differs at '+key)
        model.load_state_dict(parent['model']);optimizer.load_state_dict(parent['optimizer']);scaler.load_state_dict(parent['scaler'])
        start_epoch=parent['epoch']+1;best=parent['best_validation_metric']
        import random,numpy as np
        torch.set_rng_state(parent['rng']['torch']);random.setstate(parent['rng']['python']);np.random.set_state(parent['rng']['numpy'])
        if device.type=='cuda' and parent['rng'].get('cuda'):
            torch.cuda.set_rng_state(parent['rng']['cuda'][min(local,len(parent['rng']['cuda'])-1)],device)
    cfg['_code_sha256']=code_fingerprint()
    if rank==0:
        write_json(attempt_dir/'run.json',{'config':cfg,'command':sys.argv,'config_sha256':sha256(path),'code_sha256':cfg['_code_sha256'],
            'train_manifest_sha256':sha256(ROOT/cfg['train_manifest']),'validation_manifest_sha256':sha256(ROOT/cfg['validation_manifest']),
            'executor':'manual_replica_gradient_sum','world_size':world,'microbatch_per_rank':1,
            'microbatches_per_rank_per_step':accumulation,'effective_global_batch':effective,
            'example_RNG':'seed+1000003*epoch+97409*global_shuffled_position modulo 2**31-1',
            'no_padding_or_dropped_training_samples':True,'started_utc':utcnow(),'start_epoch_zero_based':start_epoch,
            'initialization_checkpoint_sha256':sha256(ROOT/parent_path) if parent else None,
            'attempt_id':attempt_name,'resume_checkpoint_sha256':sha256(ROOT/parent_path) if resuming else None,
            'parent_run':parent['run_id'] if parent else None,'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'torch_version':torch.__version__,'parameters':sum(p.numel() for p in model.parameters()),'paper_protocol_verified':False})
        if parent and not resuming:
            # Keep the parent's selected model eligible for the100-epoch chain.
            best_parent=parent
            if cfg.get('parent_best_checkpoint'):best_parent=load_checkpoint(ROOT/cfg['parent_best_checkpoint'],allow_smoke=cfg.get('category')=='smoke')
            current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_parent['model'])
            optimizer.load_state_dict(best_parent['optimizer']);scaler.load_state_dict(best_parent['scaler'])
            save_checkpoint(checkpoint_dir/'best.pth',model,optimizer,scaler,best_parent['epoch'],cfg,'stage1',run,best,False)
            model.load_state_dict(current)
            optimizer.load_state_dict(parent['optimizer']);scaler.load_state_dict(parent['scaler'])
            parent_output=ROOT/'results'/best_parent['run_id']
            evidence=parent_output/f'validation_epoch_{best_parent["epoch"]+1:03d}.json'
            if not evidence.exists():raise ValueError('Missing parent selected validation evidence')
            shutil.copyfile(evidence,output/evidence.name)
            write_json(output/'parent_evidence.json',{'run_id':parent['run_id'],'prefix_epochs':start_epoch,
                'last_checkpoint_sha256':sha256(ROOT/parent_path),'selected_validation_source':str(evidence.relative_to(ROOT)),
                'selected_validation_sha256':sha256(evidence),'rng_restored':True,
                'example_rng_policy':'Per-example seeds are world-size independent; floating-point reduction order can differ.'})
    if world>1:dist.barrier()
    start=time.monotonic();global_step=start_epoch*math.ceil(len(ds)/effective)
    try:
        for epoch in range(start_epoch,cfg['epochs']):
            generator=torch.Generator().manual_seed(cfg['seed']+epoch)
            order=torch.randperm(len(ds),generator=generator).tolist()
            local_indices=order[rank::world]
            loader=DataLoader(Subset(ds,local_indices),batch_size=1,shuffle=False,num_workers=cfg.get('workers',2),collate_fn=collate,pin_memory=device.type=='cuda')
            iterator=iter(loader);model.train();epoch_start=time.monotonic();loss_sum=0.;updates=0
            for group_start in range(0,len(ds),effective):
                count=min(effective,len(ds)-group_start);optimizer.zero_grad(set_to_none=True)
                group_loss=torch.zeros((),device=device);terms={}
                for micro in range(accumulation):
                    position=group_start+micro*world+rank
                    if position>=len(ds) or position>=group_start+count:continue
                    batch=next(iterator);seed_all(example_seed(cfg['seed'],epoch,position))
                    x=batch['image'].to(device,non_blocking=True)
                    targets=[{k:v.to(device) for k,v in t.items()} for t in batch['target']]
                    with torch.autocast(device_type=device.type,enabled=scaler.is_enabled(),dtype=getattr(torch,cfg.get('amp_dtype','bfloat16'))):
                        loss,terms=stage1_objective(model,x,targets,cfg.get('afal',{}))
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite per-example training loss')
                    scaler.scale(loss/count).backward();group_loss+=loss.detach()/count
                sum_replica_gradients(model.parameters())
                if world>1:dist.all_reduce(group_loss,op=dist.ReduceOp.SUM)
                scaler.unscale_(optimizer)
                norm,gradient_details=clip_gradients(model,cfg,diagnose=(global_step%cfg.get('gradient_log_interval',10)==0),error_if_nonfinite=not scaler.is_enabled())
                finite=bool(torch.isfinite(norm))
                gradients={name:float(p.grad.float().norm()) if finite else None for name,p in model.named_parameters()
                    if p.grad is not None and (name in ['enhancement.kernel','fusion.weight'] or name.endswith('backbone.body.conv1.weight'))}
                if finite:scaler.step(optimizer);scaler.update()
                else:scaler.update(new_scale=max(scaler.get_scale()/2,1.))
                optimizer.zero_grad(set_to_none=True);global_step+=1;updates+=1;loss_sum+=float(group_loss)
                if rank==0:
                    append_jsonl(output/'train.jsonl',{'epoch':epoch+1,'attempt_id':attempt_name,'step':global_step,'global_examples_seen_this_epoch':group_start+count,
                        'loss':group_loss,'last_example_terms_rank0':terms,'preclip_grad_norm':norm if finite else None,
                        'selected_parameter_grad_norms':gradients,'optimizer_step_executed':finite,'amp_scale':scaler.get_scale(),
                        'lr':[p['lr'] for p in optimizer.param_groups],'world_size':world,'global_group_examples':count,**gradient_details,
                        'elapsed_seconds':time.monotonic()-start})
                    write_json(output/'status.json',{'status':'running','epoch':epoch+1,'epochs':cfg['epochs'],'step':global_step,
                        'batch_index':group_start+count,'batches':len(ds),'world_size':world,'elapsed_seconds':time.monotonic()-start,'last_loss':group_loss})
                if cfg.get('max_steps') and global_step>=cfg['max_steps']:break
            # Shard validation without padding. Only rank0 calculates pooled AP.
            replica_delta=replica_parameter_difference(model.parameters()) if cfg.get('verify_replicas') else None
            if replica_delta is not None and replica_delta!=0:raise AssertionError('Replica parameters diverged: '+str(replica_delta))
            vloader=DataLoader(Subset(validation,list(range(rank,len(validation),world))),batch_size=cfg.get('validation_batch',2),
                shuffle=False,num_workers=cfg.get('workers',2),collate_fn=collate)
            local_val,pred,gt=validate(model,vloader,device,'stage1')
            gathered=[None]*world if world>1 else [(local_val,pred,gt)]
            if world>1:dist.all_gather_object(gathered,(local_val,pred,gt))
            if rank==0:
                predictions=[p for _,ps,_ in gathered for p in ps];ground_truth=[g for _,_,gs in gathered for g in gs]
                if len(predictions)!=len(validation) or len({p['image_id'] for p in predictions})!=len(validation):raise ValueError('Validation sample mismatch')
                prediction_map={p['image_id']:p for p in predictions};gt_map={g['image_id']:g for g in ground_truth}
                predictions=[prediction_map[r['id']] for r in validation.records]
                ground_truth=[gt_map[r['id']] for r in validation.records]
                metrics=evaluate(predictions,ground_truth)
                score=metrics['AP50_coco101']
                if score is None:raise ValueError('Validation has no positive GT')
                val={'metrics':metrics,'loss':sum(v['loss']*len(p) for v,p,_ in gathered)/len(validation)}
                improved=score>best;best=max(best,score)
                write_json(output/f'validation_epoch_{epoch+1:03d}.json',{'summary':val,'predictions':predictions,'ground_truth':ground_truth})
                complete=epoch+1==cfg['epochs'] and not cfg.get('max_steps')
                save_checkpoint(checkpoint_dir/'last.pth',model,optimizer,scaler,epoch,cfg,'stage1',run,best,complete)
                if improved:save_checkpoint(checkpoint_dir/'best.pth',model,optimizer,scaler,epoch,cfg,'stage1',run,best,complete)
                append_jsonl(output/'epochs.jsonl',{'epoch':epoch+1,'attempt_id':attempt_name,'train_loss_mean_global_groups':loss_sum/updates,
                    'validation':val,'best_validation_AP50':best,'selected':improved,'epoch_seconds':time.monotonic()-epoch_start,
                    'world_size':world,'actual_training_examples':len(ds),'parent_prefix_epochs':start_epoch,
                    'max_replica_parameter_abs_difference':replica_delta})
                print(f'{run} epoch={epoch+1} valAP={score:.6f} seconds={time.monotonic()-epoch_start:.1f}',flush=True)
            if world>1:dist.barrier()
            if cfg.get('max_steps') and global_step>=cfg['max_steps']:
                if rank==0:write_json(output/'status.json',{'status':'smoke_complete','epoch':epoch+1,'steps':global_step,
                    'elapsed_seconds':time.monotonic()-start,'formal_training_complete':False,'world_size':world})
                return
        if rank==0:write_json(output/'status.json',{'status':'training_complete','epochs':cfg['epochs'],
            'best_validation_AP50':best,'world_size':world,'elapsed_seconds':time.monotonic()-start,
            'parent_prefix_epochs':start_epoch,'paper_protocol_verified':False})
    except BaseException as exc:
        if rank==0:write_json(output/'status.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start})
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()
if __name__=='__main__':main()
