import argparse,json,subprocess,sys,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stage1.json');p.add_argument('--gpus',type=int,default=1)
    p.add_argument('--resume');a=p.parse_args();root=Path(__file__).resolve().parent;c=json.loads((root/a.config).read_text())
    if c['effective_batch']%a.gpus:raise ValueError('Global batch must be divisible by GPU replicas')
    if a.resume:c['resume']=a.resume
    c['world_size']=a.gpus;path=root/'results/launch_configs'/f'{c["run_id"]}_{os.getpid()}.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(c,indent=2))
    command=([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(a.gpus),'--module','training.distributed']
        if a.gpus>1 else [sys.executable,'-m','training.distributed'])
    subprocess.run(command+['--config',str(path)],cwd=root,check=True)
if __name__=='__main__':main()
