import argparse,json,subprocess,sys,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/stage2.json');p.add_argument('--resume');a=p.parse_args()
    root=Path(__file__).resolve().parent;c=json.loads((root/a.config).read_text())
    if a.resume:c['resume']=a.resume
    path=root/'results/launch_configs'/f'{c["run_id"]}_{os.getpid()}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(c,indent=2))
    subprocess.run([sys.executable,'-m','training.run','--config',str(path)],cwd=root,check=True)
if __name__=='__main__':main()
