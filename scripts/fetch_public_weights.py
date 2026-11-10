"""Optional training initializer; inference bundles never need this download."""
import json,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    records=json.loads((ROOT/'configs/public_weights.json').read_text())
    for record in records:
        path=ROOT/record['file'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temporary=path.with_suffix('.download');urllib.request.urlretrieve(record['official_url'],temporary);temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Public initializer checksum mismatch')
    (ROOT/'checkpoints/public/provenance.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
