import json,tempfile,unittest
from pathlib import Path
from inference.api import load_settings,load_model

class ReleaseContracts(unittest.TestCase):
    def test_frozen_evaluator_settings_cannot_be_changed(self):
        c=load_settings();c['detector_nms_threshold']=.3
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'settings.json';p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):load_settings(p)
    def test_corrupt_weight_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.pt';p.write_bytes(b'corrupt model')
            Path(str(p)+'.json').write_text(json.dumps({'sha256':'0'*64}))
            with self.assertRaisesRegex(ValueError,'SHA256'):load_model(p)

if __name__=='__main__':unittest.main()
