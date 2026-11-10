import unittest,tempfile,json
from pathlib import Path
import torch
from PIL import Image
from models.wavelet import WaveletPyramid
from models.repair import DualDomainRepair
from losses.repair import dependency_loss
from datasets.manifest import ManifestDataset
from evaluation.metrics import evaluate,coco_reference

class PortableTests(unittest.TestCase):
    def test_thirteen_bands_roundtrip_and_dependency_gradient(self):
        torch.set_num_threads(2);x=torch.rand(1,3,64,64,requires_grad=True);w=WaveletPyramid()
        bands=w(x);self.assertEqual(len(bands),13);torch.testing.assert_close(w.inverse(bands,(64,64)),x,atol=2e-6,rtol=0)
        a=torch.tensor(.4,requires_grad=True);b=torch.tensor(.5,requires_grad=True);dependency_loss(a,b).backward()
        self.assertEqual(a.grad.item(),1.);self.assertEqual(b.grad.item(),-1.)
    def test_mask_identity_and_both_repair_branches(self):
        r=DualDomainRepair(width=4);x=torch.rand(1,3,64,64);mask=torch.zeros(1,1,64,64)
        self.assertTrue(torch.equal(r(x,mask),x));mask[:,:,16:48,16:48]=1
        r.branches='both';both=r(x,mask);r.branches='frequency';frequency=r(x,mask);r.branches='spatial';spatial=r(x,mask)
        self.assertFalse(torch.equal(both,frequency));self.assertFalse(torch.equal(both,spatial))
        self.assertTrue(torch.equal(both[:,:,0:16,:],x[:,:,0:16,:]))
    def test_portable_relative_manifest_and_COCO_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);Image.new('RGB',(64,64)).save(p/'x.png')
            (p/'manifest.jsonl').write_text(json.dumps({'id':'x','image':'x.png','patch_boxes_xyxy':[[8,8,32,32]]})+'\n')
            d=ManifestDataset(p/'manifest.jsonl',64);self.assertEqual(d[0]['image'].shape,(3,64,64))
        gt=[{'image_id':'x','boxes':[[8,8,32,32]]}];pred=[{'image_id':'x','boxes':[[8,8,32,32]],'scores':[.9]}]
        self.assertAlmostEqual(evaluate(pred,gt)['AP50_coco101'],coco_reference(pred,gt),places=12)
if __name__=='__main__':unittest.main(verbosity=2)
