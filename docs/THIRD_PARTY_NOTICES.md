# Third-party and data/model notices

These notices inventory installed dependency evidence. They do not select a project license or establish dataset/model redistribution permission. Actual wheel license texts are preserved in third_party_licenses/.

| Component | Evidence / terms | Release treatment |
|---|---|---|
| PyTorch2.1 / torchvision0.16 | Installed BSD license text; official versioned license links below | Imported dependency; no library binaries vendored |
| pytorch_wavelets1.3 | Installed license explicitly puts DWT sections under MIT; DTCWT has additional provenance conditions | Uses DWT; retains the complete upstream notice |
| PyWavelets / NumPy / SciPy / scikit-image | Installed BSD-style and bundled third-party notices | Retain notices; inspect bundled components if redistributing wheels |
| Pillow / pycocotools / PyYAML | Installed MIT-CMU / FreeBSD / MIT notices | Imported dependencies; notices preserved |
| opencv-python | Installed Apache2 and third-party notices | Imported only; wheel components retain their own terms |
| INRIA original and patch assets | No sufficiently verified image/attack-asset redistribution grant established from supplied raw assets | Images, GT and real demo images excluded from public archives; owner review pending |
| COCO | Consortium annotations terms differ from image copyright ownership | Raw images/annotations excluded; local evaluation only |
| CityPersons/Cityscapes | Dataset terms and any derived restrictions require owner review | Raw images/GT excluded |
| Official COCO person initializer / derived defense weights | Software license alone does not establish all data/model rights | Initializer not bundled; derived weights remain local pending review |
| Supplied paper / figures | Draft source; no figure use permission assumed | No paper figure copied; new Mermaid data-flow diagram |

Primary references: [PyTorch2.1 license](https://github.com/pytorch/pytorch/blob/v2.1.0/LICENSE), [torchvision0.16 license](https://github.com/pytorch/vision/blob/v0.16.0/LICENSE), [COCO Consortium terms source](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm). The actual installed pytorch_wavelets wheel license is included and takes precedence over a guessed short license label.

The project source code is available under the MIT License. No trained weights are distributed. Dataset, attack-asset and model terms must be assessed separately; the project license does not grant rights to third-party assets.
