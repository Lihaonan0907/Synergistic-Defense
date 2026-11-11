# Synergistic Frequency-Domain Detection and Repair with Adversarial Training for Adversarial Patch Defense

Yang Li, Haonan Li, Ziyan He, Shihao Wu, Xin Zhou, and Zhunga Liu

PyTorch implementation of frequency-domain adversarial patch localization,
mask-guided repair, and person detection.

## Overview

The two-stage framework follows **Patch Localization → Mask Generation →
Dual-Domain Repair → Person Detection**. Stage 1 localizes adversarial patches
using wavelet features and adaptive frequency-domain learning. Stage 2 combines
spatial and frequency-domain repair with a downstream person detector.

Some implementation details not fixed by the paper use explicit assumptions;
see [implementation notes](docs/IMPLEMENTATION.md).
The current configurations use db2 for localization and db6 for repair; they
have not been validated by retraining.

## Installation

Python 3.10 or 3.11 is required. Create an environment and install dependencies:

```bash
git clone https://github.com/Lihaonan0907/Synergistic-Defense.git
cd Synergistic-Defense
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
```

The pinned dependencies include PyTorch 2.1.0 and torchvision 0.16.0. Install
compatible PyTorch builds for your platform and CUDA setup when needed.

## Code Structure

```text
models/       Patch localizer, wavelets, repair, detector, and full defense
losses/       Frequency alignment, reconstruction, and dependency losses
training/     Training objectives, optimization, and checkpoint handling
inference/    Image inference API and command-line interface
evaluation/   Evaluation utilities and metrics
attacks/      Adversarial patch and robustness evaluation utilities
datasets/     Dataset and annotation readers
configs/      Training and inference configurations
scripts/      Data preparation and supporting utilities
tests/        Lightweight tests
```

## Usage

**Model weights and datasets are not included.** Inference requires compatible
Stage-1 and Stage-2 exported bundles with SHA-256 JSON sidecars. The Stage-1
weights must match the frozen localizer inside the Stage-2 bundle. The supplied
sidecars in `weights/` record hashes for earlier bundles, which may not match the
current configurations; see [weight requirements](weights/README.md). No download is provided.

### Inference

Place compatible bundles in `weights/` and supply your own image:

```bash
python demo.py --input examples/input.jpg --output outputs/demo --device cpu
```

For a directory of images, pass its path with `--input` to `inference.py`.
Use `--stage1-weights` and `--stage2-weights` to select other bundle paths.

```python
from datasets.manifest import read_rgb
from inference import DefensePredictor

predict = DefensePredictor(
    stage1_weights="weights/stage1_best.pt",
    stage2_weights="weights/stage2_best.pt",
    config="configs/inference.json",
    device="cpu",
)
image, original_size = read_rgb("examples/input.jpg", predict.size)
outputs = predict(image)  # patch boxes, mask, repaired image, person boxes
```

### Data, training, and evaluation

Prepare user-supplied manifests as described in [data/MANIFEST_FORMAT.md](data/MANIFEST_FORMAT.md).
The training configurations specify the expected paths and a separately obtained
person-detector initializer. After providing the required inputs:

```bash
python train_stage1.py --config configs/stage1.json --gpus 1
python scripts/fetch_public_weights.py
python train_stage2.py --config configs/stage2.json
python evaluate.py --manifest data/manifests/stage2_test.jsonl --output outputs/evaluation
```

Training saves native checkpoints; inference expects exported bundles, and this
repository does not currently provide a native-checkpoint export utility. See
[weight requirements](weights/README.md) before running inference or evaluation.

## Demonstration Videos

[Watch the supplementary demonstration videos](https://drive.google.com/drive/folders/1lpCOSaDeDZgQkG4PL94qWd_tSS_Zd3Ug?usp=drive_link).

## Citation

```bibtex
@misc{li_synergistic_frequency_defense,
  title  = {Synergistic Frequency-Domain Detection and Repair with Adversarial Training for Adversarial Patch Defense},
  author = {Li, Yang and Li, Haonan and He, Ziyan and Wu, Shihao and Zhou, Xin and Liu, Zhunga},
  note   = {Publication details to be added}
}
```

## License

The project source is released under the [MIT License](LICENSE). Model weights,
datasets, and third-party assets have separate terms; see [third-party notices](docs/THIRD_PARTY_NOTICES.md).
