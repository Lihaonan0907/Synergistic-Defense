# Synergistic Frequency-Domain Detection and Repair with Adversarial Training for Adversarial Patch Defense

Yang Li, Haonan Li, Ziyan He, Shihao Wu, Xin Zhou, and Zhunga Liu

PyTorch implementation of frequency-domain adversarial patch localization and mask-guided image repair for person detection. The method combines spectral patch localization with spatial and frequency-domain repair, adversarial training, and downstream detection feedback.

**Release contents:** source code and configurations. Model weights and datasets are not included; inference and evaluation require compatible weights supplied separately.

## Installation

The documented environment uses Python 3.10, PyTorch 2.1.0, and torchvision 0.16.0. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

See [the dependency lock](docs/requirements-lock.cpu.txt) for the previously verified CPU environment and [historical CUDA dependencies](docs/historical-training-direct-dependencies.cu118.txt) for the training environment.

## Data and weights

Obtain images and annotations under their applicable terms. Prepare the required JSONL manifests using [the data format](data/MANIFEST_FORMAT.md) and `scripts/prepare_dataset.py`. Stage 1 requires patch annotations; Stage 2 requires paired clean images and person annotations. Keep source identities disjoint across training, validation, and test splits.

The default inference configuration expects `weights/stage1_best.pt` and `weights/stage2_best.pt`, with their corresponding SHA256 sidecars. **These model files are not distributed here and no public download is currently available.** Compatible weights and matching sidecars are required. The inference commands below require compatible weights and a user-provided image.

## Inference

```bash
python demo.py --input examples/input.jpg --output outputs/single --device cpu

python inference.py --input examples/inputs \
  --stage1-weights weights/stage1_best.pt \
  --stage2-weights weights/stage2_best.pt \
  --config configs/inference.json --output outputs/directory --device cpu
```

Outputs include predicted patch masks, repaired images, person detections, and numerical predictions. Use `--device cuda:0` with a compatible CUDA installation.

## Training and evaluation

After preparing the stage-specific manifests described in [the data format](data/MANIFEST_FORMAT.md), run:

```bash
python train_stage1.py --config configs/stage1.json --gpus 1
python scripts/fetch_public_weights.py
python train_stage2.py --config configs/stage2.json

python evaluate.py --task patch --manifest data/manifests/stage1_test.jsonl --output outputs/patch_test --device cpu
python evaluate.py --manifest data/manifests/stage2_test.jsonl --output outputs/person_test --device cpu
```

Stage 2 requires a trained Stage-1 checkpoint and the person-detector initializer specified in `configs/public_weights.json`; the initializer is acquired separately and subject to its own terms. Configuration files specify the training parameters and required assets.

## Code structure

- `models/`, `losses/`: frequency-domain localization, dual-domain repair, detector, and objectives.
- `training/`, `inference/`, `evaluation/`, `attacks/`: training, inference, metrics, and attack evaluation.
- `datasets/`, `scripts/`, `configs/`, `tests/`: data preparation, utilities, settings, and lightweight checks.

Method details are described in the source code.

## Citation

```bibtex
@misc{li_synergistic_frequency_defense,
  title  = {Synergistic Frequency-Domain Detection and Repair with Adversarial Training for Adversarial Patch Defense},
  author = {Li, Yang and Li, Haonan and He, Ziyan and Wu, Shihao and Zhou, Xin and Liu, Zhunga},
  note   = {Publication details to be added}
}
```

## License

The project source code is released under the [MIT License](LICENSE). Model weights, datasets, and third-party assets are subject to their own terms; see [third-party notices](docs/THIRD_PARTY_NOTICES.md).
