# Defense Against Adversarial Patches

A two-stage defense system against adversarial patches in object detection.

## Key Components

1. **Frequency-Guided Repair Module** - Wavelet-based frequency domain analysis for patch removal
2. **Adaptive Frequency Alignment Loss (AFAL)** - Self-supervised loss for robust feature learning
3. **Dynamic Training Strategy** - Progressive curriculum learning to balance repair and detection
4. **Cross-Detector Generalization** - Support for YOLOv5, YOLOv11, Faster R-CNN

## Installation

```bash
pip install -r requirements.txt
pip install pytorch-wavelets
```

## Project Structure

```
├── train.py                         # Main training script (two-stage)
├── create_adversarial_datasets.py   # Attack implementations & dataset generation
├── evaluate.py                   # Validation in training
├── val.py                           # Standard validation
├── detector_adapters.py             # Detector adapters
├── checkpoints/
│   └── defense_model.pt             # Pre-trained defense model
├── models/
│   ├── yolo.py                      # YOLO model
│   ├── frequency_guided_repair.py   # Frequency-guided repair module
│   └── *.yaml                       # Model configurations
├── utils/                           # Utility functions
├── data/                            # Dataset configurations
└── configs/                         # Training configurations
```

## Quick Start

### Using Pre-trained Model

```bash
# Evaluate defense on adversarial datasets
python evaluate.py --model checkpoints/defense_model.pt --datasets advPatch LaVAN GNAP
```

### Training from Scratch

#### Stage 1: Patch Detector Training
```bash
python train.py --stage 1 --data data/patch.yaml --cfg models/yolov5s.yaml --epochs 100
```

#### Stage 2: Joint Training (Repair + Detection)
```bash
python train.py --stage 2 --data data/advpatch_dataset.yaml --epochs 80
```

## Attack Implementations

Generate adversarial datasets using various attack methods:

```bash
python create_adversarial_datasets.py \
    --inria-path /path/to/INRIA \
    --patches-path /path/to/adversarial_patches \
    --output-path /path/to/output
```

Supported attack methods:
- **advPatch** - Basic adversarial patch attack
- **DM-NAP** - Diffusion Model based Natural Adversarial Patch
- **GNAP** - Generative Natural Adversarial Patch
- **LaVAN** - Localized and Visible Adversarial Noise
- **T-SEA** - Transferable and Stealthy Adversarial Attack
- **diffpatch** - Diffusion-based patch generation
- **adaptive_advpatch** - Adaptive adversarial patch
- **advtexture** - Adversarial texture attack

## Evaluation

Evaluate defense performance on multiple datasets:

```bash
python val.py \
    --model checkpoints/defense_model.pt \
    --datasets advPatch DM-NAP GNAP LaVAN T-SEA diffpatch adaptive_advpatch advtexture \
    --output-dir evaluation_results
```

### Output Metrics
- **Adversarial mAP**: Detection performance on adversarial images
- **Repaired mAP**: Detection performance after frequency-guided repair
- **Clean mAP**: Upper bound (detection on clean images)
- **Recovery Rate**: (Repaired - Adversarial) / (Clean - Adversarial)
- **Latency**: End-to-end inference time breakdown

## Cross-Detector Experiments

Train and evaluate with different backbone detectors:

```bash
# YOLOv5
python train.py --detector-type yolov5 --data data/advpatch_dataset.yaml

# YOLOv11
python train.py --detector-type yolov11 --data data/advpatch_dataset.yaml

# Cross-detector evaluation
python evaluate.py --model checkpoints/defense_model.pt --detector-type faster_rcnn
```

## License

MIT License
