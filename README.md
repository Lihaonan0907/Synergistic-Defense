# Defense Against Adversarial Patches

This repository contains the official implementation for the paper **“Synergistic Frequency-Domain Detection and Repair with Adversarial Training for Adversarial Patch Defense”**. It is a unified two-stage defense framework designed to protect object detectors from localized, physically realizable adversarial patch attacks.

## Pre-trained Model

A pre-trained defense model is provided for evaluation and reproduction.

- **Model file**: `defense_model.pt`
- **Download link**:  
  https://drive.google.com/file/d/1nkCLr6su9VwigYCEALbcbmfZfUsrLqLA/view?usp=drive_link

After downloading, place the model under:
```text
checkpoints/defense_model.pt
```

## Installation

```bash
pip install -r requirements.txt
pip install pytorch-wavelets
```

## Project Structure

```
├── train.py                         # Main training script
├── evaluate.py                      # Evaluation on adversarial datasets
├── val.py                           # Standard validation
├── create_adversarial_datasets.py   # Script to generate attacked datasets
├── checkpoints/                     # Pre-trained model weights
│   └── defense_model.pt
├── models/                          # Model definitions (detector, repair module)
├── utils/                           # Utility functions and helpers
├── data/                            # Dataset configuration files
└── configs/                         # Training and experiment configurations
```

## Usage

### Training

The framework operates in two synergistic stages:

1.  **Stage 1: Multiscale Spectral Patch Detection**  
    A wavelet-based detector localizes adversarial patches using adaptive frequency-domain analysis.
    ```bash
    python train.py --stage 1 --data data/patch.yaml --cfg models/yolov5s.yaml --epochs 100
    ```

2.  **Stage 2: Joint Repair and Robust Detection**  
    A frequency-guided module repairs the detected regions while a detector is jointly trained for robust perception.
    ```bash
    python train.py --stage 2 --data data/advpatch_dataset.yaml --epochs 80
    ```

### Evaluation

Evaluate defense performance on adversarial datasets:

```bash
python val.py \
    --model checkpoints/defense_model.pt \
    --datasets adversarial datasets \
    --output-dir evaluation_results
```

## Supplementary Material: Demonstration Videos

Supplementary demonstration videos are available, providing visual comparisons of adversarial patch attacks and the repair effects of our defense framework. These videos illustrate the attack’s impact on object detectors and the subsequent recovery after applying our frequency-guided detection and repair pipeline.

-   **Video Folder (Google Drive)**: https://drive.google.com/drive/folders/1lpCOSaDeDZgQkG4PL94qWd_tSS_Zd3Ug?usp=drive_link

*Note*: The videos are provided as supplementary material to the paper **“Synergistic Frequency-Domain Detection and Repair with Adversarial Training for Adversarial Patch Defense”**.