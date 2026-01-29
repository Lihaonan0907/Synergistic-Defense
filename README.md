# Defense Against Adversarial Patches

This repository implements a two-stage defense framework against adversarial patches for object detection.

## Overview

This repository contains the official implementation for the paper “Synergistic Frequency-Domain Detection and Repair with Adversarial Training for Adversarial Patch Defense”. It is a unified two-stage defense framework designed to protect object detectors from localized, physically realizable adversarial patch attacks.

## Pre-trained Model

A pre-trained defense model is provided for evaluation and reproduction.

- **Model file**: `defense_model.pt`
- **Download link**:  
  https://drive.google.com/file/d/1ifqh-ajEr1w8Xre8svaX7Cgp7ix2txlw/view?usp=drive_link

After downloading, place the model under:
```text
checkpoints/defense_model.pt
Installation
pip install -r requirements.txt
pip install pytorch-wavelets

Project Structure
├── train.py
├── evaluate.py
├── val.py
├── create_adversarial_datasets.py
├── checkpoints/
│   └── defense_model.pt
├── models/
├── utils/
├── data/
└── configs/

Usage
Evaluation

Evaluate the pre-trained defense model on adversarial datasets:

python evaluate.py --model checkpoints/defense_model.pt

The framework operates in two synergistic stages:

Stage 1: Multiscale Spectral Patch Detection
A wavelet-based detector localizes adversarial patches using adaptive frequency-domain analysis.

bash
python train.py --stage 1 --data data/patch.yaml --cfg models/yolov5s.yaml --epochs 100
Stage 2: Joint Repair and Robust Detection
A frequency-guided module repairs the detected regions while a detector is jointly trained for robust perception.

bash
python train.py --stage 2 --data data/advpatch_dataset.yaml --epochs 80Evaluation

Evaluate defense performance on multiple datasets:

python val.py \
    --model checkpoints/defense_model.pt \
    --datasets advPatch DM-NAP GNAP LaVAN T-SEA diffpatch adaptive_advpatch advtexture \
    --output-dir evaluation_results