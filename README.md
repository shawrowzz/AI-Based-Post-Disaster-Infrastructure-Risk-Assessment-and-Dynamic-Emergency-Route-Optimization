# Pilot Run: ResNet-50 Siamese-FPN on xBD

This repository contains the standalone pilot implementation of a ResNet-50 Siamese Feature Pyramid Network (Siamese-FPN) baseline on the xBD dataset (pre- and post-disaster satellite imagery for building damage assessment).

---

## Data Structure Guide

The repository does not track dataset images or annotations. To run the pipeline, organize your dataset into `train/` and `val/` (or `hold/`) directories, with separate `images/` and `targets/` subfolders.

### 1. Recommended Directory Hierarchy

Place your data in a local `data/` folder inside the repository (automatically ignored by git), or in any external path:

```
pilot/
├── data/
│   ├── train/
│   │   ├── images/
│   │   │   ├── hurricane-harvey_00000001_pre_disaster.png
│   │   │   ├── hurricane-harvey_00000001_post_disaster.png
│   │   │   ├── socal-fire_00000002_pre_disaster.png
│   │   │   └── socal-fire_00000002_post_disaster.png
│   │   └── targets/
│   │       ├── hurricane-harvey_00000001_post_disaster_target.png
│   │       └── socal-fire_00000002_post_disaster_target.png
│   │
│   └── val/ (or hold/)
│       ├── images/
│       │   ├── mexico-earthquake_00000003_pre_disaster.png
│       │   └── mexico-earthquake_00000003_post_disaster.png
│       └── targets/
│           └── mexico-earthquake_00000003_post_disaster_target.png
```

### 2. File Naming Rules

Every sample triplet is identified by a common `{prefix}` (e.g. `hurricane-harvey_00000001`):

| File | Location | Format & Channels | Description |
| :--- | :--- | :--- | :--- |
| `{prefix}_pre_disaster.png` | `images/` | RGB (3-channel, 1024×1024) | Pre-disaster satellite optical image |
| `{prefix}_post_disaster.png` | `images/` | RGB (3-channel, 1024×1024) | Post-disaster satellite optical image |
| `{prefix}_post_disaster_target.png` | `targets/` | Grayscale / 8-bit (1-channel, 1024×1024) | Pixel-level damage label mask |

### 3. Target Mask Encoding

The target mask contains integer pixel values corresponding to the official xBD / xView2 damage taxonomy:

| Pixel Value | Class | Description |
| :---: | :--- | :--- |
| **0** | `background` | Background / non-building pixels |
| **1** | `no-damage` | Undamaged building footprint |
| **2** | `minor-damage` | Minor structural or non-structural damage |
| **3** | `major-damage` | Significant structural damage, roof partially collapsed |
| **4** | `destroyed` | Complete structural failure / collapsed |

---

## Setup & Environment

Install dependencies using Python 3.10+:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Default Run (uses `data/train` and `data/val` or local defaults)
```bash
python run_pilot.py
```

### Custom Data Locations
You can point to any local or cloud dataset paths using `--train-dir` and `--val-dir`:
```bash
python run_pilot.py \
    --train-dir /path/to/train \
    --val-dir /path/to/hold \
    --device cuda
```

### Evaluating an Existing Checkpoint
```bash
python run_pilot.py --skip-train --val-dir /path/to/hold
```

---

## Repository Structure

```
pilot/
├── dataset.py                        # xBD paired dataset loader & augmentations
├── model.py                          # Siamese-FPN architecture (ResNet-50 / ConvNeXt / Swin)
├── loss.py                           # Class-weighted focal + Dice loss
├── metrics.py                        # xView2 pixel metrics & Grad-CAM EBPG computation
├── run_pilot.py                      # Main entrypoint (train, benchmark, eval, attribution)
├── requirements.txt                  # Python dependencies
├── pilot_report.md                   # Full experimental results report
├── results/
│   ├── pilot_results.json            # Numerical evaluation metrics
│   └── training_log.json             # Per-epoch loss log
└── pilot_prediction_and_gradcam.png  # Qualitative Grad-CAM figure
```

---

## Pilot Results Summary

_Measured on local Apple M4 (MPS)_

### Inference Latency (batch 1)
| Device | Input | Median | IQR |
| :-- | :-: | :-: | :-: |
| M4 GPU (MPS) | 512x512 | 113 ms | 113-113 ms |
| M4 GPU (MPS) | 1024x1024 | 448 ms | 448-448 ms |
| M4 CPU | 512x512 | 358 ms | 356-374 ms |
| M4 CPU | 1024x1024 | 1989 ms | 1488-2439 ms |

### Validation Metrics (xView2-Style)
- **Localization F1**: 0.681
- **Damage F1 (harmonic mean)**: 0.000 (macro mean 0.509)
- **Composite Score (0.3 loc + 0.7 dmg)**: 0.204

For full details, see [pilot_report.md](pilot_report.md).
