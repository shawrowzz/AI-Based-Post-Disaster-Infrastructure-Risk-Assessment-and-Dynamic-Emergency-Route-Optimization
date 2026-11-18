# Pilot results: ResNet-50 Siamese-FPN on xBD (local Apple M4)

_All numbers below are written automatically by `pilot/run_pilot.py` from measured values._

## Setup

- Backbone `resnet50.a1_in1k`, 36.3M parameters, shared FPN decoder (d=256)
- Train: 400 pairs from `train 3/` (10 events), 512 random crops, batch 4, 5 epochs, AdamW lr 0.00015
- Loss: focal (class weights [0.2, 1.0, 2.0, 2.0, 2.0]) + Dice on building classes
- Validation: 80 full 1024x1024 tiles from `hold/`, events: guatemala-volcano, hurricane-florence, hurricane-harvey, hurricane-matthew, hurricane-michael, mexico-earthquake, midwest-flooding, palu-tsunami, santa-rosa-wildfire, socal-fire
- Device: mps, seed 0 (single seed)

## 1. Inference latency (one pre/post pair, batch 1)

| Device | Input | Median | IQR | Runs |
| :-- | :-: | :-: | :-: | :-: |
| M4 GPU (MPS) | 512x512 | 113 ms | 113-113 ms | 30 |
| M4 GPU (MPS) | 1024x1024 | 448 ms | 448-448 ms | 20 |
| M4 CPU | 512x512 | 358 ms | 356-374 ms | 15 |
| M4 CPU | 1024x1024 | 1989 ms | 1488-2439 ms | 8 |

## 2. Training

| Epoch | Loss | Time |
| :-: | :-: | :-: |
| 1 | 1.3907 | 188 s |
| 2 | 1.2458 | 579 s |
| 3 | 1.2014 | 405 s |
| 4 | 1.1727 | 254 s |
| 5 | 1.1417 | 252 s |

- Throughput: 1.19 training pairs/s (512 crops) on the M4
- MPS driver memory after training: 6.8 GB
- UCloud GPU time per epoch has **not** been measured yet; it will be timed on the first cloud run.

## 3. Validation metrics (pixel level, xView2-style)

Damage F1 is computed only on ground-truth building pixels, as in the xView2 scorer. These are not official leaderboard numbers (localization comes from the same post-disaster prediction).

- Localization F1: **0.681**
- Damage F1, harmonic mean: **0.000** (macro mean 0.509)
- Composite (0.3 loc + 0.7 dmg): **0.204**

| Class | F1 | GT pixels |
| :-- | :-: | --: |
| no_damage | 0.873 | 3,507,842 |
| minor_damage | 0.000 | 460,833 |
| major_damage | 0.559 | 824,207 |
| destroyed | 0.605 | 219,005 |

## 4. Grad-CAM attribution

- Layer: head[0] (fused FPN, stride 4); gradient-weighted (Seg-Grad-CAM), target = logits of the predicted building classes
- Tiles analysed: 20 (all validation tiles with >=0.5% building area, in order)
- Mean EBPG (attribution inside footprints): **50.4%**
- Mean building area of those tiles (= score of a uniform heatmap): **8.1%**
- Median EBPG / area ratio: **13.03** (1.0 = no better than chance)
- Tiles where attribution is concentrated above chance: 20/20
- Figure: `pilot_prediction_and_gradcam.png`, tile `hurricane-harvey_00000404` (selection rule: tile with most ground-truth damaged pixels among Grad-CAM tiles)

## Caveats

- Small pilot: 400 training pairs, 5 epochs, one seed. Not a benchmark result.
- Validation uses `hold/`; the final study will use the official test/hold protocol with a separate validation split for checkpoint selection.
