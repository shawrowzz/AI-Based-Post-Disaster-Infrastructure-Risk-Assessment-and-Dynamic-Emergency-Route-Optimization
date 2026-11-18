Subject: xBD Pilot Run Results: ResNet-50 Siamese-FPN Baseline

Hi Radwa,

Following your advice, we implemented the Siamese-FPN pipeline and ran a small pilot with the ResNet-50 baseline on my laptop (Apple M4, 16 GB). The goal was to check that the full pipeline works end-to-end (data loading, training, xView2-style evaluation, Grad-CAM) before moving to UCloud. This is a small pilot (400 training pairs, 5 epochs, one seed), so the numbers are a sanity check, not benchmark results.

---

### 1. Inference latency (one pre/post pair, batch 1, median of repeated runs)

| Device | Input | Median latency | Pairs per second |
| :--- | :---: | :---: | :---: |
| M4 GPU (MPS) | 512×512 | 113 ms | 8.8 |
| M4 GPU (MPS) | 1024×1024 | 448 ms | 2.2 |
| M4 CPU | 512×512 | 358 ms | 2.8 |
| M4 CPU | 1024×1024 | 1,989 ms | 0.5 |

A full 1024×1024 xBD tile runs in about 2 s on the M4 CPU alone, so on-device inference without a dedicated GPU looks feasible. These figures are for the M4 only and will differ on other hardware.

---

### 2. Training

* The loss went down every epoch, from 1.391 (epoch 1) to 1.142 (epoch 5), using a class-weighted focal + Dice loss.
* Local throughput was about 1.2 training pairs/s (512 crops), with about 6.8 GB of GPU memory in use after training.
* We have not timed training on UCloud yet. We will measure it on the first cloud run and confirm the compute budget then.

---

### 3. Validation (80 full 1024×1024 tiles from the hold set, 10 disaster events)

Damage F1 is computed only on ground-truth building pixels, following the xView2 scorer. These are xView2-style scores, not official leaderboard numbers.

| Metric | F1 |
| :--- | :---: |
| Building localization | 0.681 |
| No damage | 0.873 |
| Minor damage | 0.000 |
| Major damage | 0.559 |
| Destroyed | 0.605 |
| Damage F1, harmonic mean (xView2) | 0.000 |
| xView2-style composite (0.3 loc + 0.7 dmg) | 0.204 |

The model never correctly predicts minor damage. In the example figure, buildings labelled minor are predicted as major. Because the xView2 damage score is a harmonic mean, this one zero pulls the damage score, and therefore the composite, down to almost zero. (The plain average of the four damage F1s is 0.509.) Separating minor from major damage is the hardest part of the task, and it is one of the questions the full benchmark is meant to answer.

---

### 4. Grad-CAM check

* We ran Grad-CAM on the fused FPN layer for 20 validation tiles. For each tile we measured the share of attribution that falls inside building footprints (EBPG) and compared it with the share of the tile that buildings cover, which is what an uninformative heatmap would score.
* On average, 50% of the attribution falls inside footprints, while buildings cover 8% of the tiles. All 20 tiles were above chance. The ratio ranged from 1.7× (dense neighbourhoods) to 124× (sparse tiles), with a median of 13×.
* Caveat: Grad-CAM explains the building classes the model predicted. So this shows that the model's evidence comes from the buildings themselves rather than the surrounding area. It does not yet show that the model is looking at the damage.
* The attached 5-panel figure (Hurricane Harvey, tile 00000404, chosen as the tile with the most damaged buildings) shows pre-disaster image, post-disaster image, ground-truth damage, model prediction, and Grad-CAM with building outlines.

---

The pipeline works end-to-end, and every number above is written automatically by the script from measured values. Next, we plan to run the full benchmark on UCloud (ResNet-50 vs. ConvNeXt-Tiny vs. Swin-Tiny, same decoder and training setup). That run will use the full training set and a separate validation split for choosing checkpoints.

Best regards,  
Samarpan Oli
