"""
Pilot run: ResNet-50 Siamese-FPN on a subset of xBD, on the local Apple M4 (MPS).

Everything written to the report is produced by this script from measured values.
Nothing is hand-edited or projected.

Usage:
    python pilot/run_pilot.py              # train + evaluate + explain
    python pilot/run_pilot.py --skip-train # reuse saved checkpoint
"""
import os
import sys
import time
import json
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    from pilot.model import SiameseFPN
    from pilot.dataset import XBDDataset
    from pilot.loss import XBDLoss
    from pilot.metrics import XBDMetrics, compute_ebpg
except ImportError:
    from model import SiameseFPN
    from dataset import XBDDataset
    from loss import XBDLoss
    from metrics import XBDMetrics, compute_ebpg

# Default paths
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(OUT_DIR, "checkpoints", "resnet50_pilot.pth")
BACKBONE = "resnet50.a1_in1k"

# Default data paths: looks for local parent directory or repo data/ directory
_LOCAL_ROOT = "/Users/kabishbhattarai/Desktop/oliz"
_REPO_DATA = os.path.join(OUT_DIR, "data")
if os.path.exists(os.path.join(_LOCAL_ROOT, "train 3")):
    TRAIN_DIR = os.path.join(_LOCAL_ROOT, "train 3")
    VAL_DIR = os.path.join(_LOCAL_ROOT, "hold")
else:
    TRAIN_DIR = os.path.join(_REPO_DATA, "train")
    VAL_DIR = os.path.join(_REPO_DATA, "val")

CFG = dict(
    seed=0,
    n_train=400,
    n_val=80,
    crop=512,
    batch_size=4,
    epochs=5,
    lr=1.5e-4,
    weight_decay=1e-4,
    class_weights=[0.2, 1.0, 2.0, 2.0, 2.0],
    n_gradcam_tiles=20,
)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- latency
def benchmark_latency(model, device, size, n_runs, warmup=5):
    """Per-call wall time for one pre/post pair. Returns median, p25, p75 in ms."""
    model.eval()
    a = torch.randn(1, 3, size, size, device=device)
    b = torch.randn(1, 3, size, size, device=device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(a, b); sync(device)
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(a, b); sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return dict(median_ms=float(np.median(t)), p25_ms=float(np.percentile(t, 25)),
                p75_ms=float(np.percentile(t, 75)), n_runs=n_runs)


# ---------------------------------------------------------------- Grad-CAM
def grad_cam(model, pre, post, layer):
    """
    Seg-Grad-CAM on `layer`. Target score = sum of the logits of the class the model
    predicts at each building pixel (classes 1-4). Channel weights = spatially averaged
    gradients; CAM = ReLU(sum_k w_k * A_k), upsampled to input size and scaled to [0, 1].
    """
    store = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(fwd_hook)
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = model(pre, post)
        pred = logits.argmax(1)
        mask = torch.zeros_like(logits)
        for c in range(1, 5):
            mask[:, c] = (pred == c).float()
        if mask.sum() == 0:            # model predicts no buildings: explain all building logits
            mask[:, 1:] = 1.0
        (logits * mask).sum().backward()
    h.remove()

    A = store["act"][0].detach()
    G = store["grad"][0].detach()
    w = G.mean(dim=(1, 2))
    cam = F.relu((w[:, None, None] * A).sum(0))
    cam = F.interpolate(cam[None, None], size=pre.shape[-2:], mode="bilinear",
                        align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), pred[0].cpu().numpy()


# ---------------------------------------------------------------- figure
DAMAGE_CMAP = ListedColormap(np.array([
    [0.00, 0.00, 0.00, 0.00],  # 0 background (transparent)
    [0.18, 0.80, 0.44, 0.90],  # 1 no damage
    [0.95, 0.77, 0.06, 0.90],  # 2 minor
    [0.90, 0.49, 0.13, 0.90],  # 3 major
    [0.91, 0.30, 0.24, 0.95],  # 4 destroyed
]))
DAMAGE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], DAMAGE_CMAP.N)
LEGEND = [
    mpatches.Patch(color="#2ecc71", label="1: No damage"),
    mpatches.Patch(color="#f1c40f", label="2: Minor"),
    mpatches.Patch(color="#e67e22", label="3: Major"),
    mpatches.Patch(color="#e74c3c", label="4: Destroyed"),
]


def denorm(t):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return np.clip((t.numpy() * std + mean).transpose(1, 2, 0), 0, 1)


def save_figure(sample, pred, cam, ebpg, area_frac, path):
    pre_v, post_v = denorm(sample["pre"]), denorm(sample["post"])
    gt = sample["target"].numpy()

    fig, ax = plt.subplots(1, 5, figsize=(27, 6))
    ax[0].imshow(pre_v); ax[0].set_title("1. Pre-disaster RGB", fontweight="bold")
    ax[1].imshow(post_v); ax[1].set_title("2. Post-disaster RGB", fontweight="bold")
    ax[2].imshow(post_v); ax[2].imshow(gt, cmap=DAMAGE_CMAP, norm=DAMAGE_NORM)
    ax[2].set_title("3. Ground truth damage", fontweight="bold")
    ax[2].legend(handles=LEGEND, loc="lower left", fontsize=9, framealpha=0.9, title="Damage index")
    ax[3].imshow(post_v); ax[3].imshow(pred, cmap=DAMAGE_CMAP, norm=DAMAGE_NORM)
    ax[3].set_title("4. Model prediction (pilot)", fontweight="bold")
    ax[4].imshow(post_v); im = ax[4].imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax[4].contour(gt > 0, levels=[0.5], colors="white", linewidths=0.6)
    ax[4].set_title(f"5. Grad-CAM (fused FPN layer)\nEBPG {ebpg*100:.1f}% vs building area {area_frac*100:.1f}%",
                    fontweight="bold")
    cb = fig.colorbar(im, ax=ax[4], fraction=0.046, pad=0.04)
    cb.set_label("Relative attribution (0 = none, 1 = max)")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"Tile: {sample['id']}  (white outlines = ground-truth building footprints)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="Skip training and reuse saved checkpoint")
    ap.add_argument("--train-dir", default=TRAIN_DIR, help="Path to training directory (contains images/ and targets/)")
    ap.add_argument("--val-dir", default=VAL_DIR, help="Path to validation directory (contains images/ and targets/)")
    ap.add_argument("--device", default=None, help="Device to use ('cuda', 'mps', or 'cpu')")
    args = ap.parse_args()

    set_seed(CFG["seed"])
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True)
    results = {"config": CFG, "backbone": BACKBONE, "device": str(device)}
    print(f"Device: {device}")

    model = SiameseFPN(backbone_name=BACKBONE, num_classes=5, pretrained=True).to(device)
    results["n_params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {results['n_params_M']:.1f}M")

    # 1. Latency (median of repeated runs)
    print("\n[1] Latency")
    lat = {
        "gpu_512": benchmark_latency(model, device, 512, n_runs=30),
        "gpu_1024": benchmark_latency(model, device, 1024, n_runs=20),
    }
    model_cpu = SiameseFPN(backbone_name=BACKBONE, num_classes=5, pretrained=False).eval()
    lat["cpu_512"] = benchmark_latency(model_cpu, torch.device("cpu"), 512, n_runs=15, warmup=2)
    lat["cpu_1024"] = benchmark_latency(model_cpu, torch.device("cpu"), 1024, n_runs=8, warmup=1)
    del model_cpu
    for k, v in lat.items():
        print(f"  {k:9s}: median {v['median_ms']:.0f} ms  (IQR {v['p25_ms']:.0f}-{v['p75_ms']:.0f}, n={v['n_runs']})")
    results["latency"] = lat

    # 2. Training
    train_ds = XBDDataset(args.train_dir, img_size=CFG["crop"], is_train=True, max_samples=CFG["n_train"])
    val_ds = XBDDataset(args.val_dir, is_train=False, max_samples=CFG["n_val"])
    events = lambda ds: sorted({s["id"].rsplit("_", 1)[0] for s in ds.samples})
    results["train_events"], results["val_events"] = events(train_ds), events(val_ds)
    print(f"\nTrain: {len(train_ds)} pairs from {len(results['train_events'])} events")
    print(f"Val:   {len(val_ds)} pairs from {len(results['val_events'])} events: {results['val_events']}")

    if args.skip_train and os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        results["training"] = json.load(open(os.path.join(OUT_DIR, "results", "training_log.json")))
        print("Loaded checkpoint, skipping training.")
    else:
        print("\n[2] Training")
        loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True)
        cw = torch.tensor(CFG["class_weights"], device=device)
        criterion = XBDLoss(class_weights=cw, ignore_bg_dice=True)
        opt = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        log = {"epoch_loss": [], "epoch_sec": []}
        for ep in range(1, CFG["epochs"] + 1):
            model.train()
            t0, tot = time.perf_counter(), 0.0
            for batch in loader:
                pre, post, tgt = batch["pre"].to(device), batch["post"].to(device), batch["target"].to(device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(model(pre, post), tgt)
                loss.backward()
                opt.step()
                tot += loss.item()
            sync(device)
            log["epoch_loss"].append(tot / len(loader))
            log["epoch_sec"].append(time.perf_counter() - t0)
            print(f"  epoch {ep}/{CFG['epochs']}  loss {log['epoch_loss'][-1]:.4f}  {log['epoch_sec'][-1]:.0f}s")
        if device.type == "mps":
            log["mps_driver_mem_GB"] = torch.mps.driver_allocated_memory() / 1e9
        log["train_pairs_per_sec"] = len(train_ds) / float(np.mean(log["epoch_sec"]))
        torch.save(model.state_dict(), CKPT_PATH)
        json.dump(log, open(os.path.join(OUT_DIR, "results", "training_log.json"), "w"), indent=2)
        results["training"] = log

    # 3. Evaluation on full 1024 tiles
    print("\n[3] Evaluation (full 1024x1024 tiles, native resolution)")
    model.eval()
    metrics = XBDMetrics()
    with torch.no_grad():
        for batch in DataLoader(val_ds, batch_size=2, shuffle=False):
            logits = model(batch["pre"].to(device), batch["post"].to(device))
            metrics.update(logits.argmax(1), batch["target"])
    m = metrics.compute()
    results["metrics"] = m
    print(f"  Localization F1          : {m['f1_localization']:.4f}")
    print(f"  Damage F1 (harmonic)     : {m['f1_damage_harmonic']:.4f}")
    print(f"  Damage F1 (macro mean)   : {m['f1_damage_macro_mean']:.4f}")
    print(f"  xView2-style composite   : {m['xview2_style_composite']:.4f}")
    for k, v in m["per_class_f1"].items():
        print(f"    {k:13s} F1 {v:.4f}   (GT pixels: {m['gt_pixel_support'][k]:,})")

    # 4. Grad-CAM over several tiles (not one hand-picked tile)
    print("\n[4] Grad-CAM")
    layer = model.head[0]   # fused FPN features (post pre/post fusion), stride 4
    rows = []
    for i, s in enumerate(val_ds.samples):
        if len(rows) >= CFG["n_gradcam_tiles"]:
            break
        sample = val_ds[i]
        gt = sample["target"].numpy()
        area = float((gt > 0).mean())
        if area < 0.005:            # skip tiles with almost no buildings
            continue
        cam, pred = grad_cam(model, sample["pre"][None].to(device), sample["post"][None].to(device), layer)
        e = compute_ebpg(cam, gt > 0)
        rows.append(dict(idx=i, id=sample["id"], ebpg=e, area=area, ratio=e / area,
                         damaged_px=int((gt >= 2).sum())))
    ebpgs = np.array([r["ebpg"] for r in rows])
    areas = np.array([r["area"] for r in rows])
    ratios = np.array([r["ratio"] for r in rows])
    results["gradcam"] = dict(
        layer="head[0] (fused FPN, stride 4)",
        n_tiles=len(rows),
        mean_ebpg=float(ebpgs.mean()), mean_building_area=float(areas.mean()),
        median_ratio_ebpg_over_area=float(np.median(ratios)),
        tiles_above_chance=int((ratios > 1).sum()),
        per_tile=rows,
    )
    g = results["gradcam"]
    print(f"  Tiles: {g['n_tiles']}  mean EBPG {g['mean_ebpg']*100:.1f}%  vs mean building area {g['mean_building_area']*100:.1f}%")
    print(f"  Median EBPG/area ratio {g['median_ratio_ebpg_over_area']:.2f}  (1.0 = no better than uniform)")
    print(f"  Tiles above chance: {g['tiles_above_chance']}/{g['n_tiles']}")

    # Figure: pre-registered rule = tile with the MOST ground-truth damaged pixels (not best heatmap)
    pick = max(rows, key=lambda r: r["damaged_px"])
    sample = val_ds[pick["idx"]]
    cam, pred = grad_cam(model, sample["pre"][None].to(device), sample["post"][None].to(device), layer)
    fig_path = os.path.join(OUT_DIR, "pilot_prediction_and_gradcam.png")
    save_figure(sample, pred, cam, pick["ebpg"], pick["area"], fig_path)
    results["figure"] = dict(path=fig_path, tile=pick["id"],
                             selection_rule="tile with most ground-truth damaged pixels among Grad-CAM tiles")
    print(f"  Figure: {fig_path}  (tile {pick['id']})")

    json.dump(results, open(os.path.join(OUT_DIR, "results", "pilot_results.json"), "w"), indent=2)
    write_report(results)
    print("\nDone.")


def write_report(r):
    lat, tr, m, g = r["latency"], r["training"], r["metrics"], r["gradcam"]
    L = []
    L.append("# Pilot results: ResNet-50 Siamese-FPN on xBD (local Apple M4)\n")
    L.append("_All numbers below are written automatically by `pilot/run_pilot.py` from measured values._\n")
    L.append("## Setup\n")
    L.append(f"- Backbone `{r['backbone']}`, {r['n_params_M']:.1f}M parameters, shared FPN decoder (d=256)")
    L.append(f"- Train: {r['config']['n_train']} pairs from `train 3/` ({len(r['train_events'])} events), "
             f"512 random crops, batch {r['config']['batch_size']}, {r['config']['epochs']} epochs, AdamW lr {r['config']['lr']}")
    L.append(f"- Loss: focal (class weights {r['config']['class_weights']}) + Dice on building classes")
    L.append(f"- Validation: {r['config']['n_val']} full 1024x1024 tiles from `hold/`, events: {', '.join(r['val_events'])}")
    L.append(f"- Device: {r['device']}, seed {r['config']['seed']} (single seed)\n")
    L.append("## 1. Inference latency (one pre/post pair, batch 1)\n")
    L.append("| Device | Input | Median | IQR | Runs |")
    L.append("| :-- | :-: | :-: | :-: | :-: |")
    for k, name in [("gpu_512", "M4 GPU (MPS)"), ("gpu_1024", "M4 GPU (MPS)"),
                    ("cpu_512", "M4 CPU"), ("cpu_1024", "M4 CPU")]:
        v = lat[k]; size = k.split("_")[1]
        L.append(f"| {name} | {size}x{size} | {v['median_ms']:.0f} ms | {v['p25_ms']:.0f}-{v['p75_ms']:.0f} ms | {v['n_runs']} |")
    L.append("")
    L.append("## 2. Training\n")
    L.append("| Epoch | Loss | Time |")
    L.append("| :-: | :-: | :-: |")
    for i, (l, s) in enumerate(zip(tr["epoch_loss"], tr["epoch_sec"]), 1):
        L.append(f"| {i} | {l:.4f} | {s:.0f} s |")
    L.append(f"\n- Throughput: {tr['train_pairs_per_sec']:.2f} training pairs/s (512 crops) on the M4")
    if "mps_driver_mem_GB" in tr:
        L.append(f"- MPS driver memory after training: {tr['mps_driver_mem_GB']:.1f} GB")
    L.append("- UCloud GPU time per epoch has **not** been measured yet; it will be timed on the first cloud run.\n")
    L.append("## 3. Validation metrics (pixel level, xView2-style)\n")
    L.append("Damage F1 is computed only on ground-truth building pixels, as in the xView2 scorer. "
             "These are not official leaderboard numbers (localization comes from the same post-disaster prediction).\n")
    L.append(f"- Localization F1: **{m['f1_localization']:.3f}**")
    L.append(f"- Damage F1, harmonic mean: **{m['f1_damage_harmonic']:.3f}** (macro mean {m['f1_damage_macro_mean']:.3f})")
    L.append(f"- Composite (0.3 loc + 0.7 dmg): **{m['xview2_style_composite']:.3f}**\n")
    L.append("| Class | F1 | GT pixels |")
    L.append("| :-- | :-: | --: |")
    for k, v in m["per_class_f1"].items():
        L.append(f"| {k} | {v:.3f} | {m['gt_pixel_support'][k]:,} |")
    L.append("")
    L.append("## 4. Grad-CAM attribution\n")
    L.append(f"- Layer: {g['layer']}; gradient-weighted (Seg-Grad-CAM), target = logits of the predicted building classes")
    L.append(f"- Tiles analysed: {g['n_tiles']} (all validation tiles with >=0.5% building area, in order)")
    L.append(f"- Mean EBPG (attribution inside footprints): **{g['mean_ebpg']*100:.1f}%**")
    L.append(f"- Mean building area of those tiles (= score of a uniform heatmap): **{g['mean_building_area']*100:.1f}%**")
    L.append(f"- Median EBPG / area ratio: **{g['median_ratio_ebpg_over_area']:.2f}** (1.0 = no better than chance)")
    L.append(f"- Tiles where attribution is concentrated above chance: {g['tiles_above_chance']}/{g['n_tiles']}")
    L.append(f"- Figure: `pilot_prediction_and_gradcam.png`, tile `{r['figure']['tile']}` "
             f"(selection rule: {r['figure']['selection_rule']})\n")
    L.append("## Caveats\n")
    L.append("- Small pilot: 400 training pairs, 5 epochs, one seed. Not a benchmark result.")
    L.append("- Validation uses `hold/`; the final study will use the official test/hold protocol with a separate validation split for checkpoint selection.")
    open(os.path.join(OUT_DIR, "pilot_report.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
