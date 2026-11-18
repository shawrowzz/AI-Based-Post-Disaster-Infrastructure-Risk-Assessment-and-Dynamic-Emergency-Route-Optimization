import torch
import numpy as np

CLASS_NAMES = ["background", "no_damage", "minor_damage", "major_damage", "destroyed"]


class XBDMetrics:
    """
    Pixel-level metrics following the xView2 scoring convention:
      - Localization F1: building (classes 1-4) vs background (class 0), over all pixels.
      - Damage F1 (per class 1-4): computed ONLY on pixels that are buildings in the ground truth.
        A building pixel predicted as background counts as a false negative for its true class.
      - Damage score: harmonic mean of the four per-class damage F1s.
      - Composite: 0.3 * localization F1 + 0.7 * damage harmonic F1.
    TP/FP/FN are accumulated over the whole evaluation set before computing F1 (as in xView2).

    Note: the official xView2 scorer takes localization from a separate prediction on the
    pre-disaster image. Here one post-disaster mask provides both, so these are
    "xView2-style" scores, not official leaderboard numbers.
    """
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # rows = ground truth, cols = prediction
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        p = preds.flatten().astype(np.int64)
        t = targets.flatten().astype(np.int64)
        valid = (t >= 0) & (t < self.num_classes) & (p >= 0) & (p < self.num_classes)
        hist = np.bincount(self.num_classes * t[valid] + p[valid],
                           minlength=self.num_classes ** 2)
        self.confusion_matrix += hist.reshape(self.num_classes, self.num_classes)

    @staticmethod
    def _f1(tp, fp, fn):
        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else 0.0

    def compute(self):
        cm = self.confusion_matrix

        # Localization: any building class vs background, all pixels
        tp_loc = cm[1:, 1:].sum()
        fp_loc = cm[0, 1:].sum()
        fn_loc = cm[1:, 0].sum()
        f1_loc = self._f1(tp_loc, fp_loc, fn_loc)

        # Damage: restricted to ground-truth building pixels (rows 1..4)
        per_class_f1 = {}
        support = {}
        for c in range(1, self.num_classes):
            tp = cm[c, c]
            fp = cm[1:, c].sum() - tp      # predicted c on building pixels of another class
            fn = cm[c, :].sum() - tp       # true c predicted as anything else (incl. background)
            per_class_f1[CLASS_NAMES[c]] = self._f1(tp, fp, fn)
            support[CLASS_NAMES[c]] = int(cm[c, :].sum())

        f1s = list(per_class_f1.values())
        f1_damage = len(f1s) / sum(1.0 / (f + 1e-6) for f in f1s)
        composite = 0.3 * f1_loc + 0.7 * f1_damage

        return {
            "f1_localization": f1_loc,
            "f1_damage_harmonic": float(f1_damage),
            "f1_damage_macro_mean": float(np.mean(f1s)),
            "xview2_style_composite": float(composite),
            "per_class_f1": per_class_f1,
            "gt_pixel_support": support,
        }


def compute_ebpg(attribution_map, building_mask):
    """
    Energy-Based Pointing Game: share of (non-negative) attribution energy inside building footprints.
    Always compare against the building area fraction of the tile, which is what a uniform
    (uninformative) heatmap would score.
    """
    if isinstance(attribution_map, torch.Tensor):
        attribution_map = attribution_map.detach().cpu().numpy()
    if isinstance(building_mask, torch.Tensor):
        building_mask = building_mask.detach().cpu().numpy()

    attribution_map = np.clip(attribution_map, 0, None)
    inside = (attribution_map * (building_mask > 0)).sum()
    total = attribution_map.sum() + 1e-12
    return float(inside / total)
