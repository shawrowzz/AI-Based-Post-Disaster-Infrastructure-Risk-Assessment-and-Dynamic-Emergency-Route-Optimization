import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [B, C, H, W]
        # targets: [B, H, W] with class indices 0..C-1
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dice_total = 0.0
        classes_counted = 0

        for c in range(num_classes):
            if self.ignore_index is not None and c == self.ignore_index:
                continue

            p = probs[:, c, :, :]
            t = targets_one_hot[:, c, :, :]

            intersection = (p * t).sum()
            union = p.sum() + t.sum()

            dice_c = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_total += (1.0 - dice_c)
            classes_counted += 1

        return dice_total / max(classes_counted, 1)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # can be a tensor of class weights

    def forward(self, logits, targets):
        # pt must come from the UNweighted CE (it is the true-class probability);
        # class weights are applied afterwards, then normalised by total weight.
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            w = self.alpha.to(logits.device)[targets]
            return (focal_loss * w).sum() / w.sum().clamp_min(1e-8)
        return focal_loss.mean()

class XBDLoss(nn.Module):
    """
    Combined Focal Loss + Multi-class Dice Loss for xBD Building Damage Assessment.
    Ignore background (class 0) in Dice Loss so the model optimizes specifically for building structures.
    """
    def __init__(self, focal_weight=1.0, dice_weight=1.0, gamma=2.0, class_weights=None, ignore_bg_dice=True):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(gamma=gamma, alpha=class_weights)
        self.dice = DiceLoss(ignore_index=0 if ignore_bg_dice else None)

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.focal_weight * loss_focal + self.dice_weight * loss_dice
