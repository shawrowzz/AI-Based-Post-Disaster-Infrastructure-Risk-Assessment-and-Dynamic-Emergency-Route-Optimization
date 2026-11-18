import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class XBDDataset(Dataset):
    """
    Dataset loader for xBD paired pre- and post-disaster satellite imagery.
    Loads:
      - Pre-disaster RGB (3 channels)
      - Post-disaster RGB (3 channels)
      - Damage target mask (1 channel, values 0-4: background, no damage, minor, major, destroyed)
    """
    def __init__(self, data_dir, img_size=512, is_train=True, max_samples=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train

        images_dir = os.path.join(data_dir, "images")
        targets_dir = os.path.join(data_dir, "targets")

        if not os.path.exists(images_dir) or not os.path.exists(targets_dir):
            raise ValueError(f"Directory {data_dir} must contain 'images' and 'targets' folders.")

        # Match post-disaster images to pre-disaster images and targets
        all_post_targets = [f for f in os.listdir(targets_dir) if f.endswith("_post_disaster_target.png")]
        all_post_targets.sort()

        self.samples = []
        for target_file in all_post_targets:
            prefix = target_file.replace("_post_disaster_target.png", "")
            pre_img_name = f"{prefix}_pre_disaster.png"
            post_img_name = f"{prefix}_post_disaster.png"

            pre_path = os.path.join(images_dir, pre_img_name)
            post_path = os.path.join(images_dir, post_img_name)
            target_path = os.path.join(targets_dir, target_file)

            if os.path.exists(pre_path) and os.path.exists(post_path) and os.path.exists(target_path):
                self.samples.append({
                    "pre_path": pre_path,
                    "post_path": post_path,
                    "target_path": target_path,
                    "id": prefix
                })

        # Subsample evenly across the sorted list so every disaster event is represented
        # (taking the first N files would only cover the first one or two events alphabetically).
        if max_samples is not None and max_samples < len(self.samples):
            idx = np.linspace(0, len(self.samples) - 1, max_samples).round().astype(int)
            self.samples = [self.samples[i] for i in idx]

        # Normalization (ImageNet stats)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if self.is_train:
            self.transform = A.Compose([
                A.RandomCrop(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ], additional_targets={'image_post': 'image'})
        else:
            # Evaluate on the full 1024x1024 tile at native 0.8 m resolution (no resizing),
            # so validation sees the same ground resolution as the 512 training crops.
            self.transform = A.Compose([
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ], additional_targets={'image_post': 'image'})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        pre_img = np.array(Image.open(sample["pre_path"]).convert("RGB"))
        post_img = np.array(Image.open(sample["post_path"]).convert("RGB"))
        target_mask = np.array(Image.open(sample["target_path"]))

        # Apply paired transforms
        augmented = self.transform(image=pre_img, image_post=post_img, mask=target_mask)

        pre_tensor = augmented['image']          # [3, H, W]
        post_tensor = augmented['image_post']    # [3, H, W]
        target_tensor = augmented['mask'].long() # [H, W] with classes 0..4

        return {
            "pre": pre_tensor,
            "post": post_tensor,
            "target": target_tensor,
            "id": sample["id"]
        }
