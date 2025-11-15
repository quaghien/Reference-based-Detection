"""Dataset for patch-based Siamese detection."""

import random
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset

from .transforms import build_transforms
from .geometry import make_patch_heatmaps, encode_patch_bbox_deltas


class PatchRetrievalDataset(Dataset):
    """Dataset for patch-based Siamese retrieval detection."""

    def __init__(self, root: str, split: str = "train", augment: bool = True, augment_prob: float = 0.75, img_size: int = 640, hard_mining: bool = True):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.augment_prob = augment_prob  # Probability of applying augmentation (0.75 = 75%)
        self.img_size = img_size
        self.hard_mining = hard_mining  # Enable hard mining for training

        # Patch configuration (fixed for 640x640, 4x4 grid)
        self.patch_grid_info = {
            'grid_h': 4,
            'grid_w': 4,
            'num_patches': 16,
            'img_size': 640
        }

        self.template_dir = self.root / split / "templates"
        self.search_images_dir = self.root / split / "search" / "images"
        self.search_labels_dir = self.root / split / "search" / "labels"

        self.template_paths = self._collect_templates()
        self.samples = self._collect_samples()
        
        # Hard mining: categorize samples by difficulty
        if self.hard_mining and split == "train":
            self.hard_samples = self._identify_hard_samples()
        else:
            self.hard_samples = []

        self.transform = build_transforms(img_size=self.img_size, augment=augment)
        self.augment = augment

    @staticmethod
    def _extract_video_id(filename: str) -> str:
        """Extract video_id from filename."""
        name = Path(filename).stem
        parts = name.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]

    def _collect_templates(self) -> Dict[str, list]:
        """Collect templates from flat structure. Group by video_id."""
        templates = {}
        template_files = list(self.template_dir.glob("*.jpg")) + list(self.template_dir.glob("*.png"))
        
        for template_path in template_files:
            video_id = self._extract_video_id(template_path.name)
            if video_id not in templates:
                templates[video_id] = []
            templates[video_id].append(template_path)
        
        if not templates:
            raise RuntimeError(f"No templates found in {self.template_dir}")
        return templates

    def _collect_samples(self):
        """Collect samples from flat structure. Group by video_id."""
        samples = []
        image_files = sorted(self.search_images_dir.glob("*.jpg")) + sorted(self.search_images_dir.glob("*.png"))
        
        for img_path in image_files:
            label_path = self.search_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                video_id = self._extract_video_id(img_path.name)
                samples.append((video_id, img_path, label_path))
        
        if not samples:
            raise RuntimeError(f"No search frames found in {self.search_images_dir}")
        return samples
    
    def _identify_hard_samples(self):
        """Identify hard samples based on object size and position."""
        hard_samples = []
        
        for i, (video_id, img_path, label_path) in enumerate(self.samples):
            try:
                with open(label_path) as f:
                    line = f.readline().strip()
                parts = line.split()
                _, x_c, y_c, w, h = map(float, parts[:5])
                
                # Hard criteria for drone surveillance:
                # 1. Small objects (w*h < 0.01 = objects < 64px in 640x640)
                # 2. Objects near patch boundaries
                # 3. Objects with aspect ratio far from 1:1
                
                object_area = w * h
                aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                
                # Check if object is near patch boundaries (4x4 grid)
                patch_x = int(x_c * 4)
                patch_y = int(y_c * 4)
                near_boundary = (abs(x_c * 4 - patch_x - 0.5) > 0.3 or 
                               abs(y_c * 4 - patch_y - 0.5) > 0.3)
                
                # Hard sample criteria
                is_hard = (object_area < 0.01 or  # Small objects
                          aspect_ratio > 3.0 or   # Elongated objects  
                          near_boundary)           # Near patch boundaries
                
                if is_hard:
                    hard_samples.append(i)
                    
            except Exception:
                continue  # Skip corrupted labels
                
        return hard_samples

    def __len__(self) -> int:
        # For hard mining, oversample hard samples (33% of batch)
        if self.hard_mining and self.split == "train" and len(self.hard_samples) > 0:
            return len(self.samples) + len(self.hard_samples) // 3
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample for patch-based training.
        
        Returns:
            Dict với keys:
            - "template": (3, 640, 640) - Template image tensor
            - "search": (3, 640, 640) - Search frame tensor  
            - "patch_heatmaps": (16,) - Heatmaps for each patch (IoU values)
            - "patch_bbox_deltas": (16, 4) - Bbox deltas for each patch
            - "patch_pos_mask": (16,) - Positive mask for each patch
            - "gt_box": (4,) - [x_c, y_c, w, h] normalized coordinates
        """
        # Hard mining: oversample hard samples
        if (self.hard_mining and self.split == "train" and 
            len(self.hard_samples) > 0 and idx >= len(self.samples)):
            # Use hard sample
            hard_idx = (idx - len(self.samples)) % len(self.hard_samples)
            actual_idx = self.hard_samples[hard_idx]
        else:
            actual_idx = idx % len(self.samples)
            
        # Get search image and label
        video_id, img_path, label_path = self.samples[actual_idx]
        
        # Choose random template from same video_id
        template_path = random.choice(self.template_paths[video_id])

        # Transform both images with same augmentation parameters (if augmenting)
        # Only apply augmentation with probability augment_prob to preserve data distribution
        should_augment = self.augment and (random.random() < self.augment_prob)
        
        if should_augment:
            # Generate augmentation parameters once and apply to both images
            aug_params = {
                'angle': random.uniform(-10, 10),
                'flip_h': random.random() < 0.5,
                'flip_v': random.random() < 0.3,
                'brightness': random.uniform(0.6, 1.4),
                'contrast': random.uniform(0.7, 1.3),
                'saturation': random.uniform(0.7, 1.3),
                'affine_angle': random.uniform(-5, 5),
                'affine_translate': (random.uniform(-0.1, 0.1) * self.img_size,
                                    random.uniform(-0.1, 0.1) * self.img_size),
                'affine_scale': random.uniform(0.9, 1.1),
                'affine_shear': (random.uniform(-5, 5), random.uniform(-5, 5)),
            }
            search_img = self.transform(img_path, aug_params=aug_params)
            template_img = self.transform(template_path, aug_params=aug_params)
        else:
            search_img = self.transform(img_path, aug_params=None)
            template_img = self.transform(template_path, aug_params=None)
            aug_params = None

        # Read ground truth bbox from label file (YOLO format)
        with open(label_path) as f:
            line = f.readline().strip()
        parts = line.split()
        _, x_c, y_c, w, h = map(float, parts[:5])  # Normalized coordinates (0-1)
        
        # Transform bbox coordinates if geometric augmentation was applied
        if should_augment and aug_params is not None:
            from .transforms import transform_bbox
            
            # Prepare affine params
            affine_params = None
            if 'affine_angle' in aug_params:
                affine_params = (
                    aug_params['affine_angle'],
                    aug_params['affine_translate'],
                    aug_params['affine_scale'],
                    aug_params['affine_shear'],
                )
            
            # Transform bbox with all geometric augmentations
            x_c, y_c, w, h = transform_bbox(
                x_c, y_c, w, h,
                img_size=self.img_size,
                angle=aug_params.get('angle'),
                flip_h=aug_params.get('flip_h', False),
                flip_v=aug_params.get('flip_v', False),
                affine_params=affine_params
            )
        
        # Create patch-based targets with soft labels (IoU-based)
        patch_heatmaps = make_patch_heatmaps((x_c, y_c), self.patch_grid_info, box_wh=(w, h))
        patch_bbox_deltas, patch_pos_mask = encode_patch_bbox_deltas((x_c, y_c, w, h), self.patch_grid_info)

        return {
            "template": template_img,
            "search": search_img,
            "patch_heatmaps": patch_heatmaps,           # (16,) - Target for classification
            "patch_bbox_deltas": patch_bbox_deltas,     # (16, 4) - Target for regression
            "patch_pos_mask": patch_pos_mask,           # (16,) - Mask for regression loss
            "gt_box": torch.tensor([x_c, y_c, w, h], dtype=torch.float32),
        }
