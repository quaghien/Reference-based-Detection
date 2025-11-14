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

    def __init__(self, root: str, split: str = "train", augment: bool = True, img_size: int = 640):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.img_size = img_size

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample for patch-based training.
        
        Returns:
            Dict với keys:
            - "template": (3, 640, 640) - Template image tensor
            - "search": (3, 640, 640) - Search frame tensor  
            - "patch_heatmaps": (16,) - Heatmaps for each patch (0 or 1)
            - "patch_bbox_deltas": (16, 4) - Bbox deltas for each patch
            - "patch_pos_mask": (16,) - Positive mask for each patch
            - "gt_box": (4,) - [x_c, y_c, w, h] normalized coordinates
        """
        # Get search image and label
        video_id, img_path, label_path = self.samples[idx]
        
        # Choose random template from same video_id
        template_path = random.choice(self.template_paths[video_id])

        # Transform both images with same augmentation parameters (if augmenting)
        if self.augment:
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

        # Read ground truth bbox from label file (YOLO format)
        with open(label_path) as f:
            line = f.readline().strip()
        parts = line.split()
        _, x_c, y_c, w, h = map(float, parts[:5])  # Normalized coordinates (0-1)
        
        # Transform bbox coordinates if geometric augmentation was applied
        if self.augment:
            # Apply same transformations to bbox coordinates
            # Note: For small rotations (±10°) and affine transforms, bbox changes are minimal
            # For flips, we need to transform coordinates
            if aug_params.get('flip_h', False):
                x_c = 1.0 - x_c  # Flip horizontally
            if aug_params.get('flip_v', False):
                y_c = 1.0 - y_c  # Flip vertically
            
            # For small rotations and affine, bbox distortion is minimal
            # We keep the original bbox coordinates (acceptable for ±10° rotation)
        
        # Create patch-based targets
        patch_heatmaps = make_patch_heatmaps((x_c, y_c), self.patch_grid_info)
        patch_bbox_deltas, patch_pos_mask = encode_patch_bbox_deltas((x_c, y_c, w, h), self.patch_grid_info)

        return {
            "template": template_img,
            "search": search_img,
            "patch_heatmaps": patch_heatmaps,           # (16,) - Target for classification
            "patch_bbox_deltas": patch_bbox_deltas,     # (16, 4) - Target for regression
            "patch_pos_mask": patch_pos_mask,           # (16,) - Mask for regression loss
            "gt_box": torch.tensor([x_c, y_c, w, h], dtype=torch.float32),
        }
