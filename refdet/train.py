"""Training script for Enhanced Siamese Detector V2."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EnhancedSiameseDetector
from utils.dataset import PatchRetrievalDataset
from utils.metrics import compute_iou, compute_patch_accuracy
from utils.geometry import decode_patch_bbox


def focal_loss(pred_probs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 0.05, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute Focal Loss with label smoothing for addressing class imbalance.
    
    Args:
        pred_probs: (B, 16) - Predicted probabilities (already sigmoid)
        targets: (B, 16) - Soft targets (IoU values 0-1)
        alpha: Weighting factor for positive class (0.25 means 25% weight for positive)
        gamma: Focusing parameter (higher = more focus on hard examples)
        smooth: Label smoothing factor
        reduction: 'mean' (average over batch) or 'none' (per-sample loss)
        
    Returns:
        loss: Focal loss value (scalar if reduction='mean', (B,) if reduction='none')
    """
    eps = 1e-7
    pred_probs = torch.clamp(pred_probs, eps, 1 - eps)
    
    # Apply label smoothing to targets
    # smooth / 2 gives: target=1 → 0.975, target=0 → 0.025 (with smooth=0.05)
    targets_smooth = targets * (1 - smooth) + smooth / 2
    
    # Compute binary cross entropy with smoothed targets
    ce_loss = -(targets_smooth * torch.log(pred_probs) + (1 - targets_smooth) * torch.log(1 - pred_probs))
    
    # Compute p_t (probability of true class) - use original targets for focal weighting
    p_t = pred_probs * targets + (1 - pred_probs) * (1 - targets)
    
    # Compute alpha_t (class weighting) - use original targets
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    
    # Compute focal weight: alpha_t * (1 - p_t)^gamma
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    # Apply focal weight to cross entropy loss
    focal_loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'none':
        return focal_loss.mean(dim=1)  # (B,) - average over 16 patches per sample
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def patch_classification_loss(pred_probs: torch.Tensor, target_heatmaps: torch.Tensor, 
                            use_focal: bool = True, smooth: float = 0.05) -> torch.Tensor:
    """
    Compute patch classification loss with option for Focal Loss.
    
    Args:
        pred_probs: (B, 16, 1) - Predicted probabilities (already sigmoid)
        target_heatmaps: (B, 16) - Target heatmaps (0 or 1)
        use_focal: Whether to use Focal Loss (True) or smooth BCE (False)
        smooth: Label smoothing factor (only used if use_focal=False)
        
    Returns:
        loss: Classification loss
    """
    pred_probs = pred_probs.squeeze(-1)  # (B, 16)
    
    if use_focal:
        # Use Focal Loss with label smoothing to handle class imbalance
        return focal_loss(pred_probs, target_heatmaps, alpha=0.25, gamma=2.0, smooth=smooth)
    else:
        # Use smooth BCE loss (original approach)
        target_smooth = target_heatmaps * (1 - smooth) + smooth / 2
        eps = 1e-7
        pred_probs = torch.clamp(pred_probs, eps, 1 - eps)
        loss = -(target_smooth * torch.log(pred_probs) + (1 - target_smooth) * torch.log(1 - pred_probs))
        return loss.mean()


def patch_regression_loss(pred_deltas: torch.Tensor, target_deltas: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute patch regression loss only at positive patches.
    
    Args:
        pred_deltas: (B, 16, 4) - Predicted bbox deltas
        target_deltas: (B, 16, 4) - Target bbox deltas
        pos_mask: (B, 16) - Positive mask
        
    Returns:
        loss: Smooth L1 loss
    """
    # Apply mask: only compute loss at positive patches
    pred_masked = pred_deltas[pos_mask]  # (N, 4) where N = number of positive patches
    target_masked = target_deltas[pos_mask]  # (N, 4)
    
    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=pred_deltas.device, requires_grad=True)
    
    return F.smooth_l1_loss(pred_masked, target_masked)


def train_one_epoch(model, loader, optimizer, device, epoch: int = 0) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_cls, total_reg = 0.0, 0.0
    total_acc = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", ncols=100)
    for batch in pbar:
        template = batch["template"].to(device)
        search = batch["search"].to(device)
        target_heatmaps = batch["patch_heatmaps"].to(device)
        target_deltas = batch["patch_bbox_deltas"].to(device)
        pos_mask = batch["patch_pos_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass (no AMP)
        cls_probs, bbox_deltas = model(template, search)
        
        # Compute losses
        # Classification loss with per-sample weighting
        cls_probs_squeezed = cls_probs.squeeze(-1)  # (B, 16)
        cls_loss_per_sample = focal_loss(cls_probs_squeezed, target_heatmaps, alpha=0.25, gamma=2.0, smooth=0.05, reduction='none')  # (B,)
        
        # Object size normalization: weight by number of positive patches
        # Small objects (1 pos patch) get higher weight than large objects (4 pos patches)
        num_pos_patches = pos_mask.sum(dim=1).float()  # (B,) - number of positive patches per sample
        size_weights = 1.0 / torch.sqrt(num_pos_patches + 1e-6)  # Inverse sqrt weighting
        
        # Apply per-sample size weighting to classification loss
        weighted_cls_loss = (size_weights * cls_loss_per_sample).mean()  # Weight each sample then average
        
        # Regression loss (already handles positive patches internally)
        reg_loss = patch_regression_loss(bbox_deltas, target_deltas, pos_mask)
        
        reg_weight = 2.0
        loss = weighted_cls_loss + reg_weight * reg_loss
        
        # For logging, compute unweighted cls_loss
        cls_loss = cls_loss_per_sample.mean()

        # Backprop with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Compute accuracy (cls_probs already sigmoid)
        acc = compute_patch_accuracy(cls_probs, target_heatmaps)

        total_cls += cls_loss.item()
        total_reg += reg_loss.item()
        total_acc += acc
        
        # Update progress bar with better formatting
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cls": f"{cls_loss.item():.4f}",
            "reg": f"{reg_loss.item():.4f}",
            "acc": f"{acc:.3f}"
        })

    steps = len(loader)
    return {
        "cls_loss": total_cls / steps,
        "reg_loss": total_reg / steps,
        "patch_accuracy": total_acc / steps,
    }


def evaluate(model, loader, device, epoch: int = 0) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_iou = 0.0
    total_samples = 0
    total_acc = 0.0
    
    patch_grid_info = model.get_patch_grid_info()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", ncols=100)
        for batch in pbar:
            template = batch["template"].to(device)
            search = batch["search"].to(device)
            gt_boxes = batch["gt_box"].to(device)
            target_heatmaps = batch["patch_heatmaps"].to(device)

            cls_probs, bbox_deltas = model(template, search)
            
            # Compute patch accuracy
            acc = compute_patch_accuracy(cls_probs, target_heatmaps)
            total_acc += acc
            
            # Find best patch for each sample (cls_probs already sigmoid)
            patch_probs = cls_probs.squeeze(-1)  # (B, 16)
            best_patch_idx = patch_probs.argmax(dim=1)  # (B,)
            
            # Decode bbox from best patch
            pred_boxes = []
            for i in range(len(best_patch_idx)):
                patch_idx = best_patch_idx[i].item()
                deltas = bbox_deltas[i, patch_idx]  # (4,)
                pred_box = decode_patch_bbox(patch_idx, deltas, patch_grid_info)
                pred_boxes.append(pred_box)
            
            pred_boxes = torch.stack(pred_boxes).to(device)  # (B, 4)
            
            # Compute IoU
            iou = compute_iou(pred_boxes, gt_boxes)
            total_iou += iou.sum().item()
            total_samples += gt_boxes.size(0)
            
            # Update progress bar
            current_iou = total_iou / total_samples if total_samples > 0 else 0.0
            current_acc = total_acc / (pbar.n + 1)
            pbar.set_postfix({"mIoU": f"{current_iou:.4f}", "acc": f"{current_acc:.3f}"})

    return {
        "mean_iou": total_iou / total_samples,
        "patch_accuracy": total_acc / len(loader)
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = EnhancedSiameseDetector(
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    model.to(device)

    # Datasets
    train_dataset = PatchRetrievalDataset(args.data_dir, split="train", augment=True, augment_prob=args.augment_prob, img_size=640)
    val_dataset = PatchRetrievalDataset(args.data_dir, split="val", augment=False, img_size=640)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.workers, pin_memory=True)

    # Optimizer (no AMP scaler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if args.lr_schedule == "cosine":
        min_lr = args.min_lr if args.min_lr is not None else args.lr * 0.01
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=min_lr
        )
    elif args.lr_schedule == "linear":
        min_lr = args.min_lr if args.min_lr is not None else args.lr * 0.01
        end_factor = min_lr / args.lr
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=args.epochs
        )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_miou = 0.0
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        checkpoint_state = checkpoint['model']
        
        # Detect num_layers from checkpoint
        checkpoint_keys = checkpoint_state.keys()
        max_layer_idx = -1
        for key in checkpoint_keys:
            if 'self_attn_layers' in key:
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    max_layer_idx = max(max_layer_idx, layer_idx)
        
        checkpoint_num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else args.num_layers
        
        # Check if adaptive_proj exists in checkpoint
        has_adaptive_proj = any('patch_embed.adaptive_proj' in key for key in checkpoint_state.keys())
        
        # If checkpoint has adaptive_proj, create it in model before loading
        if has_adaptive_proj and model.patch_embed.adaptive_proj is None:
            adaptive_proj_weight = checkpoint_state.get('patch_embed.adaptive_proj.weight')
            if adaptive_proj_weight is not None:
                patch_dim = adaptive_proj_weight.shape[1]
                embed_dim = adaptive_proj_weight.shape[0]
                model.patch_embed.adaptive_proj = nn.Linear(patch_dim, embed_dim).to(device)
                model.patch_embed.add_module('adaptive_proj', model.patch_embed.adaptive_proj)
        
        # Load state dict
        try:
            model.load_state_dict(checkpoint_state, strict=True)
        except RuntimeError:
            model.load_state_dict(checkpoint_state, strict=False)
        
        start_epoch = 0
        print(f"✓ Loaded checkpoint: {Path(args.checkpoint_path).name}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging
    training_log_file = output_dir / "training_log.json"
    training_log = []
    
    print(f"Training: {len(train_dataset)} train, {len(val_dataset)} val | Device: {device} | Batch: {args.batch_size} | Epochs: {args.epochs}")
    if args.checkpoint_path:
        print(f"Resuming from epoch {start_epoch}\n")
    else:
        print()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        
        # Training
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, epoch)
        
        mIoU = val_metrics["mean_iou"]
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler to reduce learning rate (if enabled)
        if scheduler is not None:
            scheduler.step()
        
        # Log epoch info
        epoch_log = {
            "epoch": epoch_num,
            "train_cls_loss": train_metrics["cls_loss"],
            "train_reg_loss": train_metrics["reg_loss"],
            "train_patch_accuracy": train_metrics["patch_accuracy"],
            "val_mIoU": mIoU,
            "val_patch_accuracy": val_metrics["patch_accuracy"],
            "learning_rate": current_lr,
            "timestamp": datetime.now().isoformat(),
        }
        training_log.append(epoch_log)
        
        # Save training log
        with open(training_log_file, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # Save best model (model only, no optimizer/scheduler)
        if mIoU > best_miou:
            best_miou = mIoU
            best_model_path = output_dir / f"best_model_epoch_{epoch_num}_mIoU_{mIoU:.4f}.pth"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch_num,
                "val_mIoU": mIoU,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": vars(args),
            }, best_model_path)
            print(f"New best model saved: mIoU={mIoU:.4f}")
        
        # Save last model (model only, no optimizer/scheduler)
        last_model_path = output_dir / f"last_model_epoch_{epoch_num}.pth"
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch_num,
            "val_mIoU": mIoU,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": vars(args),
        }, last_model_path)
        
        # Save checkpoint every 5 epochs (model only, no optimizer/scheduler)
        if epoch_num % 5 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch_num}.pth"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch_num,
                "val_mIoU": mIoU,
                "best_miou": best_miou,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": vars(args),
            }, checkpoint_path)
        
        # Print epoch summary (only every 5 epochs or first epoch)
        if epoch_num % 5 == 0 or epoch_num == 1:
            print(f"Epoch {epoch_num}/{args.epochs}: "
                  f"cls={train_metrics['cls_loss']:.4f} "
                  f"reg={train_metrics['reg_loss']:.4f} "
                  f"acc={train_metrics['patch_accuracy']:.3f} "
                  f"mIoU={mIoU:.4f} "
                  f"(best: {best_miou:.4f}) "
                  f"LR={current_lr:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Enhanced Siamese Detector V2")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to retrieval dataset root")
    parser.add_argument("--output_dir", type=str, default="outputs_v2", help="Where to store checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--augment_prob", type=float, default=0.2, help="Probability of applying augmentation (0.2 = 20% of data)")
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine", "linear"], 
                       help="LR schedule: constant (fixed), cosine (annealing), or linear (decay)")
    parser.add_argument("--min_lr", type=float, default=None, help="Minimum learning rate for cosine/linear schedule (default: lr * 0.01)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()

    main(args)
