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


def patch_classification_loss(pred_probs: torch.Tensor, target_heatmaps: torch.Tensor, smooth: float = 0.1) -> torch.Tensor:
    """
    Compute smooth patch classification loss.
    
    Args:
        pred_probs: (B, 16, 1) - Predicted probabilities (already sigmoid)
        target_heatmaps: (B, 16) - Target heatmaps (0 or 1)
        smooth: Label smoothing factor
        
    Returns:
        loss: Smooth binary cross-entropy loss
    """
    pred_probs = pred_probs.squeeze(-1)  # (B, 16)
    
    # Label smoothing: 0 -> smooth/2, 1 -> 1 - smooth/2
    target_smooth = target_heatmaps * (1 - smooth) + smooth / 2
    
    # BCE loss with smooth targets (pred_probs already sigmoid)
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


def train_one_epoch(model, loader, optimizer, scaler, device, epoch: int = 0) -> Dict[str, float]:
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
        
        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            cls_probs, bbox_deltas = model(template, search)
            
            cls_loss = patch_classification_loss(cls_probs, target_heatmaps, smooth=0.05)
            reg_loss = patch_regression_loss(bbox_deltas, target_deltas, pos_mask)
            loss = cls_loss + reg_loss

        # Check for NaN/Inf loss before backward
        if not torch.isfinite(loss):
            # Log and skip this batch to avoid poisoning the optimizer/scaler
            pbar.write(f"Warning: non-finite loss detected (loss={loss}). Skipping batch.")
            # zero grads already set; skip update and continue
            continue

        # Backprop with AMP scaler. Unscale and clip grads to prevent explosion.
        scaler.scale(loss).backward()
        # Unscale gradients before clipping (required for AMP)
        try:
            scaler.unscale_(optimizer)
        except Exception:
            # some older torch versions may not have unscale_; ignore if unavailable
            pass
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Compute accuracy (cls_probs already sigmoid)
        acc = compute_patch_accuracy(cls_probs, target_heatmaps)

        total_cls += cls_loss.item()
        total_reg += reg_loss.item()
        total_acc += acc
        
        # Update progress bar with better formatting
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cls": f"{cls_loss.item():.4f}",
            "reg": f"{reg_loss.item():.4f}" if reg_loss.item() > 0 else "0.0000",
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

    # Optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")
    
    # Learning rate scheduler: Cosine annealing to gradually reduce LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    best_miou = 0.0
    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Detect num_layers from checkpoint
        checkpoint_keys = checkpoint['model'].keys()
        max_layer_idx = -1
        for key in checkpoint_keys:
            if 'self_attn_layers' in key:
                # Extract layer index from key like "self_attn_layers.2.norm1.weight"
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    max_layer_idx = max(max_layer_idx, layer_idx)
        
        checkpoint_num_layers = max_layer_idx + 1 if max_layer_idx >= 0 else args.num_layers
        
        if checkpoint_num_layers != args.num_layers:
            print(f"\n⚠️  WARNING: Checkpoint has {checkpoint_num_layers} layers but model was created with {args.num_layers} layers!")
            print(f"   Please use --num_layers {checkpoint_num_layers} to match the checkpoint.")
            print(f"   Continuing with strict=False (missing layers will be randomly initialized)...\n")
        
        # Check if adaptive_proj exists in checkpoint
        checkpoint_state = checkpoint['model']
        has_adaptive_proj = any('patch_embed.adaptive_proj' in key for key in checkpoint_state.keys())
        
        # If checkpoint has adaptive_proj, create it in model before loading
        if has_adaptive_proj and model.patch_embed.adaptive_proj is None:
            # Get patch_dim from checkpoint weight shape
            adaptive_proj_weight = checkpoint_state.get('patch_embed.adaptive_proj.weight')
            if adaptive_proj_weight is not None:
                patch_dim = adaptive_proj_weight.shape[1]
                embed_dim = adaptive_proj_weight.shape[0]
                model.patch_embed.adaptive_proj = nn.Linear(patch_dim, embed_dim).to(device)
                model.patch_embed.add_module('adaptive_proj', model.patch_embed.adaptive_proj)
                print(f"   Created adaptive_proj from checkpoint (patch_dim={patch_dim}, embed_dim={embed_dim})")
        
        # Try to load state dict
        try:
            model.load_state_dict(checkpoint_state, strict=True)
            print("✓ Checkpoint loaded successfully (strict=True)")
        except RuntimeError as e:
            print(f"⚠️  Loading with strict=False due to key mismatches...")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
            if missing_keys:
                print(f"   Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
            if unexpected_keys:
                print(f"   Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0)
        best_miou = checkpoint.get('best_miou', checkpoint.get('val_mIoU', 0.0))
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")

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
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, epoch)
        
        mIoU = val_metrics["mean_iou"]
        current_lr = optimizer.param_groups[0]['lr']
        
        # Step scheduler to reduce learning rate
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
        
        # Save best model
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
        
        # Save last model
        last_model_path = output_dir / f"last_model_epoch_{epoch_num}.pth"
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch_num,
            "val_mIoU": mIoU,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": vars(args),
        }, last_model_path)
        
        # Save checkpoint every 20 epochs
        if epoch_num % 20 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch_num}.pth"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
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
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    args = parser.parse_args()

    main(args)
