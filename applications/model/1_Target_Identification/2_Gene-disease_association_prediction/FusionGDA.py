"""
FusionGDA: Unified Model for Gene-Disease Association Prediction
================================================================
This is the single model file for automated architecture search.
All trainable components are exposed as __init__ parameters:
- FusionModule (cross-attention)
- Projection layers (prot_reg, dis_reg)
- Aggregation strategy (cls / mean_all_tok / mean)
- Contrastive loss type (infoNCE, ms_loss, triplet_loss, etc.)
- Dropout, miner settings

The model accepts pre-computed embeddings (protein and disease)
and outputs fused representations for contrastive learning or classification.

Training and fine-tuning pipelines are also included here,
but data loading is delegated to external utils to prevent leakage.

Usage:
    python scripts/model.py --mode pretrain    (pre-training only)
    python scripts/model.py --mode finetune   (fine-tuning only)
    python scripts/model.py --mode full       (pre-train then fine-tune)
"""
import argparse
import os
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime
import random
import string

import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve
from pytorch_metric_learning import losses, miners

# Add parent directory to sys.path to import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import GDA_Pretrain_Dataset, GDA_Cached_Dataset
from utils.commons import set_random_seed
from utils.downstream_disgenet_processor import DisGeNETProcessor as DisGeNETFoldProcessor
from utils.tdc_disgenet_processor import DisGeNETProcessor as TDCDisGeNETProcessor
from utils.protein_encoder_factory import build_protein_encoder_bundle
from transformers import BertTokenizer, BertModel


# ============================================================================
# 1. MODEL DEFINITION (All optimizable components)
# ============================================================================
class FusionModule(nn.Module):
    """Cross-attention fusion module with self- and cross-attention."""
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _alpha_from_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)
        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def _heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, input1, input2, mask1, mask2):
        query1 = self._heads(self.query1(input1), self.num_heads, self.head_size)
        key1 = self._heads(self.key1(input1), self.num_heads, self.head_size)
        query2 = self._heads(self.query2(input2), self.num_heads, self.head_size)
        key2 = self._heads(self.key2(input2), self.num_heads, self.head_size)

        logits11 = torch.einsum('blhd, bkhd->blkh', query1, key1)
        logits12 = torch.einsum('blhd, bkhd->blkh', query1, key2)
        logits21 = torch.einsum('blhd, bkhd->blkh', query2, key1)
        logits22 = torch.einsum('blhd, bkhd->blkh', query2, key2)

        alpha11 = self._alpha_from_logits(logits11, mask1, mask1)
        alpha12 = self._alpha_from_logits(logits12, mask1, mask2)
        alpha21 = self._alpha_from_logits(logits21, mask2, mask1)
        alpha22 = self._alpha_from_logits(logits22, mask2, mask2)

        value1 = self._heads(self.value1(input1), self.num_heads, self.head_size)
        value2 = self._heads(self.value2(input2), self.num_heads, self.head_size)

        output1 = (torch.einsum('blkh, bkhd->blhd', alpha11, value1).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha12, value2).flatten(-2)) / 2
        output2 = (torch.einsum('blkh, bkhd->blhd', alpha21, value1).flatten(-2) +
                   torch.einsum('blkh, bkhd->blhd', alpha22, value2).flatten(-2)) / 2
        return output1, output2


class GDA_Metric_Learning(nn.Module):
    """
    Main model for Gene-Disease Association prediction.
    All optimizable hyperparameters are exposed.
    """
    def __init__(
        self,
        prot_dim: int = 1152,
        dis_dim: int = 768,
        fusion_heads: int = 8,
        agg_mode: str = "mean_all_tok",
        loss_type: str = "infoNCE",
        use_miner: bool = False,
        miner_margin: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.prot_dim = prot_dim
        self.dis_dim = dis_dim
        self.agg_mode = agg_mode
        self.loss_type = loss_type
        self.use_miner = use_miner

        # Projection layers
        self.prot_reg = nn.Linear(prot_dim, 1024)
        self.dis_reg = nn.Linear(dis_dim, 1024)

        # Fusion module
        self.fusion_layer = FusionModule(1024, num_heads=fusion_heads)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Loss function (for pre-training)
        if loss_type == "infoNCE":
            self.loss_fn = losses.NTXentLoss(temperature=0.07)
        elif loss_type == "ms_loss":
            self.loss_fn = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)
        elif loss_type == "triplet_loss":
            self.loss_fn = losses.TripletMarginLoss(margin=0.05, swap=False, smooth_loss=False)
        elif loss_type == "circle_loss":
            self.loss_fn = losses.CircleLoss(m=0.4, gamma=80)
        elif loss_type == "lifted_structure_loss":
            self.loss_fn = losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)
        elif loss_type == "nca_loss":
            self.loss_fn = losses.NCALoss(softmax_scale=1)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        if use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets="all")
        else:
            self.miner = None

    def forward(self, prot_emb, dis_emb, labels=None, return_embeddings=False):
        """
        Forward pass for pre-training or feature extraction.

        Args:
            prot_emb: [B, prot_dim]
            dis_emb:  [B, dis_dim]
            labels:   Dummy, kept for interface
            return_embeddings: If True, return fused embeddings [B, 2048]
                               else return contrastive loss (scalar)
        """
        B = prot_emb.size(0)
        device = prot_emb.device

        prot_proj = self.prot_reg(prot_emb)      # [B, 1024]
        dis_proj = self.dis_reg(dis_emb)        # [B, 1024]

        # Add sequence dimension (length=1)
        prot_proj = prot_proj.unsqueeze(1)
        dis_proj = dis_proj.unsqueeze(1)
        mask_prot = torch.ones(B, 1, dtype=torch.bool, device=device)
        mask_dis = torch.ones(B, 1, dtype=torch.bool, device=device)

        prot_fused, dis_fused = self.fusion_layer(prot_proj, dis_proj, mask_prot, mask_dis)

        prot_out = self.dropout(prot_fused.squeeze(1))   # [B, 1024]
        dis_out = self.dropout(dis_fused.squeeze(1))     # [B, 1024]

        if return_embeddings:
            return torch.cat([prot_out, dis_out], dim=1)  # [B, 2048]

        # Contrastive learning
        query_embed = torch.cat([prot_out, dis_out], dim=0)  # [2B, 1024]
        labels_contrast = torch.cat([torch.arange(B), torch.arange(B)], dim=0).to(device)

        if self.use_miner:
            hard_pairs = self.miner(query_embed, labels_contrast)
            loss = self.loss_fn(query_embed, labels_contrast, hard_pairs)
        else:
            loss = self.loss_fn(query_embed, labels_contrast)
        return loss

    def predict(self, prot_emb, dis_emb):
        """Extract fused embeddings for LightGBM."""
        return self.forward(prot_emb, dis_emb, return_embeddings=True)


# ============================================================================
# 2. TRAINING / FINE-TUNING / EVALUATION PIPELINES
# ============================================================================

def collate_cached(batch):
    """Collate function for cached embeddings."""
    prot_emb, dis_emb, scores = zip(*batch)
    # Ensure all elements are tensors
    prot_emb = [p if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32) for p in prot_emb]
    dis_emb = [d if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=torch.float32) for d in dis_emb]
    return torch.stack(prot_emb), torch.stack(dis_emb), torch.tensor(scores)


# def collate_cached(batch):
#     """Collate function for cached embeddings."""
#     prot_emb, dis_emb, scores = zip(*batch)
#     return torch.stack(prot_emb), torch.stack(dis_emb), torch.tensor(scores)


def save_checkpoint(model, save_path, step, lr):
    """Save model state_dict only."""
    if hasattr(model, 'module'):
        model_to_save = model.module
    else:
        model_to_save = model
    os.makedirs(save_path, exist_ok=True)
    torch.save(model_to_save.state_dict(),
               os.path.join(save_path, f"step_{step}_lr_{lr}.pt"))
    print(f"Checkpoint saved at step {step}")

def run_pretrain(args):
    """Pre-training: train fusion module + projection layers."""
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"use_cache: {args.use_cache}")

    # Data
    train_data = GDA_Pretrain_Dataset(data_dir=args.data_dir, use_cache=args.use_cache)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_cached)

    # Model
    model = GDA_Metric_Learning(
        prot_dim=args.prot_dim,
        dis_dim=args.dis_dim,
        fusion_heads=args.fusion_heads,
        agg_mode=args.agg_mode,
        loss_type=args.loss_type,
        use_miner=args.use_miner,
        miner_margin=args.miner_margin,
        dropout=args.dropout,
    ).to(device)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.fp16)

    total_steps = len(train_loader) * args.max_epoch // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    # Logging
    os.makedirs(args.save_path, exist_ok=True)
    log_file = os.path.join(args.save_path, "pretrain_log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "batch", "loss", "lr", "time"])

    global_step = 0
    start_time = time.time()
    pbar = tqdm(total=len(train_loader) * args.max_epoch, desc="Pre-training")
    
    for epoch in range(args.max_epoch):
        model.train()
        epoch_loss_sum = 0
        epoch_batches = 0
        
        for step, batch in enumerate(train_loader):
            prot_emb, dis_emb, _ = batch
            prot_emb = prot_emb.to(device)
            dis_emb = dis_emb.to(device)

            with autocast(enabled=args.fp16 or args.bf16,
                          dtype=torch.bfloat16 if args.bf16 else torch.float16):
                loss = model(prot_emb, dis_emb)
                loss = loss.mean()  # DataParallel

            loss = loss / args.grad_accum_steps
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Record loss
                current_loss = loss.item() * args.grad_accum_steps
                current_lr = scheduler.get_last_lr()[0]
                epoch_loss_sum += current_loss
                epoch_batches += 1

                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, epoch, step, current_loss, 
                                   current_lr, time.time() - start_time])

                # Print every 20 steps
                if global_step % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"[Pre-train] Step {global_step}: loss={current_loss:.6f}, lr={current_lr:.2e}, time={elapsed:.1f}s")

                if global_step % args.save_step == 0:
                    save_checkpoint(model, args.save_path, global_step, args.lr)

                if global_step >= total_steps:
                    break
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum_steps:.4f}"})
        
        # Print epoch summary
        if epoch_batches > 0:
            avg_loss = epoch_loss_sum / epoch_batches
            print(f"[Epoch {epoch+1}/{args.max_epoch}] Average loss: {avg_loss:.6f}")

    pbar.close()
    save_checkpoint(model, args.save_path, "final", args.lr)
    print(f"Pre-training completed. Log saved to {log_file}")


def find_checkpoints(ckpt_dir, start_step=100, step_interval=500):
    """
    Find checkpoints from start_step, every step_interval steps.
    Also includes the final checkpoint if it exists.
    """
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    
    # Extract step numbers from filenames
    step_files = {}
    for f in ckpt_files:
        if f.startswith("step_"):
            # Parse step number: step_100_lr_0.0001.pt
            parts = f.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                step = int(parts[1])
                step_files[step] = f
        elif f.startswith("step_final"):
            step_files[999999] = f  # Use large number for final
    
    if not step_files:
        return []
    
    # Select checkpoints at regular intervals
    max_step = max(step_files.keys())
    selected_steps = []
    current = start_step
    while current <= max_step:
        # Find the closest checkpoint >= current
        closest = min([s for s in step_files.keys() if s >= current], default=None)
        if closest is not None and closest not in selected_steps:
            selected_steps.append(closest)
        current += step_interval
    
    # Always include the final checkpoint if it exists (999999)
    if 999999 in step_files and 999999 not in selected_steps:
        selected_steps.append(999999)
    
    # Sort by step number
    selected_steps.sort()
    return [step_files[s] for s in selected_steps]


def run_finetune_all_checkpoints(args):
    """
    Run fine-tuning on all selected checkpoints.
    - Save complete result for each checkpoint to checkpoint_comparison.csv
    - Save only the best checkpoint (by val_auc) to all_results.csv (append)
    """
    # ===== 获取 run_id（与 run_finetune 保持一致） =====
    run_id = os.environ.get("PIPELINE_RUN_ID")
    if run_id is None:
        run_id = f"standalone_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        print(f"Warning: PIPELINE_RUN_ID not set, using: {run_id}")
    # ===================================================

    ckpt_dir = args.save_path
    if not os.path.exists(ckpt_dir):
        print(f"Checkpoint directory not found: {ckpt_dir}")
        return

    # Find checkpoints
    checkpoint_files = find_checkpoints(ckpt_dir, start_step=100, step_interval=500)
    if not checkpoint_files:
        print("No checkpoints found.")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(checkpoint_files)} checkpoints for {args.dataset}")
    print(f"Checkpoints: {[f for f in checkpoint_files]}")
    print(f"{'='*60}\n")

    # Prepare directory
    os.makedirs(args.result_dir, exist_ok=True)

    # Define complete field list (must match result dict)
    fieldnames = [
        "timestamp", "run_id", "dataset", "fold", "checkpoint",
        "val_auc", "test_auc", "test_aupr", "test_fmax",
        "test_accuracy", "test_f1",
        "num_leaves", "max_depth", "lr_gbm",
        "fusion_heads", "agg_mode", "loss_type"
    ]

    # Store all checkpoint results for comparison
    all_checkpoint_results = []
    best_val_auc = -1.0
    best_result = None

    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        print(f"\n--- Evaluating checkpoint: {ckpt_file} ---")

        # Evaluate this checkpoint (return_results=True)
        args.checkpoint = ckpt_path
        val_auc, metrics = run_finetune(args, return_results=True)

        # Build complete result dict for this checkpoint
        result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "run_id": run_id,
            "dataset": args.dataset,
            "fold": args.fold if args.dataset == "DisGeNET-EVAL" else 0,
            "checkpoint": ckpt_file,
            "val_auc": val_auc,
            "test_auc": metrics["test_auc"],
            "test_aupr": metrics["test_aupr"],
            "test_fmax": metrics["test_fmax"],
            "test_accuracy": metrics["test_accuracy"],
            "test_f1": metrics["test_f1"],
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "lr_gbm": args.lr_gbm,
            "fusion_heads": args.fusion_heads,
            "agg_mode": args.agg_mode,
            "loss_type": args.loss_type,
        }

        # Save to list for checkpoint_comparison.csv
        all_checkpoint_results.append(result)

        # Update best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_result = result

        print(f"  Val AUC: {val_auc:.6f} | Test AUC: {metrics['test_auc']:.6f}")

    # ------------------------------------------------------------------
    # 1. Write ALL checkpoints to checkpoint_comparison.csv (overwrite)
    # ------------------------------------------------------------------
    comparison_csv = os.path.join(args.result_dir, "checkpoint_comparison.csv")
    with open(comparison_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_checkpoint_results)

    print(f"\nAll checkpoint results saved to {comparison_csv}")

    # ------------------------------------------------------------------
    # 2. Append ONLY the BEST checkpoint to all_results.csv
    # ------------------------------------------------------------------
    if best_result is not None:
        csv_file = os.path.join(args.result_dir, "all_results.csv")
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(best_result)
        print(f"Best checkpoint result appended to {csv_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("CHECKPOINT COMPARISON SUMMARY")
    print(f"{'='*60}")
    for r in all_checkpoint_results:
        print(f"{r['checkpoint']}: Val AUC={r['val_auc']:.6f}")
    print(f"{'-'*60}")
    print(f"BEST CHECKPOINT: {best_result['checkpoint']} with Val AUC={best_val_auc:.6f}")
    print(f"Added to all_results.csv with run_id={run_id}")
    print(f"{'='*60}")


# def run_finetune_all_checkpoints(args):
#     """
#     Run fine-tuning on all selected checkpoints, then choose the best one.
#     """
    
#     ckpt_dir = args.save_path
#     if not os.path.exists(ckpt_dir):
#         print(f"Checkpoint directory not found: {ckpt_dir}")
#         return
    
#     # Find checkpoints
#     checkpoint_files = find_checkpoints(ckpt_dir, start_step=100, step_interval=500)
#     if not checkpoint_files:
#         print("No checkpoints found.")
#         return
    
#     print(f"\n{'='*60}")
#     print(f"Found {len(checkpoint_files)} checkpoints for {args.dataset}")
#     print(f"Checkpoints: {[f for f in checkpoint_files]}")
#     print(f"{'='*60}\n")
    
#     # Store results for each checkpoint
#     results = []
#     best_val_auc = -1
#     best_checkpoint = None
    
#     for ckpt_file in checkpoint_files:
#         ckpt_path = os.path.join(ckpt_dir, ckpt_file)
#         print(f"\n--- Evaluating checkpoint: {ckpt_file} ---")
        
#         # Set checkpoint and run finetune (return results only)
#         args.checkpoint = ckpt_path
#         val_auc, metrics = run_finetune(args, return_results=True)
        
#         # Store results
#         result_entry = {
#             "checkpoint": ckpt_file,
#             "val_auc": val_auc,
#             "test_auc": metrics["test_auc"],
#             "test_aupr": metrics["test_aupr"],
#             "test_fmax": metrics["test_fmax"],
#             "test_accuracy": metrics["test_accuracy"],
#             "test_f1": metrics["test_f1"],
#         }
#         results.append(result_entry)
        
#         # Track best
#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             best_checkpoint = ckpt_file
        
#         # Print intermediate result
#         print(f"  Val AUC: {val_auc:.6f} | Test AUC: {metrics['test_auc']:.6f}")
    
#     # Save checkpoint comparison to CSV
#     comparison_csv = os.path.join(args.result_dir, "checkpoint_comparison.csv")
#     os.makedirs(args.result_dir, exist_ok=True)
#     file_exists = os.path.isfile(comparison_csv)
#     with open(comparison_csv, 'a', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
#         if not file_exists:
#             writer.writeheader()
#         writer.writerows(results)
    
#     print(f"\n{'='*60}")
#     print("CHECKPOINT COMPARISON SUMMARY")
#     print(f"{'='*60}")
#     for r in results:
#         print(f"{r['checkpoint']}: Val AUC={r['val_auc']:.6f}")
#     print(f"{'-'*60}")
#     print(f"BEST CHECKPOINT: {best_checkpoint} with Val AUC={best_val_auc:.6f}")
#     print(f"Comparison saved to {comparison_csv}")
    
#     # Now run final fine-tuning with the best checkpoint
#     print(f"\n--- Running final fine-tuning with best checkpoint: {best_checkpoint} ---")
#     args.checkpoint = os.path.join(ckpt_dir, best_checkpoint)
#     run_finetune(args, return_results=False)


def run_finetune(args, return_results=False):
    """
    Fine-tune: extract features and train LightGBM.
    If return_results=True, returns (val_auc, test_metrics) without saving final model.
    """

    # ===== 从环境变量获取 pipeline run_id =====
    run_id = os.environ.get("PIPELINE_RUN_ID")
    if run_id is None:
        # 如果环境变量不存在（例如单独运行测试），生成一个临时 ID
        run_id = f"standalone_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        print(f"Warning: PIPELINE_RUN_ID not set, using: {run_id}")
    # ==========================================


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load data processor (same as before)
    if args.dataset == "DisGeNET-EVAL":
        processor = DisGeNETFoldProcessor(data_dir=args.data_dir, fold_id=args.fold)
        cache_name = f"Fold_{args.fold}_train"
    else:
        processor = TDCDisGeNETProcessor(dataset_name=args.dataset, data_dir=args.data_dir)
        cache_name = args.dataset

    train_ex = processor.get_train_examples(args.test)
    val_ex = processor.get_val_examples(args.test)
    test_ex = processor.get_test_examples(args.test)

    train_prot, train_dis, train_scores, train_idx, train_cache_name = train_ex
    val_prot, val_dis, val_scores, val_idx, val_cache_name = val_ex
    test_prot, test_dis, test_scores, test_idx, test_cache_name = test_ex

    # Build datasets (same as before)
    train_dataset = GDA_Cached_Dataset(
        (train_prot, train_dis, train_scores), args.cache_dir, train_cache_name,
        indices=train_idx, use_cache=True
    )
    val_dataset = GDA_Cached_Dataset(
        (val_prot, val_dis, val_scores), args.cache_dir, val_cache_name,
        indices=val_idx, use_cache=True
    )
    test_dataset = GDA_Cached_Dataset(
        (test_prot, test_dis, test_scores), args.cache_dir, test_cache_name,
        indices=test_idx, use_cache=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_cached)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_cached)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_cached)

    # Load model (same as before)
    model = GDA_Metric_Learning(
        prot_dim=args.prot_dim,
        dis_dim=args.dis_dim,
        fusion_heads=args.fusion_heads,
        agg_mode=args.agg_mode,
        loss_type="infoNCE",
        use_miner=False,
    ).to(device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    else:
        ckpt_dir = args.save_path
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if ckpt_files:
            latest = sorted(ckpt_files)[-1]
            ckpt_path = os.path.join(ckpt_dir, latest)
            state_dict = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

    # Extract features (same as before)
    def extract(loader, name="dataset"):
        model.eval()
        X, y = [], []
        with torch.no_grad():
            for prot_emb, dis_emb, labels in tqdm(loader, desc=f"Extracting {name}"):
                prot_emb = prot_emb.to(device)
                dis_emb = dis_emb.to(device)
                feat = model.predict(prot_emb, dis_emb)
                X.append(feat.cpu().numpy())
                y.append(labels.numpy())
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

    X_train, y_train = extract(train_loader, "train")
    X_val, y_val = extract(val_loader, "val")
    X_test, y_test = extract(test_loader, "test")

    # Train LightGBM
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'num_leaves': args.num_leaves,
        'max_depth': args.max_depth,
        'learning_rate': args.lr_gbm,
        'early_stopping_round': 30,
        'verbose': 0,
    }
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_val, y_val, reference=lgb_train)
    gbm = lgb.train(params, lgb_train, valid_sets=[lgb_valid])

    # Evaluate on validation set
    val_y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    val_auc = roc_auc_score(y_val, val_y_pred)

    # Evaluate on test set
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_bin = (y_pred >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred)
    aupr = average_precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred_bin)
    f1 = f1_score(y_test, y_pred_bin)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    fmax = (2 * precision * recall / (precision + recall + 1e-12)).max()

    metrics = {
        "val_auc": val_auc,
        "test_auc": auc,
        "test_aupr": aupr,
        "test_fmax": fmax,
        "test_accuracy": acc,
        "test_f1": f1,
        "best_iteration": gbm.best_iteration,
    }

    if return_results:
        return val_auc, metrics
    else:
        # Print results and save to CSV (existing logic)
        print(f"\n{'='*60}")
        print(f"RESULTS - {args.dataset} (fold={args.fold if args.dataset == 'DisGeNET-EVAL' else 'N/A'})")
        print(f"{'='*60}")
        print(f"Val AUC: {val_auc:.6f}")
        print(f"Test AUC: {auc:.6f}, AUPR: {aupr:.6f}, Fmax: {fmax:.6f}, Acc: {acc:.6f}, F1: {f1:.6f}")
        print(f"{'='*60}\n")

        # Save to CSV
        os.makedirs(args.result_dir, exist_ok=True)


        result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "run_id": run_id,  # ← 新增
            "dataset": args.dataset,
            "fold": args.fold if args.dataset == "DisGeNET-EVAL" else 0,
            "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else "unknown",
            "val_auc": val_auc,
            "test_auc": auc,
            "test_aupr": aupr,
            "test_fmax": fmax,
            "test_accuracy": acc,
            "test_f1": f1,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "lr_gbm": args.lr_gbm,
            "fusion_heads": args.fusion_heads,
            "agg_mode": args.agg_mode,
            "loss_type": args.loss_type,
        }

        # result = {
        #     "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        #     "dataset": args.dataset,
        #     "fold": args.fold if args.dataset == "DisGeNET-EVAL" else 0,
        #     "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else "unknown",
        #     "val_auc": val_auc,
        #     "test_auc": auc,
        #     "test_aupr": aupr,
        #     "test_fmax": fmax,
        #     "test_accuracy": acc,
        #     "test_f1": f1,
        #     "num_leaves": args.num_leaves,
        #     "max_depth": args.max_depth,
        #     "lr_gbm": args.lr_gbm,
        #     "fusion_heads": args.fusion_heads,
        #     "agg_mode": args.agg_mode,
        #     "loss_type": args.loss_type,
        # }
        csv_file = os.path.join(args.result_dir, "all_results.csv")
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)
        print(f"Results saved to {csv_file}")
        return None


def run_full(args):


    """Run pre-training then fine-tuning on all datasets."""
    print("\n" + "="*60)
    print("STARTING FULL PIPELINE")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Use cache: {args.use_cache}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max epochs: {args.max_epoch}")
    print(f"  - Fusion heads: {args.fusion_heads}")
    print(f"  - Aggregation: {args.agg_mode}")
    print(f"  - Loss type: {args.loss_type}")
    print("="*60 + "\n")

    start_time = time.time()

    if not args.skip_pretrain:
        # Pre-train (skip if checkpoint directory already has files)
        run_pretrain(args)
        # if not os.path.exists(args.save_path) or len(os.listdir(args.save_path)) < 2:
        #     run_pretrain(args)
        # else:
        #     print(f"Checkpoints already exist in {args.save_path}, skipping pre-training.")

    else:
        print("Skipping pre-training as requested.")


    # Run checkpoint selection and fine-tuning on all datasets
    # datasets = ["DisGeNET-EVAL", "TDC"]
    datasets = ["DisGeNET-EVAL"]
    for dataset in datasets:
        args.dataset = dataset
        if dataset == "DisGeNET-EVAL":
            for fold in range(1, 2):
                args.fold = fold
                run_finetune_all_checkpoints(args)
        else:
            args.fold = 0
            run_finetune_all_checkpoints(args)

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"FULL PIPELINE COMPLETED in {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
    print(f"Results saved to {args.result_dir}/all_results.csv")
    print(f"Checkpoint comparison saved to {args.result_dir}/checkpoint_comparison.csv")
    print("="*60)


# ============================================================================
# 3. COMMAND LINE INTERFACE
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["pretrain", "finetune", "full"],
                        help="Run mode: pretrain, finetune, or full (both)")

    # Data and cache paths
    parser.add_argument("--data_dir", type=str, default="../data/ALLGDA_Data",
                        help="Root data directory")
    parser.add_argument("--cache_dir", type=str, default="../data/ALLGDA_Data/cached_embeddings",
                        help="Cached embedding directory")
    parser.add_argument("--save_path", type=str, default="../checkpoints/pretrain",
                        help="Directory to save model checkpoints")
    parser.add_argument("--result_dir", type=str, default="../results/finetune",
                        help="Directory to save fine-tune results")

    # Model architecture (optimizable)
    parser.add_argument("--prot_dim", type=int, default=1152,
                        help="Protein embedding dimension")
    parser.add_argument("--dis_dim", type=int, default=768,
                        help="Disease embedding dimension")
    parser.add_argument("--fusion_heads", type=int, default=8,
                        help="Number of attention heads in fusion")
    parser.add_argument("--agg_mode", type=str, default="mean_all_tok",
                        choices=["cls", "mean_all_tok", "mean"],
                        help="Pooling strategy")
    parser.add_argument("--loss_type", type=str, default="infoNCE",
                        choices=["infoNCE", "ms_loss", "triplet_loss", "circle_loss",
                                 "lifted_structure_loss", "nca_loss"],
                        help="Contrastive loss")
    parser.add_argument("--use_miner", action="store_true",
                        help="Use hard mining for triplet loss")
    parser.add_argument("--miner_margin", type=float, default=0.2,
                        help="Margin for miner")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")

    # Pre-training hyperparameters
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Training batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Pre-training learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_epoch", type=int, default=2,
                        help="Max training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--save_step", type=int, default=100,
                        help="Save checkpoint every N updates")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable BF16 mixed precision")
    parser.add_argument("--use_cache", action="store_true",
                        help="Use cached PLM embeddings for faster training")

    # Fine-tuning hyperparameters
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pre-trained checkpoint (.pt) for finetuning")
    parser.add_argument("--dataset", type=str, default="DisGeNET-EVAL",
                        choices=["DisGeNET-EVAL", "TDC", "stomach", "alzheimer"],
                        help="Dataset for fine-tuning")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold ID for DisGeNET-EVAL (1-5)")
    parser.add_argument("--step", type=int, default=0,
                        help="Step number for checkpoint loading (if multiple)")
    parser.add_argument("--test", type=int, default=0,
                        help="If >0, use subset for fast testing")
    parser.add_argument("--num_leaves", type=int, default=32,
                        help="LightGBM num_leaves")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="LightGBM max_depth")
    parser.add_argument("--lr_gbm", type=float, default=0.15,
                        help="LightGBM learning rate")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pre-training if checkpoints already exist")
    # General
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device")
    parser.add_argument("--seed", type=int, default=2022,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "pretrain":
        run_pretrain(args)
    elif args.mode == "finetune":
        if args.checkpoint is None:
            # Auto-find latest checkpoint
            ckpt_dir = args.save_path
            if os.path.exists(ckpt_dir):
                ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
                if ckpt_files:
                    latest = sorted(ckpt_files)[-1]
                    args.checkpoint = os.path.join(ckpt_dir, latest)
                    print(f"Auto-selected checkpoint: {args.checkpoint}")
        run_finetune(args)
    elif args.mode == "full":
        run_full(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()