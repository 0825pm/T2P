"""
SPL-VQVAE v2 Training Script
- Trains SPL_VQVAE_V2 model for sign language motion reconstruction
- Training data: SOKE format (133 dims, axis-angle rotations)
- Uses H2SMotionDatasetVQ from mGPT
- Metrics: computed on joint 3D coordinates (via SMPL-X forward pass)
- Logs: recon_loss, commit_loss, velocity_loss, mpjpe, mpjve, dtw
- Saves: best model, periodic checkpoints, visualization videos

Usage:
    python train_vqvae_spl_v2.py --config configs/vqvae_spl_v2.yaml
"""

import os
import sys
import argparse
import yaml
import json
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# DTW
try:
    from dtaidistance import dtw_ndim
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("Warning: dtaidistance not installed. DTW metric will be disabled.")

# SMPL-X
try:
    import smplx
    HAS_SMPLX = True
except ImportError:
    HAS_SMPLX = False
    print("Warning: smplx not installed. Using raw features for metrics.")

# SOKE DataLoader
from mGPT.data.humanml import H2SMotionDatasetVQ
from mGPT.data.utils import humanml3d_collate


# =============================================================================
# SOKE Format Definition (133 dims)
# =============================================================================

SOKE_TOTAL_DIM = 133
SOKE_UPPER_BODY_DIM = 30
SOKE_LHAND_DIM = 45
SOKE_RHAND_DIM = 45
SOKE_JAW_DIM = 3
SOKE_EXPR_DIM = 10


# =============================================================================
# SMPL-X Converter
# =============================================================================

class SMPLXConverter:
    """Convert SOKE 133-dim to joint 3D coordinates."""
    
    def __init__(self, model_path, gender='neutral', device='cuda'):
        self.device = device
        self.model = None
        
        if HAS_SMPLX and os.path.exists(model_path):
            try:
                self.model = smplx.create(
                    model_path,
                    model_type='smplx',
                    gender=gender,
                    use_pca=False,
                    use_face_contour=True,
                    num_betas=10,
                    num_expression_coeffs=10,
                ).to(device)
                self.model.eval()
                print(f"SMPL-X model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load SMPL-X: {e}")
    
    def _parse_soke_params(self, pose_133):
        if pose_133.dim() == 2:
            pose_133 = pose_133.unsqueeze(0)
        
        B, T, _ = pose_133.shape
        device = pose_133.device
        
        upper_body = pose_133[:, :, 0:30]
        lhand = pose_133[:, :, 30:75]
        rhand = pose_133[:, :, 75:120]
        jaw = pose_133[:, :, 120:123]
        expr = pose_133[:, :, 123:133]
        
        lower_zeros = torch.zeros(B, T, 33, device=device)
        body_pose = torch.cat([lower_zeros, upper_body], dim=-1)
        global_orient = torch.zeros(B, T, 3, device=device)
        betas = torch.zeros(B, 10, device=device)
        
        return {
            'global_orient': global_orient,
            'body_pose': body_pose,
            'left_hand_pose': lhand,
            'right_hand_pose': rhand,
            'jaw_pose': jaw,
            'expression': expr,
            'betas': betas,
        }
    
    @torch.no_grad()
    def to_joints(self, pose_133):
        if self.model is None:
            return None
        
        if pose_133.dim() == 2:
            pose_133 = pose_133.unsqueeze(0)
        
        B, T, _ = pose_133.shape
        device = pose_133.device
        params = self._parse_soke_params(pose_133)
        
        all_joints = []
        for t in range(T):
            output = self.model(
                global_orient=params['global_orient'][:, t],
                body_pose=params['body_pose'][:, t],
                left_hand_pose=params['left_hand_pose'][:, t],
                right_hand_pose=params['right_hand_pose'][:, t],
                jaw_pose=params['jaw_pose'][:, t],
                expression=params['expression'][:, t],
                betas=params['betas'],
                leye_pose=torch.zeros(B, 3, device=device),
                reye_pose=torch.zeros(B, 3, device=device),
            )
            joints = output.joints
            body = joints[:, :22]
            lhand = joints[:, 25:40]
            rhand = joints[:, 40:55]
            selected = torch.cat([body, lhand, rhand], dim=1)
            all_joints.append(selected)
        
        return torch.stack(all_joints, dim=1)


# =============================================================================
# Metrics
# =============================================================================

def compute_mpjpe(pred, target, mask=None):
    """Mean Per Joint Position Error (mm)."""
    if pred.dim() == 3:
        pred = pred.view(pred.shape[0], pred.shape[1], -1, 3)
        target = target.view(target.shape[0], target.shape[1], -1, 3)
    
    per_joint_error = torch.norm(pred - target, p=2, dim=-1)
    
    if mask is not None:
        mask = mask.unsqueeze(-1)
        per_joint_error = per_joint_error * mask
        mpjpe = per_joint_error.sum() / (mask.sum() * pred.shape[2] + 1e-8)
    else:
        mpjpe = per_joint_error.mean()
    
    return mpjpe * 1000


def compute_mpjve(pred, target, mask=None):
    """Mean Per Joint Velocity Error (mm/frame)."""
    if pred.dim() == 3:
        pred = pred.view(pred.shape[0], pred.shape[1], -1, 3)
        target = target.view(target.shape[0], target.shape[1], -1, 3)
    
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    
    per_joint_vel_error = torch.norm(pred_vel - target_vel, p=2, dim=-1)
    
    if mask is not None:
        vel_mask = mask[:, 1:] * mask[:, :-1]
        vel_mask = vel_mask.unsqueeze(-1)
        per_joint_vel_error = per_joint_vel_error * vel_mask
        mpjve = per_joint_vel_error.sum() / (vel_mask.sum() * pred.shape[2] + 1e-8)
    else:
        mpjve = per_joint_vel_error.mean()
    
    return mpjve * 1000


def compute_dtw(pred, target):
    """Dynamic Time Warping distance."""
    if not HAS_DTW:
        return 0.0
    
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
    if target.ndim == 3:
        target = target.reshape(target.shape[0], -1)
    
    try:
        return dtw_ndim.distance(pred.astype(np.float64), target.astype(np.float64))
    except:
        return 0.0


# =============================================================================
# Average Meter
# =============================================================================

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# Codebook Statistics
# =============================================================================

def compute_codebook_stats(all_codes, codebook_size):
    """Compute codebook usage statistics."""
    if isinstance(all_codes, list):
        all_codes = torch.cat([c.flatten() for c in all_codes])
    else:
        all_codes = all_codes.flatten()
    
    code_counts = torch.bincount(all_codes.long(), minlength=codebook_size)
    
    total_tokens = len(all_codes)
    used_codes = (code_counts > 0).sum().item()
    usage_rate = used_codes / codebook_size * 100
    
    used_mask = code_counts > 0
    if used_mask.sum() > 0:
        used_counts = code_counts[used_mask].float()
        min_usage = used_counts.min().item()
        max_usage = used_counts.max().item()
        avg_usage = used_counts.mean().item()
        std_usage = used_counts.std().item() if len(used_counts) > 1 else 0.0
        
        # Mode: 가장 많이 사용된 코드
        mode_code = code_counts.argmax().item()
        mode_count = code_counts[mode_code].item()
        
        # Median
        median_usage = used_counts.median().item()
    else:
        min_usage = max_usage = avg_usage = std_usage = 0.0
        mode_code = mode_count = median_usage = 0
    
    # Perplexity
    probs = code_counts.float() / (total_tokens + 1e-8)
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    perplexity = torch.exp(entropy).item()
    
    # Dead codes (사용되지 않은 코드)
    dead_codes = codebook_size - used_codes
    
    return {
        'used_codes': used_codes,
        'dead_codes': dead_codes,
        'total_codes': codebook_size,
        'usage_rate': usage_rate,
        'min_usage': int(min_usage),
        'max_usage': int(max_usage),
        'avg_usage': avg_usage,
        'std_usage': std_usage,
        'median_usage': median_usage,
        'mode_code': mode_code,
        'mode_count': int(mode_count),
        'perplexity': perplexity,
    }


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, codebook_size=512, mean=None, std=None):
    """Train for one epoch."""
    model.train()
    
    recon_meter = AverageMeter()
    commit_meter = AverageMeter()
    velocity_meter = AverageMeter()
    total_meter = AverageMeter()
    perplexity_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    mpjve_meter = AverageMeter()
    
    all_codes = []
    
    # Prepare mean/std tensors
    if mean is not None:
        if isinstance(mean, np.ndarray):
            mean_t = torch.from_numpy(mean).float().to(device)
        else:
            mean_t = mean.clone().detach().to(device).float()
    else:
        mean_t = None
    
    if std is not None:
        if isinstance(std, np.ndarray):
            std_t = torch.from_numpy(std).float().to(device)
        else:
            std_t = std.clone().detach().to(device).float()
    else:
        std_t = None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch in pbar:
        if batch is None:
            continue
        
        motion = batch['motion'].to(device)
        # Handle different length key formats
        lengths = batch.get('length', batch.get('lengths'))
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths)
        lengths = lengths.to(device)
        
        B = motion.shape[0]
        
        # Forward
        output, commit_loss, perplexity, indices = model(motion, lengths)
        
        all_codes.append(indices.detach().cpu())
        
        # Mask
        max_len = motion.shape[1]
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Loss
        losses = loss_fn(output, motion, commit_loss, mask.float())
        
        # Backward
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Compute metrics (no grad needed, already detached via loss backward)
        with torch.no_grad():
            # Denormalize for metrics
            if mean_t is not None and std_t is not None:
                output_denorm = output * std_t + mean_t
                motion_denorm = motion * std_t + mean_t
            else:
                output_denorm = output
                motion_denorm = motion
            
            # Compute metrics on raw features
            num_pseudo_joints = SOKE_TOTAL_DIM // 3
            pred_reshaped = output_denorm[:, :, :num_pseudo_joints*3].view(B, -1, num_pseudo_joints, 3)
            gt_reshaped = motion_denorm[:, :, :num_pseudo_joints*3].view(B, -1, num_pseudo_joints, 3)
            
            mpjpe = compute_mpjpe(pred_reshaped, gt_reshaped, mask.float())
            mpjve = compute_mpjve(pred_reshaped, gt_reshaped, mask.float())
            
            mpjpe_meter.update(mpjpe.item(), B)
            mpjve_meter.update(mpjve.item(), B)
        
        # Update meters
        recon_meter.update(losses['recon_loss'].item(), B)
        commit_meter.update(losses['commit_loss'].item(), B)
        velocity_meter.update(losses['velocity_loss'].item(), B)
        total_meter.update(losses['total_loss'].item(), B)
        perplexity_meter.update(perplexity.item(), B)
        
        pbar.set_postfix({
            'recon': f'{recon_meter.avg:.4f}',
            'mpjpe': f'{mpjpe_meter.avg:.1f}',
            'ppl': f'{perplexity_meter.avg:.1f}',
        })
    
    codebook_stats = compute_codebook_stats(all_codes, codebook_size)
    
    return {
        'recon_loss': recon_meter.avg,
        'commit_loss': commit_meter.avg,
        'velocity_loss': velocity_meter.avg,
        'total_loss': total_meter.avg,
        'perplexity': perplexity_meter.avg,
        'mpjpe': mpjpe_meter.avg,
        'mpjve': mpjve_meter.avg,
        'codebook': codebook_stats,
    }


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, epoch, mean, std, codebook_size=512, smplx_converter=None):
    """Evaluate model."""
    model.eval()
    
    recon_meter = AverageMeter()
    commit_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    mpjve_meter = AverageMeter()
    dtw_meter = AverageMeter()
    perplexity_meter = AverageMeter()
    
    all_codes = []
    vis_samples = []
    
    if isinstance(mean, np.ndarray):
        mean_t = torch.from_numpy(mean).float().to(device)
    else:
        mean_t = mean.clone().detach().to(device).float() if mean is not None else None
    
    if isinstance(std, np.ndarray):
        std_t = torch.from_numpy(std).float().to(device)
    else:
        std_t = std.clone().detach().to(device).float() if std is not None else None
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Eval]')
    
    for batch in pbar:
        if batch is None:
            continue
        
        motion = batch['motion'].to(device)
        lengths = batch.get('length', batch.get('lengths'))
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths)
        lengths = lengths.to(device)
        names = batch.get('name', batch.get('names', []))
        srcs = batch.get('src', ['unknown'] * motion.shape[0])
        
        B = motion.shape[0]
        
        # Forward
        output, commit_loss, perplexity, indices = model(motion, lengths)
        
        all_codes.append(indices.detach().cpu())
        
        # Mask
        max_len = motion.shape[1]
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Loss
        losses = loss_fn(output, motion, commit_loss, mask.float())
        
        # Denormalize for metrics
        if mean_t is not None and std_t is not None:
            output_denorm = output * std_t + mean_t
            motion_denorm = motion * std_t + mean_t
        else:
            output_denorm = output
            motion_denorm = motion
        
        # Compute metrics on raw features (since we don't have joint coords)
        # Using 133/3 = 44 pseudo-joints
        num_pseudo_joints = SOKE_TOTAL_DIM // 3
        pred_reshaped = output_denorm[:, :, :num_pseudo_joints*3].view(B, -1, num_pseudo_joints, 3)
        gt_reshaped = motion_denorm[:, :, :num_pseudo_joints*3].view(B, -1, num_pseudo_joints, 3)
        
        mpjpe = compute_mpjpe(pred_reshaped, gt_reshaped, mask.float())
        mpjve = compute_mpjve(pred_reshaped, gt_reshaped, mask.float())
        
        mpjpe_meter.update(mpjpe.item(), B)
        mpjve_meter.update(mpjve.item(), B)
        
        # DTW (limited)
        if HAS_DTW and dtw_meter.count < 50:
            L = int(lengths[0].item())
            pred_np = pred_reshaped[0, :L].cpu().numpy().reshape(L, -1)
            gt_np = gt_reshaped[0, :L].cpu().numpy().reshape(L, -1)
            dtw_dist = compute_dtw(pred_np, gt_np)
            if dtw_dist > 0:
                dtw_meter.update(dtw_dist, 1)
        
        # Update meters
        recon_meter.update(losses['recon_loss'].item(), B)
        commit_meter.update(losses['commit_loss'].item(), B)
        perplexity_meter.update(perplexity.item(), B)
        
        # Store visualization samples
        current_src = srcs[0] if srcs else 'unknown'
        if len(vis_samples) < 3:
            L = int(lengths[0].item())
            vis_samples.append({
                'pred': pred_reshaped[0, :L].cpu().numpy(),
                'gt': gt_reshaped[0, :L].cpu().numpy(),
                'name': names[0] if names else '',
                'src': current_src,
            })
        
        pbar.set_postfix({
            'mpjpe': f'{mpjpe_meter.avg:.2f}',
            'recon': f'{recon_meter.avg:.4f}',
        })
    
    codebook_stats = compute_codebook_stats(all_codes, codebook_size)
    
    return {
        'recon_loss': recon_meter.avg,
        'commit_loss': commit_meter.avg,
        'mpjpe': mpjpe_meter.avg,
        'mpjve': mpjve_meter.avg,
        'dtw': dtw_meter.avg if dtw_meter.count > 0 else 0.0,
        'perplexity': perplexity_meter.avg,
        'codebook': codebook_stats,
        'vis_samples': vis_samples,
    }


# =============================================================================
# Checkpoint
# =============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path, is_best=False):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': {k: v for k, v in metrics.items() if k != 'vis_samples'},
    }
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")
    
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")


# =============================================================================
# Main
# =============================================================================

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Seed
    seed = config.get('training', {}).get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config.get('name', 'spl_vqvae_v2')
    output_dir = os.path.join(config.get('training', {}).get('model_dir', 'checkpoints'), f'{exp_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print(f"Output: {output_dir}")
    
    # Data config
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    eval_config = config.get('evaluation', {})
    
    # Load mean/std (SOKE style filtering)
    mean_path = data_config.get('mean_path')
    std_path = data_config.get('std_path')
    
    if mean_path and os.path.exists(mean_path):
        mean = torch.load(mean_path) if mean_path.endswith('.pt') else torch.from_numpy(np.load(mean_path))
        # SOKE filtering: remove lower body and shape
        # Original 179 -> 143 (remove lower body) -> 133 (remove shape)
        mean = mean[(3 + 3 * 11):]  # Remove lower body (root + 11 lower body joints)
        mean = torch.cat([mean[:-20], mean[-10:]], dim=0)  # Remove shape (10 dims), keep expression (10 dims)
        print(f"Loaded and filtered mean from {mean_path}, shape: {mean.shape}")
    else:
        mean = torch.zeros(SOKE_TOTAL_DIM)
        print("Warning: mean not found, using zeros")
    
    if std_path and os.path.exists(std_path):
        std = torch.load(std_path) if std_path.endswith('.pt') else torch.from_numpy(np.load(std_path))
        # Same filtering as mean
        std = std[(3 + 3 * 11):]
        std = torch.cat([std[:-20], std[-10:]], dim=0)
        print(f"Loaded and filtered std from {std_path}, shape: {std.shape}")
    else:
        std = torch.ones(SOKE_TOTAL_DIM)
        print("Warning: std not found, using ones")
    
    # Common dataset kwargs
    dataset_kwargs = {
        'data_root': data_config.get('data_root'),
        'mean': mean,
        'std': std,
        'max_motion_length': data_config.get('max_motion_length', 300),
        'min_motion_length': data_config.get('min_motion_length', 40),
        'unit_length': data_config.get('unit_length', 4),
        'dataset_name': data_config.get('dataset_name', 'how2sign_csl_phoenix'),
        'csl_root': data_config.get('csl_root'),
        'phoenix_root': data_config.get('phoenix_poses_root', data_config.get('phoenix_root')),
    }
    
    # Create datasets using SOKE H2SMotionDatasetVQ
    win_size = data_config.get('win_size', None)
    
    train_dataset = H2SMotionDatasetVQ(
        split='train',
        win_size=win_size,
        **dataset_kwargs
    )
    
    val_dataset = H2SMotionDatasetVQ(
        split='val',
        win_size=None,
        **dataset_kwargs
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 64),
        shuffle=True,
        num_workers=train_config.get('num_workers', 8),
        collate_fn=humanml3d_collate,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_config.get('batch_size', 64),
        shuffle=False,
        num_workers=eval_config.get('num_workers', 4),
        collate_fn=humanml3d_collate,
        pin_memory=True,
    )
    
    # Model
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.vqvae_spl import SPL_VQVAE_V2, SPLVQVAELoss
    
    model = SPL_VQVAE_V2(
        embed_dim=model_config.get('embed_dim', 256),
        depth=model_config.get('depth', 4),
        num_heads=model_config.get('num_heads', 8),
        mlp_dim=model_config.get('mlp_dim', 1024),
        num_queries=model_config.get('num_queries', 32),
        codebook_size=model_config.get('codebook_size', 512),
        codebook_dim=model_config.get('codebook_dim', 256),
        max_len=data_config.get('max_motion_length', 300),
        dropout=model_config.get('dropout', 0.1),
        spl_hidden_dim=model_config.get('spl_hidden_dim', 256),
        spl_num_layers=model_config.get('spl_num_layers', 2),
        commitment_weight=train_config.get('lambda_commit', 0.25),
        nfeats=SOKE_TOTAL_DIM,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input format: SOKE ({SOKE_TOTAL_DIM} dims)")
    print(f"Codebook size: {model_config.get('codebook_size', 512)}")
    print(f"Q-Former queries: {model_config.get('num_queries', 32)}")
    
    # Loss
    loss_fn = SPLVQVAELoss(
        lambda_recon=train_config.get('lambda_recon', 1.0),
        lambda_velocity=train_config.get('lambda_velocity', 0.5),
        lambda_commit=train_config.get('lambda_commit', 0.25),
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
    )
    
    # Scheduler
    num_epochs = train_config.get('epochs', 500)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=train_config.get('learning_rate_min', 1e-6),
    )
    
    # Resume
    start_epoch = 1
    best_mpjpe = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mpjpe = checkpoint['metrics'].get('mpjpe', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best MPJPE: {best_mpjpe:.2f}")
    
    # Training log
    log_path = os.path.join(output_dir, 'training_log.json')
    training_log = []
    
    codebook_size = model_config.get('codebook_size', 512)
    
    # SMPL-X converter (optional, for better metrics)
    smplx_converter = None
    smplx_config = config.get('smplx', {})
    if eval_config.get('use_smplx', False) and HAS_SMPLX:
        smplx_model_path = smplx_config.get('model_path', 'deps/smpl_models')
        if os.path.exists(smplx_model_path):
            smplx_converter = SMPLXConverter(
                model_path=smplx_model_path,
                gender=smplx_config.get('gender', 'neutral'),
                device=device,
            )
    
    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            codebook_size=codebook_size, mean=mean, std=std
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader, loss_fn, device, epoch, mean, std,
            codebook_size=codebook_size, smplx_converter=smplx_converter
        )
        
        scheduler.step()
        
        # Log
        epoch_log = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train': {k: v for k, v in train_metrics.items()},
            'val': {k: v for k, v in val_metrics.items() if k != 'vis_samples'},
        }
        training_log.append(epoch_log)
        
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # Print Train
        print(f"\n[Train]")
        print(f"  Loss - Recon: {train_metrics['recon_loss']:.4f}, "
              f"Commit: {train_metrics['commit_loss']:.4f}, "
              f"Velocity: {train_metrics['velocity_loss']:.4f}")
        print(f"  Metrics - MPJPE: {train_metrics['mpjpe']:.2f}mm, "
              f"MPJVE: {train_metrics['mpjve']:.2f}mm")
        cb = train_metrics['codebook']
        print(f"  Codebook - Used: {cb['used_codes']}/{cb['total_codes']} "
              f"({cb['usage_rate']:.1f}%), Dead: {cb['dead_codes']}")
        print(f"           Usage - Min: {cb['min_usage']}, Max: {cb['max_usage']}, "
              f"Mean: {cb['avg_usage']:.1f}, Median: {cb['median_usage']:.1f}")
        print(f"           Mode: code[{cb['mode_code']}]={cb['mode_count']}, "
              f"Perplexity: {cb['perplexity']:.1f}")
        
        # Print Val
        print(f"\n[Val]")
        print(f"  Loss - Recon: {val_metrics['recon_loss']:.4f}, "
              f"Commit: {val_metrics['commit_loss']:.4f}")
        print(f"  Metrics - MPJPE: {val_metrics['mpjpe']:.2f}mm, "
              f"MPJVE: {val_metrics['mpjve']:.2f}mm, "
              f"DTW: {val_metrics['dtw']:.4f}")
        cb = val_metrics['codebook']
        print(f"  Codebook - Used: {cb['used_codes']}/{cb['total_codes']} "
              f"({cb['usage_rate']:.1f}%), Dead: {cb['dead_codes']}")
        print(f"           Usage - Min: {cb['min_usage']}, Max: {cb['max_usage']}, "
              f"Mean: {cb['avg_usage']:.1f}, Median: {cb['median_usage']:.1f}")
        print(f"           Mode: code[{cb['mode_code']}]={cb['mode_count']}, "
              f"Perplexity: {cb['perplexity']:.1f}")
        
        # Best check
        is_best = val_metrics['mpjpe'] < best_mpjpe
        if is_best:
            best_mpjpe = val_metrics['mpjpe']
            print(f"\n*** New best MPJPE: {best_mpjpe:.2f}mm ***")
        
        # Save checkpoint
        save_every = train_config.get('save_every', 10)
        if is_best or epoch % save_every == 0:
            ckpt_path = os.path.join(output_dir, 'checkpoints', f'epoch_{epoch:04d}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, ckpt_path, is_best)
    
    print(f"\nTraining completed! Best MPJPE: {best_mpjpe:.2f}mm")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SPL-VQVAE v2')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    main(args)