#!/usr/bin/env python
# coding: utf-8
"""
Train Decouple VQ-VAE with SOKE DataLoader
Supports How2Sign, CSL-Daily, Phoenix-2014T

Usage:
    python train_vqvae_soke.py --config configs/vqvae_soke.yaml --gpu 0
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime

# Parse args first to set GPU
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--train', type=int, default=1)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--resume', type=str, default='')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter, defaultdict

# SOKE DataLoader
from mGPT.data.humanml import H2SMotionDatasetVQ
from mGPT.data.utils import humanml3d_collate

# Model
from model.vqvae_soke import VQVAE


def setup_logging(checkpoint_dir):
    """Setup logging to file and console."""
    log_file = os.path.join(checkpoint_dir, 'train.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_mean_std(config):
    """Load and filter mean/std for SOKE format."""
    mean = torch.load(config['data']['mean_path'])
    std = torch.load(config['data']['std_path'])
    
    # SOKE filtering: remove lower body and shape
    # Original 179 -> 143 (remove lower body) -> 133 (remove shape)
    mean = mean[(3 + 3 * 11):]  # Remove lower body
    mean = torch.cat([mean[:-20], mean[-10:]], dim=0)  # Remove shape
    
    std = std[(3 + 3 * 11):]
    std = torch.cat([std[:-20], std[-10:]], dim=0)
    
    return mean, std


def create_dataloaders(config, mean, std):
    """Create train/val/test dataloaders."""
    data_cfg = config['data']
    train_cfg = config['training']
    
    common_kwargs = {
        'data_root': data_cfg['data_root'],
        'mean': mean,
        'std': std,
        'max_motion_length': data_cfg.get('max_motion_length', 400),
        'min_motion_length': data_cfg.get('min_motion_length', 40),
        'win_size': data_cfg.get('win_size', 64),
        'unit_length': data_cfg.get('unit_length', 4),
        'dataset_name': data_cfg.get('dataset_name', 'how2sign_csl_phoenix'),
        'csl_root': data_cfg.get('csl_root'),
        'phoenix_root': data_cfg.get('phoenix_root'),
    }
    
    train_dataset = H2SMotionDatasetVQ(split='train', **common_kwargs)
    val_dataset = H2SMotionDatasetVQ(
        split='val', 
        win_size=None,  # No window for validation
        **{k: v for k, v in common_kwargs.items() if k != 'win_size'}
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 4),
        collate_fn=humanml3d_collate,
        drop_last=True,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 4),
        collate_fn=humanml3d_collate,
        pin_memory=True,
    )
    
    return train_loader, val_loader


class AverageMeter:
    """Computes and stores the average and current value."""
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


def compute_mpjpe(pred, gt, mask=None):
    """Compute Mean Per Joint Position Error."""
    # pred, gt: (B, T, D)
    diff = pred - gt
    if mask is not None:
        diff = diff * mask.unsqueeze(-1)
    
    # Reshape to joints: (B, T, J, 3)
    # For SOKE 133 dims, approximate joint calculation
    B, T, D = diff.shape
    
    # Per-frame error (L2 norm across features)
    error = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (B, T)
    
    if mask is not None:
        error = error * mask
        return error.sum() / mask.sum()
    else:
        return error.mean()


def train_epoch(model, dataloader, optimizer, config, epoch):
    """Train for one epoch."""
    model.train()
    
    losses = {
        'total': AverageMeter(),
        'recon': AverageMeter(),
        'commit': AverageMeter(),
    }
    perplexities = defaultdict(AverageMeter)
    code_counters = defaultdict(Counter)
    mpjpe_meter = AverageMeter()
    
    commit_weight = config['training'].get('commit_weight', 0.02)
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}')
    for batch in pbar:
        if batch is None:
            continue
        
        motion = batch['motion'].cuda()  # (B, T, 133)
        lengths = batch['lengths']
        
        # Create mask
        B, T, D = motion.shape
        mask = torch.zeros(B, T, device=motion.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        
        # Forward
        optimizer.zero_grad()
        
        recon, commit_loss, ppl_dict, idx_dict = model(motion)
        
        # Reconstruction loss (masked)
        recon_loss = ((recon - motion) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum() / D
        
        # Total loss
        total_loss = recon_loss + commit_weight * commit_loss
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        N = B
        losses['total'].update(total_loss.item(), N)
        losses['recon'].update(recon_loss.item(), N)
        losses['commit'].update(commit_loss.item(), N)
        
        for part, ppl in ppl_dict.items():
            perplexities[part].update(ppl.item(), N)
        
        for part, indices in idx_dict.items():
            code_counters[part].update(indices.flatten().cpu().tolist())
        
        with torch.no_grad():
            mpjpe = compute_mpjpe(recon, motion, mask)
            mpjpe_meter.update(mpjpe.item(), N)
        
        pbar.set_postfix({
            'loss': f'{losses["total"].avg:.4f}',
            'recon': f'{losses["recon"].avg:.4f}',
            'mpjpe': f'{mpjpe_meter.avg:.4f}',
        })
    
    # Codebook usage
    usage = {}
    for part, counter in code_counters.items():
        usage[part] = len(counter)
    
    return {
        'total_loss': losses['total'].avg,
        'recon_loss': losses['recon'].avg,
        'commit_loss': losses['commit'].avg,
        'perplexity': {k: v.avg for k, v in perplexities.items()},
        'codebook_usage': usage,
        'mpjpe': mpjpe_meter.avg,
    }


@torch.no_grad()
def validate(model, dataloader, config):
    """Validation loop."""
    model.eval()
    
    losses = {
        'total': AverageMeter(),
        'recon': AverageMeter(),
        'commit': AverageMeter(),
    }
    perplexities = defaultdict(AverageMeter)
    code_counters = defaultdict(Counter)
    mpjpe_meter = AverageMeter()
    
    commit_weight = config['training'].get('commit_weight', 0.02)
    
    for batch in tqdm(dataloader, desc='Validation'):
        if batch is None:
            continue
        
        motion = batch['motion'].cuda()
        lengths = batch['lengths']
        
        B, T, D = motion.shape
        mask = torch.zeros(B, T, device=motion.device)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1.0
        
        recon, commit_loss, ppl_dict, idx_dict = model(motion)
        
        recon_loss = ((recon - motion) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum() / D
        total_loss = recon_loss + commit_weight * commit_loss
        
        N = B
        losses['total'].update(total_loss.item(), N)
        losses['recon'].update(recon_loss.item(), N)
        losses['commit'].update(commit_loss.item(), N)
        
        for part, ppl in ppl_dict.items():
            perplexities[part].update(ppl.item(), N)
        
        for part, indices in idx_dict.items():
            code_counters[part].update(indices.flatten().cpu().tolist())
        
        mpjpe = compute_mpjpe(recon, motion, mask)
        mpjpe_meter.update(mpjpe.item(), N)
    
    usage = {part: len(counter) for part, counter in code_counters.items()}
    
    return {
        'total_loss': losses['total'].avg,
        'recon_loss': losses['recon'].avg,
        'commit_loss': losses['commit'].avg,
        'perplexity': {k: v.avg for k, v in perplexities.items()},
        'codebook_usage': usage,
        'mpjpe': mpjpe_meter.avg,
    }


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # Save last
    torch.save(state, os.path.join(checkpoint_dir, 'last.pth'))
    
    # Save best
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))


def format_metrics(metrics, prefix=''):
    """Format metrics for logging."""
    ppl_str = ', '.join([f'{k}:{v:.1f}' for k, v in metrics['perplexity'].items()])
    usage_str = ', '.join([f'{k}:{v}' for k, v in metrics['codebook_usage'].items()])
    
    return (f"{prefix}loss: {metrics['total_loss']:.4f} "
            f"(recon: {metrics['recon_loss']:.4f}, commit: {metrics['commit_loss']:.4f}) | "
            f"ppl: [{ppl_str}] | codes: [{usage_str}] | mpjpe: {metrics['mpjpe']:.4f}")


def main():
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup checkpoint directory
    if args.checkpoint:
        checkpoint_dir = args.checkpoint
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = os.path.join(
            config['training'].get('model_dir', 'checkpoints'),
            f'vqvae_soke_{timestamp}'
        )
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Setup logging
    logger = setup_logging(checkpoint_dir)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Config: {args.config}")
    
    # Load mean/std
    logger.info("Loading mean/std...")
    mean, std = load_mean_std(config)
    logger.info(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, mean, std)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model_cfg = config['model']
    model = VQVAE(
        nfeats=model_cfg.get('nfeats', 133),
        body_code_num=model_cfg.get('body_code_num', 96),
        hand_code_num=model_cfg.get('hand_code_num', 192),
        code_dim=model_cfg.get('code_dim', 512),
        output_emb_width=model_cfg.get('output_emb_width', 512),
        down_t=model_cfg.get('down_t', 2),
        stride_t=model_cfg.get('stride_t', 2),
        width=model_cfg.get('width', 512),
        depth=model_cfg.get('depth', 3),
        dilation_growth_rate=model_cfg.get('dilation_growth_rate', 3),
        norm=model_cfg.get('norm', None),
        activation=model_cfg.get('activation', 'relu'),
    ).cuda()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Body codebook: {model_cfg.get('body_code_num', 96)}")
    logger.info(f"Hand codebook: {model_cfg.get('hand_code_num', 192)}")
    
    # Optimizer
    train_cfg = config['training']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.get('learning_rate', 1e-4),
        weight_decay=train_cfg.get('weight_decay', 0.01),
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg['epochs'],
        eta_min=train_cfg.get('learning_rate_min', 1e-6),
    )
    
    # Resume
    start_epoch = 1
    best_mpjpe = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'metrics' in checkpoint:
            best_mpjpe = checkpoint['metrics'].get('mpjpe', float('inf'))
    
    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, train_cfg['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{train_cfg['epochs']} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        if args.train:
            train_metrics = train_epoch(model, train_loader, optimizer, config, epoch)
            logger.info(format_metrics(train_metrics, '[TRAIN] '))
        
        # Validate
        val_metrics = validate(model, val_loader, config)
        logger.info(format_metrics(val_metrics, '[VAL]   '))
        
        # Check best
        is_best = val_metrics['mpjpe'] < best_mpjpe
        if is_best:
            best_mpjpe = val_metrics['mpjpe']
            logger.info(f"[BEST] New best MPJPE: {best_mpjpe:.4f}")
        
        # Save checkpoint
        if args.train:
            save_checkpoint(model, optimizer, epoch, val_metrics, checkpoint_dir, is_best)
        
        # Step scheduler
        scheduler.step()
    
    logger.info("=" * 80)
    logger.info(f"Training finished! Best MPJPE: {best_mpjpe:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
