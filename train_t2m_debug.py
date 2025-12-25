"""
Train Text-to-Motion (T2M) Model using mBART
SOKE-style: Separate small heads for body/lhand/rhand

Usage:
    python train_t2m.py --config configs/t2m/t2m_mbart.yaml --gpu 0 --train 1
"""
import warnings
warnings.filterwarnings("ignore")

import os
import yaml
from common.arguments import parse_args

args = parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from argparse import Namespace
from timm.scheduler import create_scheduler

from common.utils import AccumLoss, save_model
from dataset.data_t2m import load_t2m_data, make_t2m_iter
from mGPT.archs.mgpt_mbart import MBartT2M


def train(config, dataloader, model, optimizer, clip_grad_norm=1.0):
    """Training loop for one epoch."""
    loss_all = {
        "total_loss": AccumLoss(),
        "body_loss": AccumLoss(),
        "lhand_loss": AccumLoss(),
        "rhand_loss": AccumLoss(),
    }
    acc_all = {
        "body_acc": AccumLoss(),
        "lhand_acc": AccumLoss(),
        "rhand_acc": AccumLoss(),
    }
    
    model.train()
    
    for batch_dict in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        texts = batch_dict['text']
        body_codes = batch_dict['body_codes'].cuda()
        lhand_codes = batch_dict['lhand_codes'].cuda()
        rhand_codes = batch_dict['rhand_codes'].cuda()
        lengths = batch_dict['lengths']
        
        # Forward (model now returns loss and acc directly)
        outputs = model(
            texts=texts,
            motion_tokens=body_codes,
            hand_tokens=lhand_codes,
            rhand_tokens=rhand_codes,
            lengths=lengths,
        )
        
        # Backward
        outputs['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        
        # Update metrics
        N = len(texts)
        loss_all["total_loss"].update(outputs['loss'].item(), N)
        loss_all["body_loss"].update(outputs['loss_body'].item(), N)
        loss_all["lhand_loss"].update(outputs['loss_hand'].item(), N)
        loss_all["rhand_loss"].update(outputs['loss_rhand'].item(), N)
        
        acc_all["body_acc"].update(outputs['body_acc'], N)
        acc_all["lhand_acc"].update(outputs['lhand_acc'], N)
        acc_all["rhand_acc"].update(outputs['rhand_acc'], N)
    
    return {
        'total_loss': loss_all["total_loss"].avg,
        'body_loss': loss_all["body_loss"].avg,
        'lhand_loss': loss_all["lhand_loss"].avg,
        'rhand_loss': loss_all["rhand_loss"].avg,
        'body_acc': acc_all["body_acc"].avg * 100,
        'lhand_acc': acc_all["lhand_acc"].avg * 100,
        'rhand_acc': acc_all["rhand_acc"].avg * 100,
    }


@torch.no_grad()
def test(config, dataloader, model):
    """Validation loop."""
    loss_all = {
        "total_loss": AccumLoss(),
        "body_loss": AccumLoss(),
        "lhand_loss": AccumLoss(),
        "rhand_loss": AccumLoss(),
    }
    acc_all = {
        "body_acc": AccumLoss(),
        "lhand_acc": AccumLoss(),
        "rhand_acc": AccumLoss(),
    }
    
    model.eval()
    
    for batch_dict in tqdm(dataloader, desc="Testing"):
        texts = batch_dict['text']
        body_codes = batch_dict['body_codes'].cuda()
        lhand_codes = batch_dict['lhand_codes'].cuda()
        rhand_codes = batch_dict['rhand_codes'].cuda()
        lengths = batch_dict['lengths']
        
        outputs = model(
            texts=texts,
            motion_tokens=body_codes,
            hand_tokens=lhand_codes,
            rhand_tokens=rhand_codes,
            lengths=lengths,
        )
        
        N = len(texts)
        loss_all["total_loss"].update(outputs['loss'].item(), N)
        loss_all["body_loss"].update(outputs['loss_body'].item(), N)
        loss_all["lhand_loss"].update(outputs['loss_hand'].item(), N)
        loss_all["rhand_loss"].update(outputs['loss_rhand'].item(), N)
        
        acc_all["body_acc"].update(outputs['body_acc'], N)
        acc_all["lhand_acc"].update(outputs['lhand_acc'], N)
        acc_all["rhand_acc"].update(outputs['rhand_acc'], N)
    
    return {
        'total_loss': loss_all["total_loss"].avg,
        'body_loss': loss_all["body_loss"].avg,
        'lhand_loss': loss_all["lhand_loss"].avg,
        'rhand_loss': loss_all["rhand_loss"].avg,
        'body_acc': acc_all["body_acc"].avg * 100,
        'lhand_acc': acc_all["lhand_acc"].avg * 100,
        'rhand_acc': acc_all["rhand_acc"].avg * 100,
    }


if __name__ == "__main__":
    seed = 1126
    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    train_config = config["training"]
    model_config = config["model"]
    batch_size = train_config["batch_size"]
    
    # Load data
    print("Loading T2M data...")
    train_data, dev_data, test_data, _, _ = load_t2m_data(cfg=config)
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    
    train_dataloader = make_t2m_iter(
        train_data,
        batch_size=batch_size,
        train=True,
        shuffle=True
    )
    
    test_dataloader = make_t2m_iter(
        dev_data,
        batch_size=batch_size,
        train=False,
        shuffle=False
    )
    
    # Create model (SOKE-style)
    print("Creating mBART T2M model (SOKE-style)...")
    model = MBartT2M(
        model_path=model_config.get("lm_model_path", "facebook/mbart-large-cc25"),
        model_type=model_config.get("lm_model_type", "mbart_multi"),
        body_codebook_size=model_config.get("body_code_num", 96),
        hand_codebook_size=model_config.get("hand_code_num", 192),
        rhand_codebook_size=model_config.get("rhand_code_num", 192),
        max_length=model_config.get("max_length", 256),
        num_heads=model_config.get("num_heads", 3),
        dropout=model_config.get("dropout", 0.3),
        label_smoothing=model_config.get("label_smoothing", 0.1),
        down_t=model_config.get("down_t", 2),
    ).cuda()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    lr = float(train_config["learning_rate"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_config.get("weight_decay", 0.01)
    )
    
    # Scheduler
    scheduler_config = {}
    for key, value in train_config.items():
        if isinstance(value, str):
            try:
                scheduler_config[key] = float(value)
            except ValueError:
                scheduler_config[key] = value
        else:
            scheduler_config[key] = value
    
    scheduler_args = Namespace(**scheduler_config)
    scheduler, _ = create_scheduler(scheduler_args, optimizer)
    
    # Resume
    if args.previous_dir != "":
        print(f"Loading checkpoint from {args.previous_dir}")
        state_dict = torch.load(args.previous_dir, map_location='cuda')
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    
    # Training
    best_epoch = 0
    best_loss = float('inf')
    epochs = train_config["epochs"]
    clip_grad_norm = train_config.get("clip_grad_norm", 1.0)
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*80}")
        
        if args.train:
            train_metrics = train(config, train_dataloader, model, optimizer, clip_grad_norm)
            scheduler.step(epoch)
        
        with torch.no_grad():
            test_metrics = test(config, test_dataloader, model)
        
        is_best = test_metrics['total_loss'] < best_loss
        
        if args.train and is_best:
            best_epoch = epoch
            best_loss = test_metrics['total_loss']
            args.previous_name = save_model(args, epoch, test_metrics['total_loss'], model, "t2m_model")
        
        if args.train:
            print(f"[TRAIN] loss: {train_metrics['total_loss']:.4f} "
                  f"(body: {train_metrics['body_loss']:.4f}, "
                  f"lhand: {train_metrics['lhand_loss']:.4f}, "
                  f"rhand: {train_metrics['rhand_loss']:.4f}) | "
                  f"acc: body {train_metrics['body_acc']:.1f}%, "
                  f"lhand {train_metrics['lhand_acc']:.1f}%, "
                  f"rhand {train_metrics['rhand_acc']:.1f}%")
            
            print(f"[TEST]  loss: {test_metrics['total_loss']:.4f} "
                  f"(body: {test_metrics['body_loss']:.4f}, "
                  f"lhand: {test_metrics['lhand_loss']:.4f}, "
                  f"rhand: {test_metrics['rhand_loss']:.4f}) | "
                  f"acc: body {test_metrics['body_acc']:.1f}%, "
                  f"lhand {test_metrics['lhand_acc']:.1f}%, "
                  f"rhand {test_metrics['rhand_acc']:.1f}%")
            
            print(f"[BEST]  epoch: {best_epoch} | loss: {best_loss:.4f}")
            
            # LR decay
            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= args.lr_decay
        else:
            print(f"[TEST] loss: {test_metrics['total_loss']:.4f} | "
                  f"acc: body {test_metrics['body_acc']:.1f}%, "
                  f"lhand {test_metrics['lhand_acc']:.1f}%, "
                  f"rhand {test_metrics['rhand_acc']:.1f}%")
            break
    
    # Save final model
    if args.train:
        torch.save(model.state_dict(), f"{args.checkpoint}/last.pth")
        print(f"\nTraining finished! Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")