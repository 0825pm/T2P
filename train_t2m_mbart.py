"""
Train Text-to-Motion (T2M) Model using mBART (SOKE-style)

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

from common.utils import AccumLoss, save_model
from dataset.data_t2m import load_t2m_data, make_t2m_iter
from model.mgpt_mbart import MBartT2M


def compute_accuracy(logits, labels):
    """Compute token-level accuracy."""
    if logits is None or labels is None:
        return 0.0
    
    try:
        preds = logits.argmax(dim=-1)
        mask = labels != -100
        
        if mask.sum() == 0:
            return 0.0
        
        correct = (preds == labels) & mask
        return correct.sum().item() / mask.sum().item()
    except:
        return 0.0


def train(config, dataloader, model, optimizer, clip_grad_norm=1.0):
    """
    Training loop for one epoch (SOKE-style).
    
    Key difference from before:
    - Pass texts and motion codes directly to model
    - Model handles tokenization and string conversion internally
    """
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
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_dict in pbar:
        optimizer.zero_grad()
        
        # Get batch data
        texts = batch_dict['text']
        body_codes = batch_dict['body_codes'].cuda()      # (B, T) motion code indices
        lhand_codes = batch_dict['lhand_codes'].cuda()    # (B, T) 
        rhand_codes = batch_dict['rhand_codes'].cuda()    # (B, T)
        lengths = batch_dict['lengths']                    # List of valid lengths
        
        # Get data source if available
        data_src = batch_dict.get('src', ['phoenix'] * len(texts))
        
        # Forward (SOKE-style: model handles tokenization internally)
        outputs = model(
            texts=texts,
            motion_tokens=body_codes,
            hand_tokens=lhand_codes,
            rhand_tokens=rhand_codes,
            lengths=lengths,
            data_src=data_src,
        )
        
        # Compute total loss with weights from config
        loss_cfg = config.get('loss', {})
        lambda_body = loss_cfg.get('lambda_body', 1.0)
        lambda_lhand = loss_cfg.get('lambda_lhand', 0.4)
        lambda_rhand = loss_cfg.get('lambda_rhand', 0.4)
        
        total_loss = lambda_body * outputs['loss']
        if outputs['loss_hand'] is not None:
            total_loss = total_loss + lambda_lhand * outputs['loss_hand']
        if outputs['loss_rhand'] is not None:
            total_loss = total_loss + lambda_rhand * outputs['loss_rhand']
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        
        # Update metrics
        N = len(texts)
        loss_all["total_loss"].update(total_loss.item(), N)
        loss_all["body_loss"].update(outputs['loss'].item(), N)
        
        if outputs['loss_hand'] is not None:
            loss_all["lhand_loss"].update(outputs['loss_hand'].item(), N)
        if outputs['loss_rhand'] is not None:
            loss_all["rhand_loss"].update(outputs['loss_rhand'].item(), N)
        
        # Compute accuracy if logits available
        if outputs.get('logits_body') is not None:
            body_acc = compute_accuracy(outputs['logits_body'], outputs.get('labels'))
            acc_all["body_acc"].update(body_acc, N)
        
        if outputs.get('logits_lhand') is not None:
            lhand_acc = compute_accuracy(outputs['logits_lhand'], outputs.get('labels_hand'))
            acc_all["lhand_acc"].update(lhand_acc, N)
        
        if outputs.get('logits_rhand') is not None:
            rhand_acc = compute_accuracy(outputs['logits_rhand'], outputs.get('labels_rhand'))
            acc_all["rhand_acc"].update(rhand_acc, N)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_all['total_loss'].avg:.4f}",
            'body': f"{loss_all['body_loss'].avg:.4f}",
            'lh': f"{loss_all['lhand_loss'].avg:.4f}",
            'rh': f"{loss_all['rhand_loss'].avg:.4f}",
            'acc_b': f"{acc_all['body_acc'].avg*100:.1f}%",
            'acc_lh': f"{acc_all['lhand_acc'].avg*100:.1f}%",
            'acc_rh': f"{acc_all['rhand_acc'].avg*100:.1f}%"
        })
    
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
    """Validation loop (loss & accuracy only)."""
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
    
    pbar = tqdm(dataloader, desc="Testing")
    for batch_dict in pbar:
        texts = batch_dict['text']
        body_codes = batch_dict['body_codes'].cuda()
        lhand_codes = batch_dict['lhand_codes'].cuda()
        rhand_codes = batch_dict['rhand_codes'].cuda()
        lengths = batch_dict['lengths']
        data_src = batch_dict.get('src', ['phoenix'] * len(texts))
        
        # Forward
        outputs = model(
            texts=texts,
            motion_tokens=body_codes,
            hand_tokens=lhand_codes,
            rhand_tokens=rhand_codes,
            lengths=lengths,
            data_src=data_src,
        )
        
        # Loss weights
        loss_cfg = config.get('loss', {})
        lambda_body = loss_cfg.get('lambda_body', 1.0)
        lambda_lhand = loss_cfg.get('lambda_lhand', 0.4)
        lambda_rhand = loss_cfg.get('lambda_rhand', 0.4)
        
        total_loss = lambda_body * outputs['loss']
        if outputs['loss_hand'] is not None:
            total_loss = total_loss + lambda_lhand * outputs['loss_hand']
        if outputs['loss_rhand'] is not None:
            total_loss = total_loss + lambda_rhand * outputs['loss_rhand']
        
        N = len(texts)
        loss_all["total_loss"].update(total_loss.item(), N)
        loss_all["body_loss"].update(outputs['loss'].item(), N)
        
        if outputs['loss_hand'] is not None:
            loss_all["lhand_loss"].update(outputs['loss_hand'].item(), N)
        if outputs['loss_rhand'] is not None:
            loss_all["rhand_loss"].update(outputs['loss_rhand'].item(), N)
        
        if outputs.get('logits_body') is not None:
            body_acc = compute_accuracy(outputs['logits_body'], outputs.get('labels'))
            acc_all["body_acc"].update(body_acc, N)
        
        if outputs.get('logits_lhand') is not None:
            lhand_acc = compute_accuracy(outputs['logits_lhand'], outputs.get('labels_hand'))
            acc_all["lhand_acc"].update(lhand_acc, N)
        
        if outputs.get('logits_rhand') is not None:
            rhand_acc = compute_accuracy(outputs['logits_rhand'], outputs.get('labels_rhand'))
            acc_all["rhand_acc"].update(rhand_acc, N)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_all['total_loss'].avg:.4f}",
            'body': f"{loss_all['body_loss'].avg:.4f}",
            'lh': f"{loss_all['lhand_loss'].avg:.4f}",
            'rh': f"{loss_all['rhand_loss'].avg:.4f}",
            'acc_b': f"{acc_all['body_acc'].avg*100:.1f}%",
            'acc_lh': f"{acc_all['lhand_acc'].avg*100:.1f}%",
            'acc_rh': f"{acc_all['rhand_acc'].avg*100:.1f}%"
        })
    
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
    
    # Get configs
    train_config = config.get("training", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    
    batch_size = train_config.get("batch_size", 64)
    
    # Load data
    print("=" * 60)
    print("Loading T2M data...")
    print("=" * 60)
    train_data, dev_data, test_data, _, _ = load_t2m_data(cfg=config)
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    
    train_dataloader = make_t2m_iter(
        train_data,
        batch_size=batch_size,
        train=True,
        shuffle=True,
        num_workers=train_config.get("num_workers", 4),
    )
    
    test_dataloader = make_t2m_iter(
        dev_data,
        batch_size=batch_size,
        train=False,
        shuffle=False,
        num_workers=train_config.get("num_workers", 4),
    )
    
    # Create model (SOKE-style)
    print("=" * 60)
    print("Creating mBART T2M model...")
    print("=" * 60)
    
    model = MBartT2M(
        model_path=model_config.get("lm_model_path", "/home/user/Projects/research/SOKE/deps/mbart-h2s-csl-phoenix"),
        model_type=model_config.get("lm_model_type", "mbart_multi"),
        stage="lm_pretrain",
        motion_codebook_size=model_config.get("body_code_num", 96),
        hand_codebook_size=model_config.get("hand_code_num", 192),
        rhand_codebook_size=model_config.get("rhand_code_num", 192),
        max_length=model_config.get("max_length", 256),
        num_heads=model_config.get("num_heads", 3),
        label_smoothing=model_config.get("label_smoothing", 0.1),
        freeze_encoder=model_config.get("freeze_encoder", True),
    ).cuda()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer
    lr = train_config.get("learning_rate", 2e-4)
    if isinstance(lr, str):
        lr = float(lr)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=train_config.get("weight_decay", 0.01)
    )
    
    # LR Scheduler
    lr_min = train_config.get("learning_rate_min", 1e-6)
    if isinstance(lr_min, str):
        lr_min = float(lr_min)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.get("epochs", 150),
        eta_min=lr_min
    )
    
    # Resume if specified
    if args.previous_dir:
        print(f"Loading checkpoint from {args.previous_dir}")
        state_dict = torch.load(args.previous_dir, map_location='cuda')
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
    
    # Create checkpoint directory
    model_dir = train_config.get("model_dir", "checkpoints/t2m_mbart")
    os.makedirs(model_dir, exist_ok=True)
    args.checkpoint = model_dir
    
    # Training
    best_epoch = 0
    best_loss = float('inf')
    epochs = train_config.get("epochs", 150)
    clip_grad_norm = train_config.get("clip_grad_norm", 1.0)
    save_every = train_config.get("save_every", 10)
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*80}")
        
        if args.train:
            train_metrics = train(config, train_dataloader, model, optimizer, clip_grad_norm)
            scheduler.step()
        
        with torch.no_grad():
            test_metrics = test(config, test_dataloader, model)
        
        is_best = test_metrics['total_loss'] < best_loss
        
        if args.train and is_best:
            best_epoch = epoch
            best_loss = test_metrics['total_loss']
            args.previous_name = save_model(args, epoch, test_metrics['total_loss'], model, "t2m_model")
        
        # Periodic save
        if args.train and epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_metrics['total_loss'],
            }, os.path.join(model_dir, f'epoch_{epoch:04d}.pth'))
        
        # Logging
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
            
            # LR info
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[LR]    {current_lr:.2e}")
        else:
            print(f"[TEST] loss: {test_metrics['total_loss']:.4f} | "
                  f"acc: body {test_metrics['body_acc']:.1f}%, "
                  f"lhand {test_metrics['lhand_acc']:.1f}%, "
                  f"rhand {test_metrics['rhand_acc']:.1f}%")
            break
    
    # Save final model
    if args.train:
        torch.save(model.state_dict(), os.path.join(model_dir, "last.pth"))
        print(f"\nTraining finished!")
        print(f"Best epoch: {best_epoch}, Best loss: {best_loss:.4f}")