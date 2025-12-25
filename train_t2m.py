"""
Train Text-to-Motion (T2M) Model using mBART
With MPJPE and DTW evaluation metrics.

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
from common.metrics import mpjpe, dtw_distance, compute_part_metrics, AverageMeter
from common.motion_decoder import load_vqvae_decoder
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
        
        outputs = model(
            texts=texts,
            motion_tokens=body_codes,
            hand_tokens=lhand_codes,
            rhand_tokens=rhand_codes,
            lengths=lengths,
        )
        
        total_loss = outputs['loss']
        if outputs['loss_hand'] is not None:
            total_loss = total_loss + outputs['loss_hand']
        if outputs['loss_rhand'] is not None:
            total_loss = total_loss + outputs['loss_rhand']
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        
        N = len(texts)
        loss_all["total_loss"].update(total_loss.item(), N)
        loss_all["body_loss"].update(outputs['loss'].item(), N)
        
        if outputs['loss_hand'] is not None:
            loss_all["lhand_loss"].update(outputs['loss_hand'].item(), N)
        if outputs['loss_rhand'] is not None:
            loss_all["rhand_loss"].update(outputs['loss_rhand'].item(), N)
        
        if outputs.get('logits') is not None and outputs.get('labels') is not None:
            body_acc = compute_accuracy(outputs['logits'], outputs['labels'])
            acc_all["body_acc"].update(body_acc, N)
        
        if outputs.get('logits_hand') is not None and outputs.get('labels_hand') is not None:
            lhand_acc = compute_accuracy(outputs['logits_hand'], outputs['labels_hand'])
            acc_all["lhand_acc"].update(lhand_acc, N)
        
        if outputs.get('logits_rhand') is not None and outputs.get('labels_rhand') is not None:
            rhand_acc = compute_accuracy(outputs['logits_rhand'], outputs['labels_rhand'])
            acc_all["rhand_acc"].update(rhand_acc, N)
    
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
        
        total_loss = outputs['loss']
        if outputs['loss_hand'] is not None:
            total_loss = total_loss + outputs['loss_hand']
        if outputs['loss_rhand'] is not None:
            total_loss = total_loss + outputs['loss_rhand']
        
        N = len(texts)
        loss_all["total_loss"].update(total_loss.item(), N)
        loss_all["body_loss"].update(outputs['loss'].item(), N)
        
        if outputs['loss_hand'] is not None:
            loss_all["lhand_loss"].update(outputs['loss_hand'].item(), N)
        if outputs['loss_rhand'] is not None:
            loss_all["rhand_loss"].update(outputs['loss_rhand'].item(), N)
        
        if outputs.get('logits') is not None and outputs.get('labels') is not None:
            body_acc = compute_accuracy(outputs['logits'], outputs['labels'])
            acc_all["body_acc"].update(body_acc, N)
        
        if outputs.get('logits_hand') is not None and outputs.get('labels_hand') is not None:
            lhand_acc = compute_accuracy(outputs['logits_hand'], outputs['labels_hand'])
            acc_all["lhand_acc"].update(lhand_acc, N)
        
        if outputs.get('logits_rhand') is not None and outputs.get('labels_rhand') is not None:
            rhand_acc = compute_accuracy(outputs['logits_rhand'], outputs['labels_rhand'])
            acc_all["rhand_acc"].update(rhand_acc, N)
    
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
def evaluate_generation(
    config,
    dataloader,
    model,
    motion_decoder,
    max_samples: int = 100,
    max_gen_length: int = 100,
):
    """
    Evaluate generation quality with MPJPE and DTW.
    
    Args:
        config: Config dict
        dataloader: Test dataloader
        model: T2M model
        motion_decoder: MotionDecoder for codes → pose
        max_samples: Maximum samples to evaluate
        max_gen_length: Maximum generation length
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    
    mpjpe_meter = AverageMeter()
    mpjpe_body_meter = AverageMeter()
    mpjpe_lhand_meter = AverageMeter()
    mpjpe_rhand_meter = AverageMeter()
    dtw_meter = AverageMeter()
    len_diff_meter = AverageMeter()
    
    num_evaluated = 0
    
    for batch_dict in tqdm(dataloader, desc="Evaluating Generation"):
        texts = batch_dict['text']
        gt_body_codes = batch_dict['body_codes']
        gt_lhand_codes = batch_dict['lhand_codes']
        gt_rhand_codes = batch_dict['rhand_codes']
        gt_poses = batch_dict.get('poses', None)  # GT poses if available
        lengths = batch_dict['lengths']
        
        # Generate codes
        gen_outputs = model.generate(
            texts=texts,
            max_length=max_gen_length,
            num_beams=1,
            do_sample=False,
        )
        
        pred_body_codes = gen_outputs['body_tokens']
        pred_lhand_codes = gen_outputs['hand_tokens']
        pred_rhand_codes = gen_outputs['rhand_tokens']
        
        # Evaluate each sample
        for i in range(len(texts)):
            if num_evaluated >= max_samples:
                break
            
            # Get predicted codes
            pred_body = pred_body_codes[i] if pred_body_codes else torch.zeros(1)
            pred_lhand = pred_lhand_codes[i] if pred_lhand_codes else torch.zeros(1)
            pred_rhand = pred_rhand_codes[i] if pred_rhand_codes else torch.zeros(1)
            
            # Skip if empty
            if len(pred_body) == 0 or len(pred_lhand) == 0 or len(pred_rhand) == 0:
                continue
            
            # Get GT codes
            gt_body = gt_body_codes[i][:lengths[i]]
            gt_lhand = gt_lhand_codes[i][:lengths[i]]
            gt_rhand = gt_rhand_codes[i][:lengths[i]]
            
            try:
                # Decode to poses
                pred_pose = motion_decoder.decode(pred_body, pred_lhand, pred_rhand)
                gt_pose = motion_decoder.decode(gt_body, gt_lhand, gt_rhand)
                
                # Compute metrics
                mpjpe_val = mpjpe(pred_pose, gt_pose)
                mpjpe_meter.update(mpjpe_val)
                
                # Part-wise MPJPE
                part_metrics = compute_part_metrics(pred_pose, gt_pose)
                mpjpe_body_meter.update(part_metrics['mpjpe_body'])
                mpjpe_lhand_meter.update(part_metrics['mpjpe_lhand'])
                mpjpe_rhand_meter.update(part_metrics['mpjpe_rhand'])
                
                # DTW
                try:
                    dtw_val = dtw_distance(pred_pose, gt_pose)
                    dtw_meter.update(dtw_val)
                except:
                    pass
                
                # Length difference
                len_diff_meter.update(abs(len(pred_pose) - len(gt_pose)))
                
                num_evaluated += 1
                
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                continue
        
        if num_evaluated >= max_samples:
            break
    
    return {
        'mpjpe': mpjpe_meter.avg,
        'mpjpe_body': mpjpe_body_meter.avg,
        'mpjpe_lhand': mpjpe_lhand_meter.avg,
        'mpjpe_rhand': mpjpe_rhand_meter.avg,
        'dtw': dtw_meter.avg,
        'len_diff': len_diff_meter.avg,
        'num_samples': num_evaluated,
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
    
    # Create model
    print("Creating mBART T2M model...")
    model = MBartT2M(
        model_path=model_config.get("lm_model_path", "facebook/mbart-large-cc25"),
        model_type=model_config.get("lm_model_type", "mbart_multi"),
        body_codebook_size=model_config.get("body_code_num", 96),
        hand_codebook_size=model_config.get("hand_code_num", 192),
        rhand_codebook_size=model_config.get("rhand_code_num", 192),
        max_length=model_config.get("max_length", 256),
        num_heads=model_config.get("num_heads", 3),
        down_t=model_config.get("down_t", 2),
        # Regularization
        label_smoothing=model_config.get("label_smoothing", 0.1),
        dropout=model_config.get("dropout", 0.1),
        freeze_encoder=model_config.get("freeze_encoder", True),
    ).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Load VQ-VAE decoder for evaluation
    motion_decoder = None
    vqvae_path = model_config.get("vqvae_path", None) or train_config.get("vqvae_path", None)
    if vqvae_path and os.path.exists(vqvae_path):
        print(f"Loading VQ-VAE decoder from {vqvae_path}...")
        motion_decoder = load_vqvae_decoder(
            vqvae_path,
            model_config,
            device='cuda'
        )
    else:
        print("VQ-VAE path not found, skipping generation evaluation")
    
    # Optimizer
    lr = float(train_config["learning_rate"])
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable params
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
    best_mpjpe = float('inf')
    epochs = train_config["epochs"]
    clip_grad_norm = train_config.get("clip_grad_norm", 1.0)
    eval_every = train_config.get("eval_every", 10)  # Generation eval every N epochs
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*80}")
        
        if args.train:
            train_metrics = train(config, train_dataloader, model, optimizer, clip_grad_norm)
            scheduler.step(epoch)
        
        with torch.no_grad():
            test_metrics = test(config, test_dataloader, model)
        
        # Generation evaluation (every eval_every epochs)
        gen_metrics = None
        if motion_decoder is not None and (epoch == 1 or epoch % eval_every == 0):
            print("\n--- Generation Evaluation ---")
            try:
                gen_metrics = evaluate_generation(
                    config,
                    test_dataloader,
                    model,
                    motion_decoder,
                    max_samples=30,  # 처음엔 적게
                    max_gen_length=100,
                )
            except Exception as e:
                print(f"Generation evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                gen_metrics = None
        
        is_best = test_metrics['total_loss'] < best_loss
        
        if args.train and is_best:
            best_epoch = epoch
            best_loss = test_metrics['total_loss']
            args.previous_name = save_model(args, epoch, test_metrics['total_loss'], model, "t2m_model")
        
        # Update best MPJPE
        if gen_metrics and gen_metrics['mpjpe'] < best_mpjpe:
            best_mpjpe = gen_metrics['mpjpe']
        
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
            
            if gen_metrics:
                print(f"[GEN]   MPJPE: {gen_metrics['mpjpe']:.4f} "
                      f"(body: {gen_metrics['mpjpe_body']:.4f}, "
                      f"lhand: {gen_metrics['mpjpe_lhand']:.4f}, "
                      f"rhand: {gen_metrics['mpjpe_rhand']:.4f}) | "
                      f"DTW: {gen_metrics['dtw']:.4f} | "
                      f"LenDiff: {gen_metrics['len_diff']:.1f}")
            
            print(f"[BEST]  epoch: {best_epoch} | loss: {best_loss:.4f} | mpjpe: {best_mpjpe:.4f}")
            
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
            if gen_metrics:
                print(f"[GEN] MPJPE: {gen_metrics['mpjpe']:.4f} | DTW: {gen_metrics['dtw']:.4f}")
            break
    
    # Save final model
    if args.train:
        torch.save(model.state_dict(), f"{args.checkpoint}/last.pth")
        print(f"\nTraining finished!")
        print(f"Best epoch: {best_epoch}, Best loss: {best_loss:.4f}, Best MPJPE: {best_mpjpe:.4f}")