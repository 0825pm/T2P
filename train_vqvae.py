"""
Train VQ-VAE for Sign Language Production
Supports both:
- vqvae: Single codebook for all joints
- vqvae_decouple: Separate codebooks for body, rhand, lhand

Usage:
    python train_vqvae.py --config configs/t2p_config.yaml --model vqvae --gpu 0 --train 1
    python train_vqvae.py --config configs/t2p_config.yaml --model vqvae_decouple --gpu 0 --train 1
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

import time
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from common.utils import *
import torch.optim as optim
from common.camera import *
from timm.utils import NativeScaler
from timm.scheduler import create_scheduler
from argparse import Namespace

import matplotlib
matplotlib.use("Agg")
from einops import rearrange

from utils.plot_videos import plot_video, alter_DTW_timing
from utils.builders import build_gradient_clipper, build_optimizer, build_scheduler
from dataset.data import load_data, make_data_iter
from dataset.batch import Batch
from torchtext.data import Dataset
from collections import Counter

# Dynamic model import based on --model argument
exec(f"from model.{args.model} import VQVAE")


def calculate_dtw(references, hypotheses):
    """Calculate DTW scores between references and hypotheses."""
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    
    dtw_scores = []
    references = references.cpu().numpy()
    hypotheses = hypotheses.cpu().numpy()
    
    for ref, hyp in zip(references, hypotheses):
        ref = ref.reshape(ref.shape[0], -1)
        hyp = hyp.reshape(hyp.shape[0], -1)
        distance, _ = fastdtw(ref, hyp, dist=euclidean)
        dtw_scores.append(distance / len(ref))
    
    return dtw_scores


class AccumLoss:
    def __init__(self):
        self.val = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val += val * n
        self.count += n
    
    @property
    def avg(self):
        return self.val / self.count if self.count > 0 else 0


def get_codebook_stats(counter, code_num=512, name="Codebook"):
    """Generate codebook usage statistics string."""
    if len(counter) == 0:
        return f"{name}: No codes used"
    
    counts = list(counter.values())
    total = sum(counts)
    unique = len(counter)
    usage_rate = unique / code_num * 100
    
    return (f"{name} | Used: {unique}/{code_num} ({usage_rate:.1f}%), "
            f"Total: {total}, Min: {min(counts)}, Max: {max(counts)}, "
            f"Avg: {sum(counts)/len(counts):.1f}")


def log_codebook_stats_with_nums(all_indices, code_nums, prefix=""):
    """Log codebook statistics for all parts with different codebook sizes."""
    for part_name, counter in all_indices.items():
        code_num = code_nums.get(part_name, 512)
        stats_str = get_codebook_stats(counter, code_num, f"{prefix}{part_name.upper()}")
        logging.info(stats_str)
        print(stats_str)


def log_codebook_stats(indices_dict, code_num, prefix=""):
    """Log codebook statistics for all parts."""
    stats_lines = []
    counters = {}
    
    for part_name, indices in indices_dict.items():
        counter = Counter(indices.detach().cpu().view(-1).tolist())
        counters[part_name] = counter
        stats_str = get_codebook_stats(counter, code_num, f"{prefix}{part_name.upper()}")
        stats_lines.append(stats_str)
        logging.info(stats_str)
        print(stats_str)
    
    return counters


@torch.no_grad()
def save_videos(config, dataloader, model, epoch, checkpoint_dir, src_vocab, num_samples=2):
    """Save reconstruction videos for visualization."""
    model.eval()
    
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("Could not get a batch from dataloader.")
        return

    results_dir = os.path.join(checkpoint_dir, f"videos/epoch_{epoch}")
    os.makedirs(results_dir, exist_ok=True)
    
    batch = Batch(torch_batch=batch, pad_index=0, model=model)
    
    # Input preparation
    pose_input = batch.trg_input[:, :, :150]  # (B, T, 150)
    pose_length = batch.trg_mask[..., 0].sum(dim=-1).ravel()
    
    # Text for filename
    text_input = [" ".join([src_vocab.itos[batch.src[i][j]] 
                           for j in range(len(batch.src[i])-1)]) 
                  for i in range(len(batch.src))]
    
    num_samples = min(num_samples, pose_input.shape[0])
    print(f"Saving {num_samples} reconstruction videos...")
    
    # Forward
    pose_output, commit_loss, perplexity, indices = model(pose_input)
    
    for i in range(num_samples):
        gt_len_i = int(pose_length[i].item())
        
        gt_pose_np_i = pose_input[i, :gt_len_i]
        pred_pose_np_i = pose_output[i, :gt_len_i]
        
        # Add counter (151st dimension)
        gt_pose_np_i = torch.cat((gt_pose_np_i, batch.trg_input[i, :gt_len_i, 150:]), dim=-1)
        pred_pose_np_i = torch.cat((pred_pose_np_i, batch.trg_input[i, :gt_len_i, 150:]), dim=-1)
        
        # DTW timing alignment
        timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(pred_pose_np_i, gt_pose_np_i)
        
        # Filename
        text_str = text_input[i]
        filename = text_str.split("</s>")[0].rstrip()[:50].replace(" ", "_").replace("/", "-")
        
        plot_video(
            joints=timing_hyp_seq,
            file_path=results_dir,
            video_name=f"{i:02d}_{filename}.mp4",
            references=ref_seq_count,
            skip_frames=1,
            sequence_ID=filename
        )
    
    print(f"Videos saved to {results_dir}")


def train(config, dataloader, model, optimizer, code_nums):
    """Training loop for one epoch."""
    loss_all = {
        "total_loss": AccumLoss(),
        "commit_loss": AccumLoss(),
        "recon_loss": AccumLoss(),
    }
    
    all_dtw = []
    all_mpjpe = []
    all_perplexity = {}
    all_indices = {}
    
    model.train()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        
        batch = Batch(torch_batch=batch, pad_index=0, model=model)
        
        pose_input = batch.trg_input[:, :, :150]  # (B, T, 150)
        pose_mask = batch.trg_mask[..., 0].squeeze().unsqueeze(-1)  # (B, T, 1)
        
        # Forward
        pose_output, commit_loss, perplexity_dict, indices_dict = model(pose_input)
        
        # Accumulate indices for each part
        for part_name, indices in indices_dict.items():
            if part_name not in all_indices:
                all_indices[part_name] = Counter()
            all_indices[part_name].update(indices.detach().cpu().view(-1).tolist())
        
        # Accumulate perplexity
        for part_name, ppl in perplexity_dict.items():
            if part_name not in all_perplexity:
                all_perplexity[part_name] = []
            all_perplexity[part_name].append(ppl.cpu().detach().numpy() if torch.is_tensor(ppl) else ppl)
        
        # Reconstruction Loss
        recon_loss = F.l1_loss(pose_output * pose_mask, pose_input * pose_mask)
        
        # Total Loss
        total_loss = recon_loss + commit_loss
        total_loss.backward()
        optimizer.step()

        # Metrics
        pose_output_masked = pose_output * pose_mask
        pose_input_masked = pose_input * pose_mask
        
        pred = pose_output_masked.view(pose_output.shape[0], pose_output.shape[1], -1, 3)
        gt = pose_input_masked.view(pose_input.shape[0], pose_input.shape[1], -1, 3)
        joint_error = torch.mean(torch.norm(pred - gt, dim=-1))
        
        dtw_scores = calculate_dtw(pose_input_masked, pose_output_masked.detach())
        
        all_mpjpe.append(joint_error.cpu().detach().numpy())
        all_dtw.extend(dtw_scores)
        
        N = pose_input.shape[0]
        loss_all["total_loss"].update(total_loss.item(), N)
        loss_all["commit_loss"].update(commit_loss.item(), N)
        loss_all["recon_loss"].update(recon_loss.item(), N)
    
    # Log codebook stats for each part with correct code_num
    print("\n[TRAIN Codebook Stats]")
    log_codebook_stats_with_nums(all_indices, code_nums, "Train-")
    
    # Compute average perplexity per part
    avg_perplexity = {k: np.mean(v) for k, v in all_perplexity.items()}
    
    return {
        "total_loss": loss_all["total_loss"].avg,
        "recon_loss": loss_all["recon_loss"].avg,
        "commit_loss": loss_all["commit_loss"].avg,
        "perplexity": avg_perplexity,
        "mpjpe": np.mean(all_mpjpe) * 1000,
        "dtw": np.mean(all_dtw),
        "codebook_usage": {k: len(v) for k, v in all_indices.items()},
    }


@torch.no_grad()
def test(config, dataloader, model, code_nums):
    """Test/Validation loop."""
    loss_all = {
        "total_loss": AccumLoss(),
        "commit_loss": AccumLoss(),
        "recon_loss": AccumLoss(),
    }
    
    all_dtw = []
    all_mpjpe = []
    all_perplexity = {}
    all_indices = {}
    
    model.eval()

    for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
        batch = Batch(torch_batch=batch, pad_index=0, model=model)
        
        pose_input = batch.trg_input[:, :, :150]
        pose_mask = batch.trg_mask[..., 0].squeeze().unsqueeze(-1)
        
        pose_output, commit_loss, perplexity_dict, indices_dict = model(pose_input)
        
        # Accumulate indices
        for part_name, indices in indices_dict.items():
            if part_name not in all_indices:
                all_indices[part_name] = Counter()
            all_indices[part_name].update(indices.detach().cpu().view(-1).tolist())
        
        # Accumulate perplexity
        for part_name, ppl in perplexity_dict.items():
            if part_name not in all_perplexity:
                all_perplexity[part_name] = []
            all_perplexity[part_name].append(ppl.cpu().detach().numpy() if torch.is_tensor(ppl) else ppl)
        
        recon_loss = F.l1_loss(pose_output * pose_mask, pose_input * pose_mask)
        total_loss = recon_loss + commit_loss
        
        pose_output_masked = pose_output * pose_mask
        pose_input_masked = pose_input * pose_mask
        
        pred = pose_output_masked.view(pose_output.shape[0], pose_output.shape[1], -1, 3)
        gt = pose_input_masked.view(pose_input.shape[0], pose_input.shape[1], -1, 3)
        joint_error = torch.mean(torch.norm(pred - gt, dim=-1))
        
        dtw_scores = calculate_dtw(pose_input_masked, pose_output_masked.detach())
        
        all_mpjpe.append(joint_error.cpu().detach().numpy())
        all_dtw.extend(dtw_scores)
        
        N = pose_input.shape[0]
        loss_all["total_loss"].update(total_loss.item(), N)
        loss_all["commit_loss"].update(commit_loss.item(), N)
        loss_all["recon_loss"].update(recon_loss.item(), N)
    
    # Log codebook stats with correct code_num for each part
    print("\n[TEST Codebook Stats]")
    log_codebook_stats_with_nums(all_indices, code_nums, "Test-")
    
    avg_perplexity = {k: np.mean(v) for k, v in all_perplexity.items()}
    
    return {
        "total_loss": loss_all["total_loss"].avg,
        "recon_loss": loss_all["recon_loss"].avg,
        "commit_loss": loss_all["commit_loss"].avg,
        "perplexity": avg_perplexity,
        "mpjpe": np.mean(all_mpjpe) * 1000,
        "dtw": np.mean(all_dtw),
        "codebook_usage": {k: len(v) for k, v in all_indices.items()},
    }


def save_model(args, epoch, mpjpe, model, name):
    """Save model checkpoint."""
    save_path = os.path.join(args.checkpoint, f"{name}_{epoch}_{mpjpe:.4f}.pth")
    torch.save(model.state_dict(), save_path)
    return save_path


def format_perplexity(ppl_dict):
    """Format perplexity dict as string."""
    return ", ".join([f"{k}: {v:.2f}" for k, v in ppl_dict.items()])


def format_codebook_usage(usage_dict):
    """Format codebook usage dict as string."""
    return ", ".join([f"{k}: {v}" for k, v in usage_dict.items()])


if __name__ == "__main__":
    seed = 1126

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    train_config = config["training"]
    model_config = config["model"]["qae"]
    batch_size = train_config["batch_size"]
    
    # Codebook size handling for different models
    if args.model == "vqvae_decouple":
        body_code_num = model_config.get("body_code_num", 96)  # SOKE default
        hand_code_num = model_config.get("hand_code_num", 192)  # SOKE default
        code_nums = {
            'body': body_code_num,
            'rhand': hand_code_num,
            'lhand': hand_code_num,
        }
    else:
        code_num = model_config.get("code_num", 512)
        code_nums = {'all': code_num}
    
    # Load data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=config)
    
    train_dataloader = make_data_iter(train_data,
                                      batch_size=batch_size,
                                      batch_type="sentence",
                                      train=True, shuffle=True)
    
    test_dataloader = make_data_iter(dev_data,
                                     batch_size=batch_size,
                                     batch_type="sentence",
                                     train=False, shuffle=False)
    
    lr = float(train_config["learning_rate"])
    
    # Model initialization
    model = VQVAE(**model_config).cuda()
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.model == "vqvae_decouple":
        print(f"Codebook sizes - Body: {code_nums['body']}, Hand: {code_nums['rhand']}")
    else:
        print(f"Codebook size: {code_nums['all']}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    scheduler_args = Namespace(**train_config)
    scheduler, _ = create_scheduler(scheduler_args, optimizer)
    
    # Resume from checkpoint
    if args.previous_dir != "":
        print(f"Loading checkpoint from {args.previous_dir}")
        state_dict = torch.load(args.previous_dir, map_location='cuda')
        model.load_state_dict(state_dict, strict=False)
        
    best_epoch = 0
    epochs = train_config["epochs"]
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{epochs} | Model: {args.model}")
        print(f"{'='*80}")
        
        # Training
        if args.train:
            train_metrics = train(config, train_dataloader, model, optimizer, code_nums)
            scheduler.step(epoch)
        
        # Testing
        with torch.no_grad():
            test_metrics = test(config, test_dataloader, model, code_nums)

        # Check best
        is_best = (test_metrics["mpjpe"] < args.previous_best and 
                   test_metrics["dtw"] < args.previous_best_dtw)
        
        if args.train and is_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, test_metrics["mpjpe"], model, "model")
            args.previous_best = test_metrics["mpjpe"]
            args.previous_best_dtw = test_metrics["dtw"]
        
        # Save videos on best or every 10 epochs
        if args.train and (is_best or epoch % 10 == 0):
            print(f"\nSaving reconstruction videos for epoch {epoch}...")
            save_videos(
                config,
                test_dataloader,
                model,
                epoch,
                args.checkpoint,
                src_vocab,
                num_samples=2,
            )
        
        # Logging
        if args.train:
            train_log = (f"[TRAIN] loss: {train_metrics['total_loss']:.4f} "
                        f"(recon: {train_metrics['recon_loss']:.4f}, "
                        f"commit: {train_metrics['commit_loss']:.4f}) | "
                        f"ppl: {format_perplexity(train_metrics['perplexity'])} | "
                        f"codes: {format_codebook_usage(train_metrics['codebook_usage'])} | "
                        f"mpjpe: {train_metrics['mpjpe']:.4f}, dtw: {train_metrics['dtw']:.4f}")
            
            test_log = (f"[TEST]  loss: {test_metrics['total_loss']:.4f} "
                       f"(recon: {test_metrics['recon_loss']:.4f}, "
                       f"commit: {test_metrics['commit_loss']:.4f}) | "
                       f"ppl: {format_perplexity(test_metrics['perplexity'])} | "
                       f"codes: {format_codebook_usage(test_metrics['codebook_usage'])} | "
                       f"mpjpe: {test_metrics['mpjpe']:.4f}, dtw: {test_metrics['dtw']:.4f}")
            
            best_log = (f"[BEST]  epoch: {best_epoch}, "
                       f"mpjpe: {args.previous_best:.4f}, dtw: {args.previous_best_dtw:.4f}")
            
            logging.info(f"Epoch {epoch}, lr: {lr:.6f}")
            logging.info(train_log)
            logging.info(test_log)
            logging.info(best_log)
            
            print(train_log)
            print(test_log)
            print(best_log)
            
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
            # Test only mode
            print(f"TEST: mpjpe: {test_metrics['mpjpe']:.4f}, dtw: {test_metrics['dtw']:.4f}")
            save_videos(
                config,
                test_dataloader,
                model,
                epoch,
                args.checkpoint,
                src_vocab,
                num_samples=2,
            )
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.checkpoint, "last.pth"))
    print(f"\n{'='*80}")
    print(f"Training finished!")
    print(f"Model: {args.model}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best MPJPE: {args.previous_best:.4f}, Best DTW: {args.previous_best_dtw:.4f}")
    print(f"{'='*80}")