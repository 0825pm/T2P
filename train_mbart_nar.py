import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import torch
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from einops import rearrange
import logging
from argparse import Namespace

from common.arguments import parse_args
from common.utils import *
from dataset.data_orthus import load_data, make_data_iter
from dataset.batch import Batch
from utils.plot_videos import plot_video, alter_DTW_timing
from timm.scheduler import create_scheduler

# [중요] 위에서 작성한 모델 임포트
from model.mbart_nar import MBartPoseNARGenerator

args = parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def train(dataloader, model, optimizer):
    loss_all = AccumLoss()
    model.train()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        
        batch = Batch(torch_batch=batch, pad_index=0, model=model)
        
        pose_input = batch.trg_input[:, :, :150]
        pose_input = rearrange(pose_input, "b f (n c) -> b f n c", c=3)
        pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
        text_input = batch.text
        
        # Forward (Parallel NAR Prediction)
        _, _, recon_loss = model(pose_input, text_input, pose_length)
        
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        N = pose_input.shape[0]
        loss_all.update(recon_loss.item() * N, N)
        
    return loss_all.avg

@torch.no_grad()
def test(dataloader, model):
    loss_all = AccumLoss()
    all_mpjpe = []
    model.eval()

    for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
        batch = Batch(torch_batch=batch, pad_index=0, model=model)
        
        pose_input = batch.trg_input[:, :, :150]
        pose_input = rearrange(pose_input, "b f (n c) -> b f n c", c=3)
        pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
        pose_mask = batch.trg_mask[...,0].squeeze().unsqueeze(-1).unsqueeze(-1)
        text_input = batch.text
        
        pose_output, recon_loss, _ = model(pose_input, text_input, pose_length)
        
        # Metric Calculation
        pose_output = pose_output.to(torch.float32) * pose_mask
        joint_error = torch.mean(torch.norm(pose_output - pose_input, dim=-1))
        
        all_mpjpe.append(joint_error.item())
        N = pose_input.shape[0]
        loss_all.update(recon_loss.item() * N, N)
        
    return loss_all.avg, np.mean(all_mpjpe) * 1000

@torch.no_grad()
def save_sample_videos(dataloader, model, epoch, checkpoint_dir):
    model.eval()
    try: batch = next(iter(dataloader))
    except StopIteration: return

    results_dir = os.path.join(checkpoint_dir, f"videos/epoch_{epoch}")
    os.makedirs(results_dir, exist_ok=True)
    
    batch = Batch(torch_batch=batch, pad_index=0, model=model)
    pose_input = rearrange(batch.trg_input[:, :, :150], "b f (n c) -> b f n c", c=3)
    text_input = batch.text
    
    # Generate (One-shot NAR)
    generated_pose = model.generate(text_input)
    
    num_samples = min(2, pose_input.shape[0])
    for i in range(num_samples):
        gt_pose = pose_input[i].cpu().numpy()
        pred_pose = generated_pose[i].cpu().numpy()
        
        # Add dummy channel for visualization tool compatibility
        gt_pose = np.concatenate([gt_pose.reshape(-1, 150), np.zeros((gt_pose.shape[0], 1))], axis=-1)
        pred_pose = np.concatenate([pred_pose.reshape(-1, 150), np.zeros((pred_pose.shape[0], 1))], axis=-1)
        
        try:
            timing_hyp_seq, _, _ = alter_DTW_timing(pred_pose, gt_pose)
            filename = text_input[i].replace(" ", "_")[:30]
            plot_video(
                joints=timing_hyp_seq,
                file_path=results_dir,
                video_name=f"{i}_{filename}.mp4",
                references=None,
                sequence_ID=filename
            )
        except Exception as e:
            print(f"Video error: {e}")

if __name__ == "__main__":
    seed = 1126
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=config)
    train_loader = make_data_iter(train_data, batch_size=config["training"]["batch_size"], train=True)
    test_loader = make_data_iter(dev_data, batch_size=config["training"]["batch_size"], train=False)
    
    model = MBartPoseNARGenerator(config["model"]).cuda()
    
    # Load Pre-trained QAE
    if args.previous_dir:
        print(f"Loading QAE weights from {args.previous_dir}")
        Load_model(args, model.qae)
    
    # Optimizer: QAE는 freeze되어 있으므로, mBART와 Query 파라미터 등만 학습
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(config["training"]["learning_rate"]))
    scheduler, _ = create_scheduler(Namespace(**config['training']), optimizer)
    
    best_mpjpe = float('inf')
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train(train_loader, model, optimizer)
        scheduler.step(epoch)
        test_loss, mpjpe = test(test_loader, model)
        
        print(f"Epoch {epoch}: Train {train_loss:.4f}, Test {test_loss:.4f}, MPJPE {mpjpe:.2f}")
        
        if mpjpe < best_mpjpe:
            best_mpjpe = mpjpe
            save_model(args, epoch, mpjpe, model, "mbart_nar_best")
            
        if epoch % 10 == 0:
            save_sample_videos(test_loader, model, epoch, args.checkpoint)