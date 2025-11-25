import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import time
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from argparse import Namespace
from einops import rearrange

# 필요한 모듈 임포트 (기존 코드 경로 가정)
from common.arguments import parse_args
from common.utils import AccumLoss, calculate_dtw, save_model, Load_model
from dataset.data_gan import load_data, make_data_iter
from dataset.batch import Batch
from utils.plot_videos import plot_video, alter_DTW_timing

# [NEW] 통합된 모델 파일에서 Generator와 Discriminator 임포트
from model.t5_nar_gan import T5_NAR, MotionDiscriminator
from torch.optim.lr_scheduler import CosineAnnealingLR

def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

@torch.no_grad()
def save_videos(dataloader, model, epoch, checkpoint_dir, src_vocab, num_samples=1):
    model.eval()
    try: batch = next(iter(dataloader))
    except StopIteration: return

    results_dir = os.path.join(checkpoint_dir, f"videos/epoch_{epoch}")
    os.makedirs(results_dir, exist_ok=True)
    
    batch = Batch(torch_batch=batch, pad_index=0, model=model)
    text_input = batch.text
    
    # NAR 생성 (Generate)
    pose_output = model.generate(text_input) # (B, T, 50, 3)
    
    # GT 준비
    pose_input = batch.trg_input[:, :, :150]
    pose_input = rearrange(pose_input, "b f (n c) -> b f n c", c=3)
    pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
    
    num_samples = min(num_samples, pose_input.shape[0])
    
    for i in range(num_samples):
        gt_len = pose_length[i].item()
        # 시각화를 위해 (T, 150) 형태로 변환 + Counter 정보 붙이기
        pred_np = pose_output[i, :gt_len].reshape(-1, 150).cpu().numpy()
        gt_np = pose_input[i, :gt_len].reshape(-1, 150).cpu().numpy()
        
        # 카운터 정보가 없으므로 더미로 붙이거나 생략 (plot_video 구현에 따라 조정 필요)
        # 여기서는 원본 배치에 있는 카운터를 빌려옴
        counter = batch.trg_input[i, :gt_len, 150:].cpu().numpy()
        pred_w_count = np.concatenate((pred_np, counter), axis=-1)
        gt_w_count = np.concatenate((gt_np, counter), axis=-1)
        
        timing_hyp, ref_seq, _ = alter_DTW_timing(torch.tensor(pred_w_count), torch.tensor(gt_w_count))
        
        filename = text_input[i].replace("/", "_")[:30]
        plot_video(joints=timing_hyp, file_path=results_dir, video_name=f"{i}_{filename}.mp4",
                   references=ref_seq, skip_frames=1, sequence_ID=filename)

def train_gan_epoch(dataloader, generator, discriminator, opt_G, opt_D, device, adv_weight=0.1):
    # ... (위의 train_gan.py 코드와 동일, 생략) ...
    # 단, tqdm desc에 loss가 잘 떨어지는지 확인하기 위해 postfix 추가 권장
    loss_log = {"g_total": AccumLoss(), "g_recon": AccumLoss(), "g_adv": AccumLoss(), "d_total": AccumLoss()}
    generator.train()
    discriminator.train()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_data in pbar:
        batch = Batch(torch_batch=batch_data, pad_index=0, model=generator)
        pose_gt = batch.trg_input[:, :, :150]
        pose_gt_flat = pose_gt.to(device)
        pose_input_3d = rearrange(pose_gt, "b f (n c) -> b f n c", c=3).to(device)
        pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
        text_input = batch.text

        # --- Discriminator Update ---
        opt_D.zero_grad()
        real_score = discriminator(pose_gt_flat)
        d_loss_real = torch.mean((real_score - 1) ** 2)
        
        with torch.no_grad():
            fake_pose_3d, _ = generator(pose_input_3d, text_input, pose_length)
            fake_pose_flat = rearrange(fake_pose_3d, "b f n c -> b f (n c)")
        
        fake_score = discriminator(fake_pose_flat.detach())
        d_loss_fake = torch.mean(fake_score ** 2)
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        opt_D.step()
        
        # --- Generator Update ---
        opt_G.zero_grad()
        fake_pose_3d, recon_loss = generator(pose_input_3d, text_input, pose_length)
        fake_pose_flat = rearrange(fake_pose_3d, "b f n c -> b f (n c)")
        g_fake_score = discriminator(fake_pose_flat)
        g_adv_loss = torch.mean((g_fake_score - 1) ** 2)
        
        total_g_loss = recon_loss + (adv_weight * g_adv_loss)
        total_g_loss.backward()
        opt_G.step()
        
        # Logging
        loss_log["g_total"].update(total_g_loss.item())
        loss_log["g_recon"].update(recon_loss.item())
        loss_log["g_adv"].update(g_adv_loss.item())
        loss_log["d_total"].update(d_loss.item())
        
        # 실시간 Loss 확인
        pbar.set_postfix({
            "G_Recon": f"{recon_loss.item():.4f}",
            "G_Adv": f"{g_adv_loss.item():.4f}"
        })
        
    return loss_log

@torch.no_grad()
def evaluate(dataloader, generator, discriminator, device):
    generator.eval()
    loss_log = {"g_recon": AccumLoss(), "mpjpe": [], "dtw": []}
    for batch_data in tqdm(dataloader, desc="Evaluating"):
        batch = Batch(torch_batch=batch_data, pad_index=0, model=generator)
        pose_gt = batch.trg_input[:, :, :150]
        pose_input_3d = rearrange(pose_gt, "b f (n c) -> b f n c", c=3).to(device)
        pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
        
        pred_pose_3d, recon_loss = generator(pose_input_3d, batch.text, pose_length)
        
        # Metric
        pose_mask = batch.trg_mask[...,0].squeeze().unsqueeze(-1).unsqueeze(-1)
        pred_pose_3d = pred_pose_3d * pose_mask
        
        mpjpe = torch.mean(torch.norm(pred_pose_3d - pose_input_3d, dim=-1))
        dtw_score = calculate_dtw(pose_input_3d, pred_pose_3d.cpu())
        
        loss_log["g_recon"].update(recon_loss.item())
        loss_log["mpjpe"].append(mpjpe.item())
        loss_log["dtw"].extend(dtw_score)
        
    avg_mpjpe = np.mean(loss_log["mpjpe"]) * 1000
    avg_dtw = np.mean(loss_log["dtw"])
    return loss_log["g_recon"].avg, avg_mpjpe, avg_dtw

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed = 1126
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Data Load
    print("Loading Data...")
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=config)
    train_loader = make_data_iter(train_data, batch_size=config["training"]["batch_size"], train=True, shuffle=True)
    test_loader = make_data_iter(dev_data, batch_size=config["training"]["batch_size"], train=False, shuffle=False)
    
    # 2. Model Init
    print("Initializing Models...")
    generator = T5_NAR(config["model"]).to(device)
    discriminator = MotionDiscriminator(input_size=150).to(device)
    
    # [핵심 수정] QAE Pre-trained Weights 로드
    # args.previous_dir에 QAE 학습 결과 폴더 경로가 있어야 함
    if args.previous_dir:
        print(f"Loading QAE weights from {args.previous_dir}...")
        # Load_model 함수는 내부적으로 args.previous_dir 경로의 .pth 파일을 찾아 로드함
        # QAE는 Frozen 상태이므로 처음에 제대로 로드하지 않으면 학습이 불가능함
        class DummyArgs: pass
        dummy = DummyArgs()
        dummy.previous_dir = args.previous_dir
        Load_model(dummy, generator.qae)
    else:
        print("\n[WARNING] 'previous_dir' not specified! QAE is initialized randomly.")
        print("Training will likely FAIL because QAE is frozen and random.")
        print("Please provide --previous_dir to load pre-trained QAE weights.\n")
    
    # 3. Optimizer
    lr = float(config["training"]["learning_rate"])
    g_params = filter(lambda p: p.requires_grad, generator.parameters())
    optimizer_G = optim.AdamW(g_params, lr=lr, weight_decay=0.01)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=0.01)
    
    # 4. Loop
    epochs = config["training"]["epochs"]
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=1e-6)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=1e-6)
    
    best_mpjpe = float('inf')
    
    print("Start Training...")
    for epoch in range(1, epochs + 1):
        # Warm-up 설정 (기존 유지)
        adv_weight = 0.01 if epoch > 5 else 0.0
        
        print(f"\nEpoch {epoch}/{epochs} | Adv Weight: {adv_weight}")
        
        # 현재 LR 확인 및 출력
        current_lr_G = scheduler_G.get_last_lr()[0]
        print(f"Current LR: {current_lr_G:.8f}")

        # Train Step
        train_losses = train_gan_epoch(
            train_loader, generator, discriminator, optimizer_G, optimizer_D, device, adv_weight
        )
        
        # [중요] 스케줄러 업데이트 (에포크가 끝날 때마다 호출)
        scheduler_G.step()
        scheduler_D.step()
        
        print(f"TRAIN | G_Total: {train_losses['g_total'].avg:.4f} (Recon: {train_losses['g_recon'].avg:.4f}) | D: {train_losses['d_total'].avg:.4f}")
        
        recon_val, mpjpe_val, dtw_val = evaluate(test_loader, generator, discriminator, device)
        print(f"VALID | MPJPE: {mpjpe_val:.2f}mm | DTW: {dtw_val:.4f}")
        
        if mpjpe_val < best_mpjpe:
            best_mpjpe = mpjpe_val
            save_path = os.path.join(args.checkpoint, "best_gan_model.pth")
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'config': config
            }, save_path)
            print(f"Saved best model (MPJPE: {best_mpjpe:.2f})")
            save_videos(test_loader, generator, epoch, args.checkpoint, src_vocab)