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
from dataset.data_qae import load_data, make_data_iter
from dataset.batch import Batch
from torchtext.data import Dataset

# torch.autograd.set_detect_anomaly(True)
exec("from model." + args.model + " import QAE")

@torch.no_grad()
def save_videos(config, dataloader, model, epoch, checkpoint_dir, src_vocab, num_samples=2):
    model.eval()
    try: batch = next(iter(dataloader))
    except StopIteration: print("Could not get a batch from dataloader."); return

    results_dir = checkpoint_dir + f"/videos/epoch_{epoch}" # [수정] 경로
    os.makedirs(results_dir, exist_ok=True)
    
    batch = Batch(torch_batch=batch,
                              pad_index=0,
                              model=model)
        
    pose_input = batch.trg_input[:, :, :150]
    pose_input = rearrange(pose_input, "b f (n c) -> b f n c", c=3)
    pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
    pose_mask = batch.trg_mask[...,0].squeeze().unsqueeze(-1).unsqueeze(-1)
    
    text_input = [" ".join([src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i])-1)]) for i in range(len(batch.src))]
    
    num_samples = min(num_samples, pose_input.shape[0])
    print(f"Saving {num_samples} seq2seq sampling videos...")
    
    pose_output, _, _ = model(pose_input, text_input, pose_length)
    
    for i in range(num_samples):
        gt_len_i = pose_length[i].item()
        gt_pose_np_i = pose_input[i, :gt_len_i].reshape(-1, 150)
        pred_pose_np_i = pose_output[i, :gt_len_i].reshape(-1, 150)
        gt_pose_np_i = torch.cat((gt_pose_np_i, batch.trg_input[i, :gt_len_i, 150:]), dim=-1)
        pred_pose_np_i = torch.cat((pred_pose_np_i, batch.trg_input[i, :gt_len_i, 150:]), dim=-1)
        
        timing_hyp_seq, ref_seq_count, dtw_score = alter_DTW_timing(pred_pose_np_i, gt_pose_np_i)
        
        text_str = text_input[i]
        filename = text_str.split("</s>")[0].rstrip()[:50]
        
        plot_video(joints=timing_hyp_seq,
                    file_path=results_dir,
                    video_name=f"{i:02d}_{filename}.mp4",
                    references=ref_seq_count,
                    skip_frames=1,
                    sequence_ID=filename)
            

def create_mask(seq_lengths, device="cpu"):
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.bool()

def train(config, dataloader, model, src_vocab, optimizer, clip_grad_fun):
    
    loss_all = {"total_loss": AccumLoss(),
                "recon_loss": AccumLoss(),
                }
    
    all_dtw_pose = list()
    all_mpjpe_pose = list()
    model.train()

    for i, batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        
        batch = Batch(torch_batch=batch,
                              pad_index=0,
                              model=model)
        
        pose_input = batch.trg_input[:, :, :150]
        pose_input = rearrange(pose_input, "b f (n c) -> b f n c", c=3)
        pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
        pose_mask = batch.trg_mask[...,0].squeeze().unsqueeze(-1).unsqueeze(-1)
        
        text_input = [" ".join([src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i])-1)]) for i in range(len(batch.src))]
        
        pose_output, recon_loss, _ = model(pose_input, text_input, pose_length)
        total_loss = recon_loss
        total_loss.backward()
        
        if clip_grad_fun is not None:
            # clip gradients (in-place)
            clip_grad_fun(params=model.parameters())
        
        optimizer.step()

        
        pose_output = pose_output.to(torch.float32) * pose_mask
        joint_error_pose = torch.mean(torch.norm(pose_output - pose_input, dim=len(pose_input.shape)-1))
        dtw_score_pose = calculate_dtw(pose_input, pose_output.cpu().detach())
        
        all_mpjpe_pose.append(np.mean(joint_error_pose.cpu().detach().numpy()))
        all_dtw_pose.extend(dtw_score_pose)
        
        N = pose_input.shape[0]
        loss_all["total_loss"].update(total_loss.detach().cpu().numpy() * N, N)
        loss_all["recon_loss"].update(recon_loss.detach().cpu().numpy() * N, N)
        
    return loss_all["total_loss"].avg, \
            loss_all["recon_loss"].avg, \
            np.mean(np.array(all_mpjpe_pose)) * 1000, np.mean(all_dtw_pose)

@torch.no_grad()
def test(config, dataloader, model, src_vocab):
    
    loss_all = {"total_loss": AccumLoss(),
                "recon_loss": AccumLoss(),
                }
    
    all_dtw_pose = list()
    all_mpjpe_pose = list()
    model.eval()

    for i, batch in enumerate(tqdm(dataloader)):
        
        batch = Batch(torch_batch=batch,
                              pad_index=0,
                              model=model)
        
        pose_input = batch.trg_input[:, :, :150]
        pose_input = rearrange(pose_input, "b f (n c) -> b f n c", c=3)
        pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
        pose_mask = batch.trg_mask[...,0].squeeze().unsqueeze(-1).unsqueeze(-1)
        
        text_input = [" ".join([src_vocab.itos[batch.src[i][j]] for j in range(len(batch.src[i])-1)]) for i in range(len(batch.src))]
        
        pose_output, recon_loss, _ = model(pose_input, text_input, pose_length)
        total_loss = recon_loss
        
        pose_output = pose_output.to(torch.float32) * pose_mask
        joint_error_pose = torch.mean(torch.norm(pose_output - pose_input, dim=len(pose_input.shape)-1))
        dtw_score_pose = calculate_dtw(pose_input, pose_output.cpu().detach())
        
        all_mpjpe_pose.append(np.mean(joint_error_pose.cpu().detach().numpy()))
        all_dtw_pose.extend(dtw_score_pose)
        
        N = pose_input.shape[0]
        loss_all["total_loss"].update(total_loss.detach().cpu().numpy() * N, N)
        loss_all["recon_loss"].update(recon_loss.detach().cpu().numpy() * N, N)
        
    return loss_all["total_loss"].avg, \
            loss_all["recon_loss"].avg, \
            np.mean(np.array(all_mpjpe_pose)) * 1000, np.mean(all_dtw_pose)
            
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
    
    model = QAE(model_config).cuda()
    
    # clip_grad_fun = build_gradient_clipper(config=train_config)
    # optimizer = build_optimizer(config=train_config, parameters=model.parameters())
    clip_grad_fun = None
 
    minimize_metric = True
    # scheduler, scheduler_step_at = build_scheduler(
    #         config=train_config,
    #         scheduler_mode="min" if minimize_metric else "max",
    #         optimizer=optimizer,
    #         hidden_size=model_config["hidden_size"])
    
    
    param_groups = [
        {"params": model.parameters(), "lr": lr, "weight_decay": 0.01},]
    
    optimizer = optim.AdamW([{'params' : model.parameters()},
                             ],
                            lr=lr, weight_decay=0.01)
    scheduler_args = Namespace(**config['training'])
    scheduler, _ = create_scheduler(scheduler_args, optimizer)
    
    # loss_scaler = NativeScaler()
    
    if args.previous_dir != "":
        Load_model(args, [model], ["model"])
        
    best_epoch = 0
    epoch = train_config["epochs"]
    loss_epochs = []
    mpjpes = []
    for epoch in range(1, epoch + 1):
        # with torch.no_grad():
        #     total_loss_test, recon_loss_test, kl_loss_test, contra_loss_test, len_loss_test, latent_loss_test, mpjpe_pose_test, dtw_pose_test, mpjpe_text_test, dtw_text_test, test_idx = test(config, test_dataloader, model, epoch)
        
        if args.train: 
            total_loss_train, recon_loss_train, mpjpe_train, dtw_train = train(config, train_dataloader, model, src_vocab, optimizer, clip_grad_fun)
            loss_epochs.append(total_loss_train * 1000)
            scheduler.step(epoch)
        with torch.no_grad():
            total_loss_test, recon_loss_test, mpjpe_test, dtw_test = test(config, test_dataloader, model, src_vocab)

        is_best = mpjpe_test < args.previous_best and dtw_test < args.previous_best_dtw
        if args.train and is_best:
            best_epoch = epoch
            args.previous_name = save_model(args, epoch, mpjpe_test, model, "model")
            
            args.previous_best = mpjpe_test
            args.previous_best_dtw = dtw_test
            
        if args.train and ((epoch % 10 == 0) or is_best):
            print(f"\nSaving reconstruction videos for epoch {epoch}...")
            save_videos(
                config, 
                test_dataloader,
                model, 
                epoch, 
                args.checkpoint,
                src_vocab,
                num_samples=1,
            )

        if args.train:
            logging.info("epoch: %d, lr: %.6f, TRAIN : total: %.4f, recon: %.4f, mpjpe: %.4f, dtw: %.4f, %d: %.4f, %.4f" % (epoch, lr, total_loss_train, recon_loss_train, mpjpe_train, dtw_train, best_epoch, args.previous_best, args.previous_best_dtw))
            logging.info("epoch: %d, lr: %.6f, TEST : total: %.4f, recon: %.4f, mpjpe: %.4f, dtw: %.4f, %d: %.4f, %.4f" % (epoch, lr, total_loss_test, recon_loss_test, mpjpe_test, dtw_test, best_epoch, args.previous_best, args.previous_best_dtw))

            print("epoch: %d, lr: %.6f, TRAIN : total: %.4f, recon: %.4f, mpjpe: %.4f, dtw: %.4f, %d: %.4f, %.4f" % (epoch, lr, total_loss_train, recon_loss_train, mpjpe_train, dtw_train, best_epoch, args.previous_best, args.previous_best_dtw))
            print("epoch: %d, lr: %.6f, TEST : total: %.4f, recon: %.4f, mpjpe: %.4f, dtw: %.4f, %d: %.4f, %.4f" % (epoch, lr, total_loss_test, recon_loss_test, mpjpe_test, dtw_test, best_epoch, args.previous_best, args.previous_best_dtw))
        
            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= args.lr_decay
        else:
            print("mpjpe_score: %.4f, dtw_score: %.4f" % (mpjpe_test, dtw_test))
            save_videos(
                config, 
                test_dataloader,
                model, 
                epoch, 
                args.checkpoint,
                src_vocab,
                num_samples=1,
            )
            break
    
    torch.save(model.state_dict(), "%s/last.pth" % (args.checkpoint))
