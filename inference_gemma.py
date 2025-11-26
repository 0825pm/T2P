import os
import torch
import numpy as np
import yaml
import argparse
import logging
from tqdm import tqdm
import einops
import types  # 메서드 패치를 위해 추가

# 프로젝트 모듈 임포트
from common.arguments import parse_args
from dataset.data_orthus import load_data, make_data_iter
from dataset.batch import Batch

# [요청하신 파일] model/gemma_nar.py에서 GEMMA 임포트
from model.gemma_nar import GEMMA 

from utils.plot_videos import plot_video, alter_DTW_timing
from common.utils import calculate_dtw

def load_config(path):
    with open(path, 'r') as ymlfile:
        return yaml.safe_load(ymlfile)

# --- [Monkey Patch] 올바른 generate 메서드 정의 ---
@torch.no_grad()
def correct_generate(self, text_input, diffusion_steps=50, target_length=None):
    self.eval()
    device = next(self.parameters()).device
    
    if isinstance(text_input, str):
        text_input = [text_input]
        
    context_vector = self._get_text_context_vector(text_input, device)
    B = context_vector.shape[0]
    
    generated_latents = []
    
    # Autoregressive Loop
    for i in range(self.total_latent_tokens):
        if len(generated_latents) == 0:
            inputs_embeds = context_vector
        else:
            # (B, i, latent_dim=64)
            prev_latents_tensor = torch.stack(generated_latents, dim=1)
            # (B, i, gemma_hidden_size=2048)
            latents_emb = self.latent_token_proj(prev_latents_tensor)
            inputs_embeds = torch.cat([context_vector, latents_emb], dim=1)

        L = inputs_embeds.shape[1]
        attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
        
        # Gemma Forward
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Condition for Diffusion (Last token hidden state)
        z_cond = outputs.hidden_states[-1][:, -1, :].unsqueeze(1) 
        
        # Diffusion Sampling
        # 중요: self.latent_dim (64) 크기로 노이즈 생성
        latent_sample = torch.randn(B, 1, self.latent_dim, device=device)
        self.train_scheduler.set_timesteps(diffusion_steps)
        
        for t in self.train_scheduler.timesteps:
            timesteps_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            model_output = self.diffusion_head(latent_sample, timesteps_tensor, z_cond)
            latent_sample = self.train_scheduler.step(
                model_output, t, latent_sample
            ).prev_sample
            
        # 결과 저장 (B, 64)
        generated_latents.append(latent_sample.squeeze(1))

    # (B, 24, 64)
    final_latents = torch.stack(generated_latents, dim=1)
    
    reshaped_latent = einops.rearrange(
        final_latents, 
        'b (t_qae n_parts) dim -> b dim t_qae n_parts', 
        t_qae=self.num_latent_tokens, 
        n_parts=self.num_latent_parts
    )
    
    if target_length is None:
        target_length = torch.full((B,), 150, device=device, dtype=torch.long)
    
    decoded_pose = self.qae.decode(reshaped_latent, target_length)
    
    return decoded_pose
# -------------------------------------------------

def evaluate(args, config, dataloader, model, src_vocab):
    model.eval()
    
    results_dir = os.path.join(os.path.dirname(args.checkpoint), "inference_results")
    os.makedirs(results_dir, exist_ok=True)
    
    logging.basicConfig(
        format='%(asctime)s %(message)s', 
        datefmt='%Y/%m/%d %H:%M:%S', 
        filename=os.path.join(results_dir, 'inference.log'), 
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    all_dtw = []
    all_mpjpe = []
    
    logging.info(f"Starting inference on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            
            batch = Batch(torch_batch=batch_data,
                          pad_index=0,
                          model=model)
            
            text_input = batch.text
            
            # GT Pose
            gt_pose_flat = batch.trg_input[:, :, :150]
            gt_pose = einops.rearrange(gt_pose_flat, "b f (n c) -> b f n c", c=3)
            
            pose_length = batch.trg_mask[...,0].sum(dim=-1).ravel()
            
            # Generate Pose (Patched generate call)
            generated_pose = model.generate(text_input, diffusion_steps=args.steps, target_length=pose_length)
            
            # Masking
            pose_mask = batch.trg_mask[...,0].squeeze().unsqueeze(-1).unsqueeze(-1)
            generated_pose = generated_pose.to(torch.float32) * pose_mask
            
            # Metrics
            joint_error = torch.mean(torch.norm(generated_pose - gt_pose, dim=-1), dim=[-1, -2])
            all_mpjpe.extend(joint_error.cpu().numpy().tolist())
            
            dtw_scores = calculate_dtw(gt_pose, generated_pose.cpu().detach())
            all_dtw.extend(dtw_scores)
            
            # Visualization (First 5 batches)
            if batch_idx < 5: 
                num_samples = min(1, generated_pose.shape[0])
                
                for i in range(num_samples):
                    gt_len_i = pose_length[i].item()
                    
                    gt_pose_np_i = gt_pose[i, :gt_len_i].reshape(-1, 150)
                    pred_pose_np_i = generated_pose[i, :gt_len_i].reshape(-1, 150)
                    
                    counter = batch.trg_input[i, :gt_len_i, 150:]
                    gt_with_counter = torch.cat((gt_pose_np_i, counter), dim=-1)
                    pred_with_counter = torch.cat((pred_pose_np_i, counter), dim=-1)
                    
                    try:
                        timing_hyp_seq, ref_seq_count, _ = alter_DTW_timing(pred_with_counter, gt_with_counter)
                        
                        text_str = text_input[i]
                        filename_str = text_str.replace(" ", "_").replace("/", "-")[:50]
                        video_name = f"batch{batch_idx}_{i:02d}_{filename_str}.mp4"
                        
                        plot_video(
                            joints=timing_hyp_seq,
                            file_path=results_dir,
                            video_name=video_name,
                            references=ref_seq_count,
                            skip_frames=1,
                            sequence_ID=text_str
                        )
                    except Exception as e:
                        logging.error(f"Video plot failed: {e}")

    avg_mpjpe = np.mean(all_mpjpe) * 1000
    avg_dtw = np.mean(all_dtw)
    
    logging.info(f"Inference Finished. MPJPE: {avg_mpjpe:.4f}, DTW: {avg_dtw:.4f}")
    print(f"MPJPE: {avg_mpjpe:.4f}, DTW: {avg_dtw:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/t2p_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--steps', type=int, default=50, help='Diffusion steps')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(cfg=config)
    
    test_dataloader = make_data_iter(test_data,
                                     batch_size=config["training"]["batch_size"],
                                     batch_type="sentence",
                                     train=False, 
                                     shuffle=False)
    
    print("Loading model (gemma_nar)...")
    model = GEMMA(config["model"]).to(device)
    
    # [Patch] 모델 인스턴스의 generate 메서드를 올바른 함수로 교체
    model.generate = types.MethodType(correct_generate, model)
    
    # Load Checkpoint
    if os.path.isdir(args.checkpoint):
        import glob
        pth_files = sorted(glob.glob(os.path.join(args.checkpoint, "*.pth")), key=os.path.getmtime)
        if pth_files:
            ckpt_path = pth_files[-1]
        else:
            raise FileNotFoundError(f"No .pth found in {args.checkpoint}")
    else:
        ckpt_path = args.checkpoint
        
    print(f"Loading weights from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    evaluate(args, config, test_dataloader, model, src_vocab)