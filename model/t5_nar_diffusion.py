import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import einops

from model.qae import QAE 
from common.loss import Loss
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from diffusers import DDPMScheduler

class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, model_channels)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, 4 * model_channels),
            nn.SiLU(),
            nn.Linear(4 * model_channels, model_channels),
        )
        self.context_proj = nn.Linear(z_channels, model_channels)
        
        self.block = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.GELU(),
            nn.Linear(model_channels * 4, model_channels)
        )
        
        self.proj_out = nn.Linear(model_channels, out_channels)
        
    def forward(self, x, timesteps, z):
        h = self.norm(x)
        h = self.proj_in(h)
        
        timesteps = timesteps.unsqueeze(-1).to(h.dtype) / 1000.0
        time_emb = self.time_embed(timesteps)
        context_emb = self.context_proj(z)

        c = time_emb + context_emb
        h = h + self.block(h) + c
        
        out = self.proj_out(h)
        return out

class GEMMA(nn.Module): 
    def __init__(self, config):
        super(GEMMA, self).__init__()
        
        self.use_cuda = True
        gemma_model_name = config["text_encoder"]["model_name"]
        self.latent_dim = config["qae"]['hidden_size']
        self.gemma_hidden_size = config["text_encoder"]['hidden_size']
        self.total_latent_tokens = 24
        self.num_latent_parts = 3
        self.num_latent_tokens = 8
        self.diffusion_batch_mul = 8
        
        self.loss = Loss(cfg=config["qae"], target_pad=0.0)
        # --- [수정] QAE 설정 변경 ---
        self.qae = QAE(config=config["qae"])
        
        # 1. 일단 전체 Freeze (Encoder 및 기본 설정)
        for param in self.qae.parameters():
            param.requires_grad = False
        self.qae.eval() # 전체를 eval 모드로 둠 (Encoder 등은 Frozen 상태 유지)
        
        # 2. Decoder 부분만 Unfreeze 및 Train 모드 전환
        # 학습시킬 Decoder 모듈 리스트
        decoder_modules = [
            self.qae.dec_spa_vit,
            self.qae.dec_tem_vit,
            self.qae.dec_ca,
            self.qae.pose_spl,
            self.qae.hand_spl
        ]
        
        print("Unfreezing QAE Decoder components...")
        for module in decoder_modules:
            module.train() # Dropout 등이 작동하도록 train 모드로 변경
            for param in module.parameters():
                param.requires_grad = True
        
        # dec_token은 nn.Parameter이므로 따로 처리
        self.qae.dec_token.requires_grad = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
        self.text_model = T5ForConditionalGeneration.from_pretrained(gemma_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.text_model.resize_token_embeddings(len(self.tokenizer))
            
        # [수정 포인트 1] TaskType을 SEQ_2_SEQ_LM으로 설정
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q", "k", "v", "o"], 
            lora_dropout=0.15, bias="none", 
            task_type=TaskType.SEQ_2_SEQ_LM 
        )
        self.text_model = get_peft_model(self.text_model, lora_config)
        
        self.latent_token_proj = nn.Linear(self.latent_dim, self.gemma_hidden_size)
        self.text_context_proj = nn.Linear(self.gemma_hidden_size, self.gemma_hidden_size)
        
        self.diffusion_head = SimpleMLPAdaLN(
            in_channels=self.latent_dim,
            model_channels=self.gemma_hidden_size,
            out_channels=self.latent_dim,
            z_channels=self.gemma_hidden_size,
            num_res_blocks=3
        )

        self.train_scheduler = DDPMScheduler(
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type="v_prediction",
            steps_offset=1,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True
        )

        self.latent_pred_head = nn.Linear(self.gemma_hidden_size, self.latent_dim)
        self.aux_loss_fn = nn.MSELoss()
        
        # Learnable Query Tokens (Decoder Input)
        self.query_tokens = nn.Parameter(torch.randn(1, self.total_latent_tokens, self.gemma_hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, pose_input, text_input, pose_length, cond_input=None):
        B, T, N, C = pose_input.shape
        device = pose_input.device

        # 1. Text Tokenization (Encoder Input 생성)
        tokenized_inputs = self.tokenizer(
            text_input, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        input_ids = tokenized_inputs.input_ids
        attention_mask = tokenized_inputs.attention_mask

        # 2. Decoder Input (Learnable Queries)
        decoder_inputs_embeds = self.query_tokens.repeat(B, 1, 1)
        
        # 3. T5 Forward (Seq2Seq)
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True, # 히든 스테이트 반환 요청
            return_dict=True
        )

        # [수정] Seq2SeqLMOutput에서 디코더의 마지막 히든 스테이트 추출
        # outputs.last_hidden_state 대신 decoder_hidden_states 튜플의 마지막 요소 사용
        z_cond = outputs.decoder_hidden_states[-1]
        
        # --- Diffusion Process ---
        body_feat, rhand_feat, lhand_feat = self.qae.encode_pose(pose_input, pose_length)
        qae_feat = self.qae.qformer(body_feat, rhand_feat, lhand_feat)
        z_target = rearrange(qae_feat, 'b h t j -> b (t j) h') 
        
        L_AR_Full = self.total_latent_tokens 
        
        x_0_flat = z_target.reshape(B * L_AR_Full, -1)
        z_cond_flat = z_cond.reshape(B * L_AR_Full, -1)
        
        x_0_repeated = x_0_flat.repeat(self.diffusion_batch_mul, 1)
        z_cond_repeated = z_cond_flat.repeat(self.diffusion_batch_mul, 1)
        
        timesteps = torch.randint(0, 1000, (x_0_repeated.shape[0],), dtype=torch.int64, device=device)
        noise = torch.randn_like(x_0_repeated, device=device)
        
        noisy_latents = self.train_scheduler.add_noise(x_0_repeated, noise, timesteps)
        target = self.train_scheduler.get_velocity(x_0_repeated, noise, timesteps)
        
        model_pred = self.diffusion_head(noisy_latents, timesteps, z_cond_repeated)
        diff_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Reconstruction Path
        latent_pred = self.latent_pred_head(z_cond)
        reshaped_latent = rearrange(latent_pred, 'b (t_qae n_parts) dim -> b dim t_qae n_parts', 
                                    t_qae=self.num_latent_tokens, n_parts=self.num_latent_parts) 
        
        pose_decoded = self.qae.decode(reshaped_latent, pose_length)
        pose_input_flat = einops.rearrange(pose_input, "b f n c -> b f (n c)")
        pose_output_flat = einops.rearrange(pose_decoded, "b f n c -> b f (n c)")
        recon_loss = self.loss(pose_output_flat, pose_input_flat)

        return pose_decoded, recon_loss, diff_loss
    
    @torch.no_grad()
    def generate(self, text_input, diffusion_steps=50, target_length=None):
        self.eval()
        device = next(self.parameters()).device
        
        if isinstance(text_input, str):
            text_input = [text_input]
            
        B = len(text_input)
        
        # 1. Text Tokenization (Encoder Input)
        tokenized_inputs = self.tokenizer(
            text_input, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        # 2. Decoder Input (Learnable Queries)
        # (B, 24, H) 크기의 쿼리를 한 번에 생성
        decoder_inputs_embeds = self.query_tokens.repeat(B, 1, 1)
        
        # 3. T5 Forward (Generate Condition - Parallel)
        # 루프 없이 한 번의 Forward로 전체 시퀀스(24개)에 대한 조건 벡터를 얻습니다.
        outputs = self.text_model(
            input_ids=tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            return_dict=True
        )
        
        # z_cond_seq: (B, 24, H) -> 전체 시퀀스 조건 벡터
        z_cond_seq = outputs.decoder_hidden_states[-1]
        
        # 4. Diffusion Sampling (Parallel)
        # 기존에는 토큰 별로 루프를 돌렸으나, 이제 전체 시퀀스(24개)를 한 번에 노이즈 제거합니다.
        
        # (B, 24, Latent_Dim) 크기의 노이즈 생성
        latent_sample = torch.randn(B, self.total_latent_tokens, self.latent_dim, device=device)
        
        self.train_scheduler.set_timesteps(diffusion_steps)
        
        for t in self.train_scheduler.timesteps:
            # Timesteps는 Batch 차원에 맞춰져야 하므로 (B,) 크기로 생성
            timesteps_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Diffusion Head에 (B, 24, Dim) 전체를 입력
            # SimpleMLPAdaLN은 (Batch, Sequence, Dim) 입력을 처리할 수 있도록 설계되어 있어야 함
            # 보통 Linear/LayerNorm은 마지막 차원만 맞으면 되므로 (B, 24, Dim)도 처리 가능
            model_output = self.diffusion_head(latent_sample, timesteps_tensor, z_cond_seq)
            
            latent_sample = self.train_scheduler.step(
                model_output, t, latent_sample
            ).prev_sample
            
        # Loop가 끝난 후 latent_sample은 Denoising이 완료된 (B, 24, Latent_Dim) 상태
        final_latents = latent_sample 

        # 5. Reshape & Decode
        # (B, 24, 64) -> (B, 64, 8, 3) 
        # 24 tokens = 8 (time) * 3 (parts: body, rhand, lhand)
        reshaped_latent = rearrange(
            final_latents, 
            'b (t_qae n_parts) dim -> b dim t_qae n_parts', 
            t_qae=self.num_latent_tokens, 
            n_parts=self.num_latent_parts
        )
        
        if target_length is None:
            target_length = torch.full((B,), 150, device=device, dtype=torch.long)
        
        decoded_pose = self.qae.decode(reshaped_latent, target_length)
        
        return decoded_pose