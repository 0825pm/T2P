import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import einops

from model.qae import QAE # QAE는 Pose Feature를 추출/디코딩하는 역할을 수행합니다.
from common.loss import Loss
from transformers import AutoTokenizer, GemmaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from diffusers import DDPMScheduler

# Orthus 논문의 Diffusion Head (SimpleMLPAdaLN) 역할을 수행하는 간소화된 클래스
# 실제 구현에서는 모델 구조와 AdaLN 로직을 정확히 맞춰야 하지만, 여기서는 필수적인 입출력과 기능을 모방합니다.
class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks):
        super().__init__()
        # Simplified MLP structure with layer normalization and ADA-LN
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, model_channels)
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 4 * model_channels),
            nn.SiLU(),
            nn.Linear(4 * model_channels, model_channels),
        )
        # Context Projection (z_cond)
        self.context_proj = nn.Linear(z_channels, model_channels)
        
        # Simplified block structure (e.g., one layer)
        self.block = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.GELU(),
            nn.Linear(model_channels * 4, model_channels)
        )
        
        self.proj_out = nn.Linear(model_channels, out_channels)
        
    def forward(self, x, timesteps, z):
        # x: noisy_latents (B*T, D_latent)
        # timesteps: (B*T)
        # z: condition (B*T, H_gemma)
        
        h = self.norm(x)
        h = self.proj_in(h)
        
        # Time and Context Embedding
        timesteps = timesteps.unsqueeze(-1).to(h.dtype) / 1000.0
        time_emb = self.time_embed(timesteps)
        context_emb = self.context_proj(z)

        # Apply simplified conditional modulation
        c = time_emb + context_emb
        h = h + self.block(h) + c
        
        out = self.proj_out(h)
        return out

# --- Orthus 아키텍처의 아이디어를 반영한 Pose-Text 생성 모델 클래스 (Gemma 기반) ---
class GEMMA(nn.Module):
    def __init__(self, config):
        super(GEMMA, self).__init__()
        
        # 설정값
        self.use_cuda = True
        gemma_model_name = config["text_encoder"]["model_name"]
        # Latent Dimension (Orthus의 Continuous Image Feature Dimension과 유사)
        self.latent_dim = config["qae"]['hidden_size'] # 예시: 64
        self.gemma_hidden_size = config["text_encoder"]['hidden_size'] # 예시: 2048
        self.total_latent_tokens = 24
        self.num_latent_parts = 3
        self.num_latent_tokens = 8
        self.diffusion_batch_mul = 8 # Orthus 논문 참고 값 (Section 4.2)
        
        self.loss = Loss(cfg=config["qae"], target_pad=0.0)
        # QAE (Pose Autoencoder)
        self.qae = QAE(config=config["qae"])
        for param in self.qae.parameters():
            param.requires_grad = False
        self.qae.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
        # Gemma는 Multimodal AR Backbone으로 사용됩니다.
        self.text_model = GemmaForCausalLM.from_pretrained(gemma_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.text_model.resize_token_embeddings(len(self.tokenizer))
            
        # PEFT 설정 (기존 gemma.py에서 복사)
        # lora_config = LoraConfig(
        #     r=16, lora_alpha=32,
        #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        #     lora_dropout=0.15, bias="none", task_type=TaskType.CAUSAL_LM
        # )
        # self.text_model = get_peft_model(self.text_model, lora_config)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()
        # print("="*50)
        # self.text_model.print_trainable_parameters()
        # print("="*50)
        
        # 1. Continuous Latent Token Embedding (D_latent -> Gemma Hidden Size)
        self.latent_token_proj = nn.Linear(self.latent_dim, self.gemma_hidden_size)
        
        # 2. Text Context Vector Projector 
        self.text_context_proj = nn.Linear(self.gemma_hidden_size, self.gemma_hidden_size)
        
        self.query_tokens = nn.Parameter(torch.randn(1, self.total_latent_tokens-1, self.gemma_hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        # 5. Output Projection for Reconstruction Path (기존 gemma.py의 latent_pred_head)
        self.latent_pred_head = nn.Linear(self.gemma_hidden_size, self.latent_dim)
        
        # 6. Auxiliary Loss (for pose reconstruction)
        self.aux_loss_fn = nn.MSELoss()


    def _get_text_context_vector(self, text_input: list, device: torch.device) -> torch.Tensor:
        """Text input을 Gemma에 통과시켜 단일 Context Vector (B, 1, H)를 추출합니다."""
        
        # 1. Text Encoding
        self.tokenizer.padding_side = 'left' 
        tokenized = self.tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Gemma CausalLM forward pass (Hidden State 반환)
        gemma_output = self.text_model(
            **tokenized, 
            output_hidden_states=True,
            return_dict=True
        )
        
        # 마지막 레이어의 Hidden State 추출 (B, L_text, H_gemma)
        last_hidden_state = gemma_output.hidden_states[-1] 
        attention_mask = tokenized['attention_mask'] # (B, L_text)

        # 2. Context Vector 생성 (Non-padding 토큰에 대한 평균 풀링)
        text_emb = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-5)
        
        # Context Vector를 AR 시퀀스의 첫 번째 토큰으로 사용하기 위해 차원을 확장
        context_vector = self.text_context_proj(text_emb).unsqueeze(1)
        
        return context_vector
    
    def forward(self, pose_input, text_input, pose_length, cond_input):
        B, T, N, C = pose_input.shape
        device = pose_input.device

        body_feat, rhand_feat, lhand_feat = self.qae.encode_pose(pose_input, pose_length)
        
        # Clustering
        qae_feat = self.qae.qformer(body_feat, rhand_feat, lhand_feat)
        
        # qae_feat_gt: (B, total_latent_tokens=24, dim=64)
        z_gt_seq = rearrange(qae_feat, 'b h t j -> b (t j) h') 
        decoder_inputs = self.query_tokens.repeat(B, 1, 1)
        # 1. Text Context Vector 추출: (B, 1, H_gemma)
        # context_vector = self._get_text_context_vector(text_input, device)
        context_vector = cond_input
        
        # 2. Latent Data 준비 및 Embedding 
        L_AR_Input = self.total_latent_tokens - 1 
        L_AR_Full = self.total_latent_tokens 

        # Target: Z_GT[0:24] (B, 24, 64)
        z_target = z_gt_seq[:, :L_AR_Full, :] 
        
        # Input: Z_GT[0:23] (B, 23, 64) 
        z_input = z_gt_seq[:, :L_AR_Input, :]
        
        # Latent Token Embedding: (B, 23, 64) -> (B, 23, H_gemma)
        # z_emb = self.latent_token_proj(z_input) 

        # 3. AR 입력 시퀀스 구성 (Context + Latent Tokens)
        # inputs_embeds: (B, 24, H_gemma)
        inputs_embeds = torch.cat([context_vector, decoder_inputs], dim=1) 
        
        # 4. Attention Mask 구성 
        attention_mask = torch.ones(B, L_AR_Full, dtype=torch.long, device=device) 
        
        # 5. Gemma Forward Pass (Autoregressive Backbone)
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 6. Condition Vector for Diffusion Head (z_cond): (B, 24, H_gemma)
        z_cond = outputs.hidden_states[-1]
        
        # # 7. Diffusion Loss 계산 (Orthus Core)
        
        # # Target latent vector (x_0): (B * L_AR_Full, D_latent)
        # x_0_flat = z_target.reshape(B * L_AR_Full, -1)
        # z_cond_flat = z_cond.reshape(B * L_AR_Full, -1)
        
        # # Repeat batch for diffusion training
        # x_0_repeated = x_0_flat.repeat(self.diffusion_batch_mul, 1)
        # z_cond_repeated = z_cond_flat.repeat(self.diffusion_batch_mul, 1)
        
        # # 7.1. Sample noise and timesteps
        # timesteps = torch.randint(0, 1000, (x_0_repeated.shape[0],), dtype=torch.int64, device=device)
        # noise = torch.randn_like(x_0_repeated, device=device)
        
        # # 7.2. Apply noise (x_t)
        # # Orthus 논문의 구현에 따라, latent vector에 scaling을 적용할 수 있습니다.
        # # 여기서는 모델링 편의를 위해 scaling을 생략합니다.
        # noisy_latents = self.train_scheduler.add_noise(x_0_repeated, noise, timesteps)
        
        # # 7.3. Predict target (v-prediction)
        # target = self.train_scheduler.get_velocity(x_0_repeated, noise, timesteps)
        
        # # 7.4. Diffusion Head Prediction
        # model_pred = self.diffusion_head(noisy_latents, timesteps, z_cond_repeated)
        
        # # 7.5. Calculate Diffusion Loss
        # diff_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # 8. Reconstruction Path (기존 API 호환성을 위한 처리. 실제 생성은 Inference 단계에서 별도 구현 필요)
        # Gemma 출력을 단순 선형 투사하여 포즈로 디코딩 (Diffusion의 결과가 아님)
        latent_pred = self.latent_pred_head(z_cond) # (B, 24, 64)
        
        # Latent Reshape: (B, 24, 64) -> (B, 64, 8, 3) for QAE decode
        reshaped_latent = rearrange(latent_pred, 'b (t_qae n_parts) dim -> b dim t_qae n_parts', 
                                    t_qae=self.num_latent_tokens, n_parts=self.num_latent_parts) 
        
        # QAE Decode
        # qae_feat = rearrange(qae_feat, 'b (t_qae n_parts) dim -> b dim t_qae n_parts', 
        #                             t_qae=self.num_latent_tokens, n_parts=self.num_latent_parts) 
        pose_decoded = self.qae.decode(reshaped_latent, pose_length)
        
        # Reconstruction loss (Auxiliary)
        pose_input_flat = einops.rearrange(pose_input, "b f n c -> b f (n c)")
        pose_output_flat = einops.rearrange(pose_decoded, "b f n c -> b f (n c)")
        # recon_loss = self.aux_loss_fn(pose_output_flat, pose_input_flat)
        recon_loss = self.loss(pose_output_flat, pose_input_flat)
        # recon_loss = torch.tensor([0.0], device=device)

        # 주 학습 손실은 Diffusion Loss입니다.
        return pose_decoded, recon_loss, torch.tensor([0.0], device=device)
    
    @torch.no_grad()
    def generate(self, text_input, diffusion_steps=50, target_length=None):
        """
        [수정] target_length를 받아 해당 길이만큼 포즈를 디코딩하도록 변경
        """
        self.eval()
        device = next(self.parameters()).device
        
        if isinstance(text_input, str):
            text_input = [text_input]
            
        context_vector = self._get_text_context_vector(text_input, device)
        B = context_vector.shape[0]
        
        generated_latents = []
        
        # AR Loop (기존 동일)
        for i in range(self.total_latent_tokens):
            if len(generated_latents) == 0:
                inputs_embeds = context_vector
            else:
                prev_latents_tensor = torch.stack(generated_latents, dim=1)
                latents_emb = self.latent_token_proj(prev_latents_tensor)
                inputs_embeds = torch.cat([context_vector, latents_emb], dim=1)

            L = inputs_embeds.shape[1]
            attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
            
            outputs = self.text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            z_cond = outputs.hidden_states[-1][:, -1, :].unsqueeze(1) 
            latent_sample = torch.randn(B, 1, self.latent_dim, device=device)
            self.train_scheduler.set_timesteps(diffusion_steps)
            
            for t in self.train_scheduler.timesteps:
                timesteps_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                
                # SimpleMLPAdaLN forward (차원 보정 로직 포함된 버전 사용 가정)
                model_output = self.diffusion_head(latent_sample, timesteps_tensor, z_cond)
                
                latent_sample = self.train_scheduler.step(
                    model_output, t, latent_sample
                ).prev_sample
                
            generated_latents.append(latent_sample.squeeze(1))

        final_latents = torch.stack(generated_latents, dim=1)
        
        reshaped_latent = rearrange(
            final_latents, 
            'b (t_qae n_parts) dim -> b dim t_qae n_parts', 
            t_qae=self.num_latent_tokens, 
            n_parts=self.num_latent_parts
        )
        
        # [수정 부분] target_length가 없으면 기본값 150 사용
        if target_length is None:
            target_length = torch.full((B,), 150, device=device, dtype=torch.long)
        
        # QAE Decode 시 target_length 사용 -> (B, max(target_length), 50, 3) 생성
        decoded_pose = self.qae.decode(reshaped_latent, target_length)
        
        return decoded_pose