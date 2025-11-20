import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import einops

from model.qae import QAE
from common.loss import Loss
from transformers import AutoTokenizer, GemmaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# --- P2PSLP 클래스를 논문에 맞게 수정한 PoseVQVAE 클래스 ---
class GEMMA(nn.Module):
    def __init__(self, config):
        super(GEMMA, self).__init__()
        
        # 설정값
        self.use_cuda = True
        gemma_model_name = config["text_encoder"]["model_name"]
        self.latent_dim = config["qae"]['hidden_size']
        self.gemma_hidden_size = config["text_encoder"]['hidden_size']
        self.total_latent_tokens = 24
        self.num_latent_parts = 3
        self.num_latent_tokens = 8
        
        self.qae = QAE(config=config["qae"])
        for param in self.qae.parameters():
            param.requires_grad = False
        self.qae.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
        self.text_model = GemmaForCausalLM.from_pretrained(gemma_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.text_model.resize_token_embeddings(len(self.tokenizer))

        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.15, bias="none", task_type=TaskType.CAUSAL_LM
        )
        self.text_model = get_peft_model(self.text_model, lora_config)
        print("="*50)
        self.text_model.print_trainable_parameters()
        print("="*50)
        
        # 1. Continuous Latent Token Embedding (64 -> Gemma Hidden Size)
        # Latent token을 Gemma의 Hidden space로 매핑합니다.
        self.latent_token_proj = nn.Linear(self.latent_dim, self.gemma_hidden_size)
        
        # 2. Text Context Vector Projector 
        # Text embedding을 AR 시퀀스의 첫 번째 Context Token으로 사용하기 위한 프로젝션.
        self.text_context_proj = nn.Linear(self.gemma_hidden_size, self.gemma_hidden_size)
        
        # 3. Autoregressive Prediction Head (Gemma Hidden Size -> 64)
        # Gemma의 출력 토큰(Hidden State)을 최종 Latent Dimension (64)로 매핑합니다.
        self.latent_pred_head = nn.Linear(self.gemma_hidden_size, self.latent_dim)
        
        # 4. Losses
        self.latent_loss_fn = nn.MSELoss() # 연속적인 예측을 위한 MSE Loss
    
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
        # (B, H_gemma) 벡터를 얻고, 이를 AR 시퀀스의 첫 번째 토큰으로 사용합니다.
        text_emb = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-5)
        
        # Context Vector를 AR 시퀀스의 첫 번째 토큰으로 사용하기 위해 차원을 확장
        # (B, H_gemma) -> (B, 1, H_gemma)
        context_vector = self.text_context_proj(text_emb).unsqueeze(1)
        
        return context_vector
    
    def forward(self, pose_input, text_input, pose_length, qae_feat):
        B, T, N, C = pose_input.shape
        device = pose_input.device

        # qae_feat_gt: QAE qformer의 출력 (B, dim=64, num_tokens=8, num_parts=3)
        # AR 모델을 위해 (B, total_latent_tokens=24, dim=64)로 변환
        z_gt_seq = qae_feat
        
        # 1. Text Context Vector 추출: (B, 1, H_gemma)
        context_vector = self._get_text_context_vector(text_input, device) 
        
        # 2. Latent Data 준비 및 Embedding (AR Sequence Length: 23)
        L_AR_Input = self.total_latent_tokens - 1 # 23
        L_AR_Full = self.total_latent_tokens # 24

        # Target: Z_GT[0:24] (B, 24, 64)
        z_target = z_gt_seq[:, :L_AR_Full, :] 
        
        # Input: Z_GT[0:23] (B, 23, 64) - Context 다음으로 들어갈 토큰
        z_input = z_gt_seq[:, :L_AR_Input, :]
        
        # Latent Token Embedding: (B, 23, 64) -> (B, 23, H_gemma)
        z_emb = self.latent_token_proj(z_input) 

        # 3. AR 입력 시퀀스 구성 (Context + Latent Tokens)
        # inputs_embeds: (B, 1 + 23, H_gemma) = (B, 24, H_gemma)
        inputs_embeds = torch.cat([context_vector, z_emb], dim=1) 
        
        # 4. Attention Mask 구성 (Causal LM이므로, Context 및 모든 Latent Input에 대해 1)
        attention_mask = torch.ones(B, L_AR_Full, dtype=torch.long, device=device) 
        
        # 5. Gemma Forward Pass (Autoregressive Backbone)
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 6. Prediction Head 적용
        # Hidden State 출력 (B, 24, H_gemma)
        hidden_states = outputs.hidden_states[-1] 
        
        # Prediction: (B, 24, H_gemma) -> (B, 24, 64)
        # 이 출력 시퀀스 전체가 Z_target (B, 24, 64)를 예측합니다.
        latent_pred = self.latent_pred_head(hidden_states) 
        
        # 7. Latent Loss 계산 (주요 손실)
        # MSE Loss: (B, 24, 64) vs (B, 24, 64)
        latent_loss = self.latent_loss_fn(latent_pred, z_target)
        
        # 8. 재구성을 위한 Latent Reshape 및 Decode
        # Latent Reshape: (B, 24, 64) -> (B, 64, 8, 3) for QAE decode
        reshaped_latent = rearrange(latent_pred, 'b (t_qae n_parts) dim -> b dim t_qae n_parts', 
                                    t_qae=self.num_latent_tokens, n_parts=self.num_latent_parts) 
        
        # QAE Decode
        pose_decoded = self.qae.decode(reshaped_latent, pose_length)
        
        # Reconstruction loss (Auxiliary)
        pose_input_flat = einops.rearrange(pose_input, "b f n c -> b f (n c)")
        pose_output_flat = einops.rearrange(pose_decoded, "b f n c -> b f (n c)")
        # QAE에 정의된 복합 손실 함수 사용
        recon_loss = self.qae.loss(pose_output_flat, pose_input_flat) 

        # 주 학습 손실은 Latent Loss입니다.
        return pose_decoded, recon_loss, latent_loss