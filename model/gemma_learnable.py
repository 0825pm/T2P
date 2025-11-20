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
        # for param in self.qae.parameters():
        #     param.requires_grad = False
        # self.qae.eval()
        
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
        
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_latent_tokens, self.gemma_hidden_size) * 0.02
        )
        
        # 2. Gemma 출력 -> QAE 잠재 공간 차원 (768 -> 64)
        # 24개 토큰을 768차원에서 64차원으로 줄입니다.
        self.final_gemma_proj = nn.Linear(self.gemma_hidden_size, self.latent_dim) # 768 -> 64
        
        # 3. 3가지 파트별 Projection Layer (사용자 요청 사항)
        # 각 8개 토큰 (64차원)을 입력받아 64차원으로 다시 투영합니다.
        self.body_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.rhand_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.lhand_proj = nn.Linear(self.latent_dim, self.latent_dim)
        
        # 4. Loss
        self.latent_loss_fn = nn.MSELoss()
    
    def _prepare_multimodal_input(self, text_input: list, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """텍스트 임베딩과 쿼리 토큰을 결합하여 Gemma 입력 임베딩과 마스크를 생성합니다."""
        
        self.tokenizer.padding_side = 'left' 
        tokenized = self.tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        
        # 1. 텍스트 토큰 임베딩 (B, L_text, H_gemma)
        text_embeddings = self.text_model.get_input_embeddings()(tokenized['input_ids'])
        
        # 2. 쿼리 토큰 임베딩 (B, N_query, H_gemma)
        # 학습 가능한 쿼리를 배치 크기만큼 복사
        query_embeddings = self.query_tokens.repeat(text_embeddings.shape[0], 1, 1)

        # 3. 입력 시퀀스 구성: [Query Tokens, Text Tokens]
        # (B, 24 + L_text, H_gemma)
        inputs_embeds = torch.cat([text_embeddings, query_embeddings], dim=1) 
        
        # 4. Attention Mask 구성: [Query Mask, Text Mask]
        # Query Mask: 모든 쿼리 토큰은 1
        query_mask = torch.ones(query_embeddings.shape[:2], dtype=torch.long, device=device)
        # Text Mask: 토크나이저가 생성한 마스크
        text_mask = tokenized['attention_mask']
        
        # 최종 Attention Mask (B, 24 + L_text)
        attention_mask = torch.cat([text_mask, query_mask], dim=1)
        
        return inputs_embeds, attention_mask
    
    def forward(self, pose_input, text_input, pose_length, qae_feat):
        B, T, N, C = pose_input.shape
        device = pose_input.device

        # 1. 멀티모달 입력 준비 및 Gemma 인코딩
        inputs_embeds, attention_mask = self._prepare_multimodal_input(text_input, device)

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # 2. 쿼리 토큰에 해당하는 Hidden State 추출
        # Query Tokens는 Text 시퀀스 (길이 L_text) 바로 뒤에 위치합니다.
        # Z_extracted: (B, 24, 768)
        Z_extracted = outputs.hidden_states[-1][:, -self.num_latent_tokens:, :] # <-- Text 길이만큼 건너뛰고 추출
        
        # 3. 최종 잠재 공간 차원으로 투영
        Z_proj = self.final_gemma_proj(Z_extracted) # (B, 24, 64)
        
        # 4. 파트별 분할 및 개별 Linear Layer 적용
        # Z_body, Z_rhand, Z_lhand = Z_proj.chunk(self.num_latent_parts, dim=1) # (B, 8, 64) x 3
        
        Z_body_proj = self.body_proj(Z_proj)
        Z_rhand_proj = self.rhand_proj(Z_proj)
        Z_lhand_proj = self.lhand_proj(Z_proj)
        
        # 5. QAE 디코더 형식에 맞게 재결합 및 재배열
        Z_final_seq = torch.cat([Z_body_proj.unsqueeze(-1), Z_rhand_proj.unsqueeze(-1), Z_lhand_proj.unsqueeze(-1)], dim=-1) # (B, 24, 64)
        
        # Reshape: (B, 24, 64) -> (B, 64, 8, 3) (QAE decode 입력 형식)
        reshaped_latent = rearrange(Z_final_seq, 'b f h c -> b h f c') 
        
        # 6. QAE Decode
        pose_decoded = self.qae.decode(reshaped_latent, pose_length)
        
        # 7. Reconstruction Loss (엔드-투-엔드 손실)
        pose_input_flat = einops.rearrange(pose_input, "b f n c -> b f (n c)")
        pose_output_flat = einops.rearrange(pose_decoded, "b f n c -> b f (n c)")
        recon_loss = self.qae.loss(pose_output_flat, pose_input_flat) 
        
        # Latent Loss는 사용하지 않으므로 0으로 반환
        latent_loss = torch.zeros(1, device=device).mean()
        
        return pose_decoded, recon_loss, latent_loss