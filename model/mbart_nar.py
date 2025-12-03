import torch
import torch.nn as nn
from transformers import MBartModel, MBart50Tokenizer
import einops
from model.qae_merge import QAE
from peft import LoraConfig, get_peft_model, TaskType

class MBartPoseNARGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = True
        
        # 1. QAE 모델 로드 (Freeze)
        self.qae = QAE(config["qae"])
        # for param in self.qae.parameters():
        #     param.requires_grad = False
        # self.qae.eval()
        
        self.latent_dim = config["qae"]["hidden_size"]
        self.num_latents = config["qae"]["num_tokens"]
        
        # 2. mBART 모델 (Backbone)
        model_name = "facebook/mbart-large-50"
        self.tokenizer = MBart50Tokenizer.from_pretrained(model_name)
        
        try:
            self.mbart = MBartModel.from_pretrained(model_name, use_safetensors=True)
        except EnvironmentError:
            print("Warning: Safetensors not found. Attempting standard load...")
            self.mbart = MBartModel.from_pretrained(model_name)
            
        # [핵심] LoRA (Low-Rank Adaptation) 적용
        # 전체 파라미터를 학습하는 대신, Attention Layer의 일부 가중치만 학습합니다.
        # 과적합을 강력하게 억제합니다.
        lora_config = LoraConfig(
            r=8,                # Rank (작을수록 파라미터 적음, 과적합 방지)
            lora_alpha=32,      # Scaling factor
            target_modules=["q_proj", "v_proj"], # Attention 모듈 타겟팅
            lora_dropout=0.1,
            bias="none"
        )
        self.mbart = get_peft_model(self.mbart, lora_config)
        print(">>> mBART LoRA Applied. Trainable Parameters:")
        self.mbart.print_trainable_parameters()
        
        self.model_dim = self.mbart.config.d_model # 1024
        
        # 3. NAR Head (학습 대상)
        self.query_embeddings = nn.Parameter(torch.randn(1, self.num_latents, self.model_dim))
        self.output_proj = nn.Linear(self.model_dim, self.latent_dim)
        
        self.criterion = nn.MSELoss()

    def forward(self, pose_input, text_input, pose_length):
        device = pose_input.device
        B = pose_input.shape[0]

        # A. Target Latent 추출
        # with torch.no_grad():
        #     encoded_feat = self.qae.encode_pose(pose_input, pose_length)
        #     # QAE qformer가 (feat, indices) 튜플을 반환할 경우를 대비해 안전하게 언패킹
        #     qae_output = self.qae.qformer(encoded_feat)
        #     if isinstance(qae_output, tuple):
        #         z_gt = qae_output[0] # (B, num_latents, latent_dim)
        #     else:
        #         z_gt = qae_output
        
        # B. Text Encoding
        text_tokens = self.tokenizer(
            text_input, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        # C. Decoder Input
        decoder_inputs_embeds = self.query_embeddings.expand(B, -1, -1)
        
        # D. mBART Forward (LoRA 적용됨)
        outputs = self.mbart(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # E. Prediction & Loss
        pred_latents = self.output_proj(last_hidden_state)
        
        # Latent Space에서의 MSE Loss
        # dist_loss = self.criterion(pred_latents, z_gt)
        dist_loss = torch.tensor([0.0], device=device)
        
        # F. Decoding (For Visualization/Debug only)
        # 학습 중에는 디코딩하지 않아도 되지만, 리턴값 유지를 위해 수행
        pose_decoded = self.qae.decode(pred_latents, pose_length)
        
        pose_input = einops.rearrange(pose_input, "b f n c -> b f (n c)")
        pose_output = einops.rearrange(pose_decoded, "b f n c -> b f (n c)")
        
        recon_loss = self.qae.loss(pose_output, pose_input)
        
        return pose_decoded, recon_loss, dist_loss

    @torch.no_grad()
    def generate(self, text_input, target_length=None):
        self.eval()
        device = next(self.parameters()).device
        B = len(text_input) if isinstance(text_input, list) else 1
        
        text_tokens = self.tokenizer(
            text_input, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        decoder_inputs_embeds = self.query_embeddings.expand(B, -1, -1)
        
        outputs = self.mbart(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            return_dict=True
        )
        
        pred_latents = self.output_proj(outputs.last_hidden_state)
        
        if target_length is None:
            target_length = torch.full((B,), 150, device=device, dtype=torch.long)
            
        decoded_pose = self.qae.decode(pred_latents, target_length)
        
        return decoded_pose