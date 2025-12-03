import torch
import torch.nn as nn
from transformers import MBartModel, MBart50Tokenizer
import einops
from model.qae_merge import QAE

class MBartPoseNARGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = True
        # ---------------------------------------------------------
        # 1. QAE 모델 로드 (qae_merge.py 기반)
        # ---------------------------------------------------------
        self.qae = QAE(config["qae"])
        for param in self.qae.parameters():
            param.requires_grad = False
        self.qae.eval()
        
        self.latent_dim = config["qae"]["hidden_size"]
        self.num_latents = config["qae"]["num_tokens"]
        
        # ---------------------------------------------------------
        # 2. mBART 모델 (Backbone)
        # ---------------------------------------------------------
        model_name = "facebook/mbart-large-50"
        self.tokenizer = MBart50Tokenizer.from_pretrained(model_name)
        
        # [수정] use_safetensors=True 추가하여 보안 에러 우회
        try:
            self.mbart = MBartModel.from_pretrained(model_name, use_safetensors=True)
        except EnvironmentError:
            # 만약 safetensors가 없다면 경고를 출력하고 일반 로드 시도 (이 경우 torch 업그레이드 필요할 수 있음)
            print("Warning: Safetensors not found. Attempting standard load (requires Torch >= 2.6)...")
            self.mbart = MBartModel.from_pretrained(model_name)
        
        self.model_dim = self.mbart.config.d_model # 1024
        
        # ---------------------------------------------------------
        # 3. NAR을 위한 Learnable Query & Projection
        # ---------------------------------------------------------
        self.query_embeddings = nn.Parameter(torch.randn(1, self.num_latents, self.model_dim))
        
        self.output_proj = nn.Linear(self.model_dim, self.latent_dim)
        
        self.criterion = nn.MSELoss()

    def forward(self, pose_input, text_input, pose_length):
        device = pose_input.device
        B = pose_input.shape[0]

        # A. Target Latent 추출
        with torch.no_grad():
            encoded_feat = self.qae.encode_pose(pose_input, pose_length)
            z_gt = self.qae.qformer(encoded_feat)
            # pose_decoded = self.qae.decode(z_gt, pose_length)
        # B. Text Encoding
        text_tokens = self.tokenizer(
            text_input, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        
        # C. Decoder Input (Learnable Query)
        decoder_inputs_embeds = self.query_embeddings.expand(B, -1, -1)
        
        # D. mBART Forward
        outputs = self.mbart(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # E. Prediction & Loss
        pred_latents = self.output_proj(last_hidden_state)
        dist_loss = self.criterion(pred_latents, z_gt)
        
        # F. Decoding
        pose_decoded = self.qae.decode(pred_latents, pose_length)
        
        pose_input = einops.rearrange(pose_input, "b f n c -> b f (n c)")
        pose_output = einops.rearrange(pose_decoded, "b f n c -> b f (n c)")
        # recon_loss = self.qae.loss(pose_output, pose_input)
        recon_loss = torch.tensor([0.0], device=device)
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