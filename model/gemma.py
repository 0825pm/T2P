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
        embed_dim = config['hidden_size']
        depth = config['depth']
        num_heads = config['num_heads']
        mlp_dim = config['ff_size']
        self.num_tokens = config['num_tokens']
        
        drop_path_rate = config['drop_path_rate']
        drop_rate = config['drop_rate']
        attn_drop_rate = config['attn_drop_rate']
        qkv_bias = config['qkv_bias']
        qk_scale = config['qk_scale']
        
        
        
        self.qae = QAE(config=config]["qae"])
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
        
    
    def forward(self, pose_input, text_input, pose_length, qae_feat):
        B, T, N, C = pose_input.shape
        device = pose_input.device

        # text input -> gemma -> qae feat pred -> decoder
        # qae feat pred, qae feat gt loss
        
        # Decode
        pose_decoded = self.qae.decode(qae_feat, pose_length)
        # Reconstruction Loss
        # recon_loss = reconstruction_loss(pose_decoded, pose_input)
        recon_loss = nn.L1Loss()(pose_decoded, pose_input)
        
        pose_input = einops.rearrange(pose_input, "b f n c -> b f (n c)")
        pose_output = einops.rearrange(pose_decoded, "b f n c -> b f (n c)")
        
        recon_loss = self.qae.loss(pose_output, pose_input)

        return pose_decoded, recon_loss