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

# --- Orthus 아키텍처의 아이디어를 반영한 Pose-Text 생성 모델 클래스 ---
# 주요 특징: 연속적인 포즈 잠재 특징(qae_feat)을 입력 및 예측 목표로 사용합니다.
class GEMMA(nn.Module):
    def __init__(self, config):
        super(GEMMA, self).__init__()
        
        # 설정값
        self.use_cuda = True
        gemma_model_name = config["text_encoder"]["model_name"]
        # Latent Dimension (Orthus의 Continuous Image Feature Dimension과 유사)
        self.latent_dim = config["qae"]['hidden_size'] 
        self.gemma_hidden_size = config["text_encoder"]['hidden_size']
        self.total_latent_tokens = 24
        self.num_latent_parts = 3
        self.num_latent_tokens = 8
        
        # QAE(Pose Autoencoder)는 학습 여부와 관계없이 연속적인 잠재 벡터를 제공하고, 
        # 최종적으로 예측된 벡터를 Pose로 디코딩하는 역할을 수행합니다.
        self.qae = QAE(config=config["qae"])
        # 예시로 QAE는 고정된 상태로 가정합니다 (Orthus 논문의 Post-training 이전 단계와 유사).
        for param in self.qae.parameters():
            param.requires_grad = False
        self.qae.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)
        # Gemma는 Multimodal AR Backbone으로 사용됩니다.
        self.text_model = GemmaForCausalLM.from_pretrained(gemma_model_name)
        
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()
        
        # self.text_model = get_peft_model(self.text_model, lora_config)
        print("="*50)
        self.text_model.print_trainable_parameters()
        print("="*50)
    
    def forward(self, pose_input, text_input, pose_length, qae_feat):
        