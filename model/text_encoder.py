import torch
import torch.nn as nn
# SentenceTransformer 사용을 위한 라이브러리 임포트
from sentence_transformers import SentenceTransformer 

class T5XXLTextEncoder(nn.Module):

    def __init__(self, model_name: str = "sentence-transformers/sentence-t5-xxl", embed_dim: int = 768):
        super().__init__()
        
        # 1. SentenceTransformer 모델 로드 및 가중치 고정 SentenceTransformer('sentencet5-xxl/')
        self.t5_model = SentenceTransformer(model_name)
        self.t5_model.eval()
        for p in self.t5_model.parameters():
            p.requires_grad = False
            
        # SentenceTransformer의 기본 출력 차원 확인 (sentencet5-xxl의 경우 768)
        self.input_dim = self.t5_model.get_sentence_embedding_dimension()

        # 2. T5 출력 차원을 QAE의 embed_dim으로 맞추는 Projection 층
        if self.input_dim != embed_dim:
            self.projection = nn.Linear(self.input_dim, embed_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, text_list: list) -> torch.Tensor:
        device = next(self.parameters()).device
        
        # SentenceTransformer의 encode 메서드를 사용하여 문장 임베딩을 계산합니다.
        embeddings = self.t5_model.encode(
            text_list, 
            convert_to_tensor=True, 
            show_progress_bar=False, # 프로그레스 바 숨김
            device=device
        ) # (B, D_t5)

        # 최종 Projection
        context_vector = self.projection(embeddings) # (B, D_embed)
        return context_vector