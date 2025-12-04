import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import einops

from model.qformer import QFormer
from common.loss import Loss

    

class BertLayerNorm(nn.Module):
    """TF 스타일의 LayerNorm (epsilon이 제곱근 안에 있음)"""
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Attention(nn.Module):
    """Standard Multi-head Attention"""
    def __init__(self, num_heads, size):
        super(Attention, self).__init__()
        assert size % num_heads == 0
        self.head_size = size // num_heads
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, size)
        self.v_layer = nn.Linear(size, size)
        self.q_layer = nn.Linear(size, size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size = k.size(0)
        k = self.k_layer(k).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_layer(v).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        q = self.q_layer(q).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        
        attention = self.softmax(scores)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        return self.output_layer(context)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = BertLayerNorm(dim)
        self.attn = Attention(heads, dim)
        self.norm2 = BertLayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.ffn(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim, heads, mlp_dim, dropout) for _ in range(depth)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

# 논문에 명시된 Structured Prediction Layer (SPL)
# modules/sp_layer.py의 코드를 가져옴
SIGN_POSE_SKELETON = [
    [(-1, 0, "neck")],
    [(0, 1, "head"), (0, 5, "LeftUpArm"), (0, 2, "RightUpArm")],
    [(2, 3, "RightElbow"), (5, 6, "LeftElbow")],
    [(3, 4, "RightWrist"), (6, 7, "LeftWrist")]]

SIGN_HAND_SKELETON = [
    [(-1, 0, "Wrist")],
    [(0, 1, "hand1"), (0, 5, "hand5"), (0, 9, "hand9"), (0, 13, "hand13"), (0, 17, "hand17")],
    [(1, 2, "hand2"), (5, 6, "hand6"), (9, 10, "hand10"), (13, 14, "hand14"), (17, 18, "hand18")],
    [(2, 3, "hand3"), (6, 7, "hand7"), (10, 11, "hand11"), (14, 15, "hand15"), (18, 19, "hand19")],
    [(3, 4, "hand4"), (7, 8, "hand8"), (11, 12, "hand12"), (15, 16, "hand16"), (19, 20, "hand20")]]

class SP_block(nn.Module):
    def __init__(self, input_size, hid_size, out_size, L_num):
        super().__init__()
        layers = [nn.Linear(input_size, hid_size), nn.ReLU()]
        for _ in range(L_num - 1):
            layers.extend([nn.Linear(hid_size, hid_size), nn.ReLU()])
        layers.append(nn.Linear(hid_size, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SPL(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units, joint_size, SKELETON):
        super().__init__()
        self.input_size = input_size
        if SKELETON == "sign_pose":
            self.skeleton = SIGN_POSE_SKELETON
            self.num_joints = 8
        elif SKELETON == "sign_hand":
            self.skeleton = SIGN_HAND_SKELETON
            self.num_joints = 21
        else:
            raise ValueError(f"{SKELETON} is not a valid skeleton type!")
        
        kinematic_tree = {entry[1]: [([entry[0]] if entry[0] > -1 else []), entry[2]] for layer in self.skeleton for entry in layer}
        self.prediction_order = list(range(self.num_joints))
        self.indexed_skeleton = {i: [kinematic_tree[i][0], i, kinematic_tree[i][1]] for i in self.prediction_order}
        
        self.joint_predictions = nn.ModuleList()
        for joint_key in self.prediction_order:
            parent_ids, _, _ = self.indexed_skeleton[joint_key]
            current_input_size = self.input_size + joint_size * len(parent_ids)
            self.joint_predictions.append(SP_block(current_input_size, hidden_units, joint_size, hidden_layers))

    def forward(self, x):
        out = {}
        for joint_key in self.prediction_order:
            parent_ids, _, _ = self.indexed_skeleton[joint_key]
            parent_feats = [out[i] for i in parent_ids]
            x_input = torch.cat([x] + parent_feats, dim=-1) if parent_feats else x
            out[joint_key] = self.joint_predictions[joint_key](x_input)
        
        return torch.cat([out[i] for i in self.prediction_order], dim=-1)

class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., length=27):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        B, N_1, C = x_3.shape

        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N_1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- P2PSLP 클래스를 논문에 맞게 수정한 PoseVQVAE 클래스 ---
class QAE(nn.Module):
    def __init__(self, config):
        super(QAE, self).__init__()
        
        # 설정값
        self.use_cuda = True
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
        
        self.loss = Loss(cfg = config, target_pad=0.0)

        # 1. 임베딩 (기존 유지)
        self.pose_emb = nn.Linear(8 * 3, embed_dim)
        self.rhand_emb = nn.Linear(21 * 3, embed_dim)
        self.lhand_emb = nn.Linear(21 * 3, embed_dim)

        # Positional Encoding (기존 유지)
        self.spa_pos_emb = nn.Parameter(torch.zeros(1, 3, embed_dim)) 
        self.tem_pos_emb = nn.Parameter(torch.zeros(1, config['max_len'], embed_dim)) 

        # 2. 인코더
        # Spatial Transformer는 3개 파트 간의 관계를 학습하므로 유지
        self.enc_spa_vit = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=mlp_dim, dropout=0.1)
        
        # [변경] Merge Projection: (Body + R + L) -> Combined
        # 3개의 파트(3 * embed_dim)를 1개의 파트(embed_dim)로 압축
        self.merge_proj = nn.Linear(embed_dim * 3, embed_dim)

        # Temporal Transformer는 이제 병합된 1개의 시퀀스만 처리합니다.
        self.enc_tem_vit = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=mlp_dim, dropout=0.1)
        
        # 3. Bottleneck (Q-Former)
        # [변경] 3개의 Q-Former 대신 1개만 사용
        self.unified_query = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.unified_query.data.normal_(mean=0.0, std=0.02)
        
        self.unified_pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.unified_pos_emb.data.normal_(mean=0.0, std=0.02)

        self.qformer_model = QFormer(embed_dim=embed_dim,
                                     drop_rate=config['drop_rate'],
                                     depth=depth,
                                     num_heads=num_heads)
        
        self.bottleneck_conv = nn.Sequential(
            # 채널 수는 그대로 유지 (embed_dim -> embed_dim)
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            # [핵심] 입력 길이 T가 얼마든 num_tokens 길이로 강제 변환
            nn.AdaptiveAvgPool1d(self.num_tokens) 
        )
        
        self.bottleneck_norm = nn.LayerNorm(embed_dim)
        
        self.bottleneck_norm = nn.LayerNorm(embed_dim)
        
        # [참고] VQAE라면 여기에 ResidualLFQ 등의 양자화 모듈이 1개만 들어갑니다.

        # 4. 디코더
        # Temporal Transformer (1개의 시퀀스 처리)
        self.dec_tem_vit = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=mlp_dim, dropout=0.1)
        self.dec_ca = Cross_Attention(embed_dim, num_heads=num_heads, qkv_bias=config['qkv_bias'], \
            qk_scale=config['qk_scale'], attn_drop=config['attn_drop_rate'], proj_drop=config['drop_rate'])
        self.dec_token = nn.Parameter(torch.zeros(1, config['max_len'], embed_dim))

        # [변경] Split Projection: Combined -> (Body + R + L)
        # 1개의 파트(embed_dim)를 다시 3개의 파트(3 * embed_dim)로 복원
        self.split_proj = nn.Linear(embed_dim, embed_dim * 3)

        # Spatial Transformer (복원된 3개 파트 간의 관계 정제)
        self.dec_spa_vit = Encoder(dim=embed_dim, depth=depth, heads=num_heads, mlp_dim=mlp_dim, dropout=0.1)

        # 5. SPL (Structured Prediction Layer) - 기존 유지
        self.pose_spl = SPL(input_size=embed_dim, hidden_layers=5, hidden_units=embed_dim, joint_size=3, SKELETON="sign_pose")
        self.hand_spl = SPL(input_size=embed_dim, hidden_layers=5, hidden_units=embed_dim, joint_size=3, SKELETON="sign_hand")      
    
    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size, device=device).unsqueeze(0).repeat(x_len.size(0), 1)
        mask = pos < x_len.unsqueeze(1)
        return mask
    
    def encode_pose(self, pose_input, pose_length):
        B, T, N, C = pose_input.shape
        device = pose_input.device

        # 1. Part Embeddings
        body_input = pose_input[:, :, :8, :].reshape(B, T, -1)
        rhand_input = pose_input[:, :, 8:29, :].reshape(B, T, -1)
        lhand_input = pose_input[:, :, 29:, :].reshape(B, T, -1)
        
        pose_emb = self.pose_emb(body_input).unsqueeze(2)   # [B, T, 1, H]
        rhand_emb = self.rhand_emb(rhand_input).unsqueeze(2) # [B, T, 1, H]
        lhand_emb = self.lhand_emb(lhand_input).unsqueeze(2) # [B, T, 1, H]
        
        # [B, T, 3, H]
        points_feat = torch.cat([pose_emb, rhand_emb, lhand_emb], dim=2) 
        
        # 2. Spatial Transformer (Inter-part correlation)
        # (B, T)를 배치 차원으로 병합하여 각 프레임별로 3개 파트 간 어텐션 수행
        points_feat = einops.rearrange(points_feat, "b t n h -> (b t) n h")
        points_feat = points_feat + self.spa_pos_emb
        points_feat = self.enc_spa_vit(points_feat, mask=None)
        
        # [변경] 3. Merge & Temporal Transformer
        # 다시 (B, T, 3, H)로 복구
        points_feat = einops.rearrange(points_feat, "(b t) n h -> b t n h", b=B, t=T)
        
        # (B, T, 3*H)로 Flatten 후 (B, T, H)로 Projection
        points_feat = einops.rearrange(points_feat, "b t n h -> b t (n h)") 
        merged_feat = self.merge_proj(points_feat) # [B, T, H]

        # Temporal PE 더하기
        merged_feat = merged_feat + self.tem_pos_emb[:, :T, :]
        
        # 마스크 생성 (B, 1, 1, T) - Head 차원 고려
        points_mask = self._get_mask(pose_length, T, device)
        points_mask = points_mask.unsqueeze(1).unsqueeze(1) 
        
        # Temporal Attention 수행
        encoded_feat = self.enc_tem_vit(merged_feat, mask=points_mask) # [B, T, H]
        
        return encoded_feat
    
    # def qformer(self, encoded_feat):
    #     # encoded_feat: [B, T, H]
    #     B = encoded_feat.shape[0]
        
    #     # Query 확장 [B, num_tokens, H]
    #     query = self.unified_query.expand(B, -1, -1)
        
    #     # Q-Former (Cross Attention: Query가 Encoded Feat를 압축)
    #     # Output: [B, num_tokens, H]
    #     qae_feat = self.qformer_model(encoded_feat, query, self.unified_pos_emb)
    #     qae_feat = self.bottleneck_norm(qae_feat)
    #     qae_feat = torch.tanh(qae_feat)
    #     # 여기서 VQ를 적용한다면 qae_feat를 통과시키면 됩니다.
    #     return qae_feat

    def qformer(self, encoded_feat):
        # encoded_feat: [B, T, H]
        
        # Conv1d는 [B, Channel, Length] 입력을 받으므로 차원 변경
        x = encoded_feat.transpose(1, 2)  # [B, H, T]
        
        # Conv + Pooling 통과 -> [B, H, num_tokens]
        x = self.bottleneck_conv(x)
        
        # 다시 원래 형태 [B, num_tokens, H]로 복구
        qae_feat = x.transpose(1, 2)
        
        # Norm & Activation (기존 유지)
        qae_feat = self.bottleneck_norm(qae_feat)
        qae_feat = torch.tanh(qae_feat)
        
        return qae_feat
    
    def decode(self, qae_feat, pose_length):
        # qae_feat: [B, num_tokens, H] (or Quantized)
        B = qae_feat.shape[0]
        T = max(pose_length)
        device = qae_feat.device
        
        # 1. Temporal Decoding
        # Decoder Token 확장 [B, T, H]
        dec_token = repeat(self.dec_token, '() f c -> b f c', b=B)[:, :T, :]
        
        # Cross Attention (Decoder Token <-> Compressed Latent)
        # qae_feat를 Key/Value로 사용
        points_feat = dec_token + self.dec_ca(dec_token, qae_feat, qae_feat)
        points_feat = points_feat + self.tem_pos_emb[:, :T, :]
        
        # Mask 처리
        points_mask = self._get_mask(pose_length, T, device)
        points_mask = points_mask.unsqueeze(1).unsqueeze(1)
        
        # Temporal Transformer
        points_feat = self.dec_tem_vit(points_feat, mask=points_mask) # [B, T, H]
        
        # [변경] 2. Split & Spatial Decoding
        # (B, T, H) -> (B, T, 3*H) -> (B*T, 3, H)
        points_feat = self.split_proj(points_feat)
        points_feat = einops.rearrange(points_feat, "b t (n h) -> (b t) n h", n=3)
        
        # Spatial Transformer (각 파트 간 관계 복원)
        points_feat = points_feat + self.spa_pos_emb
        points_feat = self.dec_spa_vit(points_feat, mask=None)
        
        # 3. SPL Head (Regression)
        # (B, T, 3, H) 형태로 변환
        rec_feat = einops.rearrange(points_feat, "(b t) n h -> b t n h", b=B, t=T)
        
        dec_pose_feat = rec_feat[:, :, 0, :]
        dec_rhand_feat = rec_feat[:, :, 1, :]
        dec_lhand_feat = rec_feat[:, :, 2, :]
        
        # SPL 통과 (B, T, Joints * 3)
        dec_pose = self.pose_spl(dec_pose_feat)
        dec_rhand = self.hand_spl(dec_rhand_feat)
        dec_lhand = self.hand_spl(dec_lhand_feat)
        
        # 최종 병합 (B, T, 50, 3)
        reconstructed_pose = torch.cat([dec_pose, dec_rhand, dec_lhand], dim=-1).view(B, T, 50, 3)

        return reconstructed_pose
    
    def forward(self, pose_input, text_input, pose_length):
        # 1. Encode
        encoded_feat = self.encode_pose(pose_input, pose_length) # [B, T, H]
        
        # 2. Bottleneck (Q-Former + VQ if needed)
        qae_feat = self.qformer(encoded_feat) # [B, num_tokens, H]
        
        # 3. Decode
        pose_decoded = self.decode(qae_feat, pose_length) # [B, T, 50, 3]
        
        # 4. Loss
        pose_input = einops.rearrange(pose_input, "b f n c -> b f (n c)")
        pose_output = einops.rearrange(pose_decoded, "b f n c -> b f (n c)")
        
        recon_loss = self.loss(pose_output, pose_input)

        return pose_decoded, recon_loss, qae_feat