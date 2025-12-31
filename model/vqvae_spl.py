"""
SPL-VQVAE v2: Structured Prediction Layer based VQ-VAE for Sign Language
- SOKE format (133 dims, axis-angle rotations)
- Q-Former based compression (no temporal convolution)
- vector_quantize_pytorch for quantization
- SPL decoder with SMPL-X kinematic chain

SOKE 133 dims structure:
    - upper_body_pose: 30 dims (10 joints * 3) - indices [0:30]
    - lhand_pose: 45 dims (15 joints * 3) - indices [30:75]
    - rhand_pose: 45 dims (15 joints * 3) - indices [75:120]
    - jaw_pose: 3 dims (1 joint * 3) - indices [120:123]
    - expression: 10 dims - indices [123:133]

Architecture:
    Input (B, T, 133)
        ↓
    Part Embedding (body, lhand, rhand, expr)
        ↓
    Spatial Transformer (inter-part correlation)
        ↓
    Merge Projection (N parts → 1)
        ↓
    Q-Former (Learnable Queries → Fixed N tokens)
        ↓
    VectorQuantize (vector_quantize_pytorch)
        ↓
    Q-Former Decoder
        ↓
    Split Projection (1 → N parts)
        ↓
    Spatial Transformer (decoder)
        ↓
    SPL (SMPL-X kinematic chain prediction)
        ↓
    Output (B, T, 133)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat
from vector_quantize_pytorch import VectorQuantize


# =============================================================================
# SOKE Format Definition (133 dims)
# =============================================================================

# SOKE 133 dims structure
SOKE_UPPER_BODY_DIM = 30   # 10 joints * 3
SOKE_LHAND_DIM = 45        # 15 joints * 3
SOKE_RHAND_DIM = 45        # 15 joints * 3
SOKE_JAW_DIM = 3           # 1 joint * 3
SOKE_EXPR_DIM = 10         # expression coefficients
SOKE_TOTAL_DIM = 133

# Part indices in SOKE format
SOKE_BODY_START = 0
SOKE_BODY_END = 30
SOKE_LHAND_START = 30
SOKE_LHAND_END = 75
SOKE_RHAND_START = 75
SOKE_RHAND_END = 120
SOKE_JAW_START = 120
SOKE_JAW_END = 123
SOKE_EXPR_START = 123
SOKE_EXPR_END = 133

# Number of joints per part (for axis-angle, each joint has 3 dims)
NUM_UPPER_BODY_JOINTS = 10
NUM_HAND_JOINTS = 15
NUM_JAW_JOINTS = 1

# SMPL-X upper body joint indices (after removing lower body)
# These are the 10 joints kept in SOKE: spine3, neck, head, l/r_shoulder, l/r_elbow, l/r_wrist, + 1 more
UPPER_BODY_JOINTS = [9, 12, 15, 13, 14, 16, 17, 18, 19, 20]  # Example mapping

# SPL connections for SOKE format (joint index within each part)
# Upper body (10 joints): spine chain + arms
UPPER_BODY_CONNECTIONS = [
    (0, 1),   # spine3 → neck
    (1, 2),   # neck → head
    (0, 3),   # spine3 → l_collar
    (0, 4),   # spine3 → r_collar
    (3, 5),   # l_collar → l_shoulder
    (4, 6),   # r_collar → r_shoulder
    (5, 7),   # l_shoulder → l_elbow
    (6, 8),   # r_shoulder → r_elbow
    (7, 9),   # l_elbow → l_wrist (connects to lhand)
]

# Hand connections (15 joints each): wrist is at index 0 of hand, fingers 1-14
# Each finger: base(0) → mid(1) → tip(2)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3),
    # Index
    (0, 4), (4, 5), (5, 6),
    # Middle
    (0, 7), (7, 8), (8, 9),
    # Ring
    (0, 10), (10, 11), (11, 12),
    # Pinky
    (0, 13), (13, 14),
]


def build_soke_kinematic_tree(part='body'):
    """Build kinematic tree for SOKE format."""
    if part == 'body':
        connections = UPPER_BODY_CONNECTIONS
        num_joints = NUM_UPPER_BODY_JOINTS
    else:  # hand
        connections = HAND_CONNECTIONS
        num_joints = NUM_HAND_JOINTS
    
    kinematic_tree = {j: [] for j in range(num_joints)}
    for parent, child in connections:
        if child < num_joints:
            kinematic_tree[child].append(parent)
    
    return kinematic_tree


def get_soke_prediction_order(part='body'):
    """Get topological order for SOKE format."""
    kinematic_tree = build_soke_kinematic_tree(part)
    
    if part == 'body':
        num_joints = NUM_UPPER_BODY_JOINTS
        connections = UPPER_BODY_CONNECTIONS
    else:
        num_joints = NUM_HAND_JOINTS
        connections = HAND_CONNECTIONS
    
    children_map = {j: [] for j in range(num_joints)}
    for parent, child in connections:
        if parent < num_joints and child < num_joints:
            children_map[parent].append(child)
    
    visited = set()
    order = []
    queue = [0]
    
    while queue:
        joint = queue.pop(0)
        if joint in visited or joint >= num_joints:
            continue
        
        parents = kinematic_tree.get(joint, [])
        if all(p in visited for p in parents) or joint == 0:
            visited.add(joint)
            order.append(joint)
            for child in children_map.get(joint, []):
                if child not in visited and child < num_joints:
                    queue.append(child)
        else:
            queue.append(joint)
    
    # Add any missing joints
    for j in range(num_joints):
        if j not in order:
            order.append(j)
    
    return order, kinematic_tree


# =============================================================================
# Basic Building Blocks
# =============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    """FFN with GELU activation"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
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


class MultiHeadAttention(nn.Module):
    """Multi-head Self Attention"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class CrossAttention(nn.Module):
    """Cross Attention for Q-Former"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None):
        B, N, C = query.shape
        _, S, _ = key_value.shape
        
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Standard Transformer Encoder Block"""
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    def __init__(self, dim, depth, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============================================================================
# Q-Former
# =============================================================================

class QFormerLayer(nn.Module):
    """Q-Former layer: self-attention + cross-attention + FFN"""
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, dropout)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        self.norm3 = LayerNorm(dim)

    def forward(self, queries, encoder_output, mask=None):
        queries = queries + self.self_attn(self.norm1(queries))
        queries = queries + self.cross_attn(self.norm2(queries), encoder_output, mask)
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class QFormer(nn.Module):
    """Q-Former: Compress variable-length sequence to fixed N tokens"""
    def __init__(self, num_queries, dim, depth, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            QFormerLayer(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, encoder_output, mask=None):
        B = encoder_output.shape[0]
        queries = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, encoder_output, mask)
        return self.norm(queries)


# =============================================================================
# SPL (Structured Prediction Layer) for SOKE format
# =============================================================================

class JointPredictor(nn.Module):
    """MLP for single joint prediction (axis-angle, 3 dims per joint)"""
    def __init__(self, input_dim, hidden_dim, output_dim=3, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PartSPL(nn.Module):
    """
    SPL for a single body part (body, lhand, or rhand).
    Predicts joint rotations following kinematic chain.
    """
    def __init__(self, input_dim, hidden_dim=256, num_joints=10, joint_dim=3, num_layers=2, part='body'):
        super().__init__()
        self.input_dim = input_dim
        self.joint_dim = joint_dim
        self.num_joints = num_joints
        
        # Get prediction order and kinematic tree for this part
        self.prediction_order, self.kinematic_tree = get_soke_prediction_order(part)
        self.prediction_order = self.prediction_order[:num_joints]
        
        # Create predictor for each joint
        self.predictors = nn.ModuleList()
        for joint_idx in self.prediction_order:
            parents = self.kinematic_tree.get(joint_idx, [])
            # Filter parents to only include joints already in prediction order
            idx = self.prediction_order.index(joint_idx)
            valid_parents = [p for p in parents if p in self.prediction_order[:idx]] if idx > 0 else []
            predictor_input_dim = input_dim + len(valid_parents) * joint_dim
            self.predictors.append(JointPredictor(
                predictor_input_dim, hidden_dim, joint_dim, num_layers
            ))

    def forward(self, x):
        """
        Args:
            x: (B*T, H) hidden features
        Returns:
            (B*T, num_joints * joint_dim)
        """
        joint_outputs = {}
        
        for idx, joint_idx in enumerate(self.prediction_order):
            parents = self.kinematic_tree.get(joint_idx, [])
            valid_parents = [p for p in parents if p in joint_outputs]
            
            if valid_parents:
                parent_feats = torch.cat([joint_outputs[p] for p in valid_parents], dim=-1)
                predictor_input = torch.cat([x, parent_feats], dim=-1)
            else:
                predictor_input = x
            
            joint_outputs[joint_idx] = self.predictors[idx](predictor_input)
        
        # Concatenate in order
        output = torch.cat([joint_outputs[j] for j in self.prediction_order], dim=-1)
        return output


class SOKE_SPL(nn.Module):
    """
    Full SPL for SOKE 133 dims format.
    Separate kinematic chains for body, lhand, rhand.
    Expression is predicted directly (no kinematic structure).
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        
        # Body SPL (10 joints)
        self.body_spl = PartSPL(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_joints=NUM_UPPER_BODY_JOINTS, joint_dim=3,
            num_layers=num_layers, part='body'
        )
        
        # Left hand SPL (15 joints)
        self.lhand_spl = PartSPL(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_joints=NUM_HAND_JOINTS, joint_dim=3,
            num_layers=num_layers, part='hand'
        )
        
        # Right hand SPL (15 joints)
        self.rhand_spl = PartSPL(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_joints=NUM_HAND_JOINTS, joint_dim=3,
            num_layers=num_layers, part='hand'
        )
        
        # Jaw predictor (1 joint, no kinematic chain)
        self.jaw_pred = JointPredictor(input_dim, hidden_dim, SOKE_JAW_DIM, num_layers)
        
        # Expression predictor (10 dims, no kinematic structure)
        self.expr_pred = JointPredictor(input_dim, hidden_dim, SOKE_EXPR_DIM, num_layers)

    def forward(self, x):
        """
        Args:
            x: (B*T, H) hidden features
        Returns:
            (B*T, 133) SOKE format output
        """
        body = self.body_spl(x)      # (B*T, 30)
        lhand = self.lhand_spl(x)    # (B*T, 45)
        rhand = self.rhand_spl(x)    # (B*T, 45)
        jaw = self.jaw_pred(x)       # (B*T, 3)
        expr = self.expr_pred(x)     # (B*T, 10)
        
        # Concatenate in SOKE order: body + lhand + rhand + jaw + expr
        output = torch.cat([body, lhand, rhand, jaw, expr], dim=-1)
        return output


# =============================================================================
# Main Model: SPL-VQVAE v2 (SOKE format)
# =============================================================================

class SPL_VQVAE_V2(nn.Module):
    """
    SPL-VQVAE v2 for SOKE 133 dims format.
    
    Input format: (B, T, 133)
        - upper_body: 30 dims (10 joints * 3)
        - lhand: 45 dims (15 joints * 3)
        - rhand: 45 dims (15 joints * 3)
        - jaw: 3 dims (1 joint * 3)
        - expression: 10 dims
    """
    def __init__(
        self,
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_dim=2048,
        num_queries=32,
        codebook_size=512,
        codebook_dim=512,
        max_len=300,
        dropout=0.1,
        spl_hidden_dim=256,
        spl_num_layers=2,
        commitment_weight=0.25,
        nfeats=133,  # SOKE format
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.max_len = max_len
        self.nfeats = nfeats
        
        # SOKE part dimensions
        self.body_dim = SOKE_UPPER_BODY_DIM + SOKE_JAW_DIM  # 30 + 3 = 33
        self.lhand_dim = SOKE_LHAND_DIM   # 45
        self.rhand_dim = SOKE_RHAND_DIM   # 45
        self.expr_dim = SOKE_EXPR_DIM     # 10
        self.total_dim = SOKE_TOTAL_DIM   # 133
        self.num_parts = 4  # body+jaw, lhand, rhand, expr
        
        # ===================== ENCODER =====================
        # 1. Part Embeddings
        self.body_embed = nn.Linear(self.body_dim, embed_dim)
        self.lhand_embed = nn.Linear(self.lhand_dim, embed_dim)
        self.rhand_embed = nn.Linear(self.rhand_dim, embed_dim)
        self.expr_embed = nn.Linear(self.expr_dim, embed_dim)
        
        # 2. Positional Embeddings
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_parts, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        # 3. Spatial Transformer (inter-part)
        self.spatial_encoder = TransformerEncoder(
            dim=embed_dim, depth=max(1, depth // 2), num_heads=num_heads,
            mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 4. Merge Projection
        self.merge_proj = nn.Linear(embed_dim * self.num_parts, embed_dim)
        
        # 5. Q-Former Encoder
        self.qformer_encoder = QFormer(
            num_queries=num_queries, dim=embed_dim, depth=max(1, depth // 2),
            num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout
        )
        
        # ===================== QUANTIZER =====================
        self.quantizer = VectorQuantize(
            dim=embed_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            commitment_weight=commitment_weight,
            decay=0.99,
            eps=1e-5,
            kmeans_init=True,
            kmeans_iters=10,
            threshold_ema_dead_code=2,
        )
        
        # ===================== DECODER =====================
        # 1. Q-Former Decoder
        self.qformer_decoder = QFormer(
            num_queries=max_len, dim=embed_dim, depth=max(1, depth // 2),
            num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 2. Split Projection
        self.split_proj = nn.Linear(embed_dim, embed_dim * self.num_parts)
        
        # 3. Spatial Transformer (decoder)
        self.spatial_decoder = TransformerEncoder(
            dim=embed_dim, depth=max(1, depth // 2), num_heads=num_heads,
            mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 4. SPL (SOKE kinematic chain prediction)
        self.spl = SOKE_SPL(
            input_dim=embed_dim, hidden_dim=spl_hidden_dim,
            num_layers=spl_num_layers
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def _split_input(self, x):
        """Split input (B, T, 133) into body+jaw, lhand, rhand, expr"""
        # body: [0:30], lhand: [30:75], rhand: [75:120], jaw: [120:123], expr: [123:133]
        body = torch.cat([
            x[:, :, SOKE_BODY_START:SOKE_BODY_END],  # upper body (30)
            x[:, :, SOKE_JAW_START:SOKE_JAW_END]      # jaw (3)
        ], dim=-1)  # (B, T, 33)
        
        lhand = x[:, :, SOKE_LHAND_START:SOKE_LHAND_END]  # (B, T, 45)
        rhand = x[:, :, SOKE_RHAND_START:SOKE_RHAND_END]  # (B, T, 45)
        expr = x[:, :, SOKE_EXPR_START:SOKE_EXPR_END]     # (B, T, 10)
        
        return body, lhand, rhand, expr

    def encode(self, x, lengths=None):
        """
        Encode motion sequence to quantized tokens.
        
        Args:
            x: (B, T, 133) SOKE format motion
            lengths: (B,) sequence lengths (optional)
        
        Returns:
            quantized: (B, N, C)
            indices: (B, N)
            commit_loss: scalar
        """
        B, T, D = x.shape
        device = x.device
        
        # 1. Split into parts and embed
        body, lhand, rhand, expr = self._split_input(x)
        
        body_emb = self.body_embed(body)      # (B, T, H)
        lhand_emb = self.lhand_embed(lhand)
        rhand_emb = self.rhand_embed(rhand)
        expr_emb = self.expr_embed(expr)
        
        # Stack: (B, T, 4, H)
        parts = torch.stack([body_emb, lhand_emb, rhand_emb, expr_emb], dim=2)
        
        # 2. Spatial Transformer (per timestep)
        parts = rearrange(parts, 'b t n h -> (b t) n h')
        parts = parts + self.spatial_pos_embed
        parts = self.spatial_encoder(parts)
        
        # 3. Merge parts
        parts = rearrange(parts, '(b t) n h -> b t (n h)', b=B, t=T)
        merged = self.merge_proj(parts)  # (B, T, H)
        
        # Add temporal positional embedding
        merged = merged + self.temporal_pos_embed[:, :T, :]
        
        # 4. Q-Former
        qformer_out = self.qformer_encoder(merged)  # (B, N, H)
        
        # 5. Quantize
        quantized, indices, commit_loss = self.quantizer(qformer_out)
        
        return quantized, indices, commit_loss

    def decode(self, quantized, target_length):
        """
        Decode quantized tokens to motion sequence.
        
        Args:
            quantized: (B, N, C) quantized features
            target_length: int
        
        Returns:
            output: (B, T, 133) SOKE format
        """
        B = quantized.shape[0]
        T = target_length
        
        # 1. Q-Former Decoder
        decoded = self.qformer_decoder.queries[:, :T, :].expand(B, -1, -1)
        for layer in self.qformer_decoder.layers:
            decoded = layer(decoded, quantized)
        decoded = self.qformer_decoder.norm(decoded)  # (B, T, H)
        
        # 2. Split to parts
        split = self.split_proj(decoded)  # (B, T, 4*H)
        split = rearrange(split, 'b t (n h) -> (b t) n h', n=self.num_parts)
        
        # 3. Spatial Transformer (decoder)
        split = split + self.spatial_pos_embed
        parts = self.spatial_decoder(split)
        parts = rearrange(parts, '(b t) n h -> b t n h', b=B, t=T)
        
        # 4. SPL prediction
        # Use mean of part features
        merged = parts.mean(dim=2)  # (B, T, H)
        merged = rearrange(merged, 'b t h -> (b t) h')
        
        soke_output = self.spl(merged)  # (B*T, 133)
        soke_output = rearrange(soke_output, '(b t) d -> b t d', b=B, t=T)
        
        return soke_output

    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: (B, T, 133) SOKE format motion
            lengths: (B,) sequence lengths
        
        Returns:
            output: (B, T, 133)
            commit_loss: scalar
            perplexity: scalar
            indices: (B, N)
        """
        B, T, D = x.shape
        
        # Encode
        quantized, indices, commit_loss = self.encode(x, lengths)
        
        # Decode
        output = self.decode(quantized, T)
        
        # Perplexity
        with torch.no_grad():
            unique_codes = indices.unique()
            perplexity = torch.tensor(float(len(unique_codes)), device=x.device)
        
        return output, commit_loss, perplexity, indices

    def encode_to_codes(self, x, lengths=None):
        """Encode motion to discrete codes"""
        _, indices, _ = self.encode(x, lengths)
        return indices

    def decode_from_codes(self, codes, target_length):
        """Decode from discrete codes"""
        quantized = self.quantizer.get_codes_from_indices(codes)
        return self.decode(quantized, target_length)


# =============================================================================
# Loss Functions
# =============================================================================

class SPLVQVAELoss(nn.Module):
    """Combined loss for SPL-VQVAE"""
    def __init__(
        self,
        lambda_recon=1.0,
        lambda_velocity=0.5,
        lambda_commit=0.25,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_velocity = lambda_velocity
        self.lambda_commit = lambda_commit
    
    def forward(self, pred, target, commit_loss, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1)
            pred = pred * mask
            target = target * mask
        
        # Reconstruction loss
        recon_loss = F.l1_loss(pred, target)
        
        # Velocity loss
        if self.lambda_velocity > 0:
            pred_vel = pred[:, 1:] - pred[:, :-1]
            target_vel = target[:, 1:] - target[:, :-1]
            if mask is not None:
                vel_mask = mask[:, 1:] * mask[:, :-1]
                pred_vel = pred_vel * vel_mask
                target_vel = target_vel * vel_mask
            velocity_loss = F.l1_loss(pred_vel, target_vel)
        else:
            velocity_loss = torch.tensor(0.0, device=pred.device)
        
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_velocity * velocity_loss +
            self.lambda_commit * commit_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'velocity_loss': velocity_loss,
            'commit_loss': commit_loss,
        }


# =============================================================================
# Test
# =============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Model
    model = SPL_VQVAE_V2(
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_dim=1024,
        num_queries=32,
        codebook_size=512,
        codebook_dim=256,
        max_len=300,
        spl_hidden_dim=256,
        spl_num_layers=2,
        nfeats=133,  # SOKE format
    ).to(device)
    
    print(f"\nParameters: {count_parameters(model):,}")
    
    # Print kinematic info for each part
    body_order, body_tree = get_soke_prediction_order('body')
    hand_order, hand_tree = get_soke_prediction_order('hand')
    print(f"\nBody joints: {len(body_order)}, order: {body_order}")
    print(f"Hand joints: {len(hand_order)}, order: {hand_order}")
    
    # Input: SOKE 133 dims
    B, T = 4, 100
    x = torch.randn(B, T, SOKE_TOTAL_DIM).to(device)  # 133 dims
    lengths = torch.tensor([100, 80, 60, 100]).to(device)
    
    print(f"\nInput: {x.shape} (SOKE 133 dims)")
    print(f"  Body: {SOKE_UPPER_BODY_DIM} dims (10 joints * 3)")
    print(f"  LHand: {SOKE_LHAND_DIM} dims (15 joints * 3)")
    print(f"  RHand: {SOKE_RHAND_DIM} dims (15 joints * 3)")
    print(f"  Jaw: {SOKE_JAW_DIM} dims (1 joint * 3)")
    print(f"  Expr: {SOKE_EXPR_DIM} dims")
    
    # Forward
    model.train()
    output, commit_loss, perplexity, codes = model(x, lengths)
    
    print(f"\nOutput: {output.shape}")
    print(f"Codes: {codes.shape}")
    print(f"Commit loss: {commit_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.1f}")
    
    # Loss
    loss_fn = SPLVQVAELoss()
    losses = loss_fn(output, x, commit_loss)
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Eval
    model.eval()
    with torch.no_grad():
        codes = model.encode_to_codes(x, lengths)
        recon = model.decode_from_codes(codes, T)
    print(f"\nEncode/decode: {recon.shape}")
    
    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_model()