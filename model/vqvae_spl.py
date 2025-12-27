"""
SPL-VQVAE: Structured Prediction Layer based VQ-VAE for Sign Language
- SMPL-X compatible skeleton structure
- Temporal Compression + Q-Former + EMA Reset Codebook

Architecture:
    Input (B, T, J*3)
        ↓
    SPL Encoder (Spatial Transformer)
        ↓
    Temporal Compression (Conv1D)
        ↓
    Q-Former (Learnable Queries → Fixed N tokens)
        ↓
    EMA Reset Codebook (SOKE style)
        ↓
    Q-Former Decoder
        ↓
    Temporal Upsampling
        ↓
    SPL Decoder
        ↓
    Output (B, T, J*3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange, repeat
import einops


# =============================================================================
# 1. SMPL-X Skeleton Definitions for Sign Language (Upper Body + Hands)
# =============================================================================

# SMPL-X Upper Body Skeleton (excluding lower body)
# Format: [(parent_idx, child_idx, name), ...]
# -1 means root (no parent)

# Upper Body: 10 joints for sign language
# pelvis(0) -> spine3(1) -> neck(2) -> head(3)
#                        -> left_shoulder(4) -> left_elbow(5) -> left_wrist(6)
#                        -> right_shoulder(7) -> right_elbow(8) -> right_wrist(9)

SMPLX_UPPER_BODY_SKELETON = [
    [(-1, 0, "pelvis")],
    [(0, 1, "spine3")],
    [(1, 2, "neck"), (1, 4, "left_shoulder"), (1, 7, "right_shoulder")],
    [(2, 3, "head")],
    [(4, 5, "left_elbow"), (7, 8, "right_elbow")],
    [(5, 6, "left_wrist"), (8, 9, "right_wrist")],
]

# SMPL-X Hand Skeleton (15 joints each, 0 is wrist from body)
# Wrist connects to 5 fingers, each with 3 joints
SMPLX_HAND_SKELETON = [
    [(-1, 0, "wrist")],
    [(0, 1, "thumb0"), (0, 4, "index0"), (0, 7, "middle0"), (0, 10, "ring0"), (0, 13, "pinky0")],
    [(1, 2, "thumb1"), (4, 5, "index1"), (7, 8, "middle1"), (10, 11, "ring1"), (13, 14, "pinky1")],
    [(2, 3, "thumb2"), (5, 6, "index2"), (8, 9, "middle2"), (11, 12, "ring2"), (14, 15, "pinky2")],
]

# Joint counts
NUM_BODY_JOINTS = 10   # Upper body
NUM_HAND_JOINTS = 15   # Each hand (excluding wrist, which is in body)
NUM_TOTAL_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS * 2  # 10 + 15 + 15 = 40

# =============================================================================
# SOKE 133 dims Structure (Axis-Angle Rotations)
# =============================================================================
# SOKE uses filtered SMPL-X axis-angle parameters:
#   - upper_body_pose: 30 dims (10 joints * 3) - indices [0:30]
#   - lhand_pose: 45 dims (15 joints * 3) - indices [30:75]
#   - rhand_pose: 45 dims (15 joints * 3) - indices [75:120]
#   - jaw_pose: 3 dims (1 joint * 3) - indices [120:123]
#   - expression: 10 dims - indices [123:133]

SOKE_UPPER_BODY_DIM = 30   # 10 joints * 3
SOKE_LHAND_DIM = 45        # 15 joints * 3
SOKE_RHAND_DIM = 45        # 15 joints * 3
SOKE_JAW_DIM = 3           # 1 joint * 3
SOKE_EXPR_DIM = 10         # expression coefficients
SOKE_TOTAL_DIM = 133       # 30 + 45 + 45 + 3 + 10

# Part indices for SOKE 133 dims
SOKE_BODY_SLICE = slice(0, 30)           # upper_body
SOKE_LHAND_SLICE = slice(30, 75)         # left hand
SOKE_RHAND_SLICE = slice(75, 120)        # right hand
SOKE_JAW_SLICE = slice(120, 123)         # jaw
SOKE_EXPR_SLICE = slice(123, 133)        # expression


# =============================================================================
# 2. Basic Building Blocks
# =============================================================================

class BertLayerNorm(nn.Module):
    """TF-style LayerNorm (epsilon inside sqrt)"""
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
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
        super().__init__()
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


class CrossAttention(nn.Module):
    """Cross Attention for Q-Former"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, mask=None):
        """
        Args:
            q: (B, N_q, C) - queries
            kv: (B, N_kv, C) - keys and values
            mask: optional attention mask
        """
        B, N_q, C = q.shape
        N_kv = kv.shape[1]

        q = self.linear_q(q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = BertLayerNorm(dim)
        self.attn = Attention(heads, dim)
        self.norm2 = BertLayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, heads, mlp_dim, dropout) 
            for _ in range(depth)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# =============================================================================
# 3. Structured Prediction Layer (SPL) for SMPL-X
# =============================================================================

class SPBlock(nn.Module):
    """Single joint prediction block"""
    def __init__(self, input_size, hid_size, out_size, num_layers):
        super().__init__()
        layers = [nn.Linear(input_size, hid_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hid_size, hid_size), nn.ReLU()])
        layers.append(nn.Linear(hid_size, out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SPL(nn.Module):
    """Structured Prediction Layer for SMPL-X skeleton"""
    def __init__(self, input_size, hidden_layers, hidden_units, joint_size, skeleton_type):
        super().__init__()
        self.input_size = input_size
        self.joint_size = joint_size
        
        if skeleton_type == "smplx_body":
            self.skeleton = SMPLX_UPPER_BODY_SKELETON
            self.num_joints = NUM_BODY_JOINTS
        elif skeleton_type == "smplx_hand":
            self.skeleton = SMPLX_HAND_SKELETON
            self.num_joints = NUM_HAND_JOINTS + 1  # +1 for wrist (index 0)
        else:
            raise ValueError(f"{skeleton_type} is not a valid skeleton type!")
        
        # Build kinematic tree
        kinematic_tree = {}
        for layer in self.skeleton:
            for entry in layer:
                parent_idx, child_idx, name = entry
                parents = [parent_idx] if parent_idx > -1 else []
                kinematic_tree[child_idx] = [parents, name]
        
        self.prediction_order = sorted(kinematic_tree.keys())
        self.indexed_skeleton = {
            i: [kinematic_tree[i][0], i, kinematic_tree[i][1]] 
            for i in self.prediction_order
        }
        
        # Create prediction modules for each joint
        self.joint_predictions = nn.ModuleList()
        for joint_key in self.prediction_order:
            parent_ids, _, _ = self.indexed_skeleton[joint_key]
            current_input_size = self.input_size + joint_size * len(parent_ids)
            self.joint_predictions.append(
                SPBlock(current_input_size, hidden_units, joint_size, hidden_layers)
            )

    def forward(self, x):
        """
        Args:
            x: (B, T, H) or (B*T, H)
        Returns:
            (B, T, num_joints * joint_size) or (B*T, num_joints * joint_size)
        """
        out = {}
        for idx, joint_key in enumerate(self.prediction_order):
            parent_ids, _, _ = self.indexed_skeleton[joint_key]
            parent_feats = [out[i] for i in parent_ids]
            x_input = torch.cat([x] + parent_feats, dim=-1) if parent_feats else x
            out[joint_key] = self.joint_predictions[idx](x_input)
        
        return torch.cat([out[i] for i in self.prediction_order], dim=-1)


# =============================================================================
# 4. Temporal Compression / Upsampling (SOKE-style Conv1D)
# =============================================================================

class Resnet1D(nn.Module):
    """1D ResNet block for temporal modeling"""
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, activation='relu', norm=None):
        super().__init__()
        blocks = []
        dilate = 1
        
        for i in range(n_depth):
            block = []
            if norm == 'batch':
                block.append(nn.BatchNorm1d(n_in))
            elif norm == 'layer':
                block.append(nn.LayerNorm(n_in))
            
            if activation == 'relu':
                block.append(nn.ReLU())
            elif activation == 'gelu':
                block.append(nn.GELU())
            
            block.append(nn.Conv1d(n_in, n_in, 3, 1, dilate, dilation=dilate))
            dilate *= dilation_growth_rate
            blocks.append(nn.Sequential(*block))
        
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class TemporalEncoder(nn.Module):
    """Temporal compression using Conv1D (SOKE-style)"""
    def __init__(self, input_dim, output_dim, down_t=2, stride_t=2, width=512, depth=3):
        super().__init__()
        blocks = []
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        
        blocks.append(nn.Conv1d(input_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            blocks.append(nn.Conv1d(width, width, filter_t, stride_t, pad_t))
            blocks.append(Resnet1D(width, depth, activation='relu'))
        
        blocks.append(nn.Conv1d(width, output_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        
        self.down_factor = stride_t ** down_t

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            (B, T//down_factor, C)
        """
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.model(x)
        x = x.permute(0, 2, 1)  # (B, T', C)
        return x


class TemporalDecoder(nn.Module):
    """Temporal upsampling using ConvTranspose1D"""
    def __init__(self, input_dim, output_dim, up_t=2, stride_t=2, width=512, depth=3):
        super().__init__()
        blocks = []
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        
        blocks.append(nn.Conv1d(input_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(up_t):
            blocks.append(nn.ConvTranspose1d(width, width, filter_t, stride_t, pad_t))
            blocks.append(Resnet1D(width, depth, activation='relu'))
        
        blocks.append(nn.Conv1d(width, output_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        
        self.up_factor = stride_t ** up_t

    def forward(self, x):
        """
        Args:
            x: (B, T', C)
        Returns:
            (B, T'*up_factor, C)
        """
        x = x.permute(0, 2, 1)  # (B, C, T')
        x = self.model(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        return x


# =============================================================================
# 5. Q-Former (Learnable Query based compression)
# =============================================================================

class QFormerEncoderLayer(nn.Module):
    """Q-Former layer with self-attention and cross-attention"""
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        # Self-attention on queries
        self.self_attn = Attention(num_heads, dim)
        self.norm1 = BertLayerNorm(dim)
        
        # Cross-attention: queries attend to encoder output
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias=True, 
                                         attn_drop=dropout, proj_drop=dropout)
        self.norm2 = BertLayerNorm(dim)
        
        # FFN
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm3 = BertLayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, encoder_output, encoder_mask=None):
        """
        Args:
            queries: (B, N, C) - learnable queries
            encoder_output: (B, T, C) - encoder features
            encoder_mask: optional mask for encoder output
        Returns:
            (B, N, C)
        """
        # Self-attention on queries
        q_norm = self.norm1(queries)
        queries = queries + self.dropout(self.self_attn(q_norm, q_norm, q_norm))
        
        # Cross-attention: queries attend to encoder output
        queries = queries + self.dropout(self.cross_attn(self.norm2(queries), encoder_output, encoder_mask))
        
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        
        return queries


class QFormer(nn.Module):
    """Q-Former: Compress variable-length sequence to fixed N tokens"""
    def __init__(self, num_queries, dim, depth, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        
        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerEncoderLayer(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.norm = BertLayerNorm(dim)

    def forward(self, encoder_output, encoder_mask=None):
        """
        Args:
            encoder_output: (B, T, C) - temporal compressed features
            encoder_mask: (B, T) - optional mask
        Returns:
            (B, N, C) - fixed N query outputs
        """
        B = encoder_output.shape[0]
        
        # Expand learnable queries for batch
        queries = self.queries.expand(B, -1, -1)
        
        # Process through Q-Former layers
        for layer in self.layers:
            queries = layer(queries, encoder_output, encoder_mask)
        
        return self.norm(queries)


# =============================================================================
# 6. EMA Reset Quantizer (SOKE-style)
# =============================================================================

class QuantizeEMAReset(nn.Module):
    """EMA-updated codebook with reset for dead codes (SOKE implementation)"""
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.init = False
        
        # Register all buffers with correct shapes
        self.register_buffer('codebook', torch.randn(nb_code, code_dim))
        self.register_buffer('code_sum', torch.zeros(nb_code, code_dim))
        self.register_buffer('code_count', torch.ones(nb_code))
        
    def reset_codebook(self):
        self.init = False
        self.codebook.zero_()
        self.code_sum.zero_()
        self.code_count.fill_(1.0)

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook.copy_(out[:self.nb_code])
        self.code_sum.copy_(self.codebook)
        self.code_count.fill_(1.0)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # EMA update (in-place)
        self.code_sum.mul_(self.mu).add_(code_sum, alpha=1. - self.mu)
        self.code_count.mul_(self.mu).add_(code_count, alpha=1. - self.mu)

        # Reset dead codes
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum / (self.code_count.view(self.nb_code, 1) + 1e-8)
        self.codebook.copy_(usage * code_update + (1 - usage) * code_rand)
        
        prob = code_count / torch.sum(code_count + 1e-8)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    def quantize(self, x):
        """Find nearest codebook entries"""
        # x: (N, C)
        k_w = self.codebook.t()  # (C, nb_code)
        distance = (
            torch.sum(x ** 2, dim=-1, keepdim=True) 
            - 2 * torch.matmul(x, k_w) 
            + torch.sum(k_w ** 2, dim=0, keepdim=True)
        )
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        """Lookup codebook entries"""
        return F.embedding(code_idx, self.codebook)

    def forward(self, x):
        """
        Args:
            x: (B, N, C) - Q-Former output (N fixed tokens)
        Returns:
            x_quantized: (B, N, C)
            commit_loss: scalar
            perplexity: scalar
        """
        B, N, C = x.shape
        x_flat = x.reshape(-1, C)  # (B*N, C)
        
        # Initialize codebook on first forward pass
        if self.training and not self.init:
            self.init_codebook(x_flat)

        # Quantize
        code_idx = self.quantize(x_flat)
        x_d = self.dequantize(code_idx)

        # Update codebook (EMA) during training
        if self.training:
            perplexity = self.update_codebook(x_flat, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        
        # Commitment loss
        commit_loss = F.mse_loss(x_flat, x_d.detach())
        
        # Straight-through estimator
        x_d = x_flat + (x_d - x_flat).detach()
        x_d = x_d.view(B, N, C)
        
        return x_d, commit_loss, perplexity, code_idx.view(B, N)


# =============================================================================
# 7. Main Model: SPL-VQVAE
# =============================================================================

class SPL_VQVAE(nn.Module):
    """
    SPL-VQVAE: Structured Prediction Layer based VQ-VAE for Sign Language
    
    Supports two input formats:
    - 'joint_coords': (B, T, J*3) - 3D joint coordinates (J joints * 3 dims)
    - 'soke': (B, T, 133) - SOKE axis-angle format
    
    Args:
        embed_dim: Hidden dimension
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_dim: FFN hidden dimension
        num_queries: Number of Q-Former queries (fixed output tokens)
        codebook_size: Number of codebook entries
        down_t: Temporal downsampling factor (2^down_t)
        max_len: Maximum sequence length
        input_format: 'joint_coords' or 'soke'
        nfeats: Input feature dimension (auto-detected if None)
    """
    def __init__(
        self,
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_dim=2048,
        num_queries=32,      # Fixed number of output tokens
        codebook_size=512,   # Codebook size
        code_dim=512,        # Codebook dimension
        down_t=2,            # Temporal downsampling
        stride_t=2,
        max_len=300,
        dropout=0.1,
        spl_hidden_layers=3,
        spl_hidden_units=512,
        input_format='soke',  # 'joint_coords' or 'soke'
        nfeats=133,           # Input feature dimension
    ):
        super().__init__()
        
        self.use_cuda = True
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.down_t = down_t
        self.stride_t = stride_t
        self.down_factor = stride_t ** down_t
        self.max_len = max_len
        self.input_format = input_format
        self.nfeats = nfeats
        
        # Set part dimensions based on input format
        if input_format == 'soke' or nfeats == 133:
            self.input_format = 'soke'
            # SOKE 133 dims: body(33) + lhand(45) + rhand(45) + expr(10)
            # We combine upper_body(30) + jaw(3) = body(33)
            self.body_dim = SOKE_UPPER_BODY_DIM + SOKE_JAW_DIM  # 33
            self.lhand_dim = SOKE_LHAND_DIM   # 45
            self.rhand_dim = SOKE_RHAND_DIM   # 45
            self.expr_dim = SOKE_EXPR_DIM     # 10
            self.total_dim = SOKE_TOTAL_DIM   # 133
            self.num_parts = 4  # body, lhand, rhand, expr
        else:
            # Joint coordinates format
            self.body_dim = NUM_BODY_JOINTS * 3       # 10 * 3 = 30
            self.lhand_dim = (NUM_HAND_JOINTS + 1) * 3  # 16 * 3 = 48
            self.rhand_dim = (NUM_HAND_JOINTS + 1) * 3  # 16 * 3 = 48
            self.expr_dim = 0
            self.total_dim = self.body_dim + self.lhand_dim + self.rhand_dim
            self.num_parts = 3  # body, lhand, rhand
        
        # ===================== ENCODER =====================
        # 1. Part Embeddings
        self.body_emb = nn.Linear(self.body_dim, embed_dim)
        self.lhand_emb = nn.Linear(self.lhand_dim, embed_dim)
        self.rhand_emb = nn.Linear(self.rhand_dim, embed_dim)
        if self.expr_dim > 0:
            self.expr_emb = nn.Linear(self.expr_dim, embed_dim)
        
        # 2. Positional Embeddings
        self.spa_pos_emb = nn.Parameter(torch.zeros(1, self.num_parts, embed_dim))
        self.tem_pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        # 3. Spatial Transformer (inter-part correlation)
        self.enc_spa_transformer = TransformerEncoder(
            dim=embed_dim, depth=depth//2, heads=num_heads, 
            mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 4. Merge Projection: N parts -> 1
        self.merge_proj = nn.Linear(embed_dim * self.num_parts, embed_dim)
        
        # 5. Temporal Transformer
        self.enc_tem_transformer = TransformerEncoder(
            dim=embed_dim, depth=depth, heads=num_heads,
            mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 6. Temporal Compression (Conv1D)
        self.temporal_encoder = TemporalEncoder(
            input_dim=embed_dim, output_dim=embed_dim,
            down_t=down_t, stride_t=stride_t, width=embed_dim, depth=3
        )
        
        # 7. Q-Former (compress to fixed N tokens)
        self.qformer = QFormer(
            num_queries=num_queries, dim=embed_dim, depth=depth//2,
            num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout
        )
        
        # ===================== QUANTIZER =====================
        # Note: code_dim should match embed_dim since Q-Former outputs embed_dim
        self.quantizer = QuantizeEMAReset(codebook_size, embed_dim, mu=0.99)
        
        # ===================== DECODER =====================
        # Calculate max compressed length
        max_compressed_len = max_len // self.down_factor + 1
        
        # 1. Q-Former Decoder (cross-attention from decoder tokens to quantized)
        self.dec_queries = nn.Parameter(torch.zeros(1, max_compressed_len, embed_dim))
        self.qformer_decoder = QFormer(
            num_queries=max_compressed_len, dim=embed_dim, depth=depth//2,
            num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 2. Temporal Upsampling
        self.temporal_decoder = TemporalDecoder(
            input_dim=embed_dim, output_dim=embed_dim,
            up_t=down_t, stride_t=stride_t, width=embed_dim, depth=3
        )
        
        # 3. Temporal Transformer (decoder)
        self.dec_tem_transformer = TransformerEncoder(
            dim=embed_dim, depth=depth, heads=num_heads,
            mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 4. Split Projection: 1 -> N parts
        self.split_proj = nn.Linear(embed_dim, embed_dim * self.num_parts)
        
        # 5. Spatial Transformer (decoder)
        self.dec_spa_transformer = TransformerEncoder(
            dim=embed_dim, depth=depth//2, heads=num_heads,
            mlp_dim=mlp_dim, dropout=dropout
        )
        
        # 6. Output Heads
        if self.input_format == 'soke':
            # Direct linear projection for SOKE format (axis-angle)
            self.body_head = nn.Linear(embed_dim, self.body_dim)
            self.lhand_head = nn.Linear(embed_dim, self.lhand_dim)
            self.rhand_head = nn.Linear(embed_dim, self.rhand_dim)
            self.expr_head = nn.Linear(embed_dim, self.expr_dim)
        else:
            # SPL Heads for joint coordinates
            self.body_spl = SPL(
                input_size=embed_dim, hidden_layers=spl_hidden_layers,
                hidden_units=spl_hidden_units, joint_size=3, skeleton_type="smplx_body"
            )
            self.hand_spl = SPL(
                input_size=embed_dim, hidden_layers=spl_hidden_layers,
                hidden_units=spl_hidden_units, joint_size=3, skeleton_type="smplx_hand"
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.spa_pos_emb, std=0.02)
        nn.init.trunc_normal_(self.tem_pos_emb, std=0.02)
        nn.init.trunc_normal_(self.dec_queries, std=0.02)
    
    def _get_mask(self, lengths, max_len, device):
        """Create attention mask from lengths"""
        pos = torch.arange(0, max_len, device=device).unsqueeze(0)
        mask = pos < lengths.unsqueeze(1)
        return mask
    
    def _split_parts_soke(self, x):
        """Split SOKE 133 dims into parts"""
        # x: (B, T, 133)
        body = torch.cat([
            x[:, :, SOKE_BODY_SLICE],   # upper_body: 30
            x[:, :, SOKE_JAW_SLICE]     # jaw: 3
        ], dim=-1)                       # body: 33
        lhand = x[:, :, SOKE_LHAND_SLICE]   # 45
        rhand = x[:, :, SOKE_RHAND_SLICE]   # 45
        expr = x[:, :, SOKE_EXPR_SLICE]     # 10
        return body, lhand, rhand, expr
    
    def _merge_parts_soke(self, body, lhand, rhand, expr):
        """Merge parts back to SOKE 133 dims"""
        # body: (B, T, 33) -> split to upper_body(30) + jaw(3)
        upper_body = body[:, :, :30]
        jaw = body[:, :, 30:33]
        # Reconstruct: upper_body(30) + lhand(45) + rhand(45) + jaw(3) + expr(10)
        return torch.cat([upper_body, lhand, rhand, jaw, expr], dim=-1)
    
    def encode(self, pose_input, pose_length):
        """
        Encode pose sequence to quantized tokens.
        
        Args:
            pose_input: (B, T, D) - motion features (133 for SOKE, J*3 for joint_coords)
            pose_length: (B,) - sequence lengths
        
        Returns:
            quantized: (B, N, C) - quantized features
            commit_loss: scalar
            perplexity: scalar
            codes: (B, N) - codebook indices
        """
        B, T, D = pose_input.shape
        device = pose_input.device
        
        # 1. Split into parts and embed
        if self.input_format == 'soke':
            body, lhand, rhand, expr = self._split_parts_soke(pose_input)
            body_emb = self.body_emb(body).unsqueeze(2)     # (B, T, 1, H)
            lhand_emb = self.lhand_emb(lhand).unsqueeze(2)
            rhand_emb = self.rhand_emb(rhand).unsqueeze(2)
            expr_emb = self.expr_emb(expr).unsqueeze(2)
            parts_feat = torch.cat([body_emb, lhand_emb, rhand_emb, expr_emb], dim=2)  # (B, T, 4, H)
        else:
            # Original joint coordinates format
            pose_input_4d = rearrange(pose_input, "b t (j c) -> b t j c", c=3)
            body_input = pose_input_4d[:, :, :NUM_BODY_JOINTS, :].reshape(B, T, -1)
            lhand_input = pose_input_4d[:, :, NUM_BODY_JOINTS:NUM_BODY_JOINTS+NUM_HAND_JOINTS+1, :].reshape(B, T, -1)
            rhand_input = pose_input_4d[:, :, NUM_BODY_JOINTS+NUM_HAND_JOINTS+1:, :].reshape(B, T, -1)
            
            body_emb = self.body_emb(body_input).unsqueeze(2)
            lhand_emb = self.lhand_emb(lhand_input).unsqueeze(2)
            rhand_emb = self.rhand_emb(rhand_input).unsqueeze(2)
            parts_feat = torch.cat([body_emb, lhand_emb, rhand_emb], dim=2)  # (B, T, 3, H)
        
        # 2. Spatial Transformer (inter-part)
        parts_feat = rearrange(parts_feat, "b t n h -> (b t) n h")
        parts_feat = parts_feat + self.spa_pos_emb
        parts_feat = self.enc_spa_transformer(parts_feat, mask=None)
        
        # 3. Merge parts
        parts_feat = rearrange(parts_feat, "(b t) n h -> b t (n h)", b=B, t=T)
        merged_feat = self.merge_proj(parts_feat)  # (B, T, H)
        
        # 4. Temporal Transformer with length masking
        merged_feat = merged_feat + self.tem_pos_emb[:, :T, :]
        mask = self._get_mask(pose_length, T, device)
        mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
        temporal_feat = self.enc_tem_transformer(merged_feat, mask=mask)
        
        # 5. Temporal Compression (Conv1D)
        compressed_feat = self.temporal_encoder(temporal_feat)  # (B, T//down_factor, H)
        
        # 6. Q-Former (compress to N tokens)
        qformer_out = self.qformer(compressed_feat)  # (B, N, H)
        
        # 7. Quantize
        quantized, commit_loss, perplexity, codes = self.quantizer(qformer_out)
        
        return quantized, commit_loss, perplexity, codes
    
    def decode(self, quantized, pose_length):
        """
        Decode quantized tokens to pose sequence.
        
        Args:
            quantized: (B, N, C) - quantized features
            pose_length: (B,) - target sequence lengths
        
        Returns:
            reconstructed: (B, T, D) - reconstructed motion
        """
        B = quantized.shape[0]
        T = int(max(pose_length).item())
        T_compressed = T // self.down_factor + 1
        device = quantized.device
        
        # 1. Q-Former Decoder (cross-attention to quantized tokens)
        dec_queries = self.dec_queries[:, :T_compressed, :].expand(B, -1, -1)
        decoded_feat = self.qformer_decoder.layers[0](dec_queries, quantized)
        for layer in self.qformer_decoder.layers[1:]:
            decoded_feat = layer(decoded_feat, quantized)
        decoded_feat = self.qformer_decoder.norm(decoded_feat)
        
        # 2. Temporal Upsampling
        upsampled_feat = self.temporal_decoder(decoded_feat)  # (B, T', H)
        upsampled_feat = upsampled_feat[:, :T, :]  # Trim to target length
        
        # Pad if needed
        if upsampled_feat.shape[1] < T:
            pad_len = T - upsampled_feat.shape[1]
            upsampled_feat = F.pad(upsampled_feat, (0, 0, 0, pad_len))
        
        # 3. Temporal Transformer (decoder) with length masking
        upsampled_feat = upsampled_feat + self.tem_pos_emb[:, :T, :]
        mask = self._get_mask(pose_length, T, device)
        mask = mask.unsqueeze(1).unsqueeze(1)
        temporal_feat = self.dec_tem_transformer(upsampled_feat, mask=mask)
        
        # 4. Split to parts
        split_feat = self.split_proj(temporal_feat)  # (B, T, num_parts*H)
        split_feat = rearrange(split_feat, "b t (n h) -> (b t) n h", n=self.num_parts)
        
        # 5. Spatial Transformer (decoder)
        split_feat = split_feat + self.spa_pos_emb
        parts_feat = self.dec_spa_transformer(split_feat, mask=None)
        parts_feat = rearrange(parts_feat, "(b t) n h -> b t n h", b=B, t=T)
        
        # 6. Output Heads
        if self.input_format == 'soke':
            body_feat = parts_feat[:, :, 0, :]
            lhand_feat = parts_feat[:, :, 1, :]
            rhand_feat = parts_feat[:, :, 2, :]
            expr_feat = parts_feat[:, :, 3, :]
            
            body_out = self.body_head(body_feat)      # (B, T, 33)
            lhand_out = self.lhand_head(lhand_feat)   # (B, T, 45)
            rhand_out = self.rhand_head(rhand_feat)   # (B, T, 45)
            expr_out = self.expr_head(expr_feat)      # (B, T, 10)
            
            reconstructed = self._merge_parts_soke(body_out, lhand_out, rhand_out, expr_out)
        else:
            body_feat = parts_feat[:, :, 0, :]
            lhand_feat = parts_feat[:, :, 1, :]
            rhand_feat = parts_feat[:, :, 2, :]
            
            body_out = self.body_spl(body_feat)    # (B, T, 10*3)
            lhand_out = self.hand_spl(lhand_feat)  # (B, T, 16*3)
            rhand_out = self.hand_spl(rhand_feat)  # (B, T, 16*3)
            
            reconstructed = torch.cat([body_out, lhand_out, rhand_out], dim=-1)
        
        return reconstructed
    
    def forward(self, pose_input, pose_length):
        """
        Forward pass: encode -> quantize -> decode
        
        Args:
            pose_input: (B, T, D) - motion features (D = 133 for SOKE, J*3 for joint_coords)
            pose_length: (B,) - sequence lengths
        
        Returns:
            pose_output: (B, T, D) - reconstructed motion
            commit_loss: scalar - commitment loss
            perplexity: scalar - codebook perplexity
            codes: (B, N) - codebook indices
        """
        # Encode
        quantized, commit_loss, perplexity, codes = self.encode(pose_input, pose_length)
        
        # Decode
        reconstructed = self.decode(quantized, pose_length)
        
        return reconstructed, commit_loss, perplexity, codes
    
    def encode_to_codes(self, pose_input, pose_length):
        """Encode poses to codebook indices only"""
        _, _, _, codes = self.encode(pose_input, pose_length)
        return codes
    
    def decode_from_codes(self, codes, pose_length):
        """Decode from codebook indices"""
        B, N = codes.shape
        quantized = self.quantizer.dequantize(codes.view(-1))
        quantized = quantized.view(B, N, -1)
        reconstructed = self.decode(quantized, pose_length)
        return reconstructed


# =============================================================================
# 8. Loss Functions
# =============================================================================

class SPLVQVAELoss(nn.Module):
    """Combined loss for SPL-VQVAE"""
    def __init__(
        self,
        lambda_recon=1.0,
        lambda_velocity=0.5,
        lambda_commit=0.02,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_velocity = lambda_velocity
        self.lambda_commit = lambda_commit
    
    def forward(self, pred, target, commit_loss, mask=None):
        """
        Args:
            pred: (B, T, J*3) - predicted poses
            target: (B, T, J*3) - ground truth poses
            commit_loss: scalar - commitment loss from quantizer
            mask: (B, T) - optional mask for variable length
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, T, 1)
            pred = pred * mask
            target = target * mask
        
        # Reconstruction loss (L1)
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
        
        # Total loss
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
# 9. Utility Functions
# =============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model with dummy data"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model config
    model = SPL_VQVAE(
        embed_dim=512,
        depth=4,
        num_heads=8,
        mlp_dim=2048,
        num_queries=32,
        codebook_size=512,
        code_dim=512,
        down_t=2,
        stride_t=2,
        max_len=300,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Dummy input
    B, T = 4, 100
    J = NUM_TOTAL_JOINTS  # 40
    pose_input = torch.randn(B, T, J * 3).to(device)
    pose_length = torch.tensor([100, 80, 60, 100]).to(device)
    
    # Forward pass
    model.train()
    pose_output, commit_loss, perplexity, codes = model(pose_input, pose_length)
    
    print(f"Input shape: {pose_input.shape}")
    print(f"Output shape: {pose_output.shape}")
    print(f"Codes shape: {codes.shape}")
    print(f"Commit loss: {commit_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.2f}")
    
    # Loss
    loss_fn = SPLVQVAELoss()
    losses = loss_fn(pose_output, pose_input, commit_loss)
    print(f"Total loss: {losses['total_loss'].item():.4f}")
    
    # Encode/decode test
    model.eval()
    with torch.no_grad():
        codes = model.encode_to_codes(pose_input, pose_length)
        reconstructed = model.decode_from_codes(codes, pose_length)
    print(f"Encode-decode test passed. Reconstructed shape: {reconstructed.shape}")


if __name__ == "__main__":
    test_model()