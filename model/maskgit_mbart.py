# coding: utf-8
"""
MaskGIT Sign Language Model with mBART Encoder

Location: /home/user/Projects/research/T2P/model/maskgit_mbart.py

Features:
1. mBART Encoder as text encoder (freeze → finetune option)
2. Motion token embeddings from mBART (reuse pretrained)
3. Language conditioning for Body & Hands

Architecture:
    [text tokens] + [lang_token]
              ↓
        mBART Encoder (frozen/finetune)
              ↓
        text_hidden + lang_embed
              ↓
        Length Predictor → N chunks
              ↓
        Body MaskGIT (lang-conditioned)
              ↓ body_hidden
        ┌─────┴─────┐
        ↓           ↓
    LHand MaskGIT  RHand MaskGIT
    (lang + body)  (lang + body)
              ↓
    (body, lhand, rhand) × N chunks
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List, Union
from einops import rearrange, repeat

from transformers import MBartModel, MBartTokenizer, MBartConfig


# ============================================================================
# Language Token Mapping (from SOKE)
# ============================================================================
LANG_MAP = {
    'how2sign': 'en_ASL',
    'csl': 'zh_CSL', 
    'phoenix': 'de_DGS',
}

# Special tokens added to mBART tokenizer
SPECIAL_TOKENS = {
    'motion': ['<motion_id_{}>'.format(i) for i in range(512)],  # body codes
    'hand': ['<hand_id_{}>'.format(i) for i in range(512)],      # lhand codes
    'rhand': ['<rhand_id_{}>'.format(i) for i in range(512)],    # rhand codes
    'lang': ['en_ASL', 'zh_CSL', 'de_DGS'],
}


# ============================================================================
# Positional Encoding
# ============================================================================
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pe(positions)
        return self.dropout(x)


# ============================================================================
# Chunk Length Predictor
# ============================================================================
class ChunkLengthPredictor(nn.Module):
    """
    Text + Language → Chunk 길이 예측.
    
    Chunk = (body, lhand, rhand) 트리플
    """
    
    def __init__(
        self,
        text_dim: int = 1024,
        lang_dim: int = 1024,
        hidden_dim: int = 256,
        max_chunks: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_chunks = max_chunks
        
        # Text + Lang fusion
        self.fusion = nn.Linear(text_dim + lang_dim, hidden_dim)
        
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_chunks),
        )
    
    def forward(self, text_hidden: Tensor, lang_embed: Tensor) -> Tensor:
        """
        Args:
            text_hidden: (B, T, D) or (B, D) text embeddings
            lang_embed: (B, D) language embeddings
            
        Returns:
            logits: (B, max_chunks)
        """
        if text_hidden.dim() == 3:
            text_hidden = text_hidden.mean(dim=1)  # (B, D)
        
        fused = self.fusion(torch.cat([text_hidden, lang_embed], dim=-1))
        return self.predictor(fused)
    
    def predict_length(self, text_hidden: Tensor, lang_embed: Tensor) -> Tensor:
        logits = self.forward(text_hidden, lang_embed)
        return logits.argmax(dim=-1) + 1  # 최소 1 chunk


# ============================================================================
# Attention Modules
# ============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if key is None:
            key = query
        if value is None:
            value = key
        
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        
        Q = self.w_q(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ============================================================================
# Transformer Blocks with Language Conditioning
# ============================================================================
class LangConditionedTransformerBlock(nn.Module):
    """
    Language-conditioned Transformer Block.
    
    - Self-attention
    - Cross-attention to text
    - Language embedding injection (adaptive layer norm 스타일)
    - FFN
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        lang_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to text
        self.text_cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Language conditioning (scale & shift)
        self.lang_proj = nn.Linear(lang_dim, d_model * 2)  # scale, shift
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        text_hidden: Tensor,
        lang_embed: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, T, D) input tokens
            text_hidden: (B, T_text, D_text) text encoder output
            lang_embed: (B, D_lang) language embedding
            mask: attention mask
        """
        # Language conditioning: scale & shift
        lang_cond = self.lang_proj(lang_embed)  # (B, 2*D)
        scale, shift = lang_cond.chunk(2, dim=-1)  # (B, D) each
        scale = scale.unsqueeze(1)  # (B, 1, D)
        shift = shift.unsqueeze(1)
        
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, mask=mask))
        
        # Apply language conditioning
        x = x * (1 + scale) + shift
        
        # Cross-attention to text
        x_norm = self.norm2(x)
        x = x + self.dropout(self.text_cross_attn(x_norm, text_hidden, text_hidden))
        
        # FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))
        
        return x


class HandTransformerBlock(nn.Module):
    """
    Hand Transformer Block with Body + Language conditioning.
    
    - Self-attention
    - Cross-attention to body
    - Cross-attention to text
    - Language conditioning
    - FFN
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        body_dim: int = 512,
        lang_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to body
        self.body_cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.body_proj = nn.Linear(body_dim, d_model) if body_dim != d_model else nn.Identity()
        
        # Cross-attention to text
        self.text_cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Language conditioning
        self.lang_proj = nn.Linear(lang_dim, d_model * 2)
        
        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        body_hidden: Tensor,
        text_hidden: Tensor,
        lang_embed: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Language conditioning
        lang_cond = self.lang_proj(lang_embed)
        scale, shift = lang_cond.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        
        # Self-attention
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, mask=mask))
        
        # Cross-attention to body
        body_hidden = self.body_proj(body_hidden)
        x_norm = self.norm2(x)
        x = x + self.dropout(self.body_cross_attn(x_norm, body_hidden, body_hidden))
        
        # Apply language conditioning
        x = x * (1 + scale) + shift
        
        # Cross-attention to text
        x_norm = self.norm3(x)
        x = x + self.dropout(self.text_cross_attn(x_norm, text_hidden, text_hidden))
        
        # FFN
        x = x + self.dropout(self.ffn(self.norm4(x)))
        
        return x


# ============================================================================
# Masked Transformers
# ============================================================================
class BodyMaskedTransformer(nn.Module):
    """
    Body token Masked Transformer with mBART embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 96,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        text_dim: int = 1024,
        lang_dim: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pretrained_embed: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = vocab_size  # [MASK] token
        
        # Token embedding
        if pretrained_embed is not None:
            # mBART embedding 재활용
            self.token_embed = nn.Embedding(vocab_size + 1, d_model)
            with torch.no_grad():
                self.token_embed.weight[:vocab_size] = pretrained_embed[:vocab_size]
        else:
            self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        
        self.pos_embed = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Text projection (mBART hidden → d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LangConditionedTransformerBlock(d_model, n_heads, d_ff, lang_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        tokens: Tensor,
        text_hidden: Tensor,
        lang_embed: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, T = tokens.shape
        
        # Embed tokens
        x = self.token_embed(tokens)
        x = self.pos_embed(x)
        
        # Project text
        text_hidden = self.text_proj(text_hidden)
        
        # Attention mask
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, T, -1)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, text_hidden, lang_embed, attn_mask)
        
        hidden = self.final_norm(x)
        logits = self.output_proj(hidden)
        
        return logits, hidden


class HandMaskedTransformer(nn.Module):
    """
    Hand token Masked Transformer with body + language conditioning.
    """
    
    def __init__(
        self,
        vocab_size: int = 192,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 8,
        d_ff: int = 1536,
        body_dim: int = 512,
        text_dim: int = 1024,
        lang_dim: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pretrained_embed: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = vocab_size
        
        # Token embedding
        if pretrained_embed is not None:
            self.token_embed = nn.Embedding(vocab_size + 1, d_model)
            with torch.no_grad():
                # Project from mBART dim to hand dim
                self.token_embed.weight[:vocab_size] = pretrained_embed[:vocab_size, :d_model]
        else:
            self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        
        self.pos_embed = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Text projection
        self.text_proj = nn.Linear(text_dim, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            HandTransformerBlock(d_model, n_heads, d_ff, body_dim, lang_dim, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        tokens: Tensor,
        body_hidden: Tensor,
        text_hidden: Tensor,
        lang_embed: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T = tokens.shape
        
        x = self.token_embed(tokens)
        x = self.pos_embed(x)
        
        text_hidden = self.text_proj(text_hidden)
        
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).expand(-1, T, -1)
        
        for layer in self.layers:
            x = layer(x, body_hidden, text_hidden, lang_embed, attn_mask)
        
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits


# ============================================================================
# Cosine Masking Schedule
# ============================================================================
class CosineScheduler:
    @staticmethod
    def get_mask_ratio(step: int, total_steps: int) -> float:
        r = (step + 1) / total_steps
        return math.cos(math.pi * r / 2)
    
    @staticmethod
    def get_num_masked(step: int, total_steps: int, seq_len: int) -> int:
        ratio = CosineScheduler.get_mask_ratio(step, total_steps)
        return max(1, int(math.ceil(ratio * seq_len)))


# ============================================================================
# Main Model
# ============================================================================
class MaskGITmBART(nn.Module):
    """
    MaskGIT + mBART Encoder for Sign Language Generation.
    
    Features:
    1. mBART Encoder as frozen/finetune text encoder
    2. Motion token embeddings from mBART
    3. Language conditioning for all transformers
    
    Args:
        mbart_path: path to pretrained mBART model
        freeze_encoder: whether to freeze mBART encoder
        body_vocab_size: body codebook size (default 96)
        hand_vocab_size: hand codebook size (default 192)
    """
    
    def __init__(
        self,
        mbart_path: str = "/home/user/Projects/research/T2P/deps/mbart-h2s-csl-phoenix",
        freeze_encoder: bool = True,
        
        # Codebook sizes
        body_vocab_size: int = 96,
        hand_vocab_size: int = 192,
        
        # Body transformer
        body_d_model: int = 512,
        body_n_heads: int = 8,
        body_n_layers: int = 12,
        body_d_ff: int = 2048,
        
        # Hand transformer
        hand_d_model: int = 384,
        hand_n_heads: int = 6,
        hand_n_layers: int = 8,
        hand_d_ff: int = 1536,
        
        # Common
        max_seq_len: int = 256,
        max_chunks: int = 128,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
        num_iterations: int = 10,
    ):
        super().__init__()
        
        self.body_vocab_size = body_vocab_size
        self.hand_vocab_size = hand_vocab_size
        self.max_seq_len = max_seq_len
        self.num_iterations = num_iterations
        self.label_smoothing = label_smoothing
        self.freeze_encoder = freeze_encoder
        
        # MASK token IDs
        self.body_mask_id = body_vocab_size
        self.hand_mask_id = hand_vocab_size
        
        # ========== Load mBART ==========
        print(f"Loading mBART from {mbart_path}...")
        self.tokenizer = MBartTokenizer.from_pretrained(mbart_path, legacy=True)
        mbart = MBartModel.from_pretrained(mbart_path)
        
        # ========== Add custom tokens (SOKE style) ==========
        # Add language tokens for sign languages
        new_lang_tokens = ['en_ASL', 'zh_CSL', 'de_DGS']
        self.tokenizer.add_tokens(new_lang_tokens, special_tokens=True)
        
        # Add motion tokens: <motion_id_0> to <motion_id_{vocab_size+2}> (+3 for BOS, EOS, PAD)
        all_motion_str = [f'<motion_id_{i}>' for i in range(body_vocab_size + 3)]
        all_hand_str = [f'<hand_id_{i}>' for i in range(hand_vocab_size + 3)]
        all_rhand_str = [f'<rhand_id_{i}>' for i in range(hand_vocab_size + 3)]
        
        self.tokenizer.add_tokens(all_motion_str + all_hand_str + all_rhand_str)
        
        print(f"  Added {len(new_lang_tokens)} language tokens: {new_lang_tokens}")
        print(f"  Added {len(all_motion_str)} motion, {len(all_hand_str)} lhand, {len(all_rhand_str)} rhand tokens")
        print(f"  New vocab size: {len(self.tokenizer)}")
        
        # mBART encoder
        self.mbart_encoder = mbart.get_encoder()
        self.mbart_config = mbart.config
        text_dim = self.mbart_config.d_model  # 1024 for mBART
        
        # Resize embeddings to match new tokenizer vocab
        tokenizer_vocab_size = len(self.tokenizer)
        embedding_vocab_size = self.mbart_encoder.embed_tokens.num_embeddings
        
        if tokenizer_vocab_size > embedding_vocab_size:
            print(f"  Resizing embeddings: {embedding_vocab_size} -> {tokenizer_vocab_size}")
            self.mbart_encoder.resize_token_embeddings(tokenizer_vocab_size)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.mbart_encoder.parameters():
                param.requires_grad = False
            print("  mBART encoder frozen")
        
        # ========== Extract motion embeddings from mBART ==========
        # Use the resized encoder's embeddings
        mbart_embed = self.mbart_encoder.embed_tokens.weight.data  # (vocab_size, 1024)
        
        # Get motion token embeddings
        body_embed = self._extract_motion_embeddings(mbart_embed, 'motion', body_vocab_size)
        lhand_embed = self._extract_motion_embeddings(mbart_embed, 'hand', hand_vocab_size)
        rhand_embed = self._extract_motion_embeddings(mbart_embed, 'rhand', hand_vocab_size)
        
        print(f"  Extracted embeddings - body: {body_embed.shape}, hand: {lhand_embed.shape}")
        
        # ========== Language embeddings ==========
        # Try to get language token IDs (might not exist in all tokenizers)
        self.lang_token_ids = {}
        lang_tokens_found = 0
        
        # Try SOKE custom tokens first, then standard mBART tokens
        token_variants = {
            'en_ASL': ['en_ASL', 'en_XX'],
            'zh_CSL': ['zh_CSL', 'zh_CN'],
            'de_DGS': ['de_DGS', 'de_DE'],
        }
        
        for lang, variants in token_variants.items():
            token_id = None
            for variant in variants:
                tid = self.tokenizer.convert_tokens_to_ids(variant)
                if tid != self.tokenizer.unk_token_id:
                    token_id = tid
                    print(f"  Found language token: {variant} -> {tid}")
                    break
            self.lang_token_ids[lang] = token_id
            if token_id is not None:
                lang_tokens_found += 1
        
        # Language embedding layer (from mBART if available, else random)
        self.lang_embed = nn.Embedding(3, text_dim)
        with torch.no_grad():
            for i, lang in enumerate(['en_ASL', 'zh_CSL', 'de_DGS']):
                token_id = self.lang_token_ids[lang]
                if token_id is not None and token_id < mbart_embed.shape[0]:
                    self.lang_embed.weight[i] = mbart_embed[token_id].clone()
                else:
                    # Random init
                    self.lang_embed.weight[i] = torch.randn(text_dim) * 0.02
        
        if lang_tokens_found == 0:
            print(f"  [Warning] No language tokens found, using random init for language embeddings")
        
        # Language to index mapping
        self.lang_to_idx = {'en_ASL': 0, 'zh_CSL': 1, 'de_DGS': 2,
                           'how2sign': 0, 'csl': 1, 'phoenix': 2}
        
        # ========== Length Predictor ==========
        self.length_predictor = ChunkLengthPredictor(
            text_dim=text_dim,
            lang_dim=text_dim,
            hidden_dim=256,
            max_chunks=max_chunks,
            dropout=dropout,
        )
        
        # ========== Body Transformer ==========
        # Project body embeddings to body_d_model
        body_embed_proj = self._project_embeddings(body_embed, body_d_model)
        
        self.body_transformer = BodyMaskedTransformer(
            vocab_size=body_vocab_size,
            d_model=body_d_model,
            n_heads=body_n_heads,
            n_layers=body_n_layers,
            d_ff=body_d_ff,
            text_dim=text_dim,
            lang_dim=text_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pretrained_embed=body_embed_proj,
        )
        
        # ========== Hand Transformers ==========
        lhand_embed_proj = self._project_embeddings(lhand_embed, hand_d_model)
        rhand_embed_proj = self._project_embeddings(rhand_embed, hand_d_model)
        
        self.lhand_transformer = HandMaskedTransformer(
            vocab_size=hand_vocab_size,
            d_model=hand_d_model,
            n_heads=hand_n_heads,
            n_layers=hand_n_layers,
            d_ff=hand_d_ff,
            body_dim=body_d_model,
            text_dim=text_dim,
            lang_dim=text_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pretrained_embed=lhand_embed_proj,
        )
        
        self.rhand_transformer = HandMaskedTransformer(
            vocab_size=hand_vocab_size,
            d_model=hand_d_model,
            n_heads=hand_n_heads,
            n_layers=hand_n_layers,
            d_ff=hand_d_ff,
            body_dim=body_d_model,
            text_dim=text_dim,
            lang_dim=text_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pretrained_embed=rhand_embed_proj,
        )
        
        # ========== Loss ==========
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
        
        # Clean up original mbart model
        del mbart
        
        print(f"MaskGITmBART initialized:")
        print(f"  Body: {body_vocab_size} codes, {body_d_model}d, {body_n_layers} layers")
        print(f"  Hand: {hand_vocab_size} codes, {hand_d_model}d, {hand_n_layers} layers")
    
    def _extract_motion_embeddings(
        self,
        mbart_embed: Tensor,
        token_type: str,
        vocab_size: int,
    ) -> Tensor:
        """mBART에서 motion token embedding 추출."""
        embeddings = []
        found_count = 0
        
        for i in range(vocab_size):
            if token_type == 'motion':
                token = f'<motion_id_{i}>'
            elif token_type == 'hand':
                token = f'<hand_id_{i}>'
            elif token_type == 'rhand':
                token = f'<rhand_id_{i}>'
            else:
                raise ValueError(f"Unknown token type: {token_type}")
            
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            
            # Check if token exists (not UNK) and ID is valid
            is_valid = (
                token_id is not None and 
                token_id != self.tokenizer.unk_token_id and
                token_id < mbart_embed.shape[0]
            )
            
            if is_valid:
                embeddings.append(mbart_embed[token_id].clone())
                found_count += 1
            else:
                # Fallback: Xavier-like random init
                rand_embed = torch.randn(mbart_embed.shape[1]) * 0.02
                embeddings.append(rand_embed)
        
        if found_count == 0:
            print(f"  [Warning] No {token_type} tokens found in tokenizer, using random init")
        elif found_count < vocab_size:
            print(f"  [Info] Found {found_count}/{vocab_size} {token_type} tokens, rest randomly initialized")
        
        return torch.stack(embeddings, dim=0)
    
    def _project_embeddings(self, embed: Tensor, target_dim: int) -> Tensor:
        """Embedding dimension projection."""
        if embed.shape[1] == target_dim:
            return embed
        # Simple linear projection
        proj = nn.Linear(embed.shape[1], target_dim, bias=False)
        with torch.no_grad():
            return proj(embed)
    
    def get_lang_embed(self, data_src: List[str]) -> Tensor:
        """데이터 소스 → language embedding."""
        device = self.lang_embed.weight.device
        indices = torch.tensor([self.lang_to_idx.get(src, 0) for src in data_src], device=device)
        return self.lang_embed(indices)
    
    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """mBART encoder로 텍스트 인코딩."""
        # Safety: clamp input_ids to valid range
        vocab_size = self.mbart_encoder.embed_tokens.num_embeddings
        if input_ids.max() >= vocab_size:
            print(f"[Warning] input_ids max ({input_ids.max().item()}) >= vocab_size ({vocab_size}), clamping...")
            input_ids = input_ids.clamp(0, vocab_size - 1)
        
        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.mbart_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.mbart_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        return outputs.last_hidden_state
    
    def apply_random_mask(
        self,
        tokens: Tensor,
        mask_token_id: int,
        valid_mask: Optional[Tensor] = None,
        fixed_mask_ratio: Optional[float] = None,  # NEW: fixed ratio for debugging
    ) -> Tuple[Tensor, Tensor, float]:
        """BERT-style 80-10-10 random masking."""
        B, T = tokens.shape
        device = tokens.device
        vocab_size = mask_token_id  # mask_token_id = vocab_size
        
        # Mask ratio: use fixed ratio or cosine schedule
        if fixed_mask_ratio is not None:
            mask_ratio = fixed_mask_ratio
            gamma = mask_ratio  # Direct ratio
        else:
            # Cosine schedule: sample random ratio, then apply cosine
            mask_ratio = torch.rand(1).item()
            gamma = math.cos(math.pi * mask_ratio / 2)
        
        if valid_mask is not None:
            valid_lens = valid_mask.sum(dim=1)
            n_mask = (gamma * valid_lens).long().clamp(min=1)
        else:
            n_mask = max(1, int(gamma * T))
            n_mask = torch.full((B,), n_mask, dtype=torch.long, device=device)
        
        masked_tokens = tokens.clone()
        target_mask = torch.zeros_like(tokens, dtype=torch.bool)
        
        for b in range(B):
            if valid_mask is not None:
                valid_pos = valid_mask[b].nonzero(as_tuple=True)[0]
            else:
                valid_pos = torch.arange(T, device=device)
            
            n = min(n_mask[b].item() if isinstance(n_mask, Tensor) else n_mask, len(valid_pos))
            perm = torch.randperm(len(valid_pos), device=device)[:n]
            mask_pos = valid_pos[perm]
            
            rand = torch.rand(len(mask_pos), device=device)
            
            for i, pos in enumerate(mask_pos):
                target_mask[b, pos] = True
                
                if rand[i] < 0.8:
                    masked_tokens[b, pos] = mask_token_id
                elif rand[i] < 0.9:
                    masked_tokens[b, pos] = torch.randint(0, vocab_size, (1,), device=device)
        
        return masked_tokens, target_mask, mask_ratio
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        body_tokens: Tensor,
        lhand_tokens: Tensor,
        rhand_tokens: Tensor,
        data_src: List[str],
        lengths: Optional[Tensor] = None,
        code_mask: Optional[Tensor] = None,
        debug: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Training forward pass.
        
        Args:
            input_ids: (B, T_text) tokenized text
            attention_mask: (B, T_text) text attention mask
            body_tokens: (B, T) body codes
            lhand_tokens: (B, T) left hand codes
            rhand_tokens: (B, T) right hand codes
            data_src: list of data sources ['how2sign', 'csl', 'phoenix']
            lengths: (B,) chunk lengths for loss
            code_mask: (B, T) valid code positions
            debug: print debug information
        """
        B, T = body_tokens.shape
        device = body_tokens.device
        
        # Encode text
        text_hidden = self.encode_text(input_ids, attention_mask)
        
        # Get language embeddings
        lang_embed = self.get_lang_embed(data_src)
        
        if debug:
            print(f"\n  [Model Debug]")
            print(f"    text_hidden: shape={text_hidden.shape}, mean={text_hidden.mean().item():.4f}, std={text_hidden.std().item():.4f}")
            print(f"    lang_embed: shape={lang_embed.shape}, mean={lang_embed.mean().item():.4f}")
        
        # ========== Length Prediction (DISABLED) ==========
        # length_logits = self.length_predictor(text_hidden, lang_embed)
        length_logits = torch.zeros(B, self.length_predictor.max_chunks, device=device)
        length_loss = torch.tensor(0.0, device=device, requires_grad=False)
        
        # ========== Body Masked Prediction ==========
        # Use fixed 50% mask ratio for stable training (cosine schedule was too variable)
        body_masked, body_target_mask, body_mask_ratio = self.apply_random_mask(
            body_tokens, self.body_mask_id, code_mask, fixed_mask_ratio=0.5
        )
        
        if debug:
            n_masked = body_target_mask.sum().item()
            n_total = code_mask.sum().item() if code_mask is not None else body_tokens.numel()
            print(f"    Body masking: {n_masked}/{n_total} ({100*n_masked/max(n_total,1):.1f}%)")
        
        body_logits, body_hidden = self.body_transformer(
            body_masked, text_hidden, lang_embed, code_mask
        )
        
        if debug:
            print(f"    body_logits: shape={body_logits.shape}, mean={body_logits.mean().item():.4f}, std={body_logits.std().item():.4f}")
        
        body_labels = torch.where(body_target_mask, body_tokens, 
                                  torch.full_like(body_tokens, -100))
        body_loss = self.ce_loss(body_logits.view(-1, self.body_vocab_size), body_labels.view(-1))
        
        # ========== Hand Masked Prediction ==========
        # Use GT body embeddings for teacher forcing
        body_hidden_gt = self.body_transformer.token_embed(body_tokens)
        body_hidden_gt = self.body_transformer.pos_embed(body_hidden_gt)
        
        # Left hand
        lhand_masked, lhand_target_mask, _ = self.apply_random_mask(
            lhand_tokens, self.hand_mask_id, code_mask, fixed_mask_ratio=0.5
        )
        lhand_logits = self.lhand_transformer(
            lhand_masked, body_hidden_gt, text_hidden, lang_embed, code_mask
        )
        
        lhand_labels = torch.where(lhand_target_mask, lhand_tokens,
                                   torch.full_like(lhand_tokens, -100))
        lhand_loss = self.ce_loss(lhand_logits.view(-1, self.hand_vocab_size), lhand_labels.view(-1))
        
        # Right hand
        rhand_masked, rhand_target_mask, _ = self.apply_random_mask(
            rhand_tokens, self.hand_mask_id, code_mask, fixed_mask_ratio=0.5
        )
        rhand_logits = self.rhand_transformer(
            rhand_masked, body_hidden_gt, text_hidden, lang_embed, code_mask
        )
        
        rhand_labels = torch.where(rhand_target_mask, rhand_tokens,
                                   torch.full_like(rhand_tokens, -100))
        rhand_loss = self.ce_loss(rhand_logits.view(-1, self.hand_vocab_size), rhand_labels.view(-1))
        
        # Nan/Inf check for all losses
        losses = {'body': body_loss, 'lhand': lhand_loss, 'rhand': rhand_loss}
        for name, loss in losses.items():
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] {name}_loss is nan/inf, setting to 0")
                losses[name] = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Total loss (length predictor disabled)
        total_loss = losses['body'] + losses['lhand'] + losses['rhand']
        
        return {
            'loss': total_loss,
            'length_loss': length_loss,  # Always 0 (disabled)
            'body_loss': losses['body'],
            'lhand_loss': losses['lhand'],
            'rhand_loss': losses['rhand'],
            'body_logits': body_logits,
            'lhand_logits': lhand_logits,
            'rhand_logits': rhand_logits,
            'length_logits': length_logits,
        }
    
    def unfreeze_encoder(self, unfreeze_layers: int = -1):
        """
        mBART encoder unfreeze.
        
        Args:
            unfreeze_layers: -1 = all, n = last n layers
        """
        if unfreeze_layers == -1:
            for param in self.mbart_encoder.parameters():
                param.requires_grad = True
            print("Unfroze all mBART encoder layers")
        else:
            # Unfreeze last n layers
            layers = list(self.mbart_encoder.layers)
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(f"Unfroze last {unfreeze_layers} mBART encoder layers")
        
        self.freeze_encoder = False
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        data_src: List[str],
        chunk_length: Optional[int] = None,
        num_iterations: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """MaskGIT generation."""
        B = input_ids.shape[0]
        device = input_ids.device
        num_iterations = num_iterations or self.num_iterations
        
        # Encode text
        text_hidden = self.encode_text(input_ids, attention_mask)
        lang_embed = self.get_lang_embed(data_src)
        
        # Predict length
        if chunk_length is None:
            length_logits = self.length_predictor(text_hidden, lang_embed)
            chunk_length = (length_logits.argmax(dim=-1) + 1).max().item()
        
        seq_len = chunk_length
        
        # ========== Generate Body ==========
        body_tokens = torch.full((B, seq_len), self.body_mask_id, dtype=torch.long, device=device)
        
        for step in range(num_iterations):
            body_logits, body_hidden = self.body_transformer(
                body_tokens, text_hidden, lang_embed
            )
            
            # Sample
            probs = F.softmax(body_logits / temperature, dim=-1)
            if top_k:
                top_k_probs, top_k_indices = probs.topk(top_k, dim=-1)
                probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(B, seq_len)
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            
            # Determine masking
            n_masked = CosineScheduler.get_num_masked(step, num_iterations, seq_len)
            is_masked = (body_tokens == self.body_mask_id)
            
            if step < num_iterations - 1:
                confidence = torch.where(is_masked, confidence, torch.ones_like(confidence) * 1e9)
                _, indices = confidence.sort(dim=-1)
                
                body_tokens = sampled.clone()
                for b in range(B):
                    body_tokens[b, indices[b, :n_masked]] = self.body_mask_id
            else:
                body_tokens = torch.where(is_masked, sampled, body_tokens)
        
        # Final body hidden for hand conditioning
        _, body_hidden = self.body_transformer(body_tokens, text_hidden, lang_embed)
        
        # ========== Generate Hands ==========
        lhand_tokens = torch.full((B, seq_len), self.hand_mask_id, dtype=torch.long, device=device)
        rhand_tokens = torch.full((B, seq_len), self.hand_mask_id, dtype=torch.long, device=device)
        
        for step in range(num_iterations):
            # Left hand
            lhand_logits = self.lhand_transformer(
                lhand_tokens, body_hidden, text_hidden, lang_embed
            )
            lhand_probs = F.softmax(lhand_logits / temperature, dim=-1)
            lhand_sampled = torch.multinomial(lhand_probs.view(-1, lhand_probs.size(-1)), 1).view(B, seq_len)
            lhand_conf = lhand_probs.gather(-1, lhand_sampled.unsqueeze(-1)).squeeze(-1)
            
            # Right hand
            rhand_logits = self.rhand_transformer(
                rhand_tokens, body_hidden, text_hidden, lang_embed
            )
            rhand_probs = F.softmax(rhand_logits / temperature, dim=-1)
            rhand_sampled = torch.multinomial(rhand_probs.view(-1, rhand_probs.size(-1)), 1).view(B, seq_len)
            rhand_conf = rhand_probs.gather(-1, rhand_sampled.unsqueeze(-1)).squeeze(-1)
            
            n_masked = CosineScheduler.get_num_masked(step, num_iterations, seq_len)
            
            if step < num_iterations - 1:
                # Left hand
                lhand_is_masked = (lhand_tokens == self.hand_mask_id)
                lhand_conf = torch.where(lhand_is_masked, lhand_conf, torch.ones_like(lhand_conf) * 1e9)
                _, lhand_idx = lhand_conf.sort(dim=-1)
                lhand_tokens = lhand_sampled.clone()
                for b in range(B):
                    lhand_tokens[b, lhand_idx[b, :n_masked]] = self.hand_mask_id
                
                # Right hand
                rhand_is_masked = (rhand_tokens == self.hand_mask_id)
                rhand_conf = torch.where(rhand_is_masked, rhand_conf, torch.ones_like(rhand_conf) * 1e9)
                _, rhand_idx = rhand_conf.sort(dim=-1)
                rhand_tokens = rhand_sampled.clone()
                for b in range(B):
                    rhand_tokens[b, rhand_idx[b, :n_masked]] = self.hand_mask_id
            else:
                lhand_tokens = torch.where(lhand_tokens == self.hand_mask_id, lhand_sampled, lhand_tokens)
                rhand_tokens = torch.where(rhand_tokens == self.hand_mask_id, rhand_sampled, rhand_tokens)
        
        return {
            'body_tokens': body_tokens,
            'lhand_tokens': lhand_tokens,
            'rhand_tokens': rhand_tokens,
            'chunk_length': chunk_length,
        }


# ============================================================================
# Test
# ============================================================================
if __name__ == '__main__':
    print("MaskGITmBART Test")
    print("=" * 60)
    
    # Check if mBART path exists
    mbart_path = "/home/user/Projects/research/T2P/deps/mbart-h2s-csl-phoenix"
    
    if not os.path.exists(mbart_path):
        print(f"mBART not found at {mbart_path}")
        print("Creating dummy test instead...")
        
        # Dummy test without mBART
        print("\nModel structure would be:")
        print("  - mBART Encoder (frozen) → text_hidden")
        print("  - Language Embedding → lang_embed")
        print("  - Length Predictor(text_hidden, lang_embed) → N chunks")
        print("  - Body MaskGIT(tokens, text_hidden, lang_embed) → body logits")
        print("  - Hand MaskGIT(tokens, body_hidden, text_hidden, lang_embed) → hand logits")
    else:
        # Full test
        model = MaskGITmBART(
            mbart_path=mbart_path,
            freeze_encoder=True,
            body_vocab_size=96,
            hand_vocab_size=192,
            body_n_layers=6,  # Reduced for testing
            hand_n_layers=4,
        )
        
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Test forward
        B, T_text, T_code = 2, 32, 64
        
        input_ids = torch.randint(0, 1000, (B, T_text))
        attention_mask = torch.ones(B, T_text)
        body_tokens = torch.randint(0, 96, (B, T_code))
        lhand_tokens = torch.randint(0, 192, (B, T_code))
        rhand_tokens = torch.randint(0, 192, (B, T_code))
        data_src = ['how2sign', 'csl']
        lengths = torch.randint(32, 64, (B,))
        
        print("\n=== Training Forward ===")
        outputs = model(
            input_ids, attention_mask,
            body_tokens, lhand_tokens, rhand_tokens,
            data_src, lengths
        )
        
        print(f"Total loss: {outputs['loss'].item():.4f}")
        print(f"Length loss: {outputs['length_loss'].item():.4f}")
        print(f"Body loss: {outputs['body_loss'].item():.4f}")
        
        print("\n=== Generation ===")
        model.eval()
        gen = model.generate(
            input_ids, attention_mask, data_src,
            chunk_length=32, num_iterations=10
        )
        
        print(f"Generated body: {gen['body_tokens'].shape}")
        print(f"Generated lhand: {gen['lhand_tokens'].shape}")
        print(f"Generated rhand: {gen['rhand_tokens'].shape}")