# coding: utf-8
"""
Token Embedding for Motion GPT
Adapted from SOKE: Signs as Tokens (https://github.com/2000ZRL/SOKE)

Key idea:
- Freeze pretrained text embeddings
- Only train new motion token embeddings via word2motionProj
"""

import torch
import torch.nn as nn
from torch import Tensor


class NewTokenEmb(nn.Module):
    """
    Extended token embedding that separates text and motion embeddings.
    
    From SOKE: mGPT/archs/tools/token_emb.py
    
    - Text embeddings: Frozen (from pretrained mBart)
    - Motion embeddings: Zero for old tokens, learnable for new tokens
    
    Forward: text_emb(x) + motion_emb(x)
    - For old tokens: text_emb + 0 = text_emb (frozen)
    - For new tokens: 0 + motion_emb = motion_emb (learnable)
    """

    def __init__(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: int = None
    ) -> None:
        """
        Args:
            old_embeddings: Original embedding layer from pretrained model
            new_num_tokens: Number of new motion tokens to add
        """
        super().__init__()

        self.num_tokens = old_embeddings.num_embeddings + new_num_tokens
        self.old_num_tokens = old_embeddings.num_embeddings
        self.new_num_tokens = new_num_tokens
        self.embedding_dim = old_embeddings.embedding_dim

        # =====================================================================
        # Text embeddings (frozen) - full vocab size
        # =====================================================================
        self.text_embeddings = nn.Embedding(
            self.num_tokens,
            self.embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype
        )
        
        with torch.no_grad():
            # Copy pretrained weights for old tokens
            self.text_embeddings.weight.data[:self.old_num_tokens] = \
                old_embeddings.weight.data
            # Initialize new tokens to zero (motion tokens don't use text embeddings)
            self.text_embeddings.weight.data[self.old_num_tokens:] = torch.zeros(
                self.new_num_tokens,
                self.embedding_dim,
                dtype=old_embeddings.weight.dtype,
                device=old_embeddings.weight.device
            )
        
        # Freeze text embeddings
        self.text_embeddings.weight.requires_grad_(False)

        # =====================================================================
        # Motion embeddings (learnable indirectly via projection)
        # =====================================================================
        self.motion_embeddings = nn.Embedding(
            new_num_tokens,  # Only new tokens
            self.embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype
        )
        
        # Initialize motion embeddings to zero
        with torch.no_grad():
            self.motion_embeddings.weight.data.zero_()
        
        # =====================================================================
        # Projection: text embedding space -> motion embedding space
        # This is what actually gets trained!
        # =====================================================================
        self.word2motionProj = nn.Linear(self.old_num_tokens, new_num_tokens)
        
        # Initialize projection
        nn.init.xavier_uniform_(self.word2motionProj.weight)
        nn.init.zeros_(self.word2motionProj.bias)

    @property
    def weight(self):
        """For compatibility with get_input_embeddings()."""
        return self.text_embeddings.weight
    
    @property
    def num_embeddings(self):
        """For compatibility."""
        return self.num_tokens

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass: combines frozen text embeddings with learnable motion embeddings.
        
        From SOKE:
        - Projects text embeddings to motion embedding space
        - Updates motion_embeddings weights (with gradient!)
        - Returns text_emb + motion_emb
        
        Args:
            input: Token indices (B, T)
            
        Returns:
            Embeddings (B, T, D)
        """
        device = input.device
        
        # Ensure embeddings are on correct device
        if self.text_embeddings.weight.device != device:
            self.text_embeddings = self.text_embeddings.to(device)
            self.motion_embeddings = self.motion_embeddings.to(device)
            self.word2motionProj = self.word2motionProj.to(device)
        
        # =====================================================================
        # Key insight from SOKE:
        # - Project text embeddings to get motion embeddings
        # - This projection is differentiable and gets trained
        # =====================================================================
        
        # Get text embeddings (frozen)
        text_emb = self.text_embeddings(input)  # (B, T, D)
        
        # Project old token embeddings to new token embedding space
        # word2motionProj: (old_num_tokens, new_num_tokens)
        # text_embeddings.weight[:old_num_tokens]: (old_num_tokens, D)
        # projected: (new_num_tokens, D)
        projected = self.word2motionProj(
            self.text_embeddings.weight.data[:self.old_num_tokens].permute(1, 0)  # (D, old_num_tokens)
        ).permute(1, 0)  # (new_num_tokens, D)
        
        # Update motion embeddings with projected values
        # Use in-place operation that preserves gradient
        self.motion_embeddings.weight.data.copy_(projected.data)
        
        # Create motion embedding lookup
        # For indices >= old_num_tokens, look up in motion_embeddings
        # For indices < old_num_tokens, return zeros
        
        # Shift indices to motion embedding space
        motion_indices = input - self.old_num_tokens  # (B, T)
        
        # Clamp to valid range (negative means old token, use 0 as placeholder)
        motion_indices_clamped = motion_indices.clamp(min=0, max=self.new_num_tokens - 1)
        
        # Get motion embeddings
        motion_emb = self.motion_embeddings(motion_indices_clamped)  # (B, T, D)
        
        # Mask out old tokens (they should have zero motion embedding)
        is_new_token = (input >= self.old_num_tokens).unsqueeze(-1).float()  # (B, T, 1)
        motion_emb = motion_emb * is_new_token  # (B, T, D)
        
        # Combine: text_emb + motion_emb
        # Old tokens: text_emb + 0 = text_emb
        # New tokens: 0 + motion_emb = motion_emb
        return text_emb + motion_emb


class NewTokenEmbSimple(nn.Module):
    """
    Simplified version: directly train motion embeddings without projection.
    
    Use this if word2motionProj causes issues.
    """

    def __init__(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: int = None
    ) -> None:
        super().__init__()

        self.num_tokens = old_embeddings.num_embeddings + new_num_tokens
        self.old_num_tokens = old_embeddings.num_embeddings
        self.new_num_tokens = new_num_tokens
        self.embedding_dim = old_embeddings.embedding_dim

        # Create combined embedding layer
        self.embeddings = nn.Embedding(
            self.num_tokens,
            self.embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype
        )
        
        with torch.no_grad():
            # Copy pretrained weights for old tokens (these will be frozen)
            self.embeddings.weight.data[:self.old_num_tokens] = old_embeddings.weight.data
            
            # Initialize new tokens randomly
            nn.init.normal_(
                self.embeddings.weight.data[self.old_num_tokens:],
                mean=0.0,
                std=0.02
            )
        
        # Create mask for freezing old token embeddings
        self.register_buffer(
            'freeze_mask',
            torch.cat([
                torch.ones(self.old_num_tokens),
                torch.zeros(self.new_num_tokens)
            ]).bool()
        )

    @property
    def weight(self):
        return self.embeddings.weight
    
    @property
    def num_embeddings(self):
        return self.num_tokens

    def forward(self, input: Tensor) -> Tensor:
        return self.embeddings(input)
    
    def freeze_old_embeddings(self):
        """Call this in training loop to freeze old token gradients."""
        if self.embeddings.weight.grad is not None:
            self.embeddings.weight.grad.data[:self.old_num_tokens].zero_()