# coding: utf-8
"""
SOKE-compatible Decouple VQ-VAE
https://github.com/2000ZRL/SOKE

SOKE uses 133 dims (SMPL-X upper body + hands + expression)
Structure:
    - Body (upper body): 24 dims (8 joints × 3)
    - Right Hand: 63 dims (21 joints × 3)  
    - Left Hand: 45 dims (15 joints × 3) + 1 extra
    - Expression: ~10 dims

Total: 133 dims
"""
from typing import List, Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Resnet1D(nn.Module):
    """1D ResNet block."""
    
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, 
                 reverse_dilation=False, activation='relu', norm=None):
        super().__init__()
        
        blocks = []
        for i in range(n_depth):
            dilation = dilation_growth_rate ** (n_depth - 1 - i if reverse_dilation else i)
            
            block = []
            block.append(nn.Conv1d(n_in, n_in, 3, 1, dilation, dilation=dilation))
            if norm == 'batch':
                block.append(nn.BatchNorm1d(n_in))
            elif norm == 'layer':
                block.append(nn.GroupNorm(1, n_in))
            
            if activation == 'relu':
                block.append(nn.ReLU())
            elif activation == 'gelu':
                block.append(nn.GELU())
            elif activation == 'silu':
                block.append(nn.SiLU())
            
            blocks.append(nn.Sequential(*block))
        
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class Encoder(nn.Module):
    """Temporal encoder with strided convolutions."""
    
    def __init__(self, input_dim, output_dim, down_t=2, stride_t=2,
                 width=512, depth=3, dilation_growth_rate=3,
                 activation='relu', norm=None):
        super().__init__()
        
        blocks = []
        filter_t = stride_t * 2
        pad_t = stride_t // 2
        
        # Initial conv
        blocks.append(nn.Conv1d(input_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        # Downsampling blocks
        for i in range(down_t):
            block = nn.Sequential(
                nn.Conv1d(width, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate,
                        activation=activation, norm=norm),
            )
            blocks.append(block)
        
        # Final conv
        blocks.append(nn.Conv1d(width, output_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """Temporal decoder with upsampling."""
    
    def __init__(self, input_dim, output_dim, down_t=2, stride_t=2,
                 width=512, depth=3, dilation_growth_rate=3,
                 activation='relu', norm=None):
        super().__init__()
        
        blocks = []
        
        # Initial conv
        blocks.append(nn.Conv1d(output_dim, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        # Upsampling blocks
        for i in range(down_t):
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate,
                        reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, width, 3, 1, 1),
            )
            blocks.append(block)
        
        # Final convs
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_dim, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.model(x)


class QuantizeEMAReset(nn.Module):
    """EMA-based vector quantizer with codebook reset."""
    
    def __init__(self, nb_code, code_dim, mu=0.99):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.reset_codebook()
    
    def reset_codebook(self):
        self.register_buffer('codebook', torch.randn(self.nb_code, self.code_dim))
        self.register_buffer('code_count', torch.ones(self.nb_code))
        self.register_buffer('code_sum', self.codebook.clone())
    
    def _tile(self, x):
        nb_code_x = x.shape[0]
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / (self.code_dim ** 0.5)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out
    
    def init_codebook(self, x):
        """Initialize codebook from data."""
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
    
    @torch.no_grad()
    def update_codebook(self, x, indices):
        """EMA update of codebook."""
        one_hot = F.one_hot(indices, self.nb_code).float()  # (N, nb_code)
        
        code_sum = one_hot.T @ x  # (nb_code, code_dim)
        code_count = one_hot.sum(0)  # (nb_code,)
        
        self.code_sum = self.mu * self.code_sum + (1 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1 - self.mu) * code_count
        
        # Update codebook
        usage = (self.code_count.unsqueeze(-1) >= 1.0).float()
        code_update = self.code_sum / self.code_count.clamp(min=1).unsqueeze(-1)
        self.codebook = usage * code_update + (1 - usage) * self.codebook
        
        # Reset dead codes
        usage_flat = (self.code_count >= 1.0).float()
        dead_mask = usage_flat == 0
        if dead_mask.sum() > 0:
            # Replace dead codes with random samples from input
            n_dead = dead_mask.sum().item()
            rand_idx = torch.randperm(x.shape[0])[:int(n_dead)]
            self.codebook[dead_mask] = x[rand_idx]
    
    def preprocess(self, x):
        """(B, C, T) -> (B*T, C)"""
        x = x.permute(0, 2, 1).contiguous()  # (B, T, C)
        x = x.view(-1, x.shape[-1])  # (B*T, C)
        return x
    
    def quantize(self, x):
        """Find nearest codebook entry."""
        # x: (N, C)
        dist = torch.cdist(x, self.codebook)  # (N, nb_code)
        indices = dist.argmin(dim=-1)  # (N,)
        return indices
    
    def dequantize(self, indices):
        """Get codebook entries."""
        return F.embedding(indices, self.codebook)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) encoded features
        
        Returns:
            x_q: (B, C, T) quantized features
            loss: commitment loss
            perplexity: codebook usage
        """
        B, C, T = x.shape
        x_flat = self.preprocess(x)  # (B*T, C)
        
        # Quantize
        indices = self.quantize(x_flat)  # (B*T,)
        x_q_flat = self.dequantize(indices)  # (B*T, C)
        
        # Commitment loss
        loss = F.mse_loss(x_flat, x_q_flat.detach())
        
        # Straight-through estimator
        x_q_flat = x_flat + (x_q_flat - x_flat).detach()
        
        # Update codebook (training only)
        if self.training:
            self.update_codebook(x_flat, indices)
        
        # Perplexity
        with torch.no_grad():
            encodings = F.one_hot(indices, self.nb_code).float()
            avg_probs = encodings.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape back
        x_q = x_q_flat.view(B, T, C).permute(0, 2, 1)  # (B, C, T)
        
        return x_q, loss, perplexity


class PartVQVAE(nn.Module):
    """Single-part VQ-VAE for body/hand encoding."""
    
    def __init__(self, nfeats, code_num=512, code_dim=512,
                 output_emb_width=512, down_t=2, stride_t=2,
                 width=512, depth=3, dilation_growth_rate=3,
                 norm=None, activation='relu'):
        super().__init__()
        
        self.code_num = code_num
        self.code_dim = code_dim
        self.nfeats = nfeats
        
        self.encoder = Encoder(
            nfeats, output_emb_width, down_t, stride_t,
            width, depth, dilation_growth_rate, activation, norm
        )
        
        self.decoder = Decoder(
            nfeats, output_emb_width, down_t, stride_t,
            width, depth, dilation_growth_rate, activation, norm
        )
        
        self.quantizer = QuantizeEMAReset(code_num, code_dim)
    
    def preprocess(self, x):
        return x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
    
    def postprocess(self, x):
        return x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
    
    def forward(self, features):
        """
        Args:
            features: (B, T, nfeats)
        Returns:
            x_out: (B, T, nfeats)
            loss: commitment loss
            perplexity: scalar
            indices: (B, T')
        """
        input_length = features.shape[1]
        
        # Encode
        x_in = self.preprocess(features)  # (B, C, T)
        x_enc = self.encoder(x_in)  # (B, code_dim, T')
        
        # Quantize
        x_q, loss, perplexity = self.quantizer(x_enc)
        
        # Get indices
        x_enc_flat = x_enc.permute(0, 2, 1).contiguous().view(-1, x_enc.shape[1])
        indices = self.quantizer.quantize(x_enc_flat)
        indices = indices.view(features.shape[0], -1)  # (B, T')
        
        # Decode
        x_dec = self.decoder(x_q)
        x_out = self.postprocess(x_dec)  # (B, T, C)
        
        # Adjust length
        output_length = x_out.shape[1]
        if output_length > input_length:
            x_out = x_out[:, :input_length, :]
        elif output_length < input_length:
            pad_size = input_length - output_length
            x_out = F.pad(x_out, (0, 0, 0, pad_size), mode='replicate')
        
        return x_out, loss, perplexity, indices
    
    def encode(self, features):
        """Encode to discrete codes."""
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_enc = self.encoder(x_in)
        x_enc = self.postprocess(x_enc)
        x_enc_flat = x_enc.contiguous().view(-1, x_enc.shape[-1])
        indices = self.quantizer.quantize(x_enc_flat)
        return indices.view(N, -1)
    
    def decode(self, codes):
        """Decode from discrete codes."""
        x_q = self.quantizer.dequantize(codes)  # (B, T', code_dim)
        x_q = x_q.permute(0, 2, 1)  # (B, code_dim, T')
        x_dec = self.decoder(x_q)
        return self.postprocess(x_dec)


class VQVAE(nn.Module):
    """
    SOKE-style Decouple VQ-VAE for Sign Language.
    
    Separate codebooks for:
        - Body (upper body): 24 dims
        - Right Hand: 63 dims
        - Left Hand: 46 dims (rest of 133)
    
    Input: (B, T, 133)
    """
    
    def __init__(self,
                 nfeats: int = 133,
                 quantizer: str = "ema_reset",
                 code_num: int = 512,
                 body_code_num: int = 96,
                 hand_code_num: int = 192,
                 code_dim: int = 512,
                 output_emb_width: int = 512,
                 down_t: int = 2,
                 stride_t: int = 2,
                 width: int = 512,
                 depth: int = 3,
                 dilation_growth_rate: int = 3,
                 norm: str = None,
                 activation: str = "relu",
                 **kwargs) -> None:
        super().__init__()
        
        self.nfeats = nfeats
        
        # SOKE joint structure for 133 dims
        # Body: first 24 dims (upper body)
        # RHand: next 63 dims  
        # LHand: remaining 46 dims
        self.body_nfeats = 24
        self.rhand_nfeats = 63
        self.lhand_nfeats = nfeats - self.body_nfeats - self.rhand_nfeats  # 46
        
        assert self.body_nfeats + self.rhand_nfeats + self.lhand_nfeats == nfeats
        
        self.body_code_num = body_code_num
        self.hand_code_num = hand_code_num
        self.code_dim = code_dim
        
        common_kwargs = {
            'code_dim': code_dim,
            'output_emb_width': output_emb_width,
            'down_t': down_t,
            'stride_t': stride_t,
            'width': width,
            'depth': depth,
            'dilation_growth_rate': dilation_growth_rate,
            'norm': norm,
            'activation': activation,
        }
        
        # Body VQ-VAE
        self.body_vae = PartVQVAE(
            nfeats=self.body_nfeats,
            code_num=body_code_num,
            **common_kwargs,
        )
        
        # Right Hand VQ-VAE
        self.rhand_vae = PartVQVAE(
            nfeats=self.rhand_nfeats,
            code_num=hand_code_num,
            **common_kwargs,
        )
        
        # Left Hand VQ-VAE
        self.lhand_vae = PartVQVAE(
            nfeats=self.lhand_nfeats,
            code_num=hand_code_num,
            **common_kwargs,
        )
    
    def split_pose(self, pose):
        """
        Split pose into body, right hand, left hand.
        
        Args:
            pose: (B, T, 133)
        Returns:
            body: (B, T, 24)
            rhand: (B, T, 63)
            lhand: (B, T, 46)
        """
        body = pose[..., :self.body_nfeats]
        rhand = pose[..., self.body_nfeats:self.body_nfeats + self.rhand_nfeats]
        lhand = pose[..., self.body_nfeats + self.rhand_nfeats:]
        return body, rhand, lhand
    
    def merge_pose(self, body, rhand, lhand):
        """Merge parts into full pose."""
        return torch.cat([body, rhand, lhand], dim=-1)
    
    def forward(self, features: Tensor):
        """
        Forward pass.
        
        Args:
            features: (B, T, 133)
        
        Returns:
            pose_out: (B, T, 133) reconstructed
            total_loss: scalar commitment loss
            perplexity_dict: dict of perplexities
            indices_dict: dict of code indices
        """
        body, rhand, lhand = self.split_pose(features)
        
        body_out, body_loss, body_ppl, body_idx = self.body_vae(body)
        rhand_out, rhand_loss, rhand_ppl, rhand_idx = self.rhand_vae(rhand)
        lhand_out, lhand_loss, lhand_ppl, lhand_idx = self.lhand_vae(lhand)
        
        pose_out = self.merge_pose(body_out, rhand_out, lhand_out)
        
        total_loss = (body_loss + rhand_loss + lhand_loss) / 3.0
        
        perplexity_dict = {
            'body': body_ppl,
            'rhand': rhand_ppl,
            'lhand': lhand_ppl,
        }
        
        indices_dict = {
            'body': body_idx,
            'rhand': rhand_idx,
            'lhand': lhand_idx,
        }
        
        return pose_out, total_loss, perplexity_dict, indices_dict
    
    def encode(self, features: Tensor) -> Dict[str, Tensor]:
        """Encode to discrete codes."""
        body, rhand, lhand = self.split_pose(features)
        
        return {
            'body': self.body_vae.encode(body),
            'rhand': self.rhand_vae.encode(rhand),
            'lhand': self.lhand_vae.encode(lhand),
        }
    
    def decode(self, codes: Dict[str, Tensor]) -> Tensor:
        """Decode from discrete codes."""
        body_out = self.body_vae.decode(codes['body'])
        rhand_out = self.rhand_vae.decode(codes['rhand'])
        lhand_out = self.lhand_vae.decode(codes['lhand'])
        
        return self.merge_pose(body_out, rhand_out, lhand_out)
