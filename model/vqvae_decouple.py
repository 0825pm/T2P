# Adapted from SOKE: https://github.com/2000ZRL/SOKE
# DecoupleVQVae: Separate codebooks for body, right hand, left hand
# Joint structure: 8 body + 21 rhand + 21 lhand = 50 joints

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.distribution import Distribution
from model.tools.resnet import Resnet1D
from model.tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), 
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class PartVQVAE(nn.Module):
    """Single-part VQ-VAE for body/hand encoding."""

    def __init__(self,
                 nfeats: int,
                 quantizer: str = "ema_reset",
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()
        self.code_num = code_num
        self.code_dim = code_dim
        self.nfeats = nfeats

        self.encoder = Encoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)

    def preprocess(self, x):
        return x.permute(0, 2, 1)

    def postprocess(self, x):
        return x.permute(0, 2, 1)

    def forward(self, features: Tensor):
        input_length = features.shape[1]
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        
        # Get indices
        x_encoder_post = self.postprocess(x_encoder)
        x_encoder_flat = x_encoder_post.contiguous().view(-1, x_encoder_post.shape[-1])
        indices = self.quantizer.quantize(x_encoder_flat)
        
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        
        # Adjust output length
        output_length = x_out.shape[1]
        if output_length > input_length:
            x_out = x_out[:, :input_length, :]
        elif output_length < input_length:
            pad_size = input_length - output_length
            x_out = F.pad(x_out, (0, 0, 0, pad_size), mode='replicate')

        return x_out, loss, perplexity, indices

    def encode(self, features: Tensor):
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def decode(self, z: Tensor):
        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class VQVAE(nn.Module):
    """
    Decouple VQ-VAE for Sign Language.
    Separate codebooks for body (8 joints), right hand (21 joints), left hand (21 joints).
    Total: 50 joints * 3 = 150 dims
    
    Joint order in input:
    - body: joints 0-7 (8 joints) -> dims 0-23
    - rhand: joints 8-28 (21 joints) -> dims 24-86
    - lhand: joints 29-49 (21 joints) -> dims 87-149
    
    SOKE default settings:
    - body_code_num: 96
    - hand_code_num: 192 (for both rhand and lhand)
    """

    def __init__(self,
                 nfeats: int = 150,  # Total features (50 joints * 3)
                 quantizer: str = "ema_reset",
                 code_num=512,  # fallback if body/hand not specified
                 body_code_num=96,  # SOKE default for body
                 hand_code_num=192,  # SOKE default for hands
                 code_dim=512,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 **kwargs) -> None:

        super().__init__()
        self.use_cuda = True  # for Batch class compatibility
        
        # Joint structure: 8 body + 21 rhand + 21 lhand = 50 joints
        self.body_nfeats = 8 * 3   # 24
        self.rhand_nfeats = 21 * 3  # 63
        self.lhand_nfeats = 21 * 3  # 63
        
        # Store codebook sizes for logging
        self.body_code_num = body_code_num
        self.hand_code_num = hand_code_num
        
        # Body VQ-VAE (96 codes by default, following SOKE)
        self.body_vae = PartVQVAE(
            nfeats=self.body_nfeats,
            quantizer=quantizer,
            code_num=body_code_num,
            code_dim=code_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            norm=norm,
            activation=activation,
        )
        
        # Right Hand VQ-VAE (192 codes by default, following SOKE)
        self.rhand_vae = PartVQVAE(
            nfeats=self.rhand_nfeats,
            quantizer=quantizer,
            code_num=hand_code_num,
            code_dim=code_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            norm=norm,
            activation=activation,
        )
        
        # Left Hand VQ-VAE (192 codes by default, following SOKE)
        self.lhand_vae = PartVQVAE(
            nfeats=self.lhand_nfeats,
            quantizer=quantizer,
            code_num=hand_code_num,
            code_dim=code_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            norm=norm,
            activation=activation,
        )

    def split_pose(self, pose):
        """
        Split pose into body, right hand, left hand.
        Input: (B, T, 150) where 150 = 50 joints * 3
        Output: body (B, T, 24), rhand (B, T, 63), lhand (B, T, 63)
        """
        body = pose[..., :self.body_nfeats]           # (B, T, 24)
        rhand = pose[..., self.body_nfeats:self.body_nfeats + self.rhand_nfeats]  # (B, T, 63)
        lhand = pose[..., self.body_nfeats + self.rhand_nfeats:]  # (B, T, 63)
        return body, rhand, lhand

    def merge_pose(self, body, rhand, lhand):
        """
        Merge body, right hand, left hand into full pose.
        Output: (B, T, 150)
        """
        return torch.cat([body, rhand, lhand], dim=-1)

    def forward(self, features: Tensor):
        """
        Forward pass for Decouple VQ-VAE.
        Input: (B, T, 150)
        Output: reconstructed pose, total commit loss, perplexity dict, indices dict
        """
        # Split into parts
        body, rhand, lhand = self.split_pose(features)
        
        # Forward each part
        body_out, body_loss, body_ppl, body_indices = self.body_vae(body)
        rhand_out, rhand_loss, rhand_ppl, rhand_indices = self.rhand_vae(rhand)
        lhand_out, lhand_loss, lhand_ppl, lhand_indices = self.lhand_vae(lhand)
        
        # Merge outputs
        pose_out = self.merge_pose(body_out, rhand_out, lhand_out)
        
        # Total loss (average of three parts)
        total_loss = (body_loss + rhand_loss + lhand_loss) / 3.0
        
        # Perplexity dict
        perplexity_dict = {
            'body': body_ppl,
            'rhand': rhand_ppl,
            'lhand': lhand_ppl,
        }
        
        # Indices dict
        indices_dict = {
            'body': body_indices,
            'rhand': rhand_indices,
            'lhand': lhand_indices,
        }
        
        return pose_out, total_loss, perplexity_dict, indices_dict

    def encode(self, features: Tensor):
        """Encode pose into discrete codes for each part."""
        body, rhand, lhand = self.split_pose(features)
        
        body_codes = self.body_vae.encode(body)
        rhand_codes = self.rhand_vae.encode(rhand)
        lhand_codes = self.lhand_vae.encode(lhand)
        
        return {
            'body': body_codes,
            'rhand': rhand_codes,
            'lhand': lhand_codes,
        }

    def decode(self, codes: dict):
        """Decode discrete codes back to pose."""
        body_out = self.body_vae.decode(codes['body'])
        rhand_out = self.rhand_vae.decode(codes['rhand'])
        lhand_out = self.lhand_vae.decode(codes['lhand'])
        
        return self.merge_pose(body_out, rhand_out, lhand_out)