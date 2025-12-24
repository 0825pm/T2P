# Partially from https://github.com/Mael-zys/T2M-GPT
# Adapted from SOKE: https://github.com/2000ZRL/SOKE/blob/main/mGPT/archs/mgpt_vq.py
# VQVae: Single codebook for all joints (50 joints * 3 = 150 dims)

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


class VQVAE(nn.Module):
    """
    VQ-VAE for Sign Language.
    Single codebook for all 50 joints (body + hands).
    Input: (B, T, 150) where 150 = 50 joints * 3
    """

    def __init__(self,
                 nfeats: int = 150,
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
        self.use_cuda = True  # for Batch class compatibility
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
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) -> (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        """
        Forward pass.
        Input: (B, T, 150)
        Output: reconstructed, commit_loss, perplexity, indices_dict
        """
        input_length = features.shape[1]
        x_in = self.preprocess(features)

        # Encode
        x_encoder = self.encoder(x_in)

        # Quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)
        
        # Get indices for codebook statistics
        x_encoder_post = self.postprocess(x_encoder)
        x_encoder_flat = x_encoder_post.contiguous().view(-1, x_encoder_post.shape[-1])
        indices = self.quantizer.quantize(x_encoder_flat)

        # Decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        
        # Adjust output length to match input
        output_length = x_out.shape[1]
        if output_length > input_length:
            x_out = x_out[:, :input_length, :]
        elif output_length < input_length:
            pad_size = input_length - output_length
            x_out = F.pad(x_out, (0, 0, 0, pad_size), mode='replicate')

        # Return format compatible with train script
        # indices as dict for unified interface
        indices_dict = {'all': indices}
        perplexity_dict = {'all': perplexity}

        return x_out, loss, perplexity_dict, indices_dict

    def encode(self, features: Tensor):
        """Encode features to discrete codes."""
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return {'all': code_idx}

    def decode(self, codes: dict):
        """Decode discrete codes to features."""
        z = codes['all']
        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out