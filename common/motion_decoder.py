# coding: utf-8
"""
Motion Decoder - VQ-VAE Decoder for Motion Code to Pose Conversion

Loads trained VQ-VAE decoder and converts motion codes to pose sequences.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Union


class MotionDecoder:
    """
    Motion Decoder wrapper for VQ-VAE.
    
    Loads trained VQ-VAE and provides methods to decode motion codes to poses.
    """
    
    def __init__(
        self,
        vqvae_path: str,
        model_config: dict,
        device: str = 'cuda',
    ):
        """
        Args:
            vqvae_path: Path to trained VQ-VAE checkpoint
            model_config: Model configuration dict
            device: Target device
        """
        self.device = device
        self.model_config = model_config
        
        # Load VQ-VAE model - use vqvae_soke which matches checkpoint
        try:
            from model.vqvae_soke import VQVAE
        except ImportError:
            from model.vqvae_decouple import VQVAE
        
        vqvae_config = {
            'nfeats': model_config.get('nfeats', 133),
            'body_code_num': model_config.get('body_code_num', 96),
            'hand_code_num': model_config.get('hand_code_num', 192),
            'code_dim': model_config.get('code_dim', 512),
            'output_emb_width': model_config.get('output_emb_width', 512),
            'down_t': model_config.get('down_t', 2),
            'stride_t': model_config.get('stride_t', 2),
            'width': model_config.get('width', 512),
            'depth': model_config.get('depth', 3),
            'dilation_growth_rate': model_config.get('dilation_growth_rate', 3),
            'activation': model_config.get('activation', 'relu'),
            'norm': model_config.get('norm', None),
        }
        
        self.vqvae = VQVAE(**vqvae_config)
        
        # Load checkpoint
        ckpt = torch.load(vqvae_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        self.vqvae.load_state_dict(state_dict, strict=True)
        self.vqvae = self.vqvae.to(device)
        self.vqvae.eval()
        
        print(f"Loaded VQ-VAE from {vqvae_path}")
    
    @torch.no_grad()
    def decode(
        self,
        body_codes: Union[np.ndarray, torch.Tensor],
        lhand_codes: Union[np.ndarray, torch.Tensor],
        rhand_codes: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode motion codes to pose sequence.
        
        Args:
            body_codes: Body code indices (T,) or (B, T)
            lhand_codes: Left hand code indices (T,) or (B, T)
            rhand_codes: Right hand code indices (T,) or (B, T)
            
        Returns:
            Pose sequence (B, T*4, 133) - upsampled by factor of 4
        """
        # Convert to tensor if needed
        if isinstance(body_codes, np.ndarray):
            body_codes = torch.from_numpy(body_codes).long()
        if isinstance(lhand_codes, np.ndarray):
            lhand_codes = torch.from_numpy(lhand_codes).long()
        if isinstance(rhand_codes, np.ndarray):
            rhand_codes = torch.from_numpy(rhand_codes).long()
        
        # Add batch dimension if needed
        if body_codes.dim() == 1:
            body_codes = body_codes.unsqueeze(0)
            lhand_codes = lhand_codes.unsqueeze(0)
            rhand_codes = rhand_codes.unsqueeze(0)
        
        # Move to device
        body_codes = body_codes.to(self.device)
        lhand_codes = lhand_codes.to(self.device)
        rhand_codes = rhand_codes.to(self.device)
        
        # Align sequence lengths (use minimum)
        min_len = min(body_codes.shape[1], lhand_codes.shape[1], rhand_codes.shape[1])
        body_codes = body_codes[:, :min_len]
        lhand_codes = lhand_codes[:, :min_len]
        rhand_codes = rhand_codes[:, :min_len]
        
        # Decode
        poses = self.vqvae.decode_from_codes(body_codes, lhand_codes, rhand_codes)
        
        return poses
    
    @torch.no_grad()
    def decode_combined(
        self,
        codes: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode combined motion codes.
        
        Args:
            codes: Combined codes (T, 3) or (B, T, 3) where [:, :, 0]=body, [:, :, 1]=lhand, [:, :, 2]=rhand
            
        Returns:
            Pose sequence (B, T*4, 133)
        """
        if isinstance(codes, np.ndarray):
            codes = torch.from_numpy(codes).long()
        
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)  # Add batch dim
        
        body_codes = codes[:, :, 0]
        lhand_codes = codes[:, :, 1]
        rhand_codes = codes[:, :, 2]
        
        return self.decode(body_codes, lhand_codes, rhand_codes)


def load_vqvae_decoder(
    vqvae_path: str,
    model_config: dict,
    device: str = 'cuda',
) -> MotionDecoder:
    """
    Load VQ-VAE decoder.
    
    Args:
        vqvae_path: Path to trained VQ-VAE checkpoint
        model_config: Model configuration dict
        device: Target device
        
    Returns:
        MotionDecoder instance
    """
    return MotionDecoder(vqvae_path, model_config, device)


# Alternative simpler decoder for when we don't need the full VQVAE
class SimpleMotionDecoder:
    """
    Simple motion decoder that only loads codebook embeddings.
    """
    
    def __init__(
        self,
        vqvae_path: str,
        model_config: dict,
        device: str = 'cuda',
    ):
        self.device = device
        
        # Load checkpoint
        state_dict = torch.load(vqvae_path, map_location='cpu', weights_only=False)
        
        # Extract codebook embeddings
        self.body_codebook = state_dict.get('body_quantizer.codebook.weight', 
                                            state_dict.get('quantizer_body.codebook.weight'))
        self.lhand_codebook = state_dict.get('lhand_quantizer.codebook.weight',
                                             state_dict.get('quantizer_lhand.codebook.weight'))
        self.rhand_codebook = state_dict.get('rhand_quantizer.codebook.weight',
                                             state_dict.get('quantizer_rhand.codebook.weight'))
        
        if self.body_codebook is not None:
            self.body_codebook = self.body_codebook.to(device)
        if self.lhand_codebook is not None:
            self.lhand_codebook = self.lhand_codebook.to(device)
        if self.rhand_codebook is not None:
            self.rhand_codebook = self.rhand_codebook.to(device)
        
        print(f"Loaded codebooks from {vqvae_path}")
    
    @torch.no_grad()
    def get_embeddings(
        self,
        body_codes: torch.Tensor,
        lhand_codes: torch.Tensor,
        rhand_codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get codebook embeddings for given codes.
        
        Returns:
            Tuple of (body_emb, lhand_emb, rhand_emb)
        """
        body_emb = self.body_codebook[body_codes] if self.body_codebook is not None else None
        lhand_emb = self.lhand_codebook[lhand_codes] if self.lhand_codebook is not None else None
        rhand_emb = self.rhand_codebook[rhand_codes] if self.rhand_codebook is not None else None
        
        return body_emb, lhand_emb, rhand_emb