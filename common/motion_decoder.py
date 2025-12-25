"""
Motion Decoder: Convert predicted codes to poses using VQ-VAE decoder.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional


class MotionDecoder:
    """
    Decode motion codes to poses using trained VQ-VAE.
    """
    
    def __init__(
        self,
        vqvae_model,
        device: str = 'cuda',
        down_t: int = 2,
    ):
        """
        Args:
            vqvae_model: Trained DecoupleVQVae model
            device: Device to run on
            down_t: Temporal downsampling factor
        """
        self.vqvae = vqvae_model
        self.vqvae.eval()
        self.device = device
        self.down_t = down_t
        self.upsample_factor = 2 ** down_t  # 4 for down_t=2
    
    @torch.no_grad()
    def decode(
        self,
        body_codes: torch.Tensor,
        lhand_codes: torch.Tensor,
        rhand_codes: torch.Tensor,
    ) -> np.ndarray:
        """
        Decode codes to poses.
        
        Args:
            body_codes: (T,) tensor of body codes
            lhand_codes: (T,) tensor of left hand codes
            rhand_codes: (T,) tensor of right hand codes
        
        Returns:
            poses: (T*upsample_factor, 150) numpy array
        """
        # Ensure same length
        min_len = min(len(body_codes), len(lhand_codes), len(rhand_codes))
        body_codes = body_codes[:min_len]
        lhand_codes = lhand_codes[:min_len]
        rhand_codes = rhand_codes[:min_len]
        
        # To device
        body_codes = body_codes.long().to(self.device)
        lhand_codes = lhand_codes.long().to(self.device)
        rhand_codes = rhand_codes.long().to(self.device)
        
        # Add batch dimension: (T,) -> (1, T)
        body_codes = body_codes.unsqueeze(0)
        lhand_codes = lhand_codes.unsqueeze(0)
        rhand_codes = rhand_codes.unsqueeze(0)
        
        # Decode using VQ-VAE
        # codes_dict format: {'body': (1, T), 'lhand': (1, T), 'rhand': (1, T)}
        codes_dict = {
            'body': body_codes,
            'lhand': lhand_codes,
            'rhand': rhand_codes,
        }
        
        # Get quantized embeddings from codebook
        # Body
        body_emb = self.vqvae.body_vae.quantizer.dequantize(body_codes)  # (1, T, C)
        body_emb = body_emb.permute(0, 2, 1)  # (1, C, T)
        body_pose = self.vqvae.body_vae.decoder(body_emb)  # (1, 24, T')
        body_pose = body_pose.permute(0, 2, 1)  # (1, T', 24)
        
        # LHand
        lhand_emb = self.vqvae.lhand_vae.quantizer.dequantize(lhand_codes)
        lhand_emb = lhand_emb.permute(0, 2, 1)
        lhand_pose = self.vqvae.lhand_vae.decoder(lhand_emb)
        lhand_pose = lhand_pose.permute(0, 2, 1)  # (1, T', 63)
        
        # RHand
        rhand_emb = self.vqvae.rhand_vae.quantizer.dequantize(rhand_codes)
        rhand_emb = rhand_emb.permute(0, 2, 1)
        rhand_pose = self.vqvae.rhand_vae.decoder(rhand_emb)
        rhand_pose = rhand_pose.permute(0, 2, 1)  # (1, T', 63)
        
        # Concatenate: body (24) + rhand (63) + lhand (63) = 150
        full_pose = torch.cat([body_pose, rhand_pose, lhand_pose], dim=-1)  # (1, T', 150)
        
        return full_pose[0].cpu().numpy()  # (T', 150)
    
    @torch.no_grad()
    def decode_batch(
        self,
        body_codes_list: List[torch.Tensor],
        lhand_codes_list: List[torch.Tensor],
        rhand_codes_list: List[torch.Tensor],
    ) -> List[np.ndarray]:
        """
        Decode a batch of codes to poses.
        
        Returns:
            List of pose arrays
        """
        poses = []
        for body, lhand, rhand in zip(body_codes_list, lhand_codes_list, rhand_codes_list):
            pose = self.decode(body, lhand, rhand)
            poses.append(pose)
        return poses


def load_vqvae_decoder(checkpoint_path: str, config: dict, device: str = 'cuda'):
    """
    Load trained VQ-VAE model for decoding.
    
    Args:
        checkpoint_path: Path to VQ-VAE checkpoint
        config: Model config dict
        device: Device
    
    Returns:
        MotionDecoder instance
    """
    from model.vqvae_decouple import VQVAE
    
    # Create model
    vqvae = VQVAE(
        nfeats=150,
        body_code_num=config.get('body_code_num', 96),
        hand_code_num=config.get('hand_code_num', 192),
        code_dim=config.get('code_dim', 512),
        output_emb_width=config.get('output_emb_width', 512),
        down_t=config.get('down_t', 2),
        stride_t=config.get('stride_t', 2),
        width=config.get('width', 512),
        depth=config.get('depth', 3),
        dilation_growth_rate=config.get('dilation_growth_rate', 3),
        activation=config.get('activation', 'relu'),
        norm=config.get('norm', None),
    )
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    vqvae.load_state_dict(new_state_dict, strict=False)
    vqvae = vqvae.to(device)
    vqvae.eval()
    
    print(f"Loaded VQ-VAE from {checkpoint_path}")
    
    return MotionDecoder(vqvae, device=device, down_t=config.get('down_t', 2))