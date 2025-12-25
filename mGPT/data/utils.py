# coding: utf-8
"""
SOKE Data Utils - Collate Functions
https://github.com/2000ZRL/SOKE

Batch collation for DataLoader
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Any


def humanml3d_collate(batch: List[Tuple]) -> Dict[str, Any]:
    """
    SOKE-style collate function for motion data.
    
    Input batch format (from dataset __getitem__):
        (text, motion, length, name, _, _, _, all_captions, tasks, src)
    
    Output dict keys:
        - text: List[str] - text annotations
        - motion: (B, T_max, D) - padded motion sequences
        - lengths: (B,) - original lengths
        - name: List[str] - sample names
        - all_captions: List[List[str]] - all captions
        - tasks: List[dict] - task instructions
        - src: List[str] - source datasets
    """
    # Filter out None items
    batch = [b for b in batch if b is not None and b[1] is not None]
    
    if len(batch) == 0:
        return None
    
    # Unpack
    texts = [b[0] for b in batch]
    motions = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    names = [b[3] for b in batch]
    all_captions = [b[7] for b in batch]
    tasks = [b[8] for b in batch]
    srcs = [b[9] for b in batch]
    
    # Pad motions
    max_len = max(lengths)
    
    if isinstance(motions[0], torch.Tensor):
        feat_dim = motions[0].shape[-1]
        padded = torch.zeros(len(batch), max_len, feat_dim)
        
        for i, m in enumerate(motions):
            padded[i, :lengths[i]] = m[:lengths[i]]
    else:
        # Handle numpy arrays
        feat_dim = motions[0].shape[-1]
        padded = np.zeros((len(batch), max_len, feat_dim))
        
        for i, m in enumerate(motions):
            padded[i, :lengths[i]] = m[:lengths[i]]
        padded = torch.from_numpy(padded).float()
    
    return {
        'text': texts,
        'motion': padded,
        'lengths': torch.tensor(lengths),
        'name': names,
        'all_captions': all_captions,
        'tasks': tasks,
        'src': srcs,
    }


def motion_token_collate(batch: List[Tuple]) -> Dict[str, Any]:
    """
    Collate function for motion token data (LM training).
    
    Input batch format:
        (caption, m_tokens, length, name, _, _, _, all_captions, tasks, src)
    
    Output dict keys:
        - text: List[str]
        - motion_tokens: (B, T_max) - padded token sequences
        - lengths: (B,)
        - name: List[str]
        - all_captions: List
        - tasks: List
        - src: List[str]
    """
    batch = [b for b in batch if b is not None and b[1] is not None]
    
    if len(batch) == 0:
        return None
    
    texts = [b[0] for b in batch]
    tokens = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    names = [b[3] for b in batch]
    all_captions = [b[7] for b in batch]
    tasks = [b[8] for b in batch]
    srcs = [b[9] for b in batch]
    
    max_len = max(lengths)
    
    # Pad tokens (assuming pad_id = 0)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, t in enumerate(tokens):
        if isinstance(t, torch.Tensor):
            padded[i, :lengths[i]] = t[:lengths[i]]
        else:
            padded[i, :lengths[i]] = torch.from_numpy(t[:lengths[i]]).long()
    
    return {
        'text': texts,
        'motion_tokens': padded,
        'lengths': torch.tensor(lengths),
        'name': names,
        'all_captions': all_captions,
        'tasks': tasks,
        'src': srcs,
    }


def decouple_token_collate(batch: List[Tuple]) -> Dict[str, Any]:
    """
    Collate function for decoupled motion tokens (body + lhand + rhand).
    
    Assumes motion_tokens has shape (T, 3) where:
        - [:, 0] = body tokens
        - [:, 1] = lhand tokens
        - [:, 2] = rhand tokens
    """
    batch = [b for b in batch if b is not None and b[1] is not None]
    
    if len(batch) == 0:
        return None
    
    texts = [b[0] for b in batch]
    tokens = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    names = [b[3] for b in batch]
    all_captions = [b[7] for b in batch]
    tasks = [b[8] for b in batch]
    srcs = [b[9] for b in batch]
    
    max_len = max(lengths)
    
    # Check if tokens are decoupled (T, 3) or single (T,)
    if tokens[0].ndim == 2 and tokens[0].shape[1] == 3:
        body_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
        lhand_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
        rhand_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
        
        for i, t in enumerate(tokens):
            if isinstance(t, torch.Tensor):
                body_tokens[i, :lengths[i]] = t[:lengths[i], 0]
                lhand_tokens[i, :lengths[i]] = t[:lengths[i], 1]
                rhand_tokens[i, :lengths[i]] = t[:lengths[i], 2]
            else:
                body_tokens[i, :lengths[i]] = torch.from_numpy(t[:lengths[i], 0]).long()
                lhand_tokens[i, :lengths[i]] = torch.from_numpy(t[:lengths[i], 1]).long()
                rhand_tokens[i, :lengths[i]] = torch.from_numpy(t[:lengths[i], 2]).long()
        
        return {
            'text': texts,
            'body_tokens': body_tokens,
            'lhand_tokens': lhand_tokens,
            'rhand_tokens': rhand_tokens,
            'lengths': torch.tensor(lengths),
            'name': names,
            'all_captions': all_captions,
            'tasks': tasks,
            'src': srcs,
        }
    else:
        # Fallback to single token sequence
        return motion_token_collate(batch)
