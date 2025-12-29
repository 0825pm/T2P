# coding: utf-8
"""
Motion Evaluation Metrics

MPJPE: Mean Per Joint Position Error
DTW: Dynamic Time Warping Distance
"""

import numpy as np
import torch
from typing import Dict, Optional, Union

# Try to import DTW library
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False
    print("Warning: fastdtw not installed. DTW metric will be disabled.")

try:
    from dtaidistance import dtw_ndim
    HAS_DTAIDISTANCE = True
except ImportError:
    HAS_DTAIDISTANCE = False


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        if val is None or np.isnan(val):
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mpjpe(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Compute Mean Per Joint Position Error.
    
    Args:
        pred: Predicted poses (T, D) or (B, T, D)
        gt: Ground truth poses (T, D) or (B, T, D)
        mask: Optional mask for valid frames
        
    Returns:
        MPJPE value (mm)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure same length
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    # Compute difference
    diff = pred - gt
    
    # Reshape to (T, J, 3) if possible
    if diff.ndim == 2:
        T, D = diff.shape
        if D % 3 == 0:
            J = D // 3
            diff = diff.reshape(T, J, 3)
            # Per-joint error
            error = np.sqrt((diff ** 2).sum(axis=-1))  # (T, J)
            error = error.mean(axis=-1)  # (T,)
        else:
            # Treat as flat vector
            error = np.sqrt((diff ** 2).mean(axis=-1))  # (T,)
    else:
        error = np.sqrt((diff ** 2).mean(axis=-1))
    
    if mask is not None:
        mask = mask[:min_len]
        if mask.sum() > 0:
            return (error * mask).sum() / mask.sum()
    
    return float(error.mean())


def mpjpe_per_part(
    pred: np.ndarray,
    gt: np.ndarray,
    part_dims: Dict[str, tuple] = None,
) -> Dict[str, float]:
    """
    Compute MPJPE per body part.
    
    Args:
        pred: Predicted poses (T, 133)
        gt: Ground truth poses (T, 133)
        part_dims: Dict mapping part names to (start, end) indices
        
    Returns:
        Dict with MPJPE per part
    """
    if part_dims is None:
        # SOKE 133-dim format
        part_dims = {
            'body': (0, 24),
            'rhand': (24, 69),   # 24 + 45
            'lhand': (69, 114),  # 69 + 45
            'jaw': (114, 117),   # 3 dims
            'expr': (117, 127),  # 10 dims
        }
    
    results = {}
    
    for part_name, (start, end) in part_dims.items():
        pred_part = pred[..., start:end]
        gt_part = gt[..., start:end]
        results[f'mpjpe_{part_name}'] = mpjpe(pred_part, gt_part)
    
    return results


def compute_part_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for body, left hand, right hand.
    
    Assumes SOKE 133-dim format:
    - body: 0-24
    - lhand: 69-114 (or 30-75 in original format)
    - rhand: 24-69 (or 75-120 in original format)
    """
    # SOKE filtered format (133 dims)
    # body(24) + rhand(45) + lhand(45) + jaw(3) + expr(10) = 127 + 6 extra?
    # Actually: body(24) + rhand(63) + lhand(46) = 133
    
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    # Try to detect format
    D = pred.shape[-1]
    
    if D == 133:
        # SOKE format: body(24) + rhand(63) + lhand(46)
        body_end = 24
        rhand_end = 24 + 63
        lhand_end = rhand_end + 46
    elif D == 150:
        # Original Phoenix format
        body_end = 30
        lhand_end = 75
        rhand_end = 120
    else:
        # Fallback: split evenly into 3 parts
        part_size = D // 3
        body_end = part_size
        lhand_end = 2 * part_size
        rhand_end = D
    
    results = {
        'mpjpe_body': mpjpe(pred[..., :body_end], gt[..., :body_end]),
    }
    
    if D >= 133:
        results['mpjpe_lhand'] = mpjpe(pred[..., rhand_end:lhand_end] if D == 133 else pred[..., body_end:lhand_end], 
                                       gt[..., rhand_end:lhand_end] if D == 133 else gt[..., body_end:lhand_end])
        results['mpjpe_rhand'] = mpjpe(pred[..., body_end:rhand_end] if D == 133 else pred[..., lhand_end:rhand_end],
                                       gt[..., body_end:rhand_end] if D == 133 else gt[..., lhand_end:rhand_end])
    
    return results


def dtw_distance(
    pred: Union[np.ndarray, torch.Tensor],
    gt: Union[np.ndarray, torch.Tensor],
    normalize: bool = True,
) -> float:
    """
    Compute Dynamic Time Warping distance.
    
    Args:
        pred: Predicted sequence (T1, D)
        gt: Ground truth sequence (T2, D)
        normalize: Whether to normalize by sequence length
        
    Returns:
        DTW distance
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    
    # Flatten to 2D
    if pred.ndim > 2:
        pred = pred.reshape(pred.shape[0], -1)
    if gt.ndim > 2:
        gt = gt.reshape(gt.shape[0], -1)
    
    if HAS_DTAIDISTANCE:
        try:
            distance = dtw_ndim.distance(pred, gt)
            if normalize:
                distance = distance / max(len(pred), len(gt))
            return float(distance)
        except:
            pass
    
    if HAS_FASTDTW:
        try:
            distance, _ = fastdtw(pred, gt, dist=euclidean)
            if normalize:
                distance = distance / max(len(pred), len(gt))
            return float(distance)
        except:
            pass
    
    # Fallback: simple L2 distance with padding
    max_len = max(len(pred), len(gt))
    
    pred_padded = np.zeros((max_len, pred.shape[-1]))
    pred_padded[:len(pred)] = pred
    
    gt_padded = np.zeros((max_len, gt.shape[-1]))
    gt_padded[:len(gt)] = gt
    
    distance = np.sqrt(((pred_padded - gt_padded) ** 2).sum())
    if normalize:
        distance = distance / max_len
    
    return float(distance)


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all metrics.
    
    Args:
        pred: Predicted poses
        gt: Ground truth poses
        
    Returns:
        Dict with all metrics
    """
    results = {
        'mpjpe': mpjpe(pred, gt),
        'dtw': dtw_distance(pred, gt),
    }
    
    # Add part-wise metrics
    part_metrics = compute_part_metrics(pred, gt)
    results.update(part_metrics)
    
    # Length difference
    results['len_diff'] = abs(len(pred) - len(gt))
    
    return results
