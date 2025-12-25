"""
Metrics for motion evaluation.
- MPJPE: Mean Per Joint Position Error
- DTW: Dynamic Time Warping
"""
import numpy as np
import torch
from typing import Tuple, Optional
from scipy.spatial.distance import cdist
from fastdtw import fastdtw


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Mean Per Joint Position Error.
    
    Args:
        pred: Predicted poses (T, J*3) or (T, J, 3)
        gt: Ground truth poses (T, J*3) or (T, J, 3)
    
    Returns:
        MPJPE in the same unit as input (usually mm or m)
    """
    # Reshape to (T, J, 3) if needed
    if pred.ndim == 2:
        pred = pred.reshape(pred.shape[0], -1, 3)
    if gt.ndim == 2:
        gt = gt.reshape(gt.shape[0], -1, 3)
    
    # Align lengths (take minimum)
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    # Compute per-joint error
    # (T, J, 3) -> (T, J)
    errors = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))
    
    # Mean over time and joints
    return float(np.mean(errors))


def mpjpe_per_joint(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    MPJPE per joint.
    
    Returns:
        Array of shape (J,) with error per joint
    """
    if pred.ndim == 2:
        pred = pred.reshape(pred.shape[0], -1, 3)
    if gt.ndim == 2:
        gt = gt.reshape(gt.shape[0], -1, 3)
    
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    
    errors = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))  # (T, J)
    return np.mean(errors, axis=0)  # (J,)


def dtw_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dynamic Time Warping distance.
    Handles sequences of different lengths.
    
    Args:
        pred: Predicted poses (T1, D)
        gt: Ground truth poses (T2, D)
    
    Returns:
        DTW distance (normalized by path length)
    """
    # Flatten if needed
    if pred.ndim > 2:
        pred = pred.reshape(pred.shape[0], -1)
    if gt.ndim > 2:
        gt = gt.reshape(gt.shape[0], -1)
    
    # Compute DTW
    distance, path = fastdtw(pred, gt, dist=lambda x, y: np.linalg.norm(x - y))
    
    # Normalize by path length
    return float(distance / len(path))


def compute_motion_metrics(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
    fps: float = 25.0
) -> dict:
    """
    Compute all motion metrics.
    
    Args:
        pred_poses: (T1, J*3) or (T1, J, 3)
        gt_poses: (T2, J*3) or (T2, J, 3)
        fps: Frames per second
    
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # MPJPE
    metrics['mpjpe'] = mpjpe(pred_poses, gt_poses)
    
    # DTW
    try:
        metrics['dtw'] = dtw_distance(pred_poses, gt_poses)
    except Exception as e:
        metrics['dtw'] = -1.0
        print(f"DTW computation failed: {e}")
    
    # Length difference
    metrics['len_pred'] = len(pred_poses)
    metrics['len_gt'] = len(gt_poses)
    metrics['len_diff'] = abs(len(pred_poses) - len(gt_poses))
    
    return metrics


def compute_part_metrics(
    pred_poses: np.ndarray,
    gt_poses: np.ndarray,
) -> dict:
    """
    Compute metrics for each body part separately.
    
    Joint structure (50 joints):
    - body: 0-7 (8 joints) -> dims 0-23
    - rhand: 8-28 (21 joints) -> dims 24-86
    - lhand: 29-49 (21 joints) -> dims 87-149
    """
    # Reshape to (T, 150)
    if pred_poses.ndim == 3:
        pred_poses = pred_poses.reshape(pred_poses.shape[0], -1)
    if gt_poses.ndim == 3:
        gt_poses = gt_poses.reshape(gt_poses.shape[0], -1)
    
    min_len = min(len(pred_poses), len(gt_poses))
    pred = pred_poses[:min_len]
    gt = gt_poses[:min_len]
    
    metrics = {}
    
    # Body (dims 0-23)
    pred_body = pred[:, :24].reshape(-1, 8, 3)
    gt_body = gt[:, :24].reshape(-1, 8, 3)
    metrics['mpjpe_body'] = mpjpe(pred_body, gt_body)
    
    # RHand (dims 24-86)
    pred_rhand = pred[:, 24:87].reshape(-1, 21, 3)
    gt_rhand = gt[:, 24:87].reshape(-1, 21, 3)
    metrics['mpjpe_rhand'] = mpjpe(pred_rhand, gt_rhand)
    
    # LHand (dims 87-149)
    pred_lhand = pred[:, 87:150].reshape(-1, 21, 3)
    gt_lhand = gt[:, 87:150].reshape(-1, 21, 3)
    metrics['mpjpe_lhand'] = mpjpe(pred_lhand, gt_lhand)
    
    return metrics


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
        if val < 0:  # Skip invalid values
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0