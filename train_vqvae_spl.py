"""
SPL-VQVAE Training Script
- Trains SPL_VQVAE model for sign language motion reconstruction
- Training data: SOKE format (133 dims, axis-angle rotations)
- Metrics: computed on joint 3D coordinates (via SMPL-X forward pass)
- Logs: recon_loss, commit_loss, velocity_loss, mpjpe, mpjve, dtw
- Saves: best model (by MPJPE), periodic checkpoints, visualization videos

Usage:
    python train_spl_vqvae.py --config configs/spl_vqvae.yaml
"""

import os
import sys
import argparse
import yaml
import json
import time
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# DTW
try:
    from dtaidistance import dtw_ndim
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    print("Warning: dtaidistance not installed. DTW metric will be disabled.")

# SMPL-X
try:
    import smplx
    HAS_SMPLX = True
except ImportError:
    HAS_SMPLX = False
    print("Warning: smplx not installed. Metrics will be computed on raw features.")

# SOKE DataLoader
from mGPT.data.humanml import H2SMotionDatasetVQ
from mGPT.data.utils import humanml3d_collate


# =============================================================================
# SMPL-X Conversion (Axis-angle 133 dims → Joint 3D Coordinates)
# =============================================================================

class SMPLXConverter:
    """
    Convert SOKE 133-dim axis-angle representation to joint 3D coordinates.
    
    SOKE 133 dims structure:
        - upper_body_pose: 30 dims (10 joints * 3) - indices [0:30]
        - lhand_pose: 45 dims (15 joints * 3) - indices [30:75]
        - rhand_pose: 45 dims (15 joints * 3) - indices [75:120]
        - jaw_pose: 3 dims (1 joint * 3) - indices [120:123]
        - expression: 10 dims - indices [123:133]
    """
    
    def __init__(self, model_path, gender='neutral', device='cuda'):
        self.device = device
        self.model = None
        
        if HAS_SMPLX and os.path.exists(model_path):
            try:
                self.model = smplx.create(
                    model_path,
                    model_type='smplx',
                    gender=gender,
                    use_pca=False,
                    use_face_contour=True,
                    num_betas=10,
                    num_expression_coeffs=10,
                ).to(device)
                self.model.eval()
                print(f"SMPL-X model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load SMPL-X model: {e}")
                self.model = None
        else:
            if not HAS_SMPLX:
                print("smplx library not installed")
            else:
                print(f"SMPL-X model path not found: {model_path}")
    
    def _parse_soke_params(self, pose_133):
        """
        Parse SOKE 133-dim vector into SMPL-X parameters.
        
        Args:
            pose_133: (B, T, 133) or (T, 133) tensor
        
        Returns:
            dict of SMPL-X parameters
        """
        if pose_133.dim() == 2:
            pose_133 = pose_133.unsqueeze(0)
        
        B, T, _ = pose_133.shape
        
        # SOKE 133 dims breakdown:
        # Original SMPL-X body_pose is 63 dims (21 joints)
        # SOKE removes lower body (11 joints = 33 dims), keeps upper body (10 joints = 30 dims)
        # But the mapping needs to reconstruct the full 63 dims with zeros for lower body
        
        upper_body_pose = pose_133[:, :, 0:30]      # (B, T, 30)
        lhand_pose = pose_133[:, :, 30:75]          # (B, T, 45)
        rhand_pose = pose_133[:, :, 75:120]         # (B, T, 45)
        jaw_pose = pose_133[:, :, 120:123]          # (B, T, 3)
        expression = pose_133[:, :, 123:133]        # (B, T, 10)
        
        # Reconstruct full body_pose (63 dims)
        # Lower body indices (0-10) are zeroed, upper body (11-20) comes from upper_body_pose
        # Actually SOKE removes: root(0) + lower_body(1-11) = 36 dims
        # Keeps: upper_body joints (12-21) = 30 dims
        
        # Full body_pose reconstruction
        lower_body_zeros = torch.zeros(B, T, 33, device=pose_133.device)  # 11 joints * 3
        body_pose = torch.cat([lower_body_zeros, upper_body_pose], dim=-1)  # (B, T, 63)
        
        # Global orient (root) is set to zero since SOKE removes it
        global_orient = torch.zeros(B, T, 3, device=pose_133.device)
        
        # Betas (shape) is set to zero since SOKE removes it
        betas = torch.zeros(B, 10, device=pose_133.device)
        
        return {
            'global_orient': global_orient,      # (B, T, 3)
            'body_pose': body_pose,              # (B, T, 63)
            'left_hand_pose': lhand_pose,        # (B, T, 45)
            'right_hand_pose': rhand_pose,       # (B, T, 45)
            'jaw_pose': jaw_pose,                # (B, T, 3)
            'expression': expression,            # (B, T, 10)
            'betas': betas,                      # (B, 10)
        }
    
    @torch.no_grad()
    def to_joints(self, pose_133, return_vertices=False):
        """
        Convert 133-dim pose to joint 3D coordinates via SMPL-X forward pass.
        
        Args:
            pose_133: (B, T, 133) tensor of SOKE format poses
            return_vertices: whether to return mesh vertices
        
        Returns:
            joints: (B, T, J, 3) tensor of joint coordinates
            vertices: (B, T, V, 3) tensor of vertices (if return_vertices=True)
        """
        if self.model is None:
            # Fallback: return reshaped input (not actual joint positions)
            print("Warning: SMPL-X model not available, returning raw features")
            B, T, _ = pose_133.shape if pose_133.dim() == 3 else (1, *pose_133.shape)
            if pose_133.dim() == 2:
                pose_133 = pose_133.unsqueeze(0)
            # Just reshape to (B, T, 44, 3) approximately
            num_joints = 133 // 3
            joints = pose_133[:, :, :num_joints*3].view(B, T, num_joints, 3)
            return joints if not return_vertices else (joints, None)
        
        if pose_133.dim() == 2:
            pose_133 = pose_133.unsqueeze(0)
        
        B, T, _ = pose_133.shape
        device = pose_133.device
        
        # Parse parameters
        params = self._parse_soke_params(pose_133)
        
        # Process frame by frame (SMPL-X expects batch, not sequence)
        all_joints = []
        all_vertices = [] if return_vertices else None
        
        for t in range(T):
            output = self.model(
                global_orient=params['global_orient'][:, t],
                body_pose=params['body_pose'][:, t],
                left_hand_pose=params['left_hand_pose'][:, t],
                right_hand_pose=params['right_hand_pose'][:, t],
                jaw_pose=params['jaw_pose'][:, t],
                expression=params['expression'][:, t],
                betas=params['betas'],
                leye_pose=torch.zeros(B, 3, device=device),
                reye_pose=torch.zeros(B, 3, device=device),
            )
            
            all_joints.append(output.joints)  # (B, J, 3)
            if return_vertices:
                all_vertices.append(output.vertices)  # (B, V, 3)
        
        joints = torch.stack(all_joints, dim=1)  # (B, T, J, 3)
        
        if return_vertices:
            vertices = torch.stack(all_vertices, dim=1)  # (B, T, V, 3)
            return joints, vertices
        
        return joints
    
    def get_upper_body_joints(self, joints):
        """
        Extract upper body + hand joints for visualization.
        
        SMPL-X joint indices:
            Body: 0-21 (22 joints)
            Left hand: 25-39 (15 joints)
            Right hand: 40-54 (15 joints)
        
        Args:
            joints: (B, T, J, 3) full SMPL-X joints
        
        Returns:
            upper_joints: (B, T, J', 3) upper body + hands
        """
        # Upper body indices (excluding lower body)
        upper_body_idx = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        lhand_idx = list(range(25, 40))
        rhand_idx = list(range(40, 55))
        
        selected_idx = upper_body_idx + lhand_idx + rhand_idx
        
        return joints[:, :, selected_idx, :]


# Global SMPL-X converter (initialized in main)
SMPLX_CONVERTER = None


# =============================================================================
# Skeleton Definitions for Visualization (SMPL-X format)
# =============================================================================

# SMPL-X joint indices
# Body: 0-21 (22 joints), Left hand: 25-39 (15 joints), Right hand: 40-54 (15 joints)

# Upper body indices (after extracting from full SMPL-X)
# We select: pelvis, spine, neck, head, shoulders, elbows, wrists
UPPER_BODY_JOINT_NAMES = [
    'pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head',
    'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist',
    'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist'
]

# Skeleton definitions are in the Visualization section below


# =============================================================================
# Metrics
# =============================================================================

def compute_mpjpe(pred, target, mask=None):
    """
    Mean Per Joint Position Error (MPJPE)
    
    Args:
        pred: (B, T, J, 3) or (B, T, J*3)
        target: (B, T, J, 3) or (B, T, J*3)
        mask: (B, T) optional mask
    
    Returns:
        scalar: mean MPJPE in mm (assuming input is in meters)
    """
    if pred.dim() == 3:
        pred = pred.view(pred.shape[0], pred.shape[1], -1, 3)
        target = target.view(target.shape[0], target.shape[1], -1, 3)
    
    # (B, T, J)
    per_joint_error = torch.norm(pred - target, p=2, dim=-1)
    
    if mask is not None:
        mask = mask.unsqueeze(-1)  # (B, T, 1)
        per_joint_error = per_joint_error * mask
        mpjpe = per_joint_error.sum() / (mask.sum() * pred.shape[2] + 1e-8)
    else:
        mpjpe = per_joint_error.mean()
    
    return mpjpe * 1000  # Convert to mm


def compute_mpjve(pred, target, mask=None):
    """
    Mean Per Joint Velocity Error (MPJVE)
    
    Args:
        pred: (B, T, J, 3) or (B, T, J*3)
        target: (B, T, J, 3) or (B, T, J*3)
        mask: (B, T) optional mask
    
    Returns:
        scalar: mean MPJVE in mm/frame
    """
    if pred.dim() == 3:
        pred = pred.view(pred.shape[0], pred.shape[1], -1, 3)
        target = target.view(target.shape[0], target.shape[1], -1, 3)
    
    # Velocity: difference between consecutive frames
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    
    # (B, T-1, J)
    per_joint_vel_error = torch.norm(pred_vel - target_vel, p=2, dim=-1)
    
    if mask is not None:
        vel_mask = mask[:, 1:] * mask[:, :-1]
        vel_mask = vel_mask.unsqueeze(-1)
        per_joint_vel_error = per_joint_vel_error * vel_mask
        mpjve = per_joint_vel_error.sum() / (vel_mask.sum() * pred.shape[2] + 1e-8)
    else:
        mpjve = per_joint_vel_error.mean()
    
    return mpjve * 1000  # Convert to mm/frame


def compute_dtw(pred, target):
    """
    Dynamic Time Warping distance
    
    Args:
        pred: (T, D) or (T, J, 3) numpy array
        target: (T, D) or (T, J, 3) numpy array
    
    Returns:
        scalar: DTW distance
    """
    if not HAS_DTW:
        return 0.0
    
    # Flatten if 3D
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
    if target.ndim == 3:
        target = target.reshape(target.shape[0], -1)
    
    try:
        distance = dtw_ndim.distance(pred.astype(np.float64), target.astype(np.float64))
        return distance
    except Exception as e:
        print(f"DTW error: {e}")
        return 0.0


# =============================================================================
# Average Meter
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# =============================================================================
# Dataset
# =============================================================================

class SignMotionDataset(Dataset):
    """
    Dataset for sign language motion data.
    Supports loading from .npy files with preprocessed motion features.
    
    For SOKE format: feat_dim=133 (axis-angle rotations)
    For joint coords: feat_dim=num_joints*3
    """
    
    def __init__(
        self,
        data_path,
        split='train',
        max_len=300,
        min_len=20,
        feat_dim=133,        # Feature dimension (133 for SOKE, J*3 for joint coords)
        mean=None,
        std=None,
        win_size=None,       # Window size for training (None = full sequence)
    ):
        """
        Args:
            data_path: Path to data directory containing .npy files or annotations
            split: 'train', 'val', or 'test'
            max_len: Maximum sequence length
            min_len: Minimum sequence length
            feat_dim: Feature dimension per frame (133 for SOKE format)
            mean: Mean for normalization (feat_dim,)
            std: Std for normalization (feat_dim,)
            win_size: Window size for random cropping during training (None = use full sequence)
        """
        self.data_path = data_path
        self.split = split
        self.max_len = max_len
        self.min_len = min_len
        self.feat_dim = feat_dim
        self.win_size = win_size
        
        self.mean = mean if mean is not None else np.zeros(self.feat_dim)
        self.std = std if std is not None else np.ones(self.feat_dim)
        
        # Ensure mean/std match feat_dim
        if len(self.mean) != self.feat_dim:
            print(f"Warning: mean shape {len(self.mean)} != feat_dim {self.feat_dim}, adjusting...")
            if len(self.mean) > self.feat_dim:
                self.mean = self.mean[:self.feat_dim]
            else:
                self.mean = np.pad(self.mean, (0, self.feat_dim - len(self.mean)))
        
        if len(self.std) != self.feat_dim:
            print(f"Warning: std shape {len(self.std)} != feat_dim {self.feat_dim}, adjusting...")
            if len(self.std) > self.feat_dim:
                self.std = self.std[:self.feat_dim]
            else:
                self.std = np.pad(self.std, (0, self.feat_dim - len(self.std)), constant_values=1.0)
        
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} (feat_dim={feat_dim}, win_size={win_size})")
    
    def _load_samples(self):
        """Load sample list from data directory"""
        # Try to load from annotation file
        ann_file = os.path.join(self.data_path, f'{self.split}.json')
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
            for ann in annotations:
                self.samples.append({
                    'name': ann['name'],
                    'path': os.path.join(self.data_path, ann['path']),
                    'text': ann.get('text', ''),
                })
        else:
            # Load all .npy files from directory
            npy_dir = os.path.join(self.data_path, self.split)
            if os.path.exists(npy_dir):
                for fname in os.listdir(npy_dir):
                    if fname.endswith('.npy'):
                        self.samples.append({
                            'name': fname.replace('.npy', ''),
                            'path': os.path.join(npy_dir, fname),
                            'text': '',
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            motion = np.load(sample['path'])  # (T, J*3) or (T, J, 3)
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return dummy data
            motion = np.zeros((self.min_len, self.feat_dim))
        
        # Reshape if needed
        if motion.ndim == 3:
            motion = motion.reshape(motion.shape[0], -1)
        
        # Ensure correct feature dimension
        if motion.shape[1] != self.feat_dim:
            # Pad or truncate
            if motion.shape[1] < self.feat_dim:
                motion = np.pad(motion, ((0, 0), (0, self.feat_dim - motion.shape[1])))
            else:
                motion = motion[:, :self.feat_dim]
        
        # Length handling
        m_length = motion.shape[0]
        
        if m_length < self.min_len:
            # Pad with last frame
            pad_len = self.min_len - m_length
            motion = np.concatenate([motion, np.tile(motion[-1:], (pad_len, 1))], axis=0)
            m_length = self.min_len
        
        if m_length > self.max_len:
            # Uniform sample
            idx_arr = np.linspace(0, m_length - 1, self.max_len, dtype=int)
            motion = motion[idx_arr]
            m_length = self.max_len
        
        # Window sampling for training
        if self.win_size and m_length > self.win_size and self.split == 'train':
            start = random.randint(0, m_length - self.win_size)
            motion = motion[start:start + self.win_size]
            m_length = self.win_size
        
        # Normalize
        motion = (motion - self.mean) / (self.std + 1e-8)
        
        return {
            'motion': torch.from_numpy(motion).float(),
            'length': m_length,
            'name': sample['name'],
            'text': sample['text'],
        }


def collate_fn(batch):
    """Collate function for variable length sequences"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    motions = [b['motion'] for b in batch]
    lengths = [b['length'] for b in batch]
    names = [b['name'] for b in batch]
    texts = [b['text'] for b in batch]
    
    max_len = max(lengths)
    feat_dim = motions[0].shape[-1]
    
    # Pad motions
    padded = torch.zeros(len(batch), max_len, feat_dim)
    for i, m in enumerate(motions):
        padded[i, :lengths[i]] = m[:lengths[i]]
    
    return {
        'motion': padded,
        'lengths': torch.tensor(lengths),
        'names': names,
        'texts': texts,
    }


# =============================================================================
# Video Visualization (3-Panel: GT Video + GT Pose + Pred Pose)
# =============================================================================

# =============================================================================
# Skeleton Definitions for SMPL-X 127 joints (matching ipynb)
# =============================================================================
# SMPL-X outputs 127 joints:
#   Body: 0-21 (22 joints)
#   Jaw, Left Eye, Right Eye: 22-24
#   Left Hand: 25-39 (15 joints)
#   Right Hand: 40-54 (15 joints)
#   Face contour: 55-126 (72 joints)

BODY_IDX = list(range(0, 22))      # 0-21: body
LHAND_IDX = list(range(25, 40))    # 25-39: left hand
RHAND_IDX = list(range(40, 55))    # 40-54: right hand
FACE_IDX = list(range(55, 127))    # 55-126: face (not used in visualization)

# Body connections (SMPL-X 22 body joints)
BODY_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),          # pelvis -> left_hip, right_hip, spine1
    (1, 4), (2, 5),                   # left_hip->left_knee, right_hip->right_knee
    (4, 7), (5, 8),                   # left_knee->left_ankle, right_knee->right_ankle
    (7, 10), (8, 11),                 # left_ankle->left_foot, right_ankle->right_foot
    (3, 6), (6, 9), (9, 12),          # spine chain
    (9, 13), (9, 14),                 # spine3 -> left_collar, right_collar
    (12, 15),                         # neck -> head
    (13, 16), (14, 17),               # collars -> shoulders
    (16, 18), (17, 19),               # shoulders -> elbows
    (18, 20), (19, 21),               # elbows -> wrists
]

# Left hand connections (wrist 20 -> hand joints 25-39)
LHAND_CONNECTIONS = [
    (20, 25), (25, 26), (26, 27),     # thumb
    (20, 28), (28, 29), (29, 30),     # index
    (20, 31), (31, 32), (32, 33),     # middle
    (20, 34), (34, 35), (35, 36),     # ring
    (20, 37), (37, 38), (38, 39),     # pinky
]

# Right hand connections (wrist 21 -> hand joints 40-54)
RHAND_CONNECTIONS = [
    (21, 40), (40, 41), (41, 42),     # thumb
    (21, 43), (43, 44), (44, 45),     # index
    (21, 46), (46, 47), (47, 48),     # middle
    (21, 49), (49, 50), (50, 51),     # ring
    (21, 52), (52, 53), (53, 54),     # pinky
]

ALL_CONNECTIONS = BODY_CONNECTIONS + LHAND_CONNECTIONS + RHAND_CONNECTIONS


def get_line_color(i, j):
    """Get connection line color based on joint indices"""
    if 25 <= i <= 39 or 25 <= j <= 39:
        return 'green', 1.5
    elif 40 <= i <= 54 or 40 <= j <= 54:
        return 'red', 1.5
    else:
        return 'blue', 2.0


def load_video_frames(video_path, target_fps=20, max_frames=None):
    """
    Load video frames from mp4 file.
    
    Args:
        video_path: Path to video file
        target_fps: Target FPS for output frames
        max_frames: Maximum number of frames to load
    
    Returns:
        frames: List of (H, W, 3) numpy arrays
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Cannot open video {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling
        if fps > 0 and fps != target_fps:
            sample_rate = fps / target_fps
        else:
            sample_rate = 1.0
        
        frames = []
        frame_idx = 0
        next_sample = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx >= next_sample:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                next_sample += sample_rate
                
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        return frames
        
    except ImportError:
        print("Warning: OpenCV not installed. Video loading disabled.")
        return None
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None


def load_image_sequence(image_dir, max_frames=None, extensions=('.png', '.jpg', '.jpeg')):
    """
    Load image sequence from a directory.
    
    Args:
        image_dir: Path to directory containing image files
        max_frames: Maximum number of frames to load
        extensions: Tuple of valid image extensions
    
    Returns:
        frames: List of (H, W, 3) numpy arrays or None
    """
    if not os.path.exists(image_dir):
        return None
    
    try:
        import cv2
        import re
        
        # Get list of image files
        files = [f for f in os.listdir(image_dir) if f.lower().endswith(extensions)]
        
        if not files:
            return None
        
        # Sort by number in filename (handles 'images0001.png', 'frame_001.jpg', etc.)
        def extract_number(filename):
            numbers = re.findall(r'\d+', filename)
            return int(numbers[-1]) if numbers else 0
        
        files = sorted(files, key=extract_number)
        
        frames = []
        for i, fname in enumerate(files):
            if max_frames and i >= max_frames:
                break
            
            img_path = os.path.join(image_dir, fname)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img_rgb)
        
        return frames if frames else None
        
    except ImportError:
        print("Warning: OpenCV not installed. Image loading disabled.")
        return None
    except Exception as e:
        print(f"Error loading images from {image_dir}: {e}")
        return None


def get_phoenix_image_path(name, phoenix_features_root):
    """
    Get Phoenix image sequence directory path.
    
    Phoenix-2014T structure:
        phoenix_features_root/  (e.g., .../features/fullFrame-210x260px/)
        ├── dev/
        │   └── {sample_name}/
        │       └── images{frame_id}.png
        ├── test/
        └── train/
    
    Args:
        name: Sample name (e.g., '01April_2010_Thursday_heute_default-0')
        phoenix_features_root: Phoenix features root directory (contains dev/test/train)
    
    Returns:
        image_dir: Path to image directory or None
    """
    if not phoenix_features_root:
        return None
    
    # Try different splits
    possible_paths = [
        os.path.join(phoenix_features_root, 'dev', name),
        os.path.join(phoenix_features_root, 'test', name),
        os.path.join(phoenix_features_root, 'train', name),
        # Direct path (if features_root already includes split)
        os.path.join(phoenix_features_root, name),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if directory contains images
            files = os.listdir(path)
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                return path
    
    return None


def get_image_frames_for_sample(name, src, data_config):
    """
    Get image frames for a sample based on its source dataset.
    
    Args:
        name: Sample name
        src: Source dataset ('how2sign', 'csl', 'phoenix')
        data_config: Data configuration dict with paths
    
    Returns:
        frames: List of (H, W, 3) numpy arrays or None
    """
    # Only Phoenix has original image sequences
    if src != 'phoenix':
        return None
    
    # Use phoenix_features_root for images (separate from poses)
    phoenix_features_root = data_config.get('phoenix_features_root')
    
    if not phoenix_features_root:
        print("Warning: phoenix_features_root not set in config, skipping image loading")
        return None
    
    # Find image directory
    image_dir = get_phoenix_image_path(name, phoenix_features_root)
    
    if image_dir:
        return load_image_sequence(image_dir)
    
    return None


def get_video_path_from_name(name, data_root, split='val'):
    """
    Get video path from sample name (for How2Sign).
    
    Args:
        name: Sample name (e.g., 'G19uBylwQww_0-2-rgb_front')
        data_root: Data root directory
        split: 'train', 'val', or 'test'
    
    Returns:
        video_path: Full path to video file or None
    """
    # How2Sign format: name.mp4
    possible_paths = [
        os.path.join(data_root, split, 'videos', f'{name}.mp4'),
        os.path.join(data_root, 'videos', f'{name}.mp4'),
        os.path.join(data_root, split, f'{name}.mp4'),
        os.path.join(data_root, f'{name}.mp4'),
        # Extract video ID from sample name
        os.path.join(data_root, split, 'videos', f"{name.split('_')[0]}.mp4"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def save_comparison_video_with_frames(
    pred_joints,         # (T, J, 3) - predicted joint coordinates
    gt_joints,           # (T, J, 3) - ground truth joint coordinates  
    video_frames,        # List of (H, W, 3) numpy arrays or None
    save_path,
    fps=20,
    view='front',
    title='',
    show_hands=True,
):
    """
    Save 3-panel comparison video: GT Video | GT Pose | Pred Pose
    
    Args:
        pred_joints: Predicted joint positions (T, J, 3)
        gt_joints: Ground truth joint positions (T, J, 3)
        video_frames: List of video frames or None (will show blank)
        save_path: Output video path (.mp4)
        fps: Frames per second
        view: 'front', 'side', or 'top'
        title: Video title
        show_hands: Whether to show hand joints
    """
    # Convert to numpy if needed
    if isinstance(pred_joints, torch.Tensor):
        pred_joints = pred_joints.cpu().numpy()
    if isinstance(gt_joints, torch.Tensor):
        gt_joints = gt_joints.cpu().numpy()
    
    # Reshape if needed (T, J*3) -> (T, J, 3)
    if pred_joints.ndim == 2:
        pred_joints = pred_joints.reshape(pred_joints.shape[0], -1, 3)
    if gt_joints.ndim == 2:
        gt_joints = gt_joints.reshape(gt_joints.shape[0], -1, 3)
    
    # Determine frame count
    T = min(len(pred_joints), len(gt_joints))
    if video_frames is not None:
        T = min(T, len(video_frames))
    
    pred_joints = pred_joints[:T]
    gt_joints = gt_joints[:T]
    
    # View mapping
    view_map = {'front': (0, 1), 'side': (2, 1), 'top': (0, 2)}
    xi, yi = view_map.get(view, (0, 1))
    
    # Determine connections based on joint count
    num_joints = pred_joints.shape[1]
    
    if num_joints >= 55:
        # Full SMPL-X (127 joints or more) - use standard skeleton
        body_idx = BODY_IDX
        lhand_idx = LHAND_IDX if show_hands else []
        rhand_idx = RHAND_IDX if show_hands else []
        connections = ALL_CONNECTIONS if show_hands else BODY_CONNECTIONS
    elif num_joints >= 44:
        # 44 joints format (from get_upper_body_joints)
        body_idx = list(range(14))
        lhand_idx = list(range(14, 29)) if show_hands else []
        rhand_idx = list(range(29, 44)) if show_hands else []
        # Generate connections for 44-joint format
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 7),
            (3, 5), (5, 8), (8, 10), (10, 12),
            (3, 6), (6, 9), (9, 11), (11, 13),
        ]
        if show_hands:
            for finger in range(5):
                connections.append((12, 14 + finger * 3))
                connections.append((14 + finger * 3, 14 + finger * 3 + 1))
                connections.append((14 + finger * 3 + 1, 14 + finger * 3 + 2))
                connections.append((13, 29 + finger * 3))
                connections.append((29 + finger * 3, 29 + finger * 3 + 1))
                connections.append((29 + finger * 3 + 1, 29 + finger * 3 + 2))
    else:
        # Other joint formats - adaptive
        body_idx = list(range(min(22, num_joints)))
        lhand_idx = []
        rhand_idx = []
        connections = [(i, j) for (i, j) in BODY_CONNECTIONS if i < num_joints and j < num_joints]
    
    # Calculate axis limits from all joints
    all_joints = np.concatenate([pred_joints, gt_joints], axis=0)
    x_min, x_max = all_joints[:, :, xi].min(), all_joints[:, :, xi].max()
    y_min, y_max = all_joints[:, :, yi].min(), all_joints[:, :, yi].max()
    margin = max(x_max - x_min, y_max - y_min) * 0.15
    
    # Create figure with 3 panels
    has_video = video_frames is not None and len(video_frames) > 0
    
    if has_video:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax_video, ax_gt, ax_pred = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        ax_video = None
        ax_gt, ax_pred = axes
    
    # Initialize video frame display
    if has_video:
        im = ax_video.imshow(video_frames[0])
        ax_video.axis('off')
        ax_video.set_title('GT Video', fontsize=14, fontweight='bold')
    
    frame_text = fig.suptitle('', fontsize=14)
    
    def draw_skeleton(ax, joints, title_str):
        ax.clear()
        x, y = joints[:, xi], joints[:, yi]
        
        # Draw connections
        for (i, j) in connections:
            if i < len(joints) and j < len(joints):
                color, lw = get_line_color(i, j)
                ax.plot([x[i], x[j]], [y[i], y[j]], color=color, lw=lw, alpha=0.7)
        
        # Draw joints (smaller sizes)
        valid_body = [i for i in body_idx if i < len(joints)]
        ax.scatter(x[valid_body], y[valid_body], c='blue', s=30, zorder=5, edgecolors='white', linewidths=0.3)
        
        if show_hands:
            valid_lhand = [i for i in lhand_idx if i < len(joints)]
            valid_rhand = [i for i in rhand_idx if i < len(joints)]
            if valid_lhand:
                ax.scatter(x[valid_lhand], y[valid_lhand], c='green', s=15, zorder=5, edgecolors='white', linewidths=0.3)
            if valid_rhand:
                ax.scatter(x[valid_rhand], y[valid_rhand], c='red', s=15, zorder=5, edgecolors='white', linewidths=0.3)
        
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_max + margin, y_min - margin)  # Inverted Y axis (head at top)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title_str, fontsize=14, fontweight='bold')
    
    def update(frame):
        artists = []
        
        # Update video frame
        if has_video and frame < len(video_frames):
            im.set_array(video_frames[frame])
            artists.append(im)
        
        # Update GT skeleton
        draw_skeleton(ax_gt, gt_joints[frame], 'GT Pose')
        
        # Update Pred skeleton
        draw_skeleton(ax_pred, pred_joints[frame], 'Pred Pose')
        
        # Update frame counter
        frame_text.set_text(f'{title} | Frame {frame + 1}/{T}')
        
        return artists
    
    plt.tight_layout()
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    
    # Save video
    try:
        writer = FFMpegWriter(fps=fps, metadata={'title': title}, bitrate=3000)
        anim.save(save_path, writer=writer)
        print(f"Video saved: {save_path}")
    except Exception as e:
        print(f"Error saving video with FFMpeg: {e}")
        # Fallback to GIF
        gif_path = save_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=min(fps, 10))
            print(f"GIF saved: {gif_path}")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")
    
    plt.close(fig)


def save_comparison_video(
    pred_seq,           # (T, J, 3) or (T, J*3)
    gt_seq,             # (T, J, 3) or (T, J*3)
    save_path,
    fps=20,
    view='front',
    title_prefix='',
):
    """
    Save side-by-side comparison video (2-panel: GT Pose | Pred Pose).
    Wrapper for backward compatibility.
    """
    save_comparison_video_with_frames(
        pred_joints=pred_seq,
        gt_joints=gt_seq,
        video_frames=None,
        save_path=save_path,
        fps=fps,
        view=view,
        title=title_prefix,
    )


# =============================================================================
# Training Functions
# =============================================================================

def compute_codebook_stats(all_codes, codebook_size):
    """
    Compute codebook usage statistics.
    
    Args:
        all_codes: List of code tensors or single tensor (N,) or (B, N)
        codebook_size: Total number of codes in codebook
    
    Returns:
        dict with usage statistics
    """
    # Flatten all codes
    if isinstance(all_codes, list):
        all_codes = torch.cat([c.flatten() for c in all_codes])
    else:
        all_codes = all_codes.flatten()
    
    # Count usage per code
    code_counts = torch.bincount(all_codes.long(), minlength=codebook_size)
    
    # Statistics
    total_tokens = len(all_codes)
    used_codes = (code_counts > 0).sum().item()
    usage_rate = used_codes / codebook_size * 100
    
    # Per-code usage (only for used codes)
    used_mask = code_counts > 0
    if used_mask.sum() > 0:
        used_counts = code_counts[used_mask].float()
        min_usage = used_counts.min().item()
        max_usage = used_counts.max().item()
        avg_usage = used_counts.mean().item()
        std_usage = used_counts.std().item() if len(used_counts) > 1 else 0.0
    else:
        min_usage = max_usage = avg_usage = std_usage = 0.0
    
    # Perplexity (entropy-based)
    probs = code_counts.float() / (total_tokens + 1e-8)
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    perplexity = torch.exp(entropy).item()
    
    return {
        'used_codes': used_codes,
        'total_codes': codebook_size,
        'usage_rate': usage_rate,
        'min_usage': min_usage,
        'max_usage': max_usage,
        'avg_usage': avg_usage,
        'std_usage': std_usage,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
    }


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, codebook_size=512):
    """Train for one epoch"""
    model.train()
    
    recon_meter = AverageMeter()
    commit_meter = AverageMeter()
    velocity_meter = AverageMeter()
    total_meter = AverageMeter()
    perplexity_meter = AverageMeter()
    
    # Collect all codes for statistics
    all_codes = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch in pbar:
        if batch is None:
            continue
        
        motion = batch['motion'].to(device)
        # Handle both 'lengths' (tensor) and 'length' (list) keys
        lengths = batch.get('lengths', batch.get('length'))
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths)
        lengths = lengths.to(device)
        
        B = motion.shape[0]
        
        # Forward
        output, commit_loss, perplexity, codes = model(motion, lengths)
        
        # Collect codes
        all_codes.append(codes.detach().cpu())
        
        # Create mask
        max_len = motion.shape[1]
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Loss
        losses = loss_fn(output, motion, commit_loss, mask)
        
        # Backward
        optimizer.zero_grad()
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update meters
        recon_meter.update(losses['recon_loss'].item(), B)
        commit_meter.update(losses['commit_loss'].item(), B)
        velocity_meter.update(losses['velocity_loss'].item(), B)
        total_meter.update(losses['total_loss'].item(), B)
        perplexity_meter.update(perplexity.item(), B)
        
        pbar.set_postfix({
            'recon': f'{recon_meter.avg:.4f}',
            'commit': f'{commit_meter.avg:.4f}',
            'ppl': f'{perplexity_meter.avg:.1f}',
        })
    
    # Compute codebook statistics
    codebook_stats = compute_codebook_stats(all_codes, codebook_size)
    
    return {
        'recon_loss': recon_meter.avg,
        'commit_loss': commit_meter.avg,
        'velocity_loss': velocity_meter.avg,
        'total_loss': total_meter.avg,
        'perplexity': perplexity_meter.avg,
        'codebook': codebook_stats,
    }


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, epoch, mean=None, std=None, codebook_size=512, smplx_converter=None):
    """Evaluate model with SMPL-X forward pass for metrics"""
    model.eval()
    
    recon_meter = AverageMeter()
    commit_meter = AverageMeter()
    velocity_meter = AverageMeter()
    total_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    mpjve_meter = AverageMeter()
    dtw_meter = AverageMeter()
    perplexity_meter = AverageMeter()
    
    # Collect all codes for statistics
    all_codes = []
    
    # Store samples for visualization
    vis_samples = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Eval]')
    
    for batch in pbar:
        if batch is None:
            continue
        
        motion = batch['motion'].to(device)
        # Handle both 'lengths' (tensor) and 'length' (list) keys
        lengths = batch.get('lengths', batch.get('length'))
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths)
        lengths = lengths.to(device)
        # Handle both 'names' and 'name' keys
        names = batch.get('names', batch.get('name', []))
        
        B = motion.shape[0]
        
        # Forward
        output, commit_loss, perplexity, codes = model(motion, lengths)
        
        # Collect codes
        all_codes.append(codes.detach().cpu())
        
        # Create mask
        max_len = motion.shape[1]
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Loss (computed on raw 133-dim features)
        losses = loss_fn(output, motion, commit_loss, mask)
        
        # Denormalize for SMPL-X conversion
        if mean is not None and std is not None:
            if isinstance(mean, torch.Tensor):
                mean_t = mean.clone().detach().to(device).float()
            else:
                mean_t = torch.from_numpy(mean).to(device).float()
            if isinstance(std, torch.Tensor):
                std_t = std.clone().detach().to(device).float()
            else:
                std_t = torch.from_numpy(std).to(device).float()
            output_denorm = output * std_t + mean_t
            motion_denorm = motion * std_t + mean_t
        else:
            output_denorm = output
            motion_denorm = motion
        
        # Convert to joint coordinates via SMPL-X for metrics
        if smplx_converter is not None and smplx_converter.model is not None:
            try:
                # Convert to joint coordinates
                pred_joints = smplx_converter.to_joints(output_denorm)  # (B, T, J, 3)
                gt_joints = smplx_converter.to_joints(motion_denorm)    # (B, T, J, 3)
                
                # MPJPE on joint coordinates
                mpjpe = compute_mpjpe(pred_joints, gt_joints, mask.float())
                
                # MPJVE on joint coordinates
                mpjve = compute_mpjve(pred_joints, gt_joints, mask.float())
                
                mpjpe_meter.update(mpjpe.item(), B)
                mpjve_meter.update(mpjve.item(), B)
                
                # DTW (sample-wise, expensive)
                if HAS_DTW and B <= 4:
                    for i in range(B):
                        L = int(lengths[i].item())
                        pred_np = pred_joints[i, :L].cpu().numpy().reshape(L, -1)
                        gt_np = gt_joints[i, :L].cpu().numpy().reshape(L, -1)
                        dtw_dist = compute_dtw(pred_np, gt_np)
                        dtw_meter.update(dtw_dist, 1)
                
                # Store for visualization (with joint coordinates)
                # Prioritize Phoenix samples (they have original images)
                current_src = batch.get('src', ['unknown'])[0] if batch.get('src') else 'unknown'
                should_store = False
                replace_idx = None
                
                if len(vis_samples) < 3:
                    should_store = True
                elif current_src == 'phoenix':
                    # Replace first non-phoenix sample with phoenix
                    for idx, s in enumerate(vis_samples):
                        if s.get('src') != 'phoenix':
                            replace_idx = idx
                            should_store = True
                            break
                
                if should_store:
                    L = int(lengths[0].item())
                    # Use full 127 joints for visualization (matching ipynb)
                    sample_data = {
                        'pred': pred_joints[0, :L].cpu().numpy(),
                        'gt': gt_joints[0, :L].cpu().numpy(),
                        'name': names[0] if names else '',
                        'src': current_src,
                    }
                    if replace_idx is not None:
                        vis_samples[replace_idx] = sample_data
                    else:
                        vis_samples.append(sample_data)
                    
            except Exception as e:
                print(f"SMPL-X conversion error: {e}")
                # Fallback to raw feature metrics
                mpjpe = compute_mpjpe(output_denorm.view(B, -1, 133//3, 3), 
                                      motion_denorm.view(B, -1, 133//3, 3), mask.float())
                mpjpe_meter.update(mpjpe.item(), B)
        else:
            # Fallback: compute metrics on raw features (less meaningful but works)
            # Reshape 133 dims to approximate joint structure
            num_pseudo_joints = 133 // 3  # 44
            pred_reshaped = output_denorm[:, :, :num_pseudo_joints*3].view(B, -1, num_pseudo_joints, 3)
            gt_reshaped = motion_denorm[:, :, :num_pseudo_joints*3].view(B, -1, num_pseudo_joints, 3)
            
            mpjpe = compute_mpjpe(pred_reshaped, gt_reshaped, mask.float())
            mpjve = compute_mpjve(pred_reshaped, gt_reshaped, mask.float())
            
            mpjpe_meter.update(mpjpe.item(), B)
            mpjve_meter.update(mpjve.item(), B)
            
            # Store for visualization (raw features reshaped)
            # Prioritize Phoenix samples (they have original images)
            current_src = batch.get('src', ['unknown'])[0] if batch.get('src') else 'unknown'
            should_store = False
            replace_idx = None
            
            if len(vis_samples) < 3:
                should_store = True
            elif current_src == 'phoenix':
                # Replace first non-phoenix sample with phoenix
                for idx, s in enumerate(vis_samples):
                    if s.get('src') != 'phoenix':
                        replace_idx = idx
                        should_store = True
                        break
            
            if should_store:
                L = int(lengths[0].item())
                sample_data = {
                    'pred': pred_reshaped[0, :L].cpu().numpy(),
                    'gt': gt_reshaped[0, :L].cpu().numpy(),
                    'name': names[0] if names else '',
                    'src': current_src,
                }
                if replace_idx is not None:
                    vis_samples[replace_idx] = sample_data
                else:
                    vis_samples.append(sample_data)
        
        # Update meters
        recon_meter.update(losses['recon_loss'].item(), B)
        commit_meter.update(losses['commit_loss'].item(), B)
        velocity_meter.update(losses['velocity_loss'].item(), B)
        total_meter.update(losses['total_loss'].item(), B)
        perplexity_meter.update(perplexity.item(), B)
        
        pbar.set_postfix({
            'mpjpe': f'{mpjpe_meter.avg:.2f}',
            'mpjve': f'{mpjve_meter.avg:.2f}',
            'ppl': f'{perplexity_meter.avg:.1f}',
        })
    
    # Compute codebook statistics
    codebook_stats = compute_codebook_stats(all_codes, codebook_size)
    
    return {
        'recon_loss': recon_meter.avg,
        'commit_loss': commit_meter.avg,
        'velocity_loss': velocity_meter.avg,
        'total_loss': total_meter.avg,
        'mpjpe': mpjpe_meter.avg,
        'mpjve': mpjve_meter.avg,
        'dtw': dtw_meter.avg if dtw_meter.count > 0 else 0.0,
        'perplexity': perplexity_meter.avg,
        'codebook': codebook_stats,
        'vis_samples': vis_samples,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path, is_best=False):
    """Save model checkpoint"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }
    
    torch.save(state, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")


# =============================================================================
# Main Training Loop
# =============================================================================

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract sub-configs
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    eval_config = config.get('evaluation', {})
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    seed = train_config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = config.get('name', 'spl_vqvae')
    output_dir = os.path.join(train_config.get('model_dir', 'checkpoints/spl_vqvae'), timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'videos'), exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print(f"Output directory: {output_dir}")
    
    # Load mean/std (SOKE style filtering)
    mean_path = data_config.get('mean_path')
    std_path = data_config.get('std_path')
    
    if mean_path and os.path.exists(mean_path):
        mean = torch.load(mean_path) if mean_path.endswith('.pt') else torch.from_numpy(np.load(mean_path))
        # SOKE filtering: remove lower body and shape
        # Original 179 -> 143 (remove lower body) -> 133 (remove shape)
        mean = mean[(3 + 3 * 11):]  # Remove lower body (root + 11 lower body joints)
        mean = torch.cat([mean[:-20], mean[-10:]], dim=0)  # Remove shape (10 dims), keep expression (10 dims)
        print(f"Loaded and filtered mean from {mean_path}, shape: {mean.shape}")
    else:
        mean = torch.zeros(133)
        print("Warning: mean not found, using zeros")
    
    if std_path and os.path.exists(std_path):
        std = torch.load(std_path) if std_path.endswith('.pt') else torch.from_numpy(np.load(std_path))
        # Same filtering as mean
        std = std[(3 + 3 * 11):]
        std = torch.cat([std[:-20], std[-10:]], dim=0)
        print(f"Loaded and filtered std from {std_path}, shape: {std.shape}")
    else:
        std = torch.ones(133)
        print("Warning: std not found, using ones")
    
    # Common dataset kwargs
    dataset_kwargs = {
        'data_root': data_config.get('data_root'),
        'mean': mean,
        'std': std,
        'max_motion_length': data_config.get('max_motion_length', 300),
        'min_motion_length': data_config.get('min_motion_length', 40),
        'unit_length': data_config.get('unit_length', 4),
        'dataset_name': data_config.get('dataset_name', 'how2sign_csl_phoenix'),
        'csl_root': data_config.get('csl_root'),
        # Phoenix poses path (H2SMotionDatasetVQ expects 'phoenix_root' for poses)
        'phoenix_root': data_config.get('phoenix_poses_root', data_config.get('phoenix_root')),
    }
    
    # Create datasets using SOKE H2SMotionDatasetVQ
    win_size = data_config.get('win_size', None)
    
    train_dataset = H2SMotionDatasetVQ(
        split='train',
        win_size=win_size,
        **dataset_kwargs
    )
    
    val_dataset = H2SMotionDatasetVQ(
        split='val',
        win_size=None,  # No window sampling for validation
        **dataset_kwargs
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 64),
        shuffle=True,
        num_workers=train_config.get('num_workers', 8),
        collate_fn=humanml3d_collate,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_config.get('batch_size', 64),
        shuffle=False,
        num_workers=eval_config.get('num_workers', 4),
        collate_fn=humanml3d_collate,
        pin_memory=True,
    )
    
    # Create model
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.vqvae_spl import SPL_VQVAE, SPLVQVAELoss
    
    # Determine input format based on nfeats
    nfeats = model_config.get('nfeats', 133)
    input_format = 'soke' if nfeats == 133 else 'joint_coords'
    
    model = SPL_VQVAE(
        embed_dim=model_config.get('embed_dim', 512),
        depth=model_config.get('depth', 4),
        num_heads=model_config.get('num_heads', 8),
        mlp_dim=model_config.get('mlp_dim', 2048),
        num_queries=model_config.get('num_queries', 32),
        codebook_size=model_config.get('codebook_size', 512),
        code_dim=model_config.get('code_dim', 512),
        down_t=model_config.get('down_t', 2),
        stride_t=model_config.get('stride_t', 2),
        max_len=data_config.get('max_motion_length', 300),
        dropout=model_config.get('dropout', 0.1),
        spl_hidden_layers=model_config.get('spl_hidden_layers', 3),
        spl_hidden_units=model_config.get('spl_hidden_units', 512),
        input_format=input_format,
        nfeats=nfeats,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input format: {input_format} ({nfeats} dims)")
    print(f"Codebook size: {model_config.get('codebook_size', 512)}")
    print(f"Q-Former queries: {model_config.get('num_queries', 32)}")
    
    # Loss function
    loss_fn = SPLVQVAELoss(
        lambda_recon=train_config.get('lambda_recon', 1.0),
        lambda_velocity=train_config.get('lambda_velocity', 0.5),
        lambda_commit=train_config.get('lambda_commit', 0.02),
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.get('learning_rate', 1e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
        betas=tuple(train_config.get('betas', [0.9, 0.999])),
    )
    
    # Scheduler
    num_epochs = train_config.get('epochs', 500)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=train_config.get('learning_rate_min', 1e-6),
    )
    
    # Load checkpoint if resuming
    start_epoch = 1
    best_mpjpe = float('inf')
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mpjpe = checkpoint['metrics'].get('mpjpe', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best MPJPE: {best_mpjpe:.2f}")
    
    # Training log
    log_path = os.path.join(output_dir, 'training_log.json')
    training_log = []
    
    # Codebook size for stats
    codebook_size = model_config.get('codebook_size', 512)
    
    # Initialize SMPL-X converter for metric computation
    smplx_config = config.get('smplx', {})
    eval_config = config.get('evaluation', {})
    smplx_converter = None
    
    if eval_config.get('use_smplx', True) and HAS_SMPLX:
        smplx_model_path = smplx_config.get('model_path', 'deps/smpl_models')
        if os.path.exists(smplx_model_path):
            smplx_converter = SMPLXConverter(
                model_path=smplx_model_path,
                gender=smplx_config.get('gender', 'neutral'),
                device=device,
            )
            print(f"SMPL-X converter initialized for metric computation")
        else:
            print(f"SMPL-X model path not found: {smplx_model_path}")
            print("Metrics will be computed on raw features (axis-angle)")
    else:
        print("SMPL-X disabled or not installed. Metrics will be computed on raw features.")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*80}")
        
        # Train (on raw 133-dim axis-angle data)
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch,
            codebook_size=codebook_size
        )
        
        # Evaluate (metrics on joint 3D coordinates via SMPL-X)
        val_metrics = evaluate(
            model, val_loader, loss_fn, device, epoch, mean, std,
            codebook_size=codebook_size,
            smplx_converter=smplx_converter
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        epoch_log = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train': {k: v for k, v in train_metrics.items()},
            'val': {k: v for k, v in val_metrics.items() if k != 'vis_samples'},
        }
        training_log.append(epoch_log)
        
        # Save log
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # Print metrics
        print(f"\n[Train]")
        print(f"  Loss - Recon: {train_metrics['recon_loss']:.4f}, "
              f"Commit: {train_metrics['commit_loss']:.4f}, "
              f"Velocity: {train_metrics['velocity_loss']:.4f}")
        print(f"  Codebook - Used: {train_metrics['codebook']['used_codes']}/{train_metrics['codebook']['total_codes']} "
              f"({train_metrics['codebook']['usage_rate']:.1f}%), "
              f"Perplexity: {train_metrics['codebook']['perplexity']:.1f}")
        print(f"  Code Usage - Min: {train_metrics['codebook']['min_usage']:.0f}, "
              f"Max: {train_metrics['codebook']['max_usage']:.0f}, "
              f"Avg: {train_metrics['codebook']['avg_usage']:.1f} ± {train_metrics['codebook']['std_usage']:.1f}")
        
        print(f"\n[Val]")
        print(f"  Loss - Recon: {val_metrics['recon_loss']:.4f}, "
              f"Commit: {val_metrics['commit_loss']:.4f}")
        print(f"  Metrics - MPJPE: {val_metrics['mpjpe']:.2f}mm, "
              f"MPJVE: {val_metrics['mpjve']:.2f}mm, "
              f"DTW: {val_metrics['dtw']:.4f}")
        print(f"  Codebook - Used: {val_metrics['codebook']['used_codes']}/{val_metrics['codebook']['total_codes']} "
              f"({val_metrics['codebook']['usage_rate']:.1f}%), "
              f"Perplexity: {val_metrics['codebook']['perplexity']:.1f}")
        print(f"  Code Usage - Min: {val_metrics['codebook']['min_usage']:.0f}, "
              f"Max: {val_metrics['codebook']['max_usage']:.0f}, "
              f"Avg: {val_metrics['codebook']['avg_usage']:.1f} ± {val_metrics['codebook']['std_usage']:.1f}")
        
        # Check if best
        is_best = val_metrics['mpjpe'] < best_mpjpe
        if is_best:
            best_mpjpe = val_metrics['mpjpe']
            print(f"\n*** New best MPJPE: {best_mpjpe:.2f}mm ***")
        
        # Save checkpoint
        ckpt_path = os.path.join(output_dir, 'checkpoints', f'epoch_{epoch:04d}.pth')
        save_every = train_config.get('save_every', 10)
        
        # Always save best, periodic every save_every epochs
        if is_best or epoch % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, ckpt_path, is_best)
            
            # Save visualization video
            if val_metrics['vis_samples']:
                # Prefer Phoenix samples (they have original images)
                sample = None
                for s in val_metrics['vis_samples']:
                    if s.get('src') == 'phoenix':
                        sample = s
                        break
                if sample is None:
                    sample = val_metrics['vis_samples'][0]
                
                video_path = os.path.join(output_dir, 'videos', f'epoch_{epoch:04d}.mp4')
                if is_best:
                    video_path = os.path.join(output_dir, 'videos', f'best_epoch_{epoch:04d}.mp4')
                
                # Try to load image frames (only Phoenix has original images)
                sample_name = sample.get('name', '')
                sample_src = sample.get('src', 'unknown')
                max_frames = sample['pred'].shape[0]
                
                image_frames = get_image_frames_for_sample(
                    name=sample_name,
                    src=sample_src,
                    data_config=data_config
                )
                
                # If Phoenix images found, subsample to match pose length
                if image_frames is not None and len(image_frames) > max_frames:
                    # Uniform subsample
                    indices = np.linspace(0, len(image_frames) - 1, max_frames, dtype=int)
                    image_frames = [image_frames[i] for i in indices]
                
                save_comparison_video_with_frames(
                    pred_joints=sample['pred'],
                    gt_joints=sample['gt'],
                    video_frames=image_frames,
                    save_path=video_path,
                    fps=20,
                    view='front',
                    title=f"Epoch {epoch} | MPJPE: {val_metrics['mpjpe']:.2f}mm | {sample_src}:{sample_name[:30]}",
                )
    
    print(f"\nTraining completed! Best MPJPE: {best_mpjpe:.2f}mm")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SPL-VQVAE')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    main(args)