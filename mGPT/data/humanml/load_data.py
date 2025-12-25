# coding: utf-8
"""
SOKE Data Loading Functions
https://github.com/2000ZRL/SOKE

각 데이터셋별 샘플 로딩 함수
- How2Sign: CSV + poses 디렉토리
- CSL: gzip pickle + poses 디렉토리  
- Phoenix: gzip pickle + 프레임별 pkl
"""
import pickle
import numpy as np
import os
import math
from bisect import bisect_left, bisect_right


# SMPL-X 파라미터 키
keys = [
    'smplx_root_pose',    # (3,)  - 1 Joint
    'smplx_body_pose',    # (63,) - 21 Joints
    'smplx_lhand_pose',   # (45,) - 15 Joints
    'smplx_rhand_pose',   # (45,) - 15 Joints
    'smplx_jaw_pose',     # (3,)  - 1 Joint
    'smplx_shape',        # (10,)
    'smplx_expr'          # (10,)
]
# Total: 3 + 63 + 45 + 45 + 3 + 10 + 10 = 179 dims


def sample(input, count):
    """Uniform sampling from input list."""
    ss = float(len(input)) / count
    return [input[int(math.floor(i * ss))] for i in range(count)]


def load_h2s_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """
    Load How2Sign sample.
    
    Args:
        ann: annotation dict with 'name', 'fps', 'text', optionally 'split'
        data_dir: root directory for poses
        need_pose: whether to load pose data
        code_path: path to pre-extracted motion codes
        need_code: whether to load motion codes
        
    Returns:
        clip_poses: (T, 133) pose array
        clip_text: text annotation
        name: sample name
        code: motion codes if need_code else None
    """
    name = ann['name']
    
    # Handle split in path
    if 'split' in ann:
        split = ann['split']
        base_dir = os.path.join(data_dir, split, 'poses', name)
    else:
        base_dir = os.path.join(data_dir, name)
    
    fps = ann['fps']
    
    # Get frame list
    frame_list = [
        os.path.join(base_dir, f'{name}_{frame_id}_3D.pkl') 
        for frame_id in range(len(os.listdir(base_dir)))
    ]
    
    # Downsample if fps > 24
    if fps > 24:
        frame_list = sample(frame_list, count=int(24 * len(frame_list) / fps))
    
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179])
    clip_text = ann['text']
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            with open(frame, 'rb') as f:
                poses = pickle.load(f)
            
            pose = np.concatenate([poses[key] for key in keys], 0)
            clip_poses[frame_id] = pose
        
        # Remove lower body joints: keep from index (3+3*11)=36 onwards
        # 179 - 36 = 143 dims
        clip_poses = clip_poses[:, (3 + 3 * 11):]
        
        # Remove shape (10 dims): keep :-20 and -10:
        # 143 - 10 = 133 dims
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)
    
    code = None
    if need_code:
        try:
            fname = os.path.join(code_path, 'how2sign', f'{name}.npy')
            code = np.load(fname)[0]
        except:
            fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
    
    return clip_poses, clip_text, name, code


def load_csl_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """
    Load CSL-Daily sample.
    
    Args:
        ann: annotation dict with 'name', 'text'
        data_dir: root directory (contains 'poses' subdirectory)
        need_pose: whether to load pose data
        code_path: path to pre-extracted motion codes
        need_code: whether to load motion codes
        
    Returns:
        clip_poses: (T, 133) pose array
        clip_text: text annotation
        name: sample name
        code: motion codes if need_code else None
    """
    clip_text = ann['text']
    name = ann['name']
    
    frame_list = sorted(os.listdir(os.path.join(data_dir, 'poses', name)))
    
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179])
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(data_dir, 'poses', name, frame)
            with open(frame_path, 'rb') as f:
                poses = pickle.load(f)
            
            pose = np.concatenate([poses[key] for key in keys], 0)
            clip_poses[frame_id] = pose
        
        # Remove lower body
        clip_poses = clip_poses[:, (3 + 3 * 11):]
        # Remove shape
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)
    
    code = None
    if need_code:
        try:
            fname = os.path.join(code_path, 'csl', f'{name}.npy')
            code = np.load(fname)[0]
        except:
            fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
    
    return clip_poses, clip_text, name, code


def load_phoenix_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False):
    """
    Load Phoenix-2014T sample.
    
    Args:
        ann: annotation dict with 'name', 'text'
        data_dir: root directory (contains sample directories directly)
        need_pose: whether to load pose data
        code_path: path to pre-extracted motion codes
        need_code: whether to load motion codes
        
    Returns:
        clip_poses: (T, 133) pose array
        clip_text: text annotation
        name: sample name
        code: motion codes if need_code else None
    """
    clip_text = ann['text']
    name = ann['name']
    
    frame_list = sorted(os.listdir(os.path.join(data_dir, name)))
    
    if len(frame_list) < 4:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179])
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            frame_path = os.path.join(data_dir, name, frame)
            with open(frame_path, 'rb') as f:
                poses = pickle.load(f)
            
            pose = np.concatenate([poses[key] for key in keys], 0)
            clip_poses[frame_id] = pose
        
        # Remove lower body
        clip_poses = clip_poses[:, (3 + 3 * 11):]
        # Remove shape
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)
    
    code = None
    if need_code:
        try:
            fname = os.path.join(code_path, 'phoenix', f'{name}.npy')
            code = np.load(fname)[0]
        except:
            fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
    
    return clip_poses, clip_text, name, code


def load_iso_sample(ann, data_dir, need_pose=True, code_path=None, need_code=False, dataset=None):
    """
    Load isolated sign sample (ASL/CSL/Phoenix isolated signs).
    
    Args:
        ann: annotation dict with 'label', 'name', 'start', 'end', 'video_file'
        data_dir: root directory
        need_pose: whether to load pose data
        code_path: path to pre-extracted motion codes
        need_code: whether to load motion codes
        dataset: 'csl_iso', 'how2sign_iso', or 'phoenix_iso'
        
    Returns:
        clip_poses: (T, 133) pose array
        clip_text: label
        name: sample name
        code: motion codes if need_code else None
    """
    clip_text = ann['label']
    name = ann['name']
    start, end = ann['start'], ann['end']
    video_file = ann['video_file']
    
    if dataset in ['csl_iso', 'how2sign_iso']:
        frame_list = sorted(os.listdir(os.path.join(data_dir, 'poses', video_file)))
        frame_idx = [int(x.split('.pkl')[0]) for x in frame_list]
    elif dataset == 'phoenix_iso':
        frame_list = sorted(os.listdir(os.path.join(data_dir, video_file)))
        frame_idx = [int(x.split('.pkl')[0].replace('images', '')) for x in frame_list]
    else:
        return None, None, None, None
    
    if len(frame_list) < 4:
        return None, None, None, None
    
    # Get frames within start-end range
    start_idx = bisect_left(frame_idx, start)
    end_idx = bisect_right(frame_idx, end)
    frame_list = frame_list[start_idx:end_idx]
    
    ratio = len(frame_list) / (end - start)
    if ratio < 0.5:
        return None, None, None, None
    
    clip_poses = np.zeros([len(frame_list), 179])
    
    if need_pose:
        for frame_id, frame in enumerate(frame_list):
            if dataset in ['csl_iso', 'how2sign_iso']:
                frame_path = os.path.join(data_dir, 'poses', video_file, frame)
            elif dataset == 'phoenix_iso':
                frame_path = os.path.join(data_dir, video_file, frame)
            
            with open(frame_path, 'rb') as f:
                poses = pickle.load(f)
            
            pose = np.concatenate([poses[key] for key in keys], 0)
            clip_poses[frame_id] = pose
        
        # Remove lower body
        clip_poses = clip_poses[:, (3 + 3 * 11):]
        # Remove shape
        clip_poses = np.concatenate([clip_poses[:, :-20], clip_poses[:, -10:]], axis=1)
    
    code = None
    if need_code:
        try:
            if dataset == 'csl_iso':
                fname = os.path.join(code_path, 'csl', f'{name}.npy')
            elif dataset == 'phoenix_iso':
                fname = os.path.join(code_path, 'phoenix', f'{name}.npy')
            elif dataset == 'how2sign_iso':
                fname = os.path.join(code_path, 'how2sign', f'{name}.npy')
            code = np.load(fname)[0]
        except:
            fname = os.path.join(code_path, f'{name}.npy')
            code = np.load(fname)[0]
    
    return clip_poses, clip_text, name, code
