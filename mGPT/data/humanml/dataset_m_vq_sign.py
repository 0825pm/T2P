# coding: utf-8
"""
SOKE H2SMotionDatasetVQ - VAE Training Dataset
https://github.com/2000ZRL/SOKE

VQ-VAE 학습을 위한 모션 데이터셋
- How2Sign, CSL-Daily, Phoenix-2014T 지원
- Window sampling 적용
"""
import random
import torch
import pickle
import gzip
import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils import data
from copy import deepcopy

from .load_data import (
    load_h2s_sample, 
    load_csl_sample, 
    load_phoenix_sample, 
    load_iso_sample
)

random.seed(0)

# 깨진 How2Sign 샘플들
bad_how2sign_ids = [
    '0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', 
    '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front', 
    '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front', 
    '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', 
    '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front', 
    '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front', 
    '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 
    'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front', 
    'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front', 
    'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 
    'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front', 
    'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front', 
    'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 
    'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front', 
    'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front', 
    'g3Cc_1-V31U_12-3-rgb_front'
]


class H2SMotionDatasetVQ(data.Dataset):
    """
    VQ-VAE 학습용 모션 데이터셋.
    
    Features:
        - 133 dims (SMPL-X upper body + hands + expression)
        - Window sampling for training
        - Mean/std normalization
        - Support multiple datasets
    """
    
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        dataset_name='how2sign',
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        """
        Args:
            data_root: How2Sign root directory
            split: 'train', 'val', or 'test'
            mean: (133,) mean tensor for normalization
            std: (133,) std tensor for normalization
            max_motion_length: maximum motion sequence length
            min_motion_length: minimum motion sequence length
            win_size: window size for sampling
            dataset_name: 'how2sign', 'csl', 'phoenix', or combinations
            unit_length: frame alignment unit (default 4)
        """
        self.dataset_name = dataset_name
        self.root_dir = data_root
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)
        
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.win_size = win_size
        
        assert max_motion_length % unit_length == 0
        assert min_motion_length % unit_length == 0
        
        self.all_data = []
        self.h2s_len = self.csl_len = self.phoenix_len = 0
        
        # Load How2Sign
        if 'how2sign' in dataset_name:
            self._load_how2sign(split)
        
        # Load CSL-Daily
        if 'csl' in dataset_name:
            self._load_csl(split)
        
        # Load Phoenix-2014T
        if 'phoenix' in dataset_name:
            self._load_phoenix(split)
        
        print(f'Data loading done. All: {len(self.all_data)}, '
              f'How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}')
    
    def _load_how2sign(self, split):
        """Load How2Sign annotations."""
        self.data_dir = os.path.join(self.root_dir, split, 'poses')
        csv_path = os.path.join(
            self.root_dir, split, 're_aligned',
            f'how2sign_realigned_{split}_preprocessed_fps.csv'
        )
        
        self.csv = pd.read_csv(csv_path)
        self.csv['DURATION'] = self.csv['END_REALIGNED'] - self.csv['START_REALIGNED']
        # Remove sequences longer than 30 seconds
        self.csv = self.csv[self.csv['DURATION'] < 30].reset_index(drop=True)
        ids = self.csv['SENTENCE_NAME']
        
        print(f'{split}--loading how2sign annotations...', len(ids))
        for idx in tqdm(range(len(ids))):
            name = ids[idx]
            if name in bad_how2sign_ids:
                continue
            
            self.all_data.append({
                'name': name,
                'fps': self.csv[self.csv['SENTENCE_NAME'] == name]['fps'].item(),
                'text': self.csv[self.csv['SENTENCE_NAME'] == name]['SENTENCE'].item(),
                'src': 'how2sign',
                'split': split
            })
        
        self.h2s_len = len(self.all_data)
    
    def _load_csl(self, split):
        """Load CSL-Daily annotations."""
        if split == 'train':
            ann_path = os.path.join(self.csl_root, 'csl_clean.train')
        else:
            ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
        
        with gzip.open(ann_path, 'rb') as f:
            ann = pickle.load(f)
        
        print(f'{split}--loading csl annotations...', len(ann))
        for item in tqdm(ann):
            item_copy = deepcopy(item)
            item_copy['src'] = 'csl'
            self.all_data.append(item_copy)
        
        self.csl_len = len(ann)
    
    def _load_phoenix(self, split):
        """Load Phoenix-2014T annotations."""
        if split == 'val':
            ann_path = os.path.join(self.phoenix_root, 'phoenix14t.dev')
        else:
            ann_path = os.path.join(self.phoenix_root, f'phoenix14t.{split}')
        
        with gzip.open(ann_path, 'rb') as f:
            ann = pickle.load(f)
        
        print(f'{split}--loading phoenix annotations...', len(ann))
        for item in tqdm(ann):
            item_copy = deepcopy(item)
            item_copy['src'] = 'phoenix'
            self.all_data.append(item_copy)
        
        self.phoenix_len = len(ann)
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']
        name = sample['name']
        
        # Load poses based on source
        if src == 'how2sign':
            clip_poses, text, name, _ = load_h2s_sample(sample, self.root_dir)
        elif src == 'csl':
            clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
        elif src == 'phoenix':
            clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
        elif src == 'asl_iso':
            clip_poses, text, name, _ = load_iso_sample(sample, self.root_dir, dataset='asl_iso')
            src = 'how2sign'
        elif src == 'csl_iso':
            clip_poses, text, name, _ = load_iso_sample(sample, self.csl_root, dataset='csl_iso')
            src = 'csl'
        elif src == 'phoenix_iso':
            clip_poses, text, name, _ = load_iso_sample(sample, self.phoenix_root, dataset='phoenix_iso')
            src = 'phoenix'
        else:
            raise ValueError(f"Unknown source: {src}")
        
        # Handle failed loading
        if clip_poses is None:
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Normalize
        clip_poses = (clip_poses - self.mean.numpy()) / (self.std.numpy() + 1e-10)
        
        # Adjust length
        m_length = clip_poses.shape[0]
        
        if m_length < self.min_motion_length:
            # Upsample using linear interpolation
            idx_arr = np.linspace(0, m_length - 1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        elif m_length > self.max_motion_length:
            # Downsample
            idx_arr = np.linspace(0, m_length - 1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        else:
            # Align to unit_length and center crop
            m_length = (m_length // self.unit_length) * self.unit_length
            start_idx = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[start_idx:start_idx + m_length]
        
        m_length = clip_poses.shape[0]
        
        # Window sampling for VAE training
        if self.win_size and m_length > self.win_size:
            start = random.randint(0, m_length - self.win_size)
            clip_poses = clip_poses[start:start + self.win_size]
            m_length = self.win_size
        
        # Return format compatible with SOKE collate function
        return (
            text,                                    # 0: text
            torch.from_numpy(clip_poses).float(),   # 1: motion (T, 133)
            m_length,                                # 2: length
            name,                                    # 3: name
            None, None, None, None, None,           # 4-8: placeholders
            src                                      # 9: source dataset
        )
