# coding: utf-8
"""
SOKE Text2MotionDataset
https://github.com/2000ZRL/SOKE

Text-to-Motion 학습/평가용 데이터셋
- Raw pose 로딩 (133 dims)
- Mean/std normalization
"""
import os
import math
import gzip
import pickle
import numpy as np
import torch
import pandas as pd
from torch.utils import data
from tqdm import tqdm
from copy import deepcopy
import random

from .load_data import load_csl_sample, load_h2s_sample, load_phoenix_sample

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


def sample(input, count):
    """Uniform sampling."""
    ss = float(len(input)) / count
    return [input[int(math.floor(i * ss))] for i in range(count)]


class Text2MotionDataset(data.Dataset):
    """
    Text-to-Motion 데이터셋.
    
    Raw pose를 반환 (motion codes 아님).
    평가나 VQ-VAE 없이 직접 pose 생성할 때 사용.
    """
    
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        dataset_name='how2sign',
        **kwargs,
    ):
        """
        Args:
            data_root: How2Sign root directory
            split: 'train', 'val', or 'test'
            mean: (133,) mean tensor
            std: (133,) std tensor
            max_motion_length: max sequence length
            min_motion_length: min sequence length
            unit_length: frame alignment unit
            dataset_name: dataset combination string
        """
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)
        
        self.mean = mean
        self.std = std
        
        assert max_motion_length % unit_length == 0
        assert min_motion_length % unit_length == 0
        
        self.all_data = []
        self.h2s_len = self.csl_len = self.phoenix_len = 0
        
        # Load How2Sign
        if 'how2sign' in dataset_name:
            self.data_dir = os.path.join(data_root, split, 'poses')
            csv_path = os.path.join(
                data_root, split, 're_aligned',
                f'how2sign_realigned_{split}_preprocessed_fps.csv'
            )
            
            self.csv = pd.read_csv(csv_path)
            self.csv['DURATION'] = self.csv['END_REALIGNED'] - self.csv['START_REALIGNED']
            self.csv = self.csv[self.csv['DURATION'] < 30].reset_index(drop=True)
            ids = self.csv['SENTENCE_NAME']
            
            print(f'loading how2sign data...', len(ids))
            for idx in tqdm(range(len(ids))):
                name = ids[idx]
                if name in bad_how2sign_ids:
                    continue
                self.all_data.append({
                    'name': name,
                    'fps': self.csv[self.csv['SENTENCE_NAME'] == name]['fps'].item(),
                    'text': self.csv[self.csv['SENTENCE_NAME'] == name]['SENTENCE'].item(),
                    'src': 'how2sign'
                })
            self.h2s_len = len(self.all_data)
        
        # Load CSL-Daily
        if 'csl' in dataset_name:
            if split == 'train':
                ann_path = os.path.join(self.csl_root, 'csl_clean.train')
            else:
                ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            
            with gzip.open(ann_path, 'rb') as f:
                ann = pickle.load(f)
            
            print(f'loading csl data...', len(ann))
            for item in tqdm(ann):
                item_copy = deepcopy(item)
                item_copy['src'] = 'csl'
                self.all_data.append(item_copy)
            self.csl_len = len(ann)
        
        # Load Phoenix-2014T
        if 'phoenix' in dataset_name:
            if split == 'val':
                ann_path = os.path.join(self.phoenix_root, 'phoenix14t.dev')
            else:
                ann_path = os.path.join(self.phoenix_root, f'phoenix14t.{split}')
            
            with gzip.open(ann_path, 'rb') as f:
                ann = pickle.load(f)
            
            print(f'{split}--loading phoenix data...', len(ann))
            for item in tqdm(ann):
                item_copy = deepcopy(item)
                item_copy['src'] = 'phoenix'
                self.all_data.append(item_copy)
            self.phoenix_len = len(ann)
        
        print(f'Data loading done. All: {len(self.all_data)}, '
              f'How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}')
        
        self.nfeats = 133
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']
        
        # Load poses
        if src == 'how2sign':
            clip_poses, text, name, _ = load_h2s_sample(sample, self.data_dir)
        elif src == 'csl':
            clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
        elif src == 'phoenix':
            clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
        else:
            raise ValueError(f"Unknown source: {src}")
        
        # Handle failed loading
        if clip_poses is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        all_captions = [text]
        
        # Normalize
        clip_poses = (clip_poses - self.mean.numpy()) / (self.std.numpy() + 1e-10)
        
        # Adjust length
        m_length = clip_poses.shape[0]
        
        if m_length < self.min_motion_length:
            idx_arr = np.linspace(0, m_length - 1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        elif m_length > self.max_motion_length:
            idx_arr = np.linspace(0, m_length - 1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            start_idx = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[start_idx:start_idx + m_length]
        
        m_length = clip_poses.shape[0]
        
        return (
            text,                                    # 0: text
            torch.from_numpy(clip_poses).float(),   # 1: motion
            m_length,                                # 2: length
            name,                                    # 3: name
            None, None, None,                       # 4-6: placeholders
            all_captions,                           # 7: all_captions
            None,                                   # 8: tasks
            src                                      # 9: source
        )


class Text2MotionDatasetEval(Text2MotionDataset):
    """
    평가용 Text2Motion 데이터셋.
    Text tokenization 포함.
    """
    
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        dataset_name='how2sign',
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(
            data_root, split, mean, std, max_motion_length,
            min_motion_length, unit_length, fps, tmpFile, tiny,
            debug, dataset_name=dataset_name, **kwargs
        )
        self.w_vectorizer = w_vectorizer
    
    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']
        
        if src == 'how2sign':
            clip_poses, text, name, _ = load_h2s_sample(sample, self.data_dir)
        elif src == 'csl':
            clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
        elif src == 'phoenix':
            clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
        else:
            raise ValueError(f"Unknown source: {src}")
        
        if clip_poses is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        all_captions = [text] * 3
        
        # Normalize
        clip_poses = (clip_poses - self.mean.numpy()) / (self.std.numpy() + 1e-10)
        
        # Adjust length
        m_length = clip_poses.shape[0]
        
        if m_length < self.min_motion_length:
            idx_arr = np.linspace(0, m_length - 1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        elif m_length > self.max_motion_length:
            idx_arr = np.linspace(0, m_length - 1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx_arr]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            start_idx = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[start_idx:start_idx + m_length]
        
        m_length = clip_poses.shape[0]
        
        # Text tokenization
        tokens = text.split(' ')
        max_text_len = 40
        
        if len(tokens) < max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
        
        return (
            text,
            torch.from_numpy(clip_poses).float(),
            m_length,
            name,
            None, None,
            "_".join(tokens),
            all_captions,
            None,
            src
        )
