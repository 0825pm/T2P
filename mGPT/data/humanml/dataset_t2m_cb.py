# coding: utf-8
"""
SOKE Text2MotionDatasetCB - Language Model Training Dataset
https://github.com/2000ZRL/SOKE

mBART 등 LM 학습을 위한 데이터셋
- Pre-extracted motion codes 사용
- Task instructions 지원
"""
import os
import gzip
import pickle
import json
import numpy as np
import torch
import pandas as pd
from torch.utils import data
from os.path import join as pjoin
from tqdm import tqdm
from copy import deepcopy

from .load_data import load_h2s_sample, load_csl_sample, load_phoenix_sample

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


class Text2MotionDatasetCB(data.Dataset):
    """
    LM 학습용 Motion CodeBook 데이터셋.
    
    Pre-extracted motion codes를 로딩하여 
    Text -> Motion Tokens 매핑 학습에 사용.
    """
    
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        dataset_name='how2sign',
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text=False,
        **kwargs,
    ):
        """
        Args:
            data_root: How2Sign root directory
            split: 'train', 'val', or 'test'
            mean: mean tensor (unused, for compatibility)
            std: std tensor (unused, for compatibility)
            dataset_name: dataset combination
            max_motion_length: max code sequence length (original / unit_length)
            min_motion_length: min code sequence length
            unit_length: VQ-VAE temporal downsampling factor
            stage: 'lm_pretrain' or 'lm_instruct'
            code_path: path to pre-extracted codes relative to data_root
            task_path: path to instruction templates
            std_text: whether to standardize text
        """
        self.tiny = tiny
        self.unit_length = unit_length
        self.data_root = data_root
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)
        
        self.mean = mean
        self.std = std
        
        # Code length limits (4x downsampling)
        self.max_motion_length = max_motion_length // unit_length
        self.min_motion_length = min_motion_length // unit_length
        
        self.code_path = code_path
        
        # Load instruction templates
        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain':
            instructions = pjoin('prepare/instructions', 'template_pretrain.json')
        elif stage in ['lm_instruct', 'lm_rl']:
            instructions = pjoin('prepare/instructions', 'template_instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")
        
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
            
            print(f'loading phoenix data...', len(ann))
            for item in tqdm(ann):
                item_copy = deepcopy(item)
                item_copy['src'] = 'phoenix'
                self.all_data.append(item_copy)
            self.phoenix_len = len(ann)
        
        print(f'Data loading done. All: {len(self.all_data)}, '
              f'How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}')
        
        # Load tasks
        self.std_text = std_text
        if os.path.exists(instructions):
            self.instructions = json.load(open(instructions, 'r'))
            self.tasks = []
            for task in self.instructions.keys():
                for subtask in self.instructions[task].keys():
                    self.tasks.append(self.instructions[task][subtask])
        else:
            # Default simple task
            self.tasks = [{'input': '{text}', 'output': '{motion}'}]
    
    def __len__(self):
        return len(self.all_data) * len(self.tasks)
    
    def __getitem__(self, idx):
        data_idx = idx % len(self.all_data)
        task_idx = idx // len(self.all_data)
        
        sample = self.all_data[data_idx]
        src = sample['src']
        
        # Load motion codes (not poses)
        if src == 'how2sign':
            _, caption, name, m_tokens = load_h2s_sample(
                sample, self.data_dir, 
                need_pose=False, 
                code_path=os.path.join(self.data_root, self.code_path), 
                need_code=True
            )
        elif src == 'csl':
            _, caption, name, m_tokens = load_csl_sample(
                sample, self.csl_root,
                need_pose=False,
                code_path=os.path.join(self.data_root, self.code_path),
                need_code=True
            )
        elif src == 'phoenix':
            _, caption, name, m_tokens = load_phoenix_sample(
                sample, self.phoenix_root,
                need_pose=False,
                code_path=os.path.join(self.data_root, self.code_path),
                need_code=True
            )
        else:
            raise ValueError(f"Unknown source: {src}")
        
        # Handle failed loading
        if m_tokens is None:
            return self.__getitem__((idx + 1) % len(self))
        
        all_captions = [caption]
        
        # Adjust code length
        m_length = m_tokens.shape[0]
        
        if m_length < self.min_motion_length:
            idx_arr = np.linspace(0, m_length - 1, num=self.min_motion_length, dtype=int)
            m_tokens = m_tokens[idx_arr]
        elif m_length > self.max_motion_length:
            idx_arr = np.linspace(0, m_length - 1, num=self.max_motion_length, dtype=int)
            m_tokens = m_tokens[idx_arr]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length if self.unit_length > 1 else m_length
            start_idx = (m_tokens.shape[0] - m_length) // 2
            m_tokens = m_tokens[start_idx:start_idx + m_length]
        
        # Random token drop augmentation (1/3 probability)
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        
        m_length = m_tokens.shape[0]
        tasks = self.tasks[task_idx]
        
        return (
            caption,                               # 0: text
            torch.from_numpy(m_tokens).long(),    # 1: motion tokens
            m_length,                              # 2: length
            name,                                  # 3: name
            None, None, None,                     # 4-6: placeholders
            all_captions,                         # 7: all_captions
            tasks,                                 # 8: task instructions
            src                                    # 9: source
        )
