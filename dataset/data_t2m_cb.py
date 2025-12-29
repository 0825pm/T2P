# coding: utf-8
"""
T2M CodeBook Dataset - for mBART-based Motion Language Model Training
Adapted from SOKE: Signs as Tokens (https://github.com/2000ZRL/SOKE)

Supports multiple datasets: How2Sign, CSL-Daily, Phoenix-2014T
Uses index.json with original_name for annotation matching.

Motion codes format: (T', 3) where [body, lhand, rhand]
"""

import os
import io
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm


# 깨진 How2Sign 샘플들
BAD_HOW2SIGN_IDS = [
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


class T2MCodeBookDataset(Dataset):
    """
    Text-to-Motion CodeBook Dataset for mBART training.
    
    Uses index.json to map motion codes to original annotations.
    """
    
    def __init__(
        self,
        split: str,
        motion_code_path: str,
        tokenizer,
        dataset_name: str = 'how2sign_csl_phoenix',
        how2sign_root: str = None,
        csl_root: str = None,
        phoenix_root: str = None,
        max_motion_length: int = 256,
        max_text_length: int = 128,
        body_code_num: int = 96,
        hand_code_num: int = 192,
        rhand_code_num: int = 192,
    ):
        self.split = split
        self.motion_code_path = motion_code_path
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.how2sign_root = how2sign_root
        self.csl_root = csl_root
        self.phoenix_root = phoenix_root
        self.max_motion_length = max_motion_length
        self.max_text_length = max_text_length
        self.body_code_num = body_code_num
        self.hand_code_num = hand_code_num
        self.rhand_code_num = rhand_code_num
        
        # Special token indices
        self.body_bos = body_code_num
        self.body_eos = body_code_num + 1
        self.body_pad = body_code_num + 2
        
        self.hand_bos = hand_code_num
        self.hand_eos = hand_code_num + 1
        self.hand_pad = hand_code_num + 2
        
        self.rhand_bos = rhand_code_num
        self.rhand_eos = rhand_code_num + 1
        self.rhand_pad = rhand_code_num + 2
        
        # Get token start IDs from tokenizer
        self.motion_token_start_id = tokenizer.convert_tokens_to_ids('<motion_id_0>')
        self.hand_token_start_id = tokenizer.convert_tokens_to_ids('<hand_id_0>')
        self.rhand_token_start_id = tokenizer.convert_tokens_to_ids('<rhand_id_0>')
        
        print(f"  Token IDs: motion={self.motion_token_start_id}, hand={self.hand_token_start_id}, rhand={self.rhand_token_start_id}")
        
        # Load annotations first
        self.annotations = {}  # name -> text
        if how2sign_root and 'how2sign' in dataset_name:
            self._load_how2sign_annotations(how2sign_root, split)
        if csl_root and 'csl' in dataset_name:
            self._load_csl_annotations(csl_root, split)
        if phoenix_root and 'phoenix' in dataset_name:
            self._load_phoenix_annotations(phoenix_root, split)
        
        # Load samples from index.json
        self.samples = []
        self.h2s_len = self.csl_len = self.phoenix_len = 0
        
        if 'how2sign' in dataset_name:
            h2s_samples = self._load_from_index('How2Sign', split)
            self.samples.extend(h2s_samples)
            self.h2s_len = len(h2s_samples)
            print(f"  How2Sign: {self.h2s_len} samples")
        
        if 'csl' in dataset_name:
            csl_samples = self._load_from_index('CSL-Daily', split)
            self.samples.extend(csl_samples)
            self.csl_len = len(csl_samples)
            print(f"  CSL-Daily: {self.csl_len} samples")
        
        if 'phoenix' in dataset_name:
            phoenix_samples = self._load_from_index('Phoenix_2014T', split)
            self.samples.extend(phoenix_samples)
            self.phoenix_len = len(phoenix_samples)
            print(f"  Phoenix-2014T: {self.phoenix_len} samples")
        
        print(f"  Total: {len(self.samples)} samples")
    
    def _load_how2sign_annotations(self, root: str, split: str):
        """Load How2Sign text annotations."""
        csv_split = 'val' if split == 'val' else split
        csv_path = os.path.join(root, csv_split, 're_aligned', 
                                f'how2sign_realigned_{csv_split}_preprocessed_fps.csv')
        
        if not os.path.exists(csv_path):
            print(f"  Warning: How2Sign CSV not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        for idx in range(len(df)):
            name = df.loc[idx, 'SENTENCE_NAME']
            if name not in BAD_HOW2SIGN_IDS:
                self.annotations[name] = df.loc[idx, 'SENTENCE']
    
    def _load_csl_annotations(self, root: str, split: str):
        """Load CSL-Daily text annotations."""
        ann_split = 'val' if split == 'val' else split
        ann_path = os.path.join(root, f'csl_clean.{ann_split}')
        
        if not os.path.exists(ann_path):
            print(f"  Warning: CSL annotation not found: {ann_path}")
            return
        
        with gzip.open(ann_path, 'rb') as f:
            annotations = pickle.load(f)
        
        for ann in annotations:
            self.annotations[ann['name']] = ann['text']
    
    def _load_phoenix_annotations(self, root: str, split: str):
        """Load Phoenix-2014T text annotations."""
        ann_split = 'dev' if split == 'val' else split
        ann_path = os.path.join(root, f'phoenix14t.{ann_split}')
        
        if not os.path.exists(ann_path):
            print(f"  Warning: Phoenix annotation not found: {ann_path}")
            return
        
        with gzip.open(ann_path, 'rb') as f:
            annotations = pickle.load(f)
        
        for ann in annotations:
            self.annotations[ann['name']] = ann['text']
    
    def _load_from_index(self, dataset_folder: str, split: str) -> List[Dict]:
        """Load samples from index.json."""
        samples = []
        
        # Determine the folder path
        if dataset_folder == 'CSL-Daily':
            index_dir = os.path.join(self.motion_code_path, dataset_folder, 'poses')
        elif dataset_folder == 'Phoenix_2014T':
            split_name = 'dev' if split == 'val' else split
            index_dir = os.path.join(self.motion_code_path, dataset_folder, split_name)
        else:
            index_dir = os.path.join(self.motion_code_path, dataset_folder, split)
        
        index_path = os.path.join(index_dir, 'index.json')
        
        if not os.path.exists(index_path):
            print(f"  Warning: Index not found: {index_path}")
            return samples
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Map dataset folder to src name
        folder_to_src = {
            'How2Sign': 'how2sign',
            'CSL-Daily': 'csl',
            'Phoenix_2014T': 'phoenix',
        }
        src = folder_to_src.get(dataset_folder, 'unknown')
        
        for entry in index_data:
            # Filter by split for CSL-Daily (all splits in one folder)
            if dataset_folder == 'CSL-Daily':
                entry_split = entry.get('split', '')
                if entry_split and entry_split != split:
                    continue
            
            code_path = os.path.join(index_dir, entry['path'])
            if not os.path.exists(code_path):
                continue
            
            # Get original name and text
            original_name = entry.get('original_name', entry['name'])
            text = self.annotations.get(original_name, '')
            
            if not text:
                # Skip if no text annotation found
                continue
            
            samples.append({
                'name': original_name,
                'code_path': code_path,
                'text': text,
                'src': src,
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _codes_to_token_ids(self, codes: np.ndarray, part: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert motion codes to token IDs with motion mask.
        
        Directly compute token IDs instead of using tokenizer.encode().
        """
        if part == 'body':
            base_id = self.motion_token_start_id
            bos_offset = self.body_code_num      # 96
            eos_offset = self.body_code_num + 1  # 97
            codebook_size = self.body_code_num
        elif part == 'lhand':
            base_id = self.hand_token_start_id
            bos_offset = self.hand_code_num      # 192
            eos_offset = self.hand_code_num + 1  # 193
            codebook_size = self.hand_code_num
        else:  # rhand
            base_id = self.rhand_token_start_id
            bos_offset = self.rhand_code_num      # 192
            eos_offset = self.rhand_code_num + 1  # 193
            codebook_size = self.rhand_code_num
        
        # Filter valid codes
        valid_codes = [int(c) for c in codes if 0 <= c < codebook_size]
        
        # Build token IDs directly
        token_ids = [base_id + bos_offset]  # BOS
        token_ids.extend([base_id + c for c in valid_codes])  # Motion codes
        token_ids.append(base_id + eos_offset)  # EOS
        
        # Create motion mask: 1 for actual motion tokens (not BOS/EOS)
        motion_mask = [0] + [1] * len(valid_codes) + [0]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(motion_mask, dtype=torch.float)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load motion codes: (T', 3) - [body, lhand, rhand]
        codes = np.load(sample['code_path'])
        
        if codes.ndim == 3:
            codes = codes[0]  # Remove batch dim if present
        
        # Truncate if too long
        if len(codes) > self.max_motion_length:
            codes = codes[:self.max_motion_length]
        
        # Split into parts
        body_codes = codes[:, 0]
        lhand_codes = codes[:, 1]
        rhand_codes = codes[:, 2]
        
        # Convert to token IDs with masks
        body_token_ids, body_mask = self._codes_to_token_ids(body_codes, 'body')
        lhand_token_ids, lhand_mask = self._codes_to_token_ids(lhand_codes, 'lhand')
        rhand_token_ids, rhand_mask = self._codes_to_token_ids(rhand_codes, 'rhand')
        
        # Tokenize text
        text_encoded = self.tokenizer(
            sample['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'text': sample['text'],
            'name': sample['name'],
            'input_ids': text_encoded['input_ids'].squeeze(0),
            'attention_mask': text_encoded['attention_mask'].squeeze(0),
            'labels_body': body_token_ids,
            'labels_lhand': lhand_token_ids,
            'labels_rhand': rhand_token_ids,
            'mask_body': body_mask,
            'mask_lhand': lhand_mask,
            'mask_rhand': rhand_mask,
            'src': sample['src'],
            'length': len(codes),
        }


def t2m_cb_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for T2MCodeBookDataset."""
    if not batch:
        return None
    
    # Get max lengths
    max_body_len = max(len(b['labels_body']) for b in batch)
    max_lhand_len = max(len(b['labels_lhand']) for b in batch)
    max_rhand_len = max(len(b['labels_rhand']) for b in batch)
    max_motion_len = max(max_body_len, max_lhand_len, max_rhand_len)
    
    # Stack text inputs
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    
    # Pad motion labels and masks
    labels_body, labels_lhand, labels_rhand = [], [], []
    mask_body, mask_lhand, mask_rhand = [], [], []
    pad_id = -100
    
    for b in batch:
        # Body
        body = b['labels_body']
        body_m = b['mask_body']
        pad_len = max_motion_len - len(body)
        if pad_len > 0:
            body = torch.cat([body, torch.full((pad_len,), pad_id, dtype=torch.long)])
            body_m = torch.cat([body_m, torch.zeros(pad_len)])
        labels_body.append(body)
        mask_body.append(body_m)
        
        # Left hand
        lhand = b['labels_lhand']
        lhand_m = b['mask_lhand']
        pad_len = max_motion_len - len(lhand)
        if pad_len > 0:
            lhand = torch.cat([lhand, torch.full((pad_len,), pad_id, dtype=torch.long)])
            lhand_m = torch.cat([lhand_m, torch.zeros(pad_len)])
        labels_lhand.append(lhand)
        mask_lhand.append(lhand_m)
        
        # Right hand
        rhand = b['labels_rhand']
        rhand_m = b['mask_rhand']
        pad_len = max_motion_len - len(rhand)
        if pad_len > 0:
            rhand = torch.cat([rhand, torch.full((pad_len,), pad_id, dtype=torch.long)])
            rhand_m = torch.cat([rhand_m, torch.zeros(pad_len)])
        labels_rhand.append(rhand)
        mask_rhand.append(rhand_m)
    
    return {
        'text': [b['text'] for b in batch],
        'name': [b['name'] for b in batch],
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels_body': torch.stack(labels_body),
        'labels_lhand': torch.stack(labels_lhand),
        'labels_rhand': torch.stack(labels_rhand),
        'mask_body': torch.stack(mask_body),
        'mask_lhand': torch.stack(mask_lhand),
        'mask_rhand': torch.stack(mask_rhand),
        'src': [b['src'] for b in batch],
        'length': torch.tensor([b['length'] for b in batch]),
    }


def load_t2m_cb_data(
    cfg: dict,
    tokenizer,
) -> Tuple[T2MCodeBookDataset, T2MCodeBookDataset, T2MCodeBookDataset]:
    """Load T2M CodeBook datasets for all splits."""
    data_cfg = cfg.get("data", cfg)
    model_cfg = cfg.get("model", {})
    
    common_kwargs = {
        'tokenizer': tokenizer,
        'motion_code_path': data_cfg.get('motion_code_path', '/home/user/Projects/research/T2P/data/motion_codes'),
        'dataset_name': data_cfg.get('dataset_name', 'how2sign_csl_phoenix'),
        'how2sign_root': data_cfg.get('how2sign', {}).get('root'),
        'csl_root': data_cfg.get('csl', {}).get('root'),
        'phoenix_root': data_cfg.get('phoenix', {}).get('root'),
        'max_motion_length': data_cfg.get('max_motion_length', 256),
        'max_text_length': model_cfg.get('max_text_length', 128),
        'body_code_num': model_cfg.get('body_code_num', 96),
        'hand_code_num': model_cfg.get('hand_code_num', 192),
        'rhand_code_num': model_cfg.get('rhand_code_num', 192),
    }
    
    print("Loading train data...")
    train_data = T2MCodeBookDataset(split='train', **common_kwargs)
    
    print("Loading dev data...")
    dev_data = T2MCodeBookDataset(split='val', **common_kwargs)
    
    print("Loading test data...")
    test_data = T2MCodeBookDataset(split='test', **common_kwargs)
    
    return train_data, dev_data, test_data


def make_t2m_cb_iter(
    dataset: T2MCodeBookDataset,
    batch_size: int,
    train: bool = False,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader for T2M CodeBook dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=t2m_cb_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )