# coding: utf-8
"""
T2M Dataset - SOKE style multi-dataset loader

Supports:
- How2Sign (English ASL)
- CSL-Daily (Chinese CSL)
- Phoenix-2014T (German DGS)

Loads pre-extracted motion codes for Text-to-Motion training.

Directory structure expected:
    data/motion_codes/
    ├── CSL-Daily/
    │   └── poses/
    │       ├── S000000_P0000_T00.npy
    │       └── ...
    ├── How2Sign/
    │   ├── train/
    │   │   └── _2FBDaOPYig_1-3-rgb_front.npy
    │   ├── val/
    │   └── test/
    └── Phoenix_2014T/
        ├── train/
        │   └── train_01April_2010_...npy
        └── dev/
            └── dev_01April_2010_...npy
"""
import os
import io
import gzip
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from copy import deepcopy


# Bad How2Sign samples (from SOKE)
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


class T2MDatasetSOKE(Dataset):
    """
    SOKE-style T2M Dataset with multi-dataset support.
    
    Loads pre-extracted motion codes and text annotations from:
    - How2Sign
    - CSL-Daily
    - Phoenix-2014T
    """
    
    def __init__(
        self,
        split: str,
        data_config: dict,
        max_motion_length: int = 400,
        min_motion_length: int = 40,
        unit_length: int = 4,
        **kwargs,
    ):
        """
        Args:
            split: 'train', 'dev', or 'test'
            data_config: Data configuration dict with dataset paths
            max_motion_length: Maximum motion sequence length
            min_motion_length: Minimum motion sequence length
            unit_length: VQ-VAE temporal downsampling factor
        """
        self.split = split
        self.max_motion_length = max_motion_length // unit_length  # In code space
        self.min_motion_length = min_motion_length // unit_length
        self.unit_length = unit_length
        
        # Get paths from config
        dataset_name = data_config.get('dataset_name', 'how2sign_csl_phoenix')
        self.motion_code_root = data_config.get('motion_code_path', '')
        
        self.all_data = []
        self.h2s_len = 0
        self.csl_len = 0
        self.phoenix_len = 0
        
        # Load How2Sign
        if 'how2sign' in dataset_name:
            h2s_config = data_config.get('how2sign', {})
            h2s_root = h2s_config.get('root', '')
            if h2s_root and os.path.exists(h2s_root):
                self._load_how2sign(h2s_root, split)
        
        # Load CSL-Daily
        if 'csl' in dataset_name:
            csl_config = data_config.get('csl', {})
            csl_root = csl_config.get('root', '')
            if csl_root and os.path.exists(csl_root):
                self._load_csl(csl_root, split)
        
        # Load Phoenix-2014T
        if 'phoenix' in dataset_name:
            phoenix_config = data_config.get('phoenix', {})
            phoenix_root = phoenix_config.get('root', '')
            if phoenix_root and os.path.exists(phoenix_root):
                self._load_phoenix(phoenix_root, split)
        
        print(f"[{split}] Data loaded: Total={len(self.all_data)}, "
              f"How2Sign={self.h2s_len}, CSL={self.csl_len}, Phoenix={self.phoenix_len}")
    
    def _load_how2sign(self, root: str, split: str):
        """
        Load How2Sign dataset.
        
        Motion codes structure: motion_codes/How2Sign/{train,val,test}/{name}.npy
        """
        # Map split name
        split_map = {'train': 'train', 'dev': 'val', 'test': 'test'}
        h2s_split = split_map.get(split, split)
        
        csv_path = os.path.join(
            root, h2s_split, 're_aligned',
            f'how2sign_realigned_{h2s_split}_preprocessed_fps.csv'
        )
        
        if not os.path.exists(csv_path):
            print(f"  How2Sign CSV not found: {csv_path}")
            return
        
        csv = pd.read_csv(csv_path)
        csv['DURATION'] = csv['END_REALIGNED'] - csv['START_REALIGNED']
        csv = csv[csv['DURATION'] < 30].reset_index(drop=True)  # Remove > 30s
        
        # Motion code directory: motion_codes/How2Sign/{split}/
        code_dir = os.path.join(self.motion_code_root, 'How2Sign', h2s_split)
        
        if not os.path.exists(code_dir):
            print(f"  How2Sign code dir not found: {code_dir}")
            return
        
        print(f"  Loading How2Sign ({split} -> {h2s_split})... {len(csv)} samples")
        loaded = 0
        for idx in tqdm(range(len(csv)), desc="How2Sign", leave=False):
            name = csv.iloc[idx]['SENTENCE_NAME']
            
            if name in BAD_HOW2SIGN_IDS:
                continue
            
            # Check if motion codes exist
            code_path = os.path.join(code_dir, f'{name}.npy')
            if not os.path.exists(code_path):
                continue
            
            self.all_data.append({
                'name': name,
                'text': csv.iloc[idx]['SENTENCE'],
                'src': 'how2sign',
                'code_path': code_path,
            })
            loaded += 1
        
        self.h2s_len = loaded
        print(f"  How2Sign loaded: {loaded}")
    
    def _load_csl(self, root: str, split: str):
        """
        Load CSL-Daily dataset.
        
        Motion codes structure: motion_codes/CSL-Daily/poses/{name}.npy
        (No split folders - all in poses/)
        """
        # Annotation path
        if split == 'dev':
            ann_path = os.path.join(root, 'csl_clean.val')  # CSL uses 'val' not 'dev'
        else:
            ann_path = os.path.join(root, f'csl_clean.{split}')
        
        if not os.path.exists(ann_path):
            print(f"  CSL annotation not found: {ann_path}")
            return
        
        with gzip.open(ann_path, 'rb') as f:
            annotations = pickle.load(f)
        
        # Motion code directory: motion_codes/CSL-Daily/poses/
        code_dir = os.path.join(self.motion_code_root, 'CSL-Daily', 'poses')
        
        if not os.path.exists(code_dir):
            print(f"  CSL code dir not found: {code_dir}")
            return
        
        print(f"  Loading CSL-Daily ({split})... {len(annotations)} samples")
        loaded = 0
        for ann in tqdm(annotations, desc="CSL", leave=False):
            name = ann.get('name', ann.get('id', ''))
            
            # Check if motion codes exist
            code_path = os.path.join(code_dir, f'{name}.npy')
            if not os.path.exists(code_path):
                continue
            
            self.all_data.append({
                'name': name,
                'text': ann.get('text', ann.get('sentence', '')),
                'src': 'csl',
                'code_path': code_path,
            })
            loaded += 1
        
        self.csl_len = loaded
        print(f"  CSL loaded: {loaded}")
    
    def _load_phoenix(self, root: str, split: str):
        """
        Load Phoenix-2014T dataset.
        
        Motion codes structure: motion_codes/Phoenix_2014T/{train,dev}/{name}.npy
        (dev instead of val, no test folder - use dev for test)
        """
        # Annotation path
        if split == 'val' or split == 'dev':
            ann_path = os.path.join(root, 'phoenix14t.dev')
            phoenix_split = 'dev'
        elif split == 'test':
            ann_path = os.path.join(root, 'phoenix14t.dev')  # Use dev as test
            phoenix_split = 'dev'
        else:
            ann_path = os.path.join(root, f'phoenix14t.{split}')
            phoenix_split = split
        
        if not os.path.exists(ann_path):
            print(f"  Phoenix annotation not found: {ann_path}")
            return
        
        with gzip.open(ann_path, 'rb') as f:
            annotations = pickle.load(f)
        
        # Motion code directory: motion_codes/Phoenix_2014T/{split}/
        code_dir = os.path.join(self.motion_code_root, 'Phoenix_2014T', phoenix_split)
        
        if not os.path.exists(code_dir):
            print(f"  Phoenix code dir not found: {code_dir}")
            return
        
        print(f"  Loading Phoenix-2014T ({split} -> {phoenix_split})... {len(annotations)} samples")
        loaded = 0
        for ann in tqdm(annotations, desc="Phoenix", leave=False):
            name = ann.get('name', ann.get('id', ''))
            
            # Phoenix motion code naming: {split}_{original_name}
            # e.g., dev_01April_2010_Thursday_heute-6694...
            code_name = f"{phoenix_split}_{name}"
            code_path = os.path.join(code_dir, f'{code_name}.npy')
            
            if not os.path.exists(code_path):
                # Try without prefix
                code_path = os.path.join(code_dir, f'{name}.npy')
                if not os.path.exists(code_path):
                    continue
            
            self.all_data.append({
                'name': name,
                'text': ann.get('text', ann.get('sentence', '')),
                'src': 'phoenix',
                'code_path': code_path,
            })
            loaded += 1
        
        self.phoenix_len = loaded
        print(f"  Phoenix loaded: {loaded}")
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        sample = self.all_data[idx]
        
        # Load motion codes
        codes = np.load(sample['code_path'])
        
        # Handle different code formats
        if len(codes.shape) == 3:
            # (1, T, 3) format
            codes = codes[0]  # (T, 3)
        elif len(codes.shape) == 1:
            # (T,) format - single codebook
            codes = np.stack([codes, codes, codes], axis=-1)  # (T, 3)
        
        # Ensure codes has 3 columns (body, lhand, rhand)
        if len(codes.shape) == 2 and codes.shape[1] != 3:
            # If it's (T, 1) or other shape, duplicate
            if codes.shape[1] == 1:
                codes = np.tile(codes, (1, 3))
            else:
                # Unexpected shape, try to handle
                codes = codes[:, :3] if codes.shape[1] >= 3 else np.tile(codes[:, :1], (1, 3))
        
        # Clip to max length
        T = min(len(codes), self.max_motion_length)
        
        if T < self.min_motion_length:
            # Pad short sequences by repeating
            if len(codes) < self.min_motion_length:
                pad_len = self.min_motion_length - len(codes)
                codes = np.pad(codes, ((0, pad_len), (0, 0)), mode='edge')
            T = self.min_motion_length
        
        body_codes = codes[:T, 0].astype(np.int64)
        lhand_codes = codes[:T, 1].astype(np.int64)
        rhand_codes = codes[:T, 2].astype(np.int64)
        
        return {
            'text': sample['text'],
            'name': sample['name'],
            'src': sample['src'],
            'body_codes': torch.tensor(body_codes, dtype=torch.long),
            'lhand_codes': torch.tensor(lhand_codes, dtype=torch.long),
            'rhand_codes': torch.tensor(rhand_codes, dtype=torch.long),
            'length': T,
        }


def t2m_collate_fn(batch):
    """Custom collate function for T2M batches."""
    texts = [item['text'] for item in batch]
    names = [item['name'] for item in batch]
    srcs = [item['src'] for item in batch]
    lengths = [item['length'] for item in batch]
    
    body_codes = [item['body_codes'] for item in batch]
    lhand_codes = [item['lhand_codes'] for item in batch]
    rhand_codes = [item['rhand_codes'] for item in batch]
    
    # Pad codes
    max_len = max(lengths)
    
    body_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    lhand_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    rhand_padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i in range(len(batch)):
        L = lengths[i]
        body_padded[i, :L] = body_codes[i]
        lhand_padded[i, :L] = lhand_codes[i]
        rhand_padded[i, :L] = rhand_codes[i]
    
    return {
        'text': texts,
        'names': names,
        'src': srcs,
        'body_codes': body_padded,
        'lhand_codes': lhand_padded,
        'rhand_codes': rhand_padded,
        'lengths': lengths,
    }


def load_t2m_data(cfg: dict) -> Tuple:
    """
    Load T2M datasets (SOKE-style config).
    
    Expected config structure:
    ```yaml
    data:
      dataset_name: "how2sign_csl_phoenix"
      how2sign:
        root: "/path/to/How2Sign"
      csl:
        root: "/path/to/CSL-Daily"
      phoenix:
        root: "/path/to/Phoenix_2014T"
      motion_code_path: "/path/to/motion_codes"  # Root of motion codes
      max_motion_length: 400
      min_motion_length: 40
      unit_length: 4
    ```
    
    Expected motion_codes structure:
    ```
    motion_codes/
    ├── CSL-Daily/poses/{name}.npy        # All splits in one folder
    ├── How2Sign/{train,val,test}/{name}.npy
    └── Phoenix_2014T/{train,dev}/{name}.npy
    ```
    """
    data_cfg = cfg.get("data", {})
    
    max_motion_length = data_cfg.get("max_motion_length", 400)
    min_motion_length = data_cfg.get("min_motion_length", 40)
    unit_length = data_cfg.get("unit_length", 4)
    
    common_kwargs = {
        "data_config": data_cfg,
        "max_motion_length": max_motion_length,
        "min_motion_length": min_motion_length,
        "unit_length": unit_length,
    }
    
    print("=" * 60)
    print("Loading T2M Data (SOKE style)")
    print("=" * 60)
    
    train_data = T2MDatasetSOKE(split='train', **common_kwargs)
    dev_data = T2MDatasetSOKE(split='dev', **common_kwargs)
    test_data = T2MDatasetSOKE(split='test', **common_kwargs)
    
    return train_data, dev_data, test_data, None, None


def make_t2m_iter(
    dataset: Dataset, 
    batch_size: int, 
    train: bool = False, 
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Create DataLoader for T2M dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=t2m_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )