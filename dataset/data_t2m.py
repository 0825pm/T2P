# coding: utf-8
"""
T2M Dataset - loads pre-extracted motion codes for Text-to-Motion training

Expected motion codes format:
- vqvae_decouple: (1, T', 3) array where [body, lhand, rhand]
- vqvae: (T',) array
"""
import os
import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from dataset.vocabulary import build_vocab, Vocabulary


class T2MDataset(Dataset):
    """T2M Dataset with pre-extracted motion codes (pure PyTorch Dataset)."""
    
    def __init__(self, text_path, files_path, motion_code_path, 
                 model_type='vqvae_decouple', max_motion_len=256):
        """
        Args:
            text_path: Path to text file (e.g., /data/phoenix/train.text)
            files_path: Path to files file (e.g., /data/phoenix/train.files)
            motion_code_path: Path to motion codes directory
            model_type: 'vqvae' or 'vqvae_decouple'
            max_motion_len: Maximum motion code length
        """
        self.motion_code_path = motion_code_path
        self.model_type = model_type
        self.max_motion_len = max_motion_len
        
        self.samples = []
        skipped = 0
        
        with io.open(text_path, mode='r', encoding='utf-8') as src_file, \
             io.open(files_path, mode='r', encoding='utf-8') as files_file:
            
            for src_line, files_line in zip(src_file, files_file):
                src_line = src_line.strip()
                files_line = files_line.strip()
                
                if not src_line or not files_line:
                    continue
                
                # Get sample name
                name = os.path.splitext(os.path.basename(files_line))[0]
                if not name:
                    name = files_line
                
                # Check if motion codes exist
                code_path = os.path.join(motion_code_path, f'{name}.npy')
                if not os.path.exists(code_path):
                    skipped += 1
                    continue
                
                self.samples.append({
                    'text': src_line,
                    'name': name,
                    'code_path': code_path,
                })
        
        if skipped > 0:
            print(f"  Skipped {skipped} samples (motion codes not found)")
        print(f"  Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load motion codes
        codes = np.load(sample['code_path'])
        
        if self.model_type == 'vqvae_decouple':
            # (1, T', 3) or (T', 3)
            if len(codes.shape) == 3:
                codes = codes[0]  # (T', 3)
            
            T = min(len(codes), self.max_motion_len)
            body_codes = codes[:T, 0].astype(np.int64)
            lhand_codes = codes[:T, 1].astype(np.int64)
            rhand_codes = codes[:T, 2].astype(np.int64)
        else:
            # (T',)
            codes = codes.flatten()
            T = min(len(codes), self.max_motion_len)
            body_codes = codes[:T].astype(np.int64)
            lhand_codes = body_codes.copy()
            rhand_codes = body_codes.copy()
        
        return {
            'text': sample['text'],
            'name': sample['name'],
            'body_codes': torch.tensor(body_codes, dtype=torch.long),
            'lhand_codes': torch.tensor(lhand_codes, dtype=torch.long),
            'rhand_codes': torch.tensor(rhand_codes, dtype=torch.long),
            'length': T,
        }


def t2m_collate_fn(batch):
    """Custom collate function for T2M batches."""
    texts = [item['text'] for item in batch]
    names = [item['name'] for item in batch]
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
        'body_codes': body_padded,
        'lhand_codes': lhand_padded,
        'rhand_codes': rhand_padded,
        'lengths': lengths,
    }


def load_t2m_data(cfg: dict) -> Tuple:
    """Load T2M datasets."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]
    motion_code_path = data_cfg.get("motion_code_path", "Data/motion_codes")
    max_motion_len = model_cfg.get("max_length", 256)
    model_type = model_cfg.get("type", "vqvae_decouple")
    
    print("Loading train data...")
    train_data = T2MDataset(
        text_path=train_path + ".text",
        files_path=train_path + ".files",
        motion_code_path=os.path.join(motion_code_path, 'train'),
        model_type=model_type,
        max_motion_len=max_motion_len,
    )
    
    print("Loading dev data...")
    dev_data = T2MDataset(
        text_path=dev_path + ".text",
        files_path=dev_path + ".files",
        motion_code_path=os.path.join(motion_code_path, 'dev'),
        model_type=model_type,
        max_motion_len=max_motion_len,
    )
    
    print("Loading test data...")
    test_data = T2MDataset(
        text_path=test_path + ".text",
        files_path=test_path + ".files",
        motion_code_path=os.path.join(motion_code_path, 'test'),
        model_type=model_type,
        max_motion_len=max_motion_len,
    )
    
    return train_data, dev_data, test_data, None, None


def make_t2m_iter(dataset: Dataset, batch_size: int, 
                  train: bool = False, shuffle: bool = False) -> DataLoader:
    """Create DataLoader for T2M dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=t2m_collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=train,
    )