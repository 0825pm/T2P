# coding: utf-8
"""
Train MaskGIT mBART for Sign Language Generation

Location: /home/user/Projects/research/T2P/train_maskgit.py

Features:
- Loss logging: total, body, lhand, rhand, length
- Accuracy logging: body_acc, lhand_acc, rhand_acc, length_acc
- Checkpoint save/resume
- Wandb integration
- Mixed precision training (AMP)
- Learning rate scheduling
- Encoder unfreezing schedule

Usage:
    python train_maskgit.py --config configs/maskgit_mbart.yaml --gpu 0
    python train_maskgit.py --config configs/maskgit_mbart.yaml --gpu 0 --resume checkpoints/maskgit/latest.pt
    python train_maskgit.py --config configs/maskgit_mbart.yaml --gpu 0,1 --wandb
"""

import os
import sys
import yaml
import argparse
import random
import gzip
import pickle
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Use --wandb flag after installing: pip install wandb")


# ============================================================================
# Arguments
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train MaskGIT mBART")
    parser.add_argument('--config', type=str, default='configs/maskgit_mbart.yaml',
                        help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id(s), e.g., 0 or 0,1')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode (small dataset)')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset: phoenix, csl, how2sign, or combinations like how2sign_csl')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP (mixed precision)')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ============================================================================
# Metrics
# ============================================================================
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
        if val is None:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class MetricTracker:
    """Track multiple metrics."""
    def __init__(self):
        self.metrics = defaultdict(AverageMeter)
    
    def update(self, metrics_dict, n=1):
        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].update(value, n)
    
    def reset(self):
        for meter in self.metrics.values():
            meter.reset()
    
    def get_avg(self):
        return {key: meter.avg for key, meter in self.metrics.items()}
    
    def get_str(self):
        return " | ".join([f"{k}: {v.avg:.4f}" for k, v in self.metrics.items()])


def compute_accuracy(logits, labels, ignore_index=-100):
    """Compute token-level accuracy."""
    if logits is None or labels is None:
        return 0.0
    
    preds = logits.argmax(dim=-1)
    mask = labels != ignore_index
    
    if mask.sum() == 0:
        return 0.0
    
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()


def compute_length_accuracy(length_logits, length_targets, tolerance=0):
    """
    Compute length prediction accuracy.
    
    Args:
        length_logits: (B, max_chunks) logits
        length_targets: (B,) ground truth lengths (1-indexed, actual lengths)
        tolerance: allow ±tolerance error
    """
    if length_logits is None or length_targets is None:
        return 0.0
    
    # Model predicts 0-indexed class, so argmax directly gives (length - 1)
    # preds: 0 → length=1, 99 → length=100
    preds = length_logits.argmax(dim=-1) + 1  # Convert to 1-indexed
    targets = length_targets  # Already 1-indexed
    
    if tolerance == 0:
        correct = (preds == targets)
    else:
        correct = (preds - targets).abs() <= tolerance
    
    return correct.float().mean().item()


# ============================================================================
# Data Loading
# ============================================================================
def create_dataloaders(cfg, tokenizer, debug=False):
    """Create train/val dataloaders."""
    
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('training', {})
    
    batch_size = train_cfg.get('batch_size', 32)
    num_workers = train_cfg.get('num_workers', 4)
    
    # MaskGITDataset 사용 (T2P/data 구조에 맞춤)
    train_dataset = MaskGITDataset(data_cfg, split='train', debug=debug)
    val_dataset = MaskGITDataset(data_cfg, split='val', debug=debug)
    
    collate_fn = MaskGITCollate(tokenizer, cfg)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Bad How2Sign samples (corrupted) - from H2SCode.py
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


class MaskGITDataset(torch.utils.data.Dataset):
    """
    Dataset for MaskGIT training.
    
    Based on H2SCode.py structure:
        T2P/data/
        ├── CSL-Daily/
        │   ├── codes/{name}.npy              # No split subdirs
        │   └── csl_clean.{train,val,test}
        ├── How2Sign/
        │   ├── codes/{train,val,test}/{name}.npy
        │   └── poses/{split}/re_aligned/how2sign_realigned_{split}_preprocessed_fps.csv
        └── Phoenix_2014T/
            ├── codes/{train,dev,test}/{split}_{name}.npy
            └── phoenix14t.{train,dev,test}
    """
    
    def __init__(self, data_cfg, split='train', debug=False):
        self.split = split
        self.max_length = data_cfg.get('max_motion_length', 400) // data_cfg.get('unit_length', 4)
        
        self.samples = []
        dataset_name = data_cfg.get('dataset_name', 'how2sign_csl_phoenix')
        
        # ============================================================
        # How2Sign
        # ============================================================
        if 'how2sign' in dataset_name:
            h2s_root = data_cfg.get('how2sign', {}).get('root', '')
            if h2s_root and os.path.exists(h2s_root):
                self._load_how2sign(h2s_root, split)
        
        # ============================================================
        # CSL-Daily
        # ============================================================
        if 'csl' in dataset_name:
            csl_root = data_cfg.get('csl', {}).get('root', '')
            if csl_root and os.path.exists(csl_root):
                self._load_csl(csl_root, split)
        
        # ============================================================
        # Phoenix-2014T
        # ============================================================
        if 'phoenix' in dataset_name:
            phoenix_root = data_cfg.get('phoenix', {}).get('root', '')
            if phoenix_root and os.path.exists(phoenix_root):
                self._load_phoenix(phoenix_root, split)
        
        if debug:
            self.samples = self.samples[:100]
        
        print(f"[MaskGITDataset {split}] Total: {len(self.samples)} samples")
    
    def _load_how2sign(self, root, split):
        """
        Load How2Sign.
        - CSV: {root}/poses/{split}/re_aligned/how2sign_realigned_{split}_preprocessed_fps.csv
        - Codes: {root}/codes/{split}/{name}.npy
        """
        h2s_split = 'val' if split == 'dev' else split
        
        # CSV path
        csv_path = os.path.join(
            root, 'poses', h2s_split, 're_aligned',
            f'how2sign_realigned_{h2s_split}_preprocessed_fps.csv'
        )
        
        if not os.path.exists(csv_path):
            print(f"  How2Sign CSV not found: {csv_path}")
            return
        
        # Code directory
        code_dir = os.path.join(root, 'codes', h2s_split)
        if not os.path.exists(code_dir):
            print(f"  How2Sign codes not found: {code_dir}")
            return
        
        df = pd.read_csv(csv_path)
        # Filter long samples
        df['DURATION'] = df['END_REALIGNED'] - df['START_REALIGNED']
        df = df[df['DURATION'] < 30].reset_index(drop=True)
        
        loaded = 0
        for idx in range(len(df)):
            name = df.iloc[idx]['SENTENCE_NAME']
            
            if name in BAD_HOW2SIGN_IDS:
                continue
            
            code_path = os.path.join(code_dir, f'{name}.npy')
            if not os.path.exists(code_path):
                continue
            
            self.samples.append({
                'name': name,
                'text': df.iloc[idx]['SENTENCE'],
                'src': 'how2sign',
                'code_path': code_path,
            })
            loaded += 1
        
        print(f"  How2Sign [{h2s_split}]: {loaded} samples")
    
    def _load_csl(self, root, split):
        """
        Load CSL-Daily.
        - Annotations: {root}/csl_clean.{train,val,test}
        - Codes: {root}/codes/{name}.npy (no split subdirs)
        """
        csl_split = 'val' if split == 'dev' else split
        ann_path = os.path.join(root, f'csl_clean.{csl_split}')
        
        if not os.path.exists(ann_path):
            print(f"  CSL annotation not found: {ann_path}")
            return
        
        # Code directory (no split subdirs for CSL)
        code_dir = os.path.join(root, 'codes')
        if not os.path.exists(code_dir):
            print(f"  CSL codes not found: {code_dir}")
            return
        
        with gzip.open(ann_path, 'rb') as f:
            annotations = pickle.load(f)
        
        loaded = 0
        for ann in annotations:
            name = ann.get('name', ann.get('id', ''))
            
            code_path = os.path.join(code_dir, f'{name}.npy')
            if not os.path.exists(code_path):
                continue
            
            self.samples.append({
                'name': name,
                'text': ann.get('text', ann.get('sentence', '')),
                'src': 'csl',
                'code_path': code_path,
            })
            loaded += 1
        
        print(f"  CSL-Daily [{csl_split}]: {loaded} samples")
    
    def _load_phoenix(self, root, split):
        """
        Load Phoenix-2014T.
        - Codes: {root}/codes/{split}/{split}_{name}.npy
        - Annotations: {root}/phoenix14t.{train,dev,test} (gzipped pickle)
        
        Annotation format:
            name: 'dev/11August_2010_Wednesday_tagesschau-2'
            text: 'tiefer luftdruck bestimmt...'
        Code file:
            dev_11August_2010_Wednesday_tagesschau-2.npy
        """
        # Phoenix uses 'dev' for validation
        phoenix_split = 'dev' if split in ['val', 'dev'] else split
        
        code_dir = os.path.join(root, 'codes', phoenix_split)
        if not os.path.exists(code_dir):
            print(f"  Phoenix codes not found: {code_dir}")
            return
        
        # Load annotation file
        ann_path = os.path.join(root, f'phoenix14t.{phoenix_split}')
        
        if not os.path.exists(ann_path):
            print(f"  Phoenix annotation not found: {ann_path}")
            # Fallback: scan codes directory
            self._load_phoenix_from_codes(code_dir, phoenix_split)
            return
        
        try:
            with gzip.open(ann_path, 'rb') as f:
                annotations = pickle.load(f)
        except Exception as e:
            print(f"  Phoenix: failed to load annotations: {e}")
            self._load_phoenix_from_codes(code_dir, phoenix_split)
            return
        
        loaded = 0
        for ann in annotations:
            name = ann.get('name', ann.get('id', ''))
            text = ann.get('text', ann.get('sentence', ''))
            
            # Convert name format: 'dev/xxx' -> 'xxx'
            if '/' in name:
                name = name.split('/')[-1]
            
            # Code file: {split}_{name}.npy
            code_path = os.path.join(code_dir, f'{phoenix_split}_{name}.npy')
            
            if not os.path.exists(code_path):
                # Try without prefix
                code_path = os.path.join(code_dir, f'{name}.npy')
                if not os.path.exists(code_path):
                    continue
            
            self.samples.append({
                'name': name,
                'text': text,
                'src': 'phoenix',
                'code_path': code_path,
            })
            loaded += 1
        
        print(f"  Phoenix [{phoenix_split}]: {loaded} samples")
    
    def _load_phoenix_from_codes(self, code_dir, phoenix_split):
        """Fallback: scan codes directory when annotation file not available."""
        loaded = 0
        for fname in os.listdir(code_dir):
            if not fname.endswith('.npy'):
                continue
            
            code_path = os.path.join(code_dir, fname)
            
            # Extract name: dev_01April_2010_Thursday_heute-6697.npy -> 01April_2010_Thursday_heute-6697
            name = fname.replace('.npy', '')
            if name.startswith(f'{phoenix_split}_'):
                name = name[len(f'{phoenix_split}_'):]
            
            self.samples.append({
                'name': name,
                'text': '',  # No text available
                'src': 'phoenix',
                'code_path': code_path,
            })
            loaded += 1
        
        if loaded > 0:
            print(f"  [Warning] Phoenix: No annotation file, loaded {loaded} samples without text")
        print(f"  Phoenix [{phoenix_split}]: {loaded} samples")
    
    def _load_motion_code(self, code_path):
        """
        Load motion codes from file. (from H2SCode.py)
        
        Returns:
            body_codes, lhand_codes, rhand_codes as numpy arrays
        """
        codes = np.load(code_path)
        
        # Shape handling: (T', 3) expected - [body, lhand, rhand]
        if codes.ndim == 3:
            codes = codes[0]  # Remove batch dim
        
        if codes.ndim == 1:
            # Single codebook -> duplicate to 3
            codes = np.stack([codes, codes, codes], axis=-1)
        
        if codes.shape[-1] != 3:
            if codes.shape[-1] > 3:
                codes = codes[..., :3]
            else:
                codes = np.tile(codes[..., :1], (1, 3))
        
        body_codes = codes[:, 0].astype(np.int64)
        lhand_codes = codes[:, 1].astype(np.int64)
        rhand_codes = codes[:, 2].astype(np.int64)
        
        return body_codes, lhand_codes, rhand_codes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load codes
        body_codes, lhand_codes, rhand_codes = self._load_motion_code(sample['code_path'])
        
        # Truncate
        T = min(len(body_codes), self.max_length)
        
        return {
            'text': sample['text'],
            'name': sample['name'],
            'src': sample['src'],
            'body_codes': torch.from_numpy(body_codes[:T]),
            'lhand_codes': torch.from_numpy(lhand_codes[:T]),
            'rhand_codes': torch.from_numpy(rhand_codes[:T]),
            'code_length': T,
        }


class MaskGITCollate:
    """Collate function for MaskGIT training."""
    
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.max_text_length = cfg.get('model', {}).get('max_text_length', 128)
        
        # Language token mapping (dataset -> SOKE language token)
        # These tokens are added dynamically to tokenizer in MaskGITmBART.__init__
        self.lang_map = {
            'how2sign': 'en_ASL',
            'csl': 'zh_CSL',
            'phoenix': 'de_DGS',
        }
        
        # Verify tokens exist
        print(f"[MaskGITCollate] Language token IDs:")
        for src, token in self.lang_map.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            exists = token_id != tokenizer.unk_token_id
            print(f"  {src} -> {token}: id={token_id}, valid={exists}")
    
    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        names = [item['name'] for item in batch]
        srcs = [item['src'] for item in batch]
        
        body_codes = [item['body_codes'] for item in batch]
        lhand_codes = [item['lhand_codes'] for item in batch]
        rhand_codes = [item['rhand_codes'] for item in batch]
        lengths = [item['code_length'] for item in batch]
        
        B = len(batch)
        max_code_len = max(lengths)
        
        # Pad codes
        body_padded = torch.zeros(B, max_code_len, dtype=torch.long)
        lhand_padded = torch.zeros(B, max_code_len, dtype=torch.long)
        rhand_padded = torch.zeros(B, max_code_len, dtype=torch.long)
        code_mask = torch.zeros(B, max_code_len, dtype=torch.bool)
        
        for i in range(B):
            L = lengths[i]
            body_padded[i, :L] = body_codes[i][:L]
            lhand_padded[i, :L] = lhand_codes[i][:L]
            rhand_padded[i, :L] = rhand_codes[i][:L]
            code_mask[i, :L] = True
        
        # Tokenize text with language tokens (SOKE style)
        # Format: [text] [lang_token] (e.g., "Hello world en_ASL")
        text_with_lang = []
        for text, src in zip(texts, srcs):
            lang_token = self.lang_map.get(src, 'en_ASL')
            text_with_lang.append(f"{text} {lang_token}")
        
        encoded = self.tokenizer(
            text_with_lang,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'body_codes': body_padded,
            'lhand_codes': lhand_padded,
            'rhand_codes': rhand_padded,
            'code_mask': code_mask,
            'lengths': torch.tensor(lengths, dtype=torch.long),
            'names': names,
            'srcs': srcs,
            'texts': texts,
        }


# ============================================================================
# Training Functions
# ============================================================================
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    epoch,
    cfg,
    device,
    writer=None,
    global_step=0,
):
    """Train for one epoch."""
    model.train()
    
    tracker = MetricTracker()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        body_codes = batch['body_codes'].to(device)
        lhand_codes = batch['lhand_codes'].to(device)
        rhand_codes = batch['rhand_codes'].to(device)
        code_mask = batch['code_mask'].to(device)
        lengths = batch['lengths'].to(device)
        srcs = batch['srcs']
        
        # Debug: Check for out-of-range token IDs (first batch only)
        if batch_idx == 0 and epoch == 0:
            vocab_size = model.mbart_encoder.config.vocab_size if hasattr(model, 'mbart_encoder') else \
                        model.module.mbart_encoder.config.vocab_size
            max_token_id = input_ids.max().item()
            min_token_id = input_ids.min().item()
            print(f"\n  [Debug] First Batch Analysis:")
            print(f"    Text: vocab_size={vocab_size}, token_id range=[{min_token_id}, {max_token_id}]")
            print(f"    Input shape: {input_ids.shape}, Attention mask sum: {attention_mask.sum().item()}")
            
            # Motion codes stats
            print(f"    Body codes: shape={body_codes.shape}, range=[{body_codes.min().item()}, {body_codes.max().item()}]")
            print(f"    LHand codes: shape={lhand_codes.shape}, range=[{lhand_codes.min().item()}, {lhand_codes.max().item()}]")
            print(f"    RHand codes: shape={rhand_codes.shape}, range=[{rhand_codes.min().item()}, {rhand_codes.max().item()}]")
            print(f"    Code mask: valid positions = {code_mask.sum().item()} / {code_mask.numel()}")
            print(f"    Lengths: {lengths.tolist()[:5]}...")
            
            # Sample text
            sample_text = batch['texts'][0][:100] if batch.get('texts') else "N/A"
            print(f"    Sample text: {sample_text}...")
            
            if max_token_id >= vocab_size:
                print(f"  [ERROR] Token ID {max_token_id} >= vocab_size {vocab_size}!")
                # Find which tokens are problematic
                problematic = (input_ids >= vocab_size).nonzero()
                if len(problematic) > 0:
                    print(f"  Problematic positions: {problematic[:10].tolist()}")
                # Clamp to valid range (temporary fix)
                input_ids = input_ids.clamp(0, vocab_size - 1)
        
        optimizer.zero_grad()
        
        # Forward with AMP
        with autocast(enabled=cfg.get('training', {}).get('use_amp', True)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                body_tokens=body_codes,
                lhand_tokens=lhand_codes,
                rhand_tokens=rhand_codes,
                data_src=srcs,
                lengths=lengths,
                code_mask=code_mask,
                debug=(batch_idx == 0 and epoch == 0),  # Debug first batch only
            )
            
            loss = outputs['loss']
        
        # Skip batch if loss is nan/inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Warning] Batch {batch_idx}: loss is nan/inf, skipping...")
            optimizer.zero_grad()
            continue
        
        # Backward
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.get('training', {}).get('grad_clip', 1.0)
        )
        
        # Check for gradient explosion
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            # Debug: find which parameter has nan gradient
            nan_params = []
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    nan_params.append(name)
            if nan_params and len(nan_params) <= 5:
                print(f"[Warning] Batch {batch_idx}: grad nan/inf in: {nan_params[:5]}")
            else:
                print(f"[Warning] Batch {batch_idx}: grad_norm is nan/inf, skipping...")
            optimizer.zero_grad()
            scaler.update()  # Must call update() after unscale_()
            continue
        elif grad_norm > 100:
            # Debug: show batch info for large gradient
            max_len = lengths.max().item()
            mean_len = lengths.float().mean().item()
            print(f"[Warning] Batch {batch_idx}: Large grad_norm = {grad_norm:.2f} (max_len={max_len}, mean_len={mean_len:.1f})")
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # ========== Compute Accuracies ==========
        with torch.no_grad():
            # Body accuracy (only on masked positions)
            body_logits = outputs['body_logits']
            body_acc = compute_accuracy(body_logits, body_codes, ignore_index=-100)
            
            # Hand accuracies
            lhand_logits = outputs['lhand_logits']
            lhand_acc = compute_accuracy(lhand_logits, lhand_codes, ignore_index=-100)
            
            rhand_logits = outputs['rhand_logits']
            rhand_acc = compute_accuracy(rhand_logits, rhand_codes, ignore_index=-100)
            
            # Length accuracy
            length_logits = outputs['length_logits']
            length_acc = compute_length_accuracy(length_logits, lengths, tolerance=0)
            length_acc_t2 = compute_length_accuracy(length_logits, lengths, tolerance=2)
        
        # Update metrics
        metrics = {
            'loss': loss.item(),
            'loss_body': outputs['body_loss'].item(),
            'loss_lhand': outputs['lhand_loss'].item(),
            'loss_rhand': outputs['rhand_loss'].item(),
            'loss_length': outputs['length_loss'].item(),
            'acc_body': body_acc,
            'acc_lhand': lhand_acc,
            'acc_rhand': rhand_acc,
            'acc_length': length_acc,
            'acc_length_t2': length_acc_t2,
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'lr': optimizer.param_groups[0]['lr'],
        }
        
        tracker.update(metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{tracker.metrics['loss'].avg:.4f}",
            'body': f"{tracker.metrics['acc_body'].avg:.3f}",
            'hand': f"{(tracker.metrics['acc_lhand'].avg + tracker.metrics['acc_rhand'].avg)/2:.3f}",
            'len': f"{tracker.metrics['acc_length'].avg:.3f}",
        })
        
        # Tensorboard logging
        if writer is not None and batch_idx % 10 == 0:
            for key, value in metrics.items():
                writer.add_scalar(f'train/{key}', value, global_step)
        
        global_step += 1
    
    return tracker.get_avg(), global_step


@torch.no_grad()
def validate(model, dataloader, cfg, device, writer=None, global_step=0):
    """Validation loop."""
    model.eval()
    
    tracker = MetricTracker()
    
    pbar = tqdm(dataloader, desc="Validation")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        body_codes = batch['body_codes'].to(device)
        lhand_codes = batch['lhand_codes'].to(device)
        rhand_codes = batch['rhand_codes'].to(device)
        code_mask = batch['code_mask'].to(device)
        lengths = batch['lengths'].to(device)
        srcs = batch['srcs']
        
        with autocast(enabled=cfg.get('training', {}).get('use_amp', True)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                body_tokens=body_codes,
                lhand_tokens=lhand_codes,
                rhand_tokens=rhand_codes,
                data_src=srcs,
                lengths=lengths,
                code_mask=code_mask,
            )
        
        # Accuracies
        body_acc = compute_accuracy(outputs['body_logits'], body_codes)
        lhand_acc = compute_accuracy(outputs['lhand_logits'], lhand_codes)
        rhand_acc = compute_accuracy(outputs['rhand_logits'], rhand_codes)
        length_acc = compute_length_accuracy(outputs['length_logits'], lengths)
        length_acc_t2 = compute_length_accuracy(outputs['length_logits'], lengths, tolerance=2)
        
        metrics = {
            'loss': outputs['loss'].item(),
            'loss_body': outputs['body_loss'].item(),
            'loss_lhand': outputs['lhand_loss'].item(),
            'loss_rhand': outputs['rhand_loss'].item(),
            'loss_length': outputs['length_loss'].item(),
            'acc_body': body_acc,
            'acc_lhand': lhand_acc,
            'acc_rhand': rhand_acc,
            'acc_length': length_acc,
            'acc_length_t2': length_acc_t2,
        }
        
        tracker.update(metrics)
        
        pbar.set_postfix({
            'loss': f"{tracker.metrics['loss'].avg:.4f}",
            'body': f"{tracker.metrics['acc_body'].avg:.3f}",
        })
    
    avg_metrics = tracker.get_avg()
    
    # Tensorboard logging
    if writer is not None:
        for key, value in avg_metrics.items():
            writer.add_scalar(f'val/{key}', value, global_step)
    
    return avg_metrics


# ============================================================================
# Checkpoint
# ============================================================================
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, metrics, path):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(state, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scheduler, scaler, path, device):
    """Load checkpoint."""
    print(f"Loading checkpoint from {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['global_step'], checkpoint.get('metrics', {})


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, GPU: {args.gpu}")
    
    # Load config
    cfg = load_config(args.config)
    
    # Override dataset if specified
    if args.dataset:
        cfg['data']['dataset_name'] = args.dataset
        print(f"Dataset override: {args.dataset}")
    
    # Override AMP setting
    if args.no_amp:
        cfg['training']['use_amp'] = False
        print(f"AMP disabled")
    
    # Override learning rate
    if args.lr:
        cfg['training']['learning_rate'] = args.lr
        print(f"Learning rate override: {args.lr}")
    
    # Set seed
    set_seed(args.seed)
    
    # Experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.name or f"maskgit_{timestamp}"
    
    # Directories
    model_cfg = cfg.get('model', {})
    train_cfg = cfg.get('training', {})
    
    checkpoint_dir = os.path.join(train_cfg.get('checkpoint_dir', 'checkpoints'), exp_name)
    log_dir = os.path.join(train_cfg.get('log_dir', 'logs'), exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    # Tensorboard
    writer = SummaryWriter(log_dir)
    
    # Wandb
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="maskgit-sign-language",
            name=exp_name,
            config=cfg,
        )
    
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Config: {args.config}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print("=" * 60)
    
    # ========== Create Model ==========
    from model.maskgit_mbart import MaskGITmBART
    
    model = MaskGITmBART(
        mbart_path=model_cfg.get('mbart_path', 'deps/mbart-h2s-csl-phoenix'),
        freeze_encoder=model_cfg.get('freeze_encoder', True),
        body_vocab_size=model_cfg.get('body_vocab_size', 96),
        hand_vocab_size=model_cfg.get('hand_vocab_size', 192),
        body_d_model=model_cfg.get('body_d_model', 512),
        body_n_heads=model_cfg.get('body_n_heads', 8),
        body_n_layers=model_cfg.get('body_n_layers', 12),
        body_d_ff=model_cfg.get('body_d_ff', 2048),
        hand_d_model=model_cfg.get('hand_d_model', 384),
        hand_n_heads=model_cfg.get('hand_n_heads', 6),
        hand_n_layers=model_cfg.get('hand_n_layers', 8),
        hand_d_ff=model_cfg.get('hand_d_ff', 1536),
        max_seq_len=model_cfg.get('max_seq_len', 256),
        max_chunks=model_cfg.get('max_chunks', 128),
        dropout=model_cfg.get('dropout', 0.1),
        label_smoothing=model_cfg.get('label_smoothing', 0.1),
        num_iterations=model_cfg.get('num_iterations', 10),
    )
    
    model = model.to(device)
    
    # Multi-GPU
    if ',' in args.gpu:
        model = nn.DataParallel(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Get tokenizer from model
    tokenizer = model.tokenizer if not isinstance(model, nn.DataParallel) else model.module.tokenizer
    
    # Debug: Print tokenizer info
    print(f"\n[Tokenizer Info]")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Model vocab size: {model.mbart_encoder.config.vocab_size if hasattr(model, 'mbart_encoder') else model.module.mbart_encoder.config.vocab_size}")
    
    # Check for language tokens
    for token in ['en_ASL', 'zh_CSL', 'de_DGS', 'en_XX', 'zh_CN', 'de_DE']:
        token_id = tokenizer.convert_tokens_to_ids(token)
        exists = token_id != tokenizer.unk_token_id
        print(f"  {token}: id={token_id}, exists={exists}")
    
    # Check if vocab size matches embedding size
    model_ref = model if not isinstance(model, nn.DataParallel) else model.module
    vocab_size = len(tokenizer)
    embed_size = model_ref.mbart_encoder.embed_tokens.num_embeddings
    if vocab_size != embed_size:
        print(f"\n  [WARNING] Vocab size mismatch! tokenizer={vocab_size}, embeddings={embed_size}")
        print(f"  Resizing embeddings to match tokenizer...")
        model_ref.mbart_encoder.resize_token_embeddings(vocab_size)
        print(f"  New embedding size: {model_ref.mbart_encoder.embed_tokens.num_embeddings}")
    
    # ========== Create Dataloaders ==========
    train_loader, val_loader = create_dataloaders(cfg, tokenizer, debug=args.debug)
    
    # ========== Optimizer ==========
    lr = float(train_cfg.get('learning_rate', 1e-4))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    
    # ========== Scheduler ==========
    num_epochs = train_cfg.get('epochs', 100)
    warmup_steps = train_cfg.get('warmup_steps', 4000)
    total_steps = len(train_loader) * num_epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos',
    )
    
    # ========== AMP Scaler ==========
    scaler = GradScaler(enabled=train_cfg.get('use_amp', True))
    
    # ========== Resume ==========
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        start_epoch, global_step, _ = load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume, device
        )
        start_epoch += 1
    
    # ========== Training Loop ==========
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    # Encoder unfreezing schedule
    unfreeze_epoch = train_cfg.get('unfreeze_encoder_epoch', -1)  # -1 = never
    
    for epoch in range(start_epoch, num_epochs):
        # Check if we should unfreeze encoder
        if unfreeze_epoch > 0 and epoch == unfreeze_epoch:
            print(f"\n>>> Unfreezing mBART encoder at epoch {epoch}")
            if isinstance(model, nn.DataParallel):
                model.module.unfreeze_encoder(unfreeze_layers=train_cfg.get('unfreeze_layers', -1))
            else:
                model.unfreeze_encoder(unfreeze_layers=train_cfg.get('unfreeze_layers', -1))
            
            # Reduce learning rate after unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        
        # Train
        train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, cfg, device, writer, global_step
        )
        
        # Validate
        val_metrics = validate(model, val_loader, cfg, device, writer, global_step)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"    Body Loss: {train_metrics['loss_body']:.4f} | Acc: {train_metrics['acc_body']:.4f}")
        print(f"    LHand Loss: {train_metrics['loss_lhand']:.4f} | Acc: {train_metrics['acc_lhand']:.4f}")
        print(f"    RHand Loss: {train_metrics['loss_rhand']:.4f} | Acc: {train_metrics['acc_rhand']:.4f}")
        print(f"    Length Loss: {train_metrics['loss_length']:.4f} | Acc: {train_metrics['acc_length']:.4f} (±2: {train_metrics['acc_length_t2']:.4f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"    Body Acc: {val_metrics['acc_body']:.4f}")
        print(f"    LHand Acc: {val_metrics['acc_lhand']:.4f}")
        print(f"    RHand Acc: {val_metrics['acc_rhand']:.4f}")
        print(f"    Length Acc: {val_metrics['acc_length']:.4f} (±2: {val_metrics['acc_length_t2']:.4f})")
        print(f"{'='*60}\n")
        
        # Wandb logging
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/body_loss': train_metrics['loss_body'],
                'train/lhand_loss': train_metrics['loss_lhand'],
                'train/rhand_loss': train_metrics['loss_rhand'],
                'train/length_loss': train_metrics['loss_length'],
                'train/body_acc': train_metrics['acc_body'],
                'train/lhand_acc': train_metrics['acc_lhand'],
                'train/rhand_acc': train_metrics['acc_rhand'],
                'train/length_acc': train_metrics['acc_length'],
                'val/loss': val_metrics['loss'],
                'val/body_acc': val_metrics['acc_body'],
                'val/lhand_acc': val_metrics['acc_lhand'],
                'val/rhand_acc': val_metrics['acc_rhand'],
                'val/length_acc': val_metrics['acc_length'],
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
        
        # Save every N epochs
        save_every = train_cfg.get('save_every', 5)
        if (epoch + 1) % save_every == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                {'train': train_metrics, 'val': val_metrics},
                os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
            )
        
        # Save best
        if is_best:
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, global_step,
                {'train': train_metrics, 'val': val_metrics},
                os.path.join(checkpoint_dir, 'best.pt')
            )
        
        # Save latest
        save_checkpoint(
            model, optimizer, scheduler, scaler, epoch, global_step,
            {'train': train_metrics, 'val': val_metrics},
            os.path.join(checkpoint_dir, 'latest.pt')
        )
    
    print("Training completed!")
    writer.close()
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()