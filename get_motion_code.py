"""
Extract Motion Codes from trained VQ-VAE
Uses same data loading as train_vqvae_soke.py

Output: data/motion_codes/{src}/{split}/{original_name}.npy

Usage:
    python get_motion_code.py \
        --config configs/vqvae_soke.yaml \
        --checkpoint checkpoints/vqvae_soke/best.pth \
        --output_dir data/motion_codes
"""
import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vqvae_soke import VQVAE
from mGPT.data.humanml import H2SMotionDatasetVQ
from mGPT.data.utils import humanml3d_collate
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/motion_codes')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    print(f"{'='*60}")
    print("Motion Code Extraction")
    print(f"{'='*60}")
    
    # =========================================================================
    # Load model (same as train_vqvae_soke.py)
    # =========================================================================
    model = VQVAE(
        nfeats=model_config.get('nfeats', 133),
        body_code_num=model_config.get('body_code_num', 96),
        hand_code_num=model_config.get('hand_code_num', 192),
        code_dim=model_config.get('code_dim', 512),
        output_emb_width=model_config.get('output_emb_width', 512),
        down_t=model_config.get('down_t', 2),
        stride_t=model_config.get('stride_t', 2),
        width=model_config.get('width', 512),
        depth=model_config.get('depth', 3),
        dilation_growth_rate=model_config.get('dilation_growth_rate', 3),
        activation=model_config.get('activation', 'relu'),
        norm=model_config.get('norm', None),
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded: {args.checkpoint}")
    
    # =========================================================================
    # Load mean/std (same as train_vqvae_soke.py)
    # =========================================================================
    mean_path = data_config.get('mean_path')
    std_path = data_config.get('std_path')
    
    if mean_path and os.path.exists(mean_path):
        mean = torch.load(mean_path, weights_only=False)
        mean = mean[(3 + 3 * 11):]
        mean = torch.cat([mean[:-20], mean[-10:]], dim=0)
    else:
        mean = torch.zeros(133)
    
    if std_path and os.path.exists(std_path):
        std = torch.load(std_path, weights_only=False)
        std = std[(3 + 3 * 11):]
        std = torch.cat([std[:-20], std[-10:]], dim=0)
    else:
        std = torch.ones(133)
    
    # =========================================================================
    # Dataset kwargs (same as train_vqvae_soke.py)
    # =========================================================================
    dataset_kwargs = {
        'data_root': data_config.get('data_root'),
        'mean': mean,
        'std': std,
        'max_motion_length': data_config.get('max_motion_length', 300),
        'min_motion_length': data_config.get('min_motion_length', 40),
        'unit_length': data_config.get('unit_length', 4),
        'dataset_name': data_config.get('dataset_name', 'how2sign_csl_phoenix'),
        'csl_root': data_config.get('csl_root'),
        'phoenix_root': data_config.get('phoenix_poses_root', data_config.get('phoenix_root')),
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    total_all = 0
    
    # Track indices per folder for building index.json
    folder_indices = {}
    
    # =========================================================================
    # Process each split (same as train_vqvae_soke.py)
    # =========================================================================
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split}...")
        print(f"{'='*60}")
        
        try:
            dataset = H2SMotionDatasetVQ(split=split, win_size=None, **dataset_kwargs)
            print(f"Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"Skipping {split}: {e}")
            continue
        
        if len(dataset) == 0:
            print(f"Skipping {split}: no samples")
            continue
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=humanml3d_collate,
        )
        
        split_total = 0
        src_counts = {}
        
        for batch in tqdm(dataloader, desc=split):
            if batch is None:
                continue
            
            motion = batch['motion'].to(device)  # (1, T, 133)
            
            # Get original name from batch - try multiple keys
            original_name = None
            for key in ['name', 'names']:
                if key in batch and batch[key]:
                    val = batch[key]
                    if isinstance(val, list) and len(val) > 0:
                        original_name = val[0]
                    elif isinstance(val, str):
                        original_name = val
                    break
            
            # Fallback to sample index
            if original_name is None:
                original_name = f'sample_{total_all + split_total}'
            
            src = batch['src'][0] if 'src' in batch and batch['src'] else 'unknown'
            
            # Encode
            with torch.no_grad():
                codes_dict = model.encode(motion)
            
            body = codes_dict['body'].cpu().numpy()[0]   # (T',)
            lhand = codes_dict['lhand'].cpu().numpy()[0]
            rhand = codes_dict['rhand'].cpu().numpy()[0]
            
            # Stack: (T', 3) - [body, lhand, rhand]
            codes = np.stack([body, lhand, rhand], axis=-1)
            
            # Map src to folder name (matching data structure)
            src_to_folder = {
                'how2sign': 'How2Sign',
                'csl': 'CSL-Daily',
                'phoenix': 'Phoenix_2014T',
            }
            folder_name = src_to_folder.get(src, src)
            
            # Determine output directory based on dataset structure
            if src == 'csl':
                # CSL-Daily: all data in poses/ folder (no split subdirs)
                # split_name = split
                # out_dir = os.path.join(args.output_dir, folder_name, split_name)
                out_dir = os.path.join(args.output_dir, folder_name, 'codes')
                split_name = 'codes'
            elif src == 'phoenix':
                # Phoenix: uses 'dev' instead of 'val'
                split_name = 'dev' if split == 'val' else split
                out_dir = os.path.join(args.output_dir, folder_name, split_name)
            else:
                # How2Sign: train/val/test subdirs
                split_name = split
                out_dir = os.path.join(args.output_dir, folder_name, split_name)
            
            os.makedirs(out_dir, exist_ok=True)
            
            # Save with original name (sanitize for filesystem)
            safe_name = original_name.replace('/', '_').replace('\\', '_')
            np.save(os.path.join(out_dir, f'{safe_name}.npy'), codes)
            
            # Track for index.json
            folder_key = f"{folder_name}/{split_name}"
            if folder_key not in folder_indices:
                folder_indices[folder_key] = []
            folder_indices[folder_key].append({
                'name': safe_name,
                'path': f'{safe_name}.npy',
                'original_name': original_name,
                'src': src,
                'split': split,
            })
            
            split_total += 1
            src_counts[src] = src_counts.get(src, 0) + 1
            
            if split_total <= 3:
                rel_path = os.path.relpath(out_dir, args.output_dir)
                print(f"  Example: {rel_path}/{safe_name}.npy | shape={codes.shape} | orig={original_name}")
        
        print(f"Saved {split_total} samples")
        for src, cnt in src_counts.items():
            print(f"  {src}: {cnt}")
        total_all += split_total
    
    # =========================================================================
    # Create index.json for each folder
    # =========================================================================
    print(f"\n{'='*60}")
    print("Creating index files...")
    print(f"{'='*60}")
    
    for folder_key, indices in folder_indices.items():
        out_dir = os.path.join(args.output_dir, folder_key)
        index_path = os.path.join(out_dir, 'index.json')
        with open(index_path, 'w') as f:
            json.dump(indices, f, indent=2)
        print(f"  {folder_key}/index.json: {len(indices)} entries")
    
    # Save metadata
    metadata = {
        'checkpoint': args.checkpoint,
        'body_code_num': model_config.get('body_code_num', 96),
        'hand_code_num': model_config.get('hand_code_num', 192),
        'temporal_downsample': 4,
        'format': "(T', 3) - [body, lhand, rhand]",
        'total_samples': total_all,
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Done! Total: {total_all}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
