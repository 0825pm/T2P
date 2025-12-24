"""
Extract Motion Codes from trained VQ-VAE
Adapted from SOKE's scripts/get_motion_code.py

Supports both:
- vqvae: Single codebook for all joints -> saves (T',) array
- vqvae_decouple: Separate codebooks -> saves (1, T', 3) array [body, lhand, rhand]

Usage:
    python scripts/get_motion_code.py \
        --config configs/t2p_vqvae.yaml \
        --model vqvae \
        --checkpoint checkpoint/vqvae_xxx/model_best.pth \
        --output_dir Data/motion_codes \
        --gpu 0

    python scripts/get_motion_code.py \
        --config configs/t2p_vqvae_decouple.yaml \
        --model vqvae_decouple \
        --checkpoint checkpoint/vqvae_decouple_xxx/model_best.pth \
        --output_dir Data/motion_codes_decouple \
        --gpu 0
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import io
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract motion codes from VQ-VAE')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, choices=['vqvae', 'vqvae_decouple'],
                        help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='Data/motion_codes',
                        help='Output directory for motion codes')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device id')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'dev', 'test', 'all'],
                        help='Which split to process')
    return parser.parse_args()


def load_model(args, config):
    """Load VQ-VAE model from checkpoint."""
    model_config = config["model"]["qae"]
    
    if args.model == 'vqvae':
        from model.vqvae import VQVAE
    else:
        from model.vqvae_decouple import VQVAE
    
    model = VQVAE(**model_config)
    
    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()
    model.eval()
    
    print(f"Model loaded from {args.checkpoint}")
    return model


def adjust_motion_length(motion_array, min_len=20, max_len=300):
    """Adjust motion length using linear interpolation."""
    m_length = len(motion_array)
    
    if m_length < min_len:
        idx = np.linspace(0, m_length - 1, num=min_len, dtype=int)
        motion_array = motion_array[idx]
    elif m_length > max_len:
        idx = np.linspace(0, m_length - 1, num=max_len, dtype=int)
        motion_array = motion_array[idx]
    
    return motion_array


def load_raw_data(data_path, trg_size=151, skip_frames=1, min_len=20, max_len=300):
    """
    Load raw data from files without torchtext.
    
    Returns:
        list of dicts with 'name' and 'pose' keys
    """
    # Determine file extensions
    src_path = data_path + ".txt"
    trg_path = data_path + ".skels"
    files_path = data_path + ".files"
    
    samples = []
    
    with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
         io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
         io.open(files_path, mode='r', encoding='utf-8') as files_file:
        
        for src_line, trg_line, files_line in zip(src_file, trg_file, files_file):
            src_line = src_line.strip()
            trg_line = trg_line.strip()
            files_line = files_line.strip()
            
            # Parse target
            trg_values = trg_line.split(" ")
            if len(trg_values) <= 1:
                continue
            
            trg_values = [float(v) for v in trg_values]
            
            # Reshape to frames: (T, trg_size)
            trg_frames = []
            for i in range(0, len(trg_values), trg_size * skip_frames):
                frame = trg_values[i:i + trg_size]
                if len(frame) == trg_size:
                    trg_frames.append(frame)
            
            if len(trg_frames) == 0:
                continue
            
            # Convert to numpy
            pose_array = np.array(trg_frames, dtype=np.float32)  # (T, 151)
            
            # Adjust length
            pose_array = adjust_motion_length(pose_array, min_len, max_len)
            
            # Extract pose without counter (first 150 dims)
            pose = pose_array[:, :150]  # (T, 150)
            
            # Get name from files_line (remove extension)
            name = os.path.splitext(os.path.basename(files_line))[0]
            if not name:
                name = f"sample_{len(samples)}"
            
            samples.append({
                'name': name,
                'pose': pose,
                'text': src_line,
            })
    
    return samples


@torch.no_grad()
def extract_codes_single(model, pose):
    """
    Extract codes from single codebook VQ-VAE.
    
    Args:
        model: VQ-VAE model
        pose: (1, T, 150) tensor
    
    Returns:
        codes: (T',) numpy array
    """
    codes_dict = model.encode(pose)
    codes = codes_dict['all'].cpu().numpy()[0]  # (T',)
    return codes


@torch.no_grad()
def extract_codes_decouple(model, pose):
    """
    Extract codes from decouple VQ-VAE (body, rhand, lhand).
    
    Args:
        model: Decouple VQ-VAE model
        pose: (1, T, 150) tensor
    
    Returns:
        codes: (1, T', 3) numpy array following SOKE format [body, lhand, rhand]
    """
    codes_dict = model.encode(pose)
    
    body_codes = codes_dict['body'].cpu().numpy()  # (1, T')
    rhand_codes = codes_dict['rhand'].cpu().numpy()
    lhand_codes = codes_dict['lhand'].cpu().numpy()
    
    # Stack in SOKE order: [body, lhand, rhand]
    # Shape: (1, T', 3)
    codes = np.stack([body_codes, lhand_codes, rhand_codes], axis=-1)
    
    return codes


def process_split(model, data_path, output_dir, model_type, split_name, config):
    """Process a single split and save motion codes."""
    
    data_cfg = config["data"]
    trg_size = 151  # 150 + 1 counter
    skip_frames = data_cfg.get("skip_frames", 1)
    min_len = data_cfg.get("min_motion_length", 20)
    max_len = data_cfg.get("max_motion_length", 300)
    
    # Load raw data
    print(f"  Loading data from {data_path}...")
    samples = load_raw_data(data_path, trg_size, skip_frames, min_len, max_len)
    print(f"  Loaded {len(samples)} samples")
    
    # Create output directory
    output_split_dir = os.path.join(output_dir, split_name)
    os.makedirs(output_split_dir, exist_ok=True)
    
    extract_fn = extract_codes_single if model_type == 'vqvae' else extract_codes_decouple
    
    total_samples = 0
    skipped_samples = 0
    
    for sample in tqdm(samples, desc=f'Processing {split_name}'):
        name = sample['name']
        pose = sample['pose']  # (T, 150)
        
        if pose.shape[0] == 0:
            skipped_samples += 1
            continue
        
        # To tensor
        pose_tensor = torch.tensor(pose).float().unsqueeze(0).cuda()  # (1, T, 150)
        
        # Extract codes
        codes = extract_fn(model, pose_tensor)
        
        # Save
        target_path = os.path.join(output_split_dir, f'{name}.npy')
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, codes)
        
        total_samples += 1
    
    print(f"  Saved {total_samples} samples, skipped {skipped_samples}")
    return total_samples


def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(f"Config loaded from {args.config}")
    print(f"Model type: {args.model}")
    
    # Load model
    model = load_model(args, config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data paths
    data_cfg = config["data"]
    split_paths = {
        'train': data_cfg.get("data_path", data_cfg.get("train")),
        'dev': data_cfg.get("dev_path", data_cfg.get("dev")),
        'test': data_cfg.get("test_path", data_cfg.get("test")),
    }
    
    # Process splits
    splits_to_process = ['train', 'dev', 'test'] if args.split == 'all' else [args.split]
    
    total_all = 0
    for split in splits_to_process:
        print(f"\nProcessing {split} split...")
        data_path = split_paths.get(split)
        
        if data_path is None:
            print(f"  Skipping {split}: path not found in config")
            continue
        
        try:
            total = process_split(model, data_path, args.output_dir, args.model, split, config)
            total_all += total
        except Exception as e:
            print(f"  Error processing {split}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Motion tokenization complete!")
    print(f"Total samples processed: {total_all}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")
    
    # Save metadata
    metadata = {
        'model_type': args.model,
        'checkpoint': args.checkpoint,
        'config': args.config,
        'total_samples': total_all,
    }
    
    if args.model == 'vqvae':
        metadata['code_num'] = config["model"]["qae"].get("code_num", 512)
        metadata['format'] = "(T',) - single codebook indices"
    else:
        metadata['body_code_num'] = config["model"]["qae"].get("body_code_num", 96)
        metadata['hand_code_num'] = config["model"]["qae"].get("hand_code_num", 192)
        metadata['format'] = "(1, T', 3) - [body, lhand, rhand] indices (SOKE format)"
    
    import json
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {os.path.join(args.output_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()