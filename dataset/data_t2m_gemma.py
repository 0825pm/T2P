"""
Dataset for Gemma-based Text-to-Motion Training (Interleaved Format)

Data format:
- Text: German sign language sentences (Phoenix14T)
- Motion codes: Pre-extracted from VQ-VAE decouple (body, lhand, rhand)

Sequence format:
    {text} <MOTION_START> <BODY_0> <LHAND_5> <RHAND_3> ... <MOTION_END>

Usage:
    dataset = GemmaT2MDataset(config, split='train')
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)
"""

import os
import io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer


class GemmaT2MDataset(Dataset):
    """
    Dataset for Gemma T2M training with interleaved motion tokens.
    
    Expected file structure:
        {data_path}/train.text     - Text sentences
        {data_path}/train.files    - File names
        {motion_code_path}/train/  - Motion codes (.npy files)
    """
    
    def __init__(
        self,
        text_path: str,
        files_path: str,
        motion_code_path: str,
        tokenizer: AutoTokenizer,
        max_text_length: int = 128,
        max_motion_length: int = 150,
        body_code_num: int = 96,
        lhand_code_num: int = 192,
        rhand_code_num: int = 192,
        motion_only_loss: bool = True,
        **kwargs
    ):
        """
        Args:
            text_path: Path to text file (e.g., /data/train.text)
            files_path: Path to files file (e.g., /data/train.files)
            motion_code_path: Path to motion codes directory
            tokenizer: Tokenizer with motion tokens added
            max_text_length: Maximum text sequence length
            max_motion_length: Maximum motion length in timesteps
            body_code_num: Number of body codes
            lhand_code_num: Number of left hand codes
            rhand_code_num: Number of right hand codes
            motion_only_loss: Whether to mask text tokens in loss
        """
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_motion_length = max_motion_length
        self.body_code_num = body_code_num
        self.lhand_code_num = lhand_code_num
        self.rhand_code_num = rhand_code_num
        self.motion_only_loss = motion_only_loss
        self.motion_code_path = motion_code_path
        
        # Get special token IDs
        self.motion_start_id = tokenizer.convert_tokens_to_ids("<MOTION_START>")
        self.motion_end_id = tokenizer.convert_tokens_to_ids("<MOTION_END>")
        
        # Load samples
        self.samples = []
        self._load_samples(text_path, files_path, motion_code_path)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_samples(self, text_path: str, files_path: str, motion_code_path: str):
        """Load text-motion pairs."""
        skipped = 0
        
        with io.open(text_path, mode='r', encoding='utf-8') as text_file, \
             io.open(files_path, mode='r', encoding='utf-8') as files_file:
            
            for text_line, files_line in zip(text_file, files_file):
                text_line = text_line.strip()
                files_line = files_line.strip()
                
                if not text_line or not files_line:
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
                    'text': text_line,
                    'name': name,
                    'code_path': code_path,
                })
        
        if skipped > 0:
            print(f"  Skipped {skipped} samples (motion codes not found)")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_motion_codes(self, code_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Load motion codes from .npy file.
        
        Expected format: (1, T, 3) or (T, 3) where columns are [body, lhand, rhand]
        
        Returns:
            body_codes, lhand_codes, rhand_codes, length
        """
        codes = np.load(code_path)
        
        # Handle different shapes
        if len(codes.shape) == 3:
            codes = codes[0]  # (1, T, 3) -> (T, 3)
        
        # Truncate if needed
        T = min(len(codes), self.max_motion_length)
        codes = codes[:T]
        
        body_codes = codes[:, 0].astype(np.int64)
        lhand_codes = codes[:, 1].astype(np.int64)
        rhand_codes = codes[:, 2].astype(np.int64)
        
        # Clamp to valid range
        body_codes = np.clip(body_codes, 0, self.body_code_num - 1)
        lhand_codes = np.clip(lhand_codes, 0, self.lhand_code_num - 1)
        rhand_codes = np.clip(rhand_codes, 0, self.rhand_code_num - 1)
        
        return body_codes, lhand_codes, rhand_codes, T
    
    def _create_interleaved_sequence(
        self,
        text: str,
        body_codes: np.ndarray,
        lhand_codes: np.ndarray,
        rhand_codes: np.ndarray,
    ) -> str:
        """
        Create interleaved sequence string.
        
        Format: {text} <MOTION_START> <BODY_0> <LHAND_5> <RHAND_3> ... <MOTION_END>
        """
        motion_tokens = ["<MOTION_START>"]
        
        for t in range(len(body_codes)):
            motion_tokens.append(f"<BODY_{body_codes[t]}>")
            motion_tokens.append(f"<LHAND_{lhand_codes[t]}>")
            motion_tokens.append(f"<RHAND_{rhand_codes[t]}>")
        
        motion_tokens.append("<MOTION_END>")
        
        return f"{text} {' '.join(motion_tokens)}"
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load motion codes
        body_codes, lhand_codes, rhand_codes, length = self._load_motion_codes(
            sample['code_path']
        )
        
        # Create full sequence
        full_sequence = self._create_interleaved_sequence(
            sample['text'],
            body_codes,
            lhand_codes,
            rhand_codes,
        )
        
        return {
            'text': sample['text'],
            'name': sample['name'],
            'full_sequence': full_sequence,
            'body_codes': torch.tensor(body_codes, dtype=torch.long),
            'lhand_codes': torch.tensor(lhand_codes, dtype=torch.long),
            'rhand_codes': torch.tensor(rhand_codes, dtype=torch.long),
            'motion_length': length,
        }
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate function for DataLoader.
        
        Returns tokenized and padded inputs ready for model.
        """
        texts = [item['text'] for item in batch]
        names = [item['name'] for item in batch]
        full_sequences = [item['full_sequence'] for item in batch]
        motion_lengths = [item['motion_length'] for item in batch]
        
        # Tokenize full sequences
        encoding = self.tokenizer(
            full_sequences,
            padding="longest",
            truncation=True,
            max_length=self.max_text_length + self.max_motion_length * 3 + 10,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        # Mask text tokens if motion_only_loss is True
        if self.motion_only_loss:
            batch_size = len(batch)
            for i in range(batch_size):
                # Find motion start position
                motion_start_positions = (input_ids[i] == self.motion_start_id).nonzero(as_tuple=True)[0]
                
                if len(motion_start_positions) > 0:
                    start_pos = motion_start_positions[0].item()
                    # Mask everything up to and including <MOTION_START>
                    labels[i, :start_pos + 1] = -100
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Also collect raw codes for evaluation
        body_codes_list = [item['body_codes'] for item in batch]
        lhand_codes_list = [item['lhand_codes'] for item in batch]
        rhand_codes_list = [item['rhand_codes'] for item in batch]
        
        # Pad codes
        max_motion_len = max(motion_lengths)
        body_codes_padded = torch.zeros(len(batch), max_motion_len, dtype=torch.long)
        lhand_codes_padded = torch.zeros(len(batch), max_motion_len, dtype=torch.long)
        rhand_codes_padded = torch.zeros(len(batch), max_motion_len, dtype=torch.long)
        
        for i in range(len(batch)):
            L = motion_lengths[i]
            body_codes_padded[i, :L] = body_codes_list[i]
            lhand_codes_padded[i, :L] = lhand_codes_list[i]
            rhand_codes_padded[i, :L] = rhand_codes_list[i]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'texts': texts,
            'names': names,
            'body_codes': body_codes_padded,
            'lhand_codes': lhand_codes_padded,
            'rhand_codes': rhand_codes_padded,
            'motion_lengths': motion_lengths,
        }


def load_gemma_t2m_data(
    config: dict,
    tokenizer: AutoTokenizer,
) -> Tuple[GemmaT2MDataset, GemmaT2MDataset, GemmaT2MDataset]:
    """
    Load train/dev/test datasets.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer with motion tokens
    
    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    loss_config = config.get("loss", {})
    
    train_path = data_config["train"]
    dev_path = data_config["dev"]
    test_path = data_config["test"]
    motion_code_path = data_config.get("motion_code_path", "Data/motion_codes_decouple")
    
    common_kwargs = {
        "tokenizer": tokenizer,
        "max_text_length": model_config.get("max_text_length", 128),
        "max_motion_length": model_config.get("max_motion_length", 150),
        "body_code_num": model_config.get("body_code_num", 96),
        "lhand_code_num": model_config.get("lhand_code_num", 192),
        "rhand_code_num": model_config.get("rhand_code_num", 192),
        "motion_only_loss": loss_config.get("motion_only_loss", True),
    }
    
    print("Loading train data...")
    train_dataset = GemmaT2MDataset(
        text_path=train_path + ".text",
        files_path=train_path + ".files",
        motion_code_path=os.path.join(motion_code_path, 'train'),
        **common_kwargs,
    )
    
    print("Loading dev data...")
    dev_dataset = GemmaT2MDataset(
        text_path=dev_path + ".text",
        files_path=dev_path + ".files",
        motion_code_path=os.path.join(motion_code_path, 'dev'),
        **common_kwargs,
    )
    
    print("Loading test data...")
    test_dataset = GemmaT2MDataset(
        text_path=test_path + ".text",
        files_path=test_path + ".files",
        motion_code_path=os.path.join(motion_code_path, 'test'),
        **common_kwargs,
    )
    
    return train_dataset, dev_dataset, test_dataset


def create_dataloaders(
    train_dataset: GemmaT2MDataset,
    dev_dataset: GemmaT2MDataset,
    test_dataset: Optional[GemmaT2MDataset] = None,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create DataLoaders for train/dev/test."""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dev_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, dev_loader, test_loader


# ========== For Testing ==========

class DummyGemmaT2MDataset(Dataset):
    """Dummy dataset for testing without actual data files."""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_samples: int = 100,
        max_text_length: int = 128,
        max_motion_length: int = 150,
        body_code_num: int = 96,
        lhand_code_num: int = 192,
        rhand_code_num: int = 192,
        motion_only_loss: bool = True,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_text_length = max_text_length
        self.max_motion_length = max_motion_length
        self.body_code_num = body_code_num
        self.lhand_code_num = lhand_code_num
        self.rhand_code_num = rhand_code_num
        self.motion_only_loss = motion_only_loss
        
        self.motion_start_id = tokenizer.convert_tokens_to_ids("<MOTION_START>")
        self.motion_end_id = tokenizer.convert_tokens_to_ids("<MOTION_END>")
        
        # Generate dummy data
        self.samples = self._generate_dummy_samples()
    
    def _generate_dummy_samples(self):
        """Generate random dummy samples."""
        samples = []
        
        dummy_texts = [
            "Der Mann geht nach links.",
            "Die Frau winkt mit der Hand.",
            "Ein Kind spielt im Park.",
            "Der Hund läuft schnell.",
            "Sie zeigt auf den Tisch.",
        ]
        
        for i in range(self.num_samples):
            text = dummy_texts[i % len(dummy_texts)]
            motion_length = np.random.randint(20, self.max_motion_length)
            
            body_codes = np.random.randint(0, self.body_code_num, motion_length)
            lhand_codes = np.random.randint(0, self.lhand_code_num, motion_length)
            rhand_codes = np.random.randint(0, self.rhand_code_num, motion_length)
            
            samples.append({
                'text': text,
                'name': f'dummy_{i}',
                'body_codes': body_codes,
                'lhand_codes': lhand_codes,
                'rhand_codes': rhand_codes,
                'motion_length': motion_length,
            })
        
        return samples
    
    def __len__(self):
        return self.num_samples
    
    def _create_interleaved_sequence(self, text, body_codes, lhand_codes, rhand_codes):
        motion_tokens = ["<MOTION_START>"]
        for t in range(len(body_codes)):
            motion_tokens.append(f"<BODY_{body_codes[t]}>")
            motion_tokens.append(f"<LHAND_{lhand_codes[t]}>")
            motion_tokens.append(f"<RHAND_{rhand_codes[t]}>")
        motion_tokens.append("<MOTION_END>")
        return f"{text} {' '.join(motion_tokens)}"
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        full_sequence = self._create_interleaved_sequence(
            sample['text'],
            sample['body_codes'],
            sample['lhand_codes'],
            sample['rhand_codes'],
        )
        
        return {
            'text': sample['text'],
            'name': sample['name'],
            'full_sequence': full_sequence,
            'body_codes': torch.tensor(sample['body_codes'], dtype=torch.long),
            'lhand_codes': torch.tensor(sample['lhand_codes'], dtype=torch.long),
            'rhand_codes': torch.tensor(sample['rhand_codes'], dtype=torch.long),
            'motion_length': sample['motion_length'],
        }
    
    def collate_fn(self, batch):
        """Same as GemmaT2MDataset.collate_fn"""
        texts = [item['text'] for item in batch]
        names = [item['name'] for item in batch]
        full_sequences = [item['full_sequence'] for item in batch]
        motion_lengths = [item['motion_length'] for item in batch]
        
        encoding = self.tokenizer(
            full_sequences,
            padding="longest",
            truncation=True,
            max_length=self.max_text_length + self.max_motion_length * 3 + 10,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        labels = input_ids.clone()
        
        if self.motion_only_loss:
            for i in range(len(batch)):
                motion_start_positions = (input_ids[i] == self.motion_start_id).nonzero(as_tuple=True)[0]
                if len(motion_start_positions) > 0:
                    start_pos = motion_start_positions[0].item()
                    labels[i, :start_pos + 1] = -100
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        body_codes_list = [item['body_codes'] for item in batch]
        lhand_codes_list = [item['lhand_codes'] for item in batch]
        rhand_codes_list = [item['rhand_codes'] for item in batch]
        
        max_motion_len = max(motion_lengths)
        body_codes_padded = torch.zeros(len(batch), max_motion_len, dtype=torch.long)
        lhand_codes_padded = torch.zeros(len(batch), max_motion_len, dtype=torch.long)
        rhand_codes_padded = torch.zeros(len(batch), max_motion_len, dtype=torch.long)
        
        for i in range(len(batch)):
            L = motion_lengths[i]
            body_codes_padded[i, :L] = body_codes_list[i]
            lhand_codes_padded[i, :L] = lhand_codes_list[i]
            rhand_codes_padded[i, :L] = rhand_codes_list[i]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'texts': texts,
            'names': names,
            'body_codes': body_codes_padded,
            'lhand_codes': lhand_codes_padded,
            'rhand_codes': rhand_codes_padded,
            'motion_lengths': motion_lengths,
        }


if __name__ == "__main__":
    # Test with dummy data
    from transformers import AutoTokenizer
    
    print("Testing GemmaT2MDataset...")
    
    # Load tokenizer and add motion tokens
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add motion tokens
    new_tokens = ["<MOTION_START>", "<MOTION_END>"]
    new_tokens.extend([f"<BODY_{i}>" for i in range(96)])
    new_tokens.extend([f"<LHAND_{i}>" for i in range(192)])
    new_tokens.extend([f"<RHAND_{i}>" for i in range(192)])
    tokenizer.add_tokens(new_tokens)
    
    # Create dummy dataset
    dataset = DummyGemmaT2MDataset(
        tokenizer=tokenizer,
        num_samples=100,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    
    # Test one batch
    batch = next(iter(dataloader))
    
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Body codes shape: {batch['body_codes'].shape}")
    print(f"Motion lengths: {batch['motion_lengths']}")
    
    # Check label masking
    num_masked = (batch['labels'] == -100).sum().item()
    total_labels = batch['labels'].numel()
    print(f"Masked labels: {num_masked}/{total_labels} ({100*num_masked/total_labels:.1f}%)")
    
    print("Test passed!")
