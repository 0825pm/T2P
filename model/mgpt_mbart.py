"""
mBART-based Language Model for Text-to-Motion Generation
"""
import torch
import torch.nn as nn
from typing import List, Set
from transformers import MBartTokenizer
from model.lm_multihead import LMMultiHead


class MBartT2M(nn.Module):
    """mBART-based Text-to-Motion model (SOKE style)."""
    
    def __init__(
        self,
        model_path: str = "facebook/mbart-large-cc25",
        model_type: str = "mbart_multi",
        body_codebook_size: int = 96,
        hand_codebook_size: int = 192,
        rhand_codebook_size: int = 192,
        max_length: int = 256,
        num_heads: int = 3,
        down_t: int = 2,
        # ========== Regularization ==========
        label_smoothing: float = 0.1,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.model_type = model_type
        self.num_heads = num_heads
        self.body_codebook_size = body_codebook_size
        self.hand_codebook_size = hand_codebook_size
        self.rhand_codebook_size = rhand_codebook_size
        self.max_length = max_length
        self.down_t = down_t
        
        # Initialize tokenizer
        self.tokenizer = MBartTokenizer.from_pretrained(model_path, legacy=True)
        
        # Add motion tokens
        all_motion_str = [f'<motion_id_{i}>' for i in range(body_codebook_size + 3)]
        all_hand_str = [f'<hand_id_{i}>' for i in range(hand_codebook_size + 3)]
        all_rhand_str = [f'<rhand_id_{i}>' for i in range(rhand_codebook_size + 3)]
        
        num_added = self.tokenizer.add_tokens(all_motion_str + all_hand_str + all_rhand_str)
        print(f"Added {num_added} motion tokens to tokenizer")
        
        # Get motion token IDs
        self.all_motion_ids: Set[int] = set([
            self.tokenizer.convert_tokens_to_ids(t) for t in all_motion_str
        ])
        self.all_hand_ids: Set[int] = set([
            self.tokenizer.convert_tokens_to_ids(t) for t in all_hand_str
        ])
        self.all_rhand_ids: Set[int] = set([
            self.tokenizer.convert_tokens_to_ids(t) for t in all_rhand_str
        ])
        
        print(f"=== Token ID Ranges ===")
        print(f"Motion tokens: {min(self.all_motion_ids)} - {max(self.all_motion_ids)}")
        print(f"Hand tokens: {min(self.all_hand_ids)} - {max(self.all_hand_ids)}")
        print(f"RHand tokens: {min(self.all_rhand_ids)} - {max(self.all_rhand_ids)}")
        print(f"Vocab size: {len(self.tokenizer)}")
        
        # Special token IDs
        special_ids = {0, 1, 2, 3}
        all_vocab = set(range(len(self.tokenizer)))
        
        # IDs to REMOVE for each part
        body_allowed = self.all_motion_ids | special_ids
        lhand_allowed = self.all_hand_ids | special_ids
        rhand_allowed = self.all_rhand_ids | special_ids
        
        ids_remove_motion = [[x] for x in sorted(all_vocab - body_allowed)]
        ids_remove_hand = [[x] for x in sorted(all_vocab - lhand_allowed)]
        ids_remove_rhand = [[x] for x in sorted(all_vocab - rhand_allowed)]
        
        print(f"=== Mask Info ===")
        print(f"Body: {len(body_allowed)} allowed, {len(ids_remove_motion)} masked")
        print(f"LHand: {len(lhand_allowed)} allowed, {len(ids_remove_hand)} masked")
        print(f"RHand: {len(rhand_allowed)} allowed, {len(ids_remove_rhand)} masked")
        
        self.eos_idx = self.tokenizer.convert_tokens_to_ids('</s>')
        print(f"EOS token ID: {self.eos_idx}")
        
        # Initialize language model
        self.language_model = LMMultiHead(
            model_type=model_type,
            model_path=model_path,
            len_token=len(self.tokenizer),
            ids_remove_motion=ids_remove_motion,
            ids_remove_hand=ids_remove_hand,
            ids_remove_rhand=ids_remove_rhand,
            num_heads=num_heads,
            eos_idx=self.eos_idx,
            label_smoothing=label_smoothing,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
        )
        
        print(f"=== MBartT2M Initialized ===")
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def _mask_invalid_labels(self, labels: torch.Tensor, valid_token_ids: Set[int]) -> torch.Tensor:
        """Mask labels that are not in valid_token_ids."""
        valid_mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in valid_token_ids:
            valid_mask |= (labels == token_id)
        valid_mask |= (labels == 2)  # EOS
        labels = labels.clone()
        labels[~valid_mask] = -100
        return labels
    
    def motion_token_to_string(self, motion_tokens: torch.Tensor, lengths: List[int], part: str = 'body') -> List[str]:
        """Convert motion codes to string format."""
        prefix = {'body': '<motion_id_', 'lhand': '<hand_id_', 'rhand': '<rhand_id_'}.get(part, '<motion_id_')
        
        motion_strings = []
        for i, tokens in enumerate(motion_tokens):
            length = lengths[i] if i < len(lengths) else len(tokens)
            token_strs = [f'{prefix}{int(t)}>' for t in tokens[:length]]
            motion_strings.append(' '.join(token_strs))
        
        return motion_strings
    
    def motion_string_to_token(self, motion_strings: List[str], part: str = 'body') -> tuple:
        """Convert motion string back to codes."""
        prefix = {'body': 'motion_id_', 'lhand': 'hand_id_', 'rhand': 'rhand_id_'}.get(part, 'motion_id_')
        
        all_tokens = []
        for motion_string in motion_strings:
            tokens = []
            for word in motion_string.split():
                if prefix in word:
                    try:
                        num = int(word.replace('<', '').replace('>', '').replace(prefix, ''))
                        tokens.append(num)
                    except:
                        pass
            all_tokens.append(torch.tensor(tokens, dtype=torch.long))
        
        return all_tokens
    
    def forward(
        self,
        texts: List[str],
        motion_tokens: torch.Tensor,
        hand_tokens: torch.Tensor = None,
        rhand_tokens: torch.Tensor = None,
        lengths: List[int] = None,
        **kwargs
    ) -> dict:
        """Training forward pass."""
        device = self.device
        
        # Encode text
        source_encoding = self.tokenizer(
            texts,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        source_input_ids = source_encoding.input_ids.to(device)
        source_attention_mask = source_encoding.attention_mask.to(device)
        
        # Convert motion codes to labels
        body_strings = self.motion_token_to_string(motion_tokens, lengths, 'body')
        body_encoding = self.tokenizer(
            body_strings,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels = body_encoding.input_ids.to(device)
        labels = self._mask_invalid_labels(labels, self.all_motion_ids)
        
        # Hand labels
        labels_hand = None
        labels_rhand = None
        
        if hand_tokens is not None and self.num_heads > 1:
            hand_strings = self.motion_token_to_string(hand_tokens, lengths, 'lhand')
            hand_encoding = self.tokenizer(
                hand_strings,
                padding='longest',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            labels_hand = hand_encoding.input_ids.to(device)
            labels_hand = self._mask_invalid_labels(labels_hand, self.all_hand_ids)
        
        if rhand_tokens is not None and self.num_heads > 2:
            rhand_strings = self.motion_token_to_string(rhand_tokens, lengths, 'rhand')
            rhand_encoding = self.tokenizer(
                rhand_strings,
                padding='longest',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            labels_rhand = rhand_encoding.input_ids.to(device)
            labels_rhand = self._mask_invalid_labels(labels_rhand, self.all_rhand_ids)
        
        # Forward
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            decoder_input_ids=None,
            decoder_input_ids_hand=None,
            decoder_input_ids_rhand=None,
            labels=labels,
            labels_hand=labels_hand,
            labels_rhand=labels_rhand,
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        texts: List[str],
        max_length: int = 100,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs
    ) -> dict:
        """Generate motion codes from text."""
        device = self.device
        
        source_encoding = self.tokenizer(
            texts,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        source_input_ids = source_encoding.input_ids.to(device)
        source_attention_mask = source_encoding.attention_mask.to(device)
        
        decoder_start_token_id = self.tokenizer.eos_token_id
        
        outputs = self.language_model.generate(
            inputs=source_input_ids,
            attention_mask=source_attention_mask,
            decoder_start_token_id=decoder_start_token_id,
            decoder_start_token_id_hand=decoder_start_token_id if self.num_heads > 1 else None,
            decoder_start_token_id_rhand=decoder_start_token_id if self.num_heads > 2 else None,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        
        body_strings = self.tokenizer.batch_decode(outputs['outputs_re'], skip_special_tokens=True)
        body_tokens = self.motion_string_to_token(body_strings, 'body')
        
        hand_tokens = None
        rhand_tokens = None
        
        if outputs['outputs_hand'] is not None:
            hand_strings = self.tokenizer.batch_decode(outputs['outputs_hand'], skip_special_tokens=True)
            hand_tokens = self.motion_string_to_token(hand_strings, 'lhand')
        
        if outputs['outputs_rhand'] is not None:
            rhand_strings = self.tokenizer.batch_decode(outputs['outputs_rhand'], skip_special_tokens=True)
            rhand_tokens = self.motion_string_to_token(rhand_strings, 'rhand')
        
        return {
            'body_tokens': body_tokens,
            'hand_tokens': hand_tokens,
            'rhand_tokens': rhand_tokens,
        }