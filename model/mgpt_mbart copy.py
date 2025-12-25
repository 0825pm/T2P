"""
mBART-based Language Model for Text-to-Motion Generation
Handles tokenization, motion token mapping, and generation

Adapted from SOKE: https://github.com/2000ZRL/SOKE
"""
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import MBartTokenizer
from model.lm_multihead import LMMultiHead


class MBartT2M(nn.Module):
    """
    mBART-based Text-to-Motion model.
    Converts text to motion tokens using multi-head decoding.
    """
    
    def __init__(
        self,
        model_path: str = "facebook/mbart-large-cc25",
        model_type: str = "mbart_multi",
        body_codebook_size: int = 96,
        hand_codebook_size: int = 192,
        rhand_codebook_size: int = 192,
        max_length: int = 256,
        num_heads: int = 3,
        framerate: float = 25.0,
        down_t: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.model_type = model_type
        self.num_heads = num_heads
        self.body_codebook_size = body_codebook_size
        self.hand_codebook_size = hand_codebook_size
        self.rhand_codebook_size = rhand_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.use_cuda = True  # For batch compatibility
        
        # Initialize tokenizer
        self.tokenizer = MBartTokenizer.from_pretrained(model_path, legacy=True)
        
        # Add language tokens for sign language
        new_lang_tokens = ['en_ASL', 'zh_CSL', 'de_DGS']
        self.tokenizer.add_tokens(new_lang_tokens, special_tokens=True)
        
        # Add motion tokens
        body_motion_tokens = [f'<motion_id_{i}>' for i in range(body_codebook_size + 3)]
        self.tokenizer.add_tokens(body_motion_tokens)
        
        hand_motion_tokens = []
        if hand_codebook_size > 0:
            hand_motion_tokens = [f'<hand_id_{i}>' for i in range(hand_codebook_size + 3)]
            self.tokenizer.add_tokens(hand_motion_tokens)
        
        rhand_motion_tokens = []
        if rhand_codebook_size > 0:
            rhand_motion_tokens = [f'<rhand_id_{i}>' for i in range(rhand_codebook_size + 3)]
            self.tokenizer.add_tokens(rhand_motion_tokens)
        
        # Get special token IDs
        self.lang_token_ids = list(map(
            self.tokenizer.convert_tokens_to_ids,
            ['en_XX', 'zh_CN', 'de_DE', '<mask>'] + new_lang_tokens
        ))
        
        # Build token ID mapping
        self._build_token_mapping()
        
        # Build bad word IDs
        ids_remove_motion = self._get_bad_word_ids('body')
        ids_remove_hand = self._get_bad_word_ids('lhand') if hand_codebook_size > 0 else None
        ids_remove_rhand = self._get_bad_word_ids('rhand') if rhand_codebook_size > 0 else None
        
        # EOS token ID
        self.eos_idx = self.tokenizer.convert_tokens_to_ids('</s>')
        
        # Initialize multi-head language model
        self.language_model = LMMultiHead(
            model_type=model_type,
            model_path=model_path,
            num_heads=num_heads,
            len_token=len(self.tok_id_to_emb_id),
            ids_remove_motion=ids_remove_motion,
            ids_remove_hand=ids_remove_hand,
            ids_remove_rhand=ids_remove_rhand,
            eos_idx=self.eos_idx,
        )
    
    def _build_token_mapping(self):
        """Build mapping between token IDs and embedding IDs."""
        vocab_size = len(self.tokenizer)
        self.tok_id_to_emb_id = {i: i for i in range(vocab_size)}
        self.emb_id_to_tok_id = {i: i for i in range(vocab_size)}
    
    def _get_bad_word_ids(self, part: str) -> List[List[int]]:
        """Get list of token IDs that shouldn't be generated for a specific part."""
        bad_ids = []
        
        if part == 'body':
            for i in range(self.hand_codebook_size + 3):
                token_id = self.tokenizer.convert_tokens_to_ids(f'<hand_id_{i}>')
                if token_id != self.tokenizer.unk_token_id:
                    bad_ids.append([token_id])
            for i in range(self.rhand_codebook_size + 3):
                token_id = self.tokenizer.convert_tokens_to_ids(f'<rhand_id_{i}>')
                if token_id != self.tokenizer.unk_token_id:
                    bad_ids.append([token_id])
        elif part == 'lhand':
            for i in range(self.body_codebook_size + 3):
                token_id = self.tokenizer.convert_tokens_to_ids(f'<motion_id_{i}>')
                if token_id != self.tokenizer.unk_token_id:
                    bad_ids.append([token_id])
            for i in range(self.rhand_codebook_size + 3):
                token_id = self.tokenizer.convert_tokens_to_ids(f'<rhand_id_{i}>')
                if token_id != self.tokenizer.unk_token_id:
                    bad_ids.append([token_id])
        elif part == 'rhand':
            for i in range(self.body_codebook_size + 3):
                token_id = self.tokenizer.convert_tokens_to_ids(f'<motion_id_{i}>')
                if token_id != self.tokenizer.unk_token_id:
                    bad_ids.append([token_id])
            for i in range(self.hand_codebook_size + 3):
                token_id = self.tokenizer.convert_tokens_to_ids(f'<hand_id_{i}>')
                if token_id != self.tokenizer.unk_token_id:
                    bad_ids.append([token_id])
        
        return bad_ids if bad_ids else None
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def motion_token_to_string(self, motion_tokens: torch.Tensor, lengths: List[int], part: str = 'body') -> List[str]:
        """Convert motion token indices to string format."""
        if part == 'body':
            prefix = '<motion_id_'
        elif part == 'lhand':
            prefix = '<hand_id_'
        elif part == 'rhand':
            prefix = '<rhand_id_'
        else:
            prefix = '<motion_id_'
        
        motion_strings = []
        for i, tokens in enumerate(motion_tokens):
            length = lengths[i] if i < len(lengths) else len(tokens)
            token_strs = [f'{prefix}{int(t)}>' for t in tokens[:length]]
            motion_strings.append(' '.join(token_strs))
        
        return motion_strings
    
    def motion_string_to_token(self, motion_strings: List[str], part: str = 'body') -> tuple:
        """Convert motion string back to token indices."""
        if part == 'body':
            prefix = 'motion_id_'
        elif part == 'lhand':
            prefix = 'hand_id_'
        elif part == 'rhand':
            prefix = 'rhand_id_'
        else:
            prefix = 'motion_id_'
        
        all_tokens = []
        cleaned_texts = []
        
        for motion_string in motion_strings:
            tokens = []
            cleaned = []
            for word in motion_string.split():
                if prefix in word:
                    try:
                        num = int(word.replace('<', '').replace('>', '').replace(prefix, ''))
                        tokens.append(num)
                        cleaned.append(word)
                    except:
                        pass
            all_tokens.append(torch.tensor(tokens, dtype=torch.long))
            cleaned_texts.append(' '.join(cleaned))
        
        return all_tokens, cleaned_texts
    
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
        
        # Tokenize input text
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
        
        # Convert motion tokens to token IDs
        body_strings = self.motion_token_to_string(motion_tokens, lengths, 'body')
        body_encoding = self.tokenizer(
            body_strings,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        decoder_input_ids = body_encoding.input_ids.to(device)
        
        # Prepare labels
        labels = decoder_input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Hand tokens
        decoder_input_ids_hand = None
        labels_hand = None
        decoder_input_ids_rhand = None
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
            decoder_input_ids_hand = hand_encoding.input_ids.to(device)
            labels_hand = decoder_input_ids_hand.clone()
            labels_hand[labels_hand == self.tokenizer.pad_token_id] = -100
        
        if rhand_tokens is not None and self.num_heads > 2:
            rhand_strings = self.motion_token_to_string(rhand_tokens, lengths, 'rhand')
            rhand_encoding = self.tokenizer(
                rhand_strings,
                padding='longest',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            decoder_input_ids_rhand = rhand_encoding.input_ids.to(device)
            labels_rhand = decoder_input_ids_rhand.clone()
            labels_rhand[labels_rhand == self.tokenizer.pad_token_id] = -100
        
        # Forward through language model
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_input_ids_hand=decoder_input_ids_hand,
            decoder_input_ids_rhand=decoder_input_ids_rhand,
            labels=labels,
            labels_hand=labels_hand,
            labels_rhand=labels_rhand,
        )
        
        # Add labels to outputs for accuracy computation
        outputs['labels'] = labels
        outputs['labels_hand'] = labels_hand
        outputs['labels_rhand'] = labels_rhand
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        texts: List[str],
        max_length: int = 100,
        num_beams: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> dict:
        """Generate motion tokens from text."""
        device = self.device
        
        # Tokenize input text
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
        
        # Create decoder start token IDs
        batch_size = len(texts)
        body_lang_id = self.tokenizer.convert_tokens_to_ids('en_ASL')
        decoder_start_body = torch.full((batch_size, 1), body_lang_id, dtype=torch.long, device=device)
        
        decoder_start_hand = decoder_start_body.clone() if self.num_heads > 1 else None
        decoder_start_rhand = decoder_start_body.clone() if self.num_heads > 2 else None
        
        # Generate
        outputs = self.language_model.generate_multihead(
            inputs=source_input_ids,
            attention_mask=source_attention_mask,
            decoder_start_token_id=decoder_start_body,
            decoder_start_token_id_hand=decoder_start_hand,
            decoder_start_token_id_rhand=decoder_start_rhand,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        
        # Convert outputs to token indices
        body_strings = self.tokenizer.batch_decode(outputs['outputs_body'], skip_special_tokens=True)
        body_tokens, _ = self.motion_string_to_token(body_strings, 'body')
        
        hand_tokens = None
        rhand_tokens = None
        
        if outputs['outputs_hand'] is not None:
            hand_strings = self.tokenizer.batch_decode(outputs['outputs_hand'], skip_special_tokens=True)
            hand_tokens, _ = self.motion_string_to_token(hand_strings, 'lhand')
        
        if outputs['outputs_rhand'] is not None:
            rhand_strings = self.tokenizer.batch_decode(outputs['outputs_rhand'], skip_special_tokens=True)
            rhand_tokens, _ = self.motion_string_to_token(rhand_strings, 'rhand')
        
        return {
            'body_tokens': body_tokens,
            'hand_tokens': hand_tokens,
            'rhand_tokens': rhand_tokens,
        }
