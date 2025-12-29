# coding: utf-8
"""
Mbart_Based_MLM - mBART-based Motion Language Model
Adapted from SOKE: Signs as Tokens (https://github.com/2000ZRL/SOKE)

Main wrapper class for Text-to-Motion generation using mBART.
Handles:
- Tokenizer extension with motion tokens
- Token ID to embedding ID mapping
- Forward pass through multi-head LM
- Generation of motion tokens
- LoRA fine-tuning support
"""

import os
import math
import pickle
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from torch import Tensor

from transformers import MBartTokenizer

from .lm_multihead import LMMultiHead
from .tools.token_emb import NewTokenEmb


def get_tokens_as_list(tokenizer, word_list: List[str]) -> List[int]:
    """Convert list of token strings to token IDs."""
    tokens_list = []
    for word in word_list:
        tokenized = tokenizer.tokenize(word)
        if len(tokenized) == 1:
            tokens_list.append(tokenizer.convert_tokens_to_ids(tokenized[0]))
    return tokens_list


def correct_lang_token(tokenizer, input_ids, token_len, data_src, part=None, target=False, model_type='mbart_multi'):
    """
    Correct language tokens based on data source.
    From SOKE: ensures proper language tokens for each dataset.
    """
    src2id = {
        'how2sign': {'body': 'en_ASL', 'lhand': 'en_ASL', 'rhand': 'en_ASL'},
        'csl': {'body': 'zh_CSL', 'lhand': 'zh_CSL', 'rhand': 'zh_CSL'},
        'phoenix': {'body': 'de_DGS', 'lhand': 'de_DGS', 'rhand': 'de_DGS'}
    }
    
    if part is None:
        part = 'body'
    
    for i, src in enumerate(data_src):
        if src in src2id:
            lang_token = src2id[src][part]
            lang_id = tokenizer.convert_tokens_to_ids(lang_token)
            
            if target:
                # For target: replace first token
                input_ids[i, 0] = lang_id
            else:
                # For source: replace last token before padding
                pos = min(token_len[i].item() - 1, input_ids.shape[1] - 1)
                input_ids[i, pos] = lang_id


def get_decoder_start_token_ids(tokenizer, data_src: List[str], part: str, device: torch.device) -> torch.Tensor:
    """Get decoder start token IDs based on data source and body part."""
    src2id = {
        'how2sign': {'body': 'en_ASL', 'lhand': 'en_ASL', 'rhand': 'en_ASL'},
        'csl': {'body': 'zh_CSL', 'lhand': 'zh_CSL', 'rhand': 'zh_CSL'},
        'phoenix': {'body': 'de_DGS', 'lhand': 'de_DGS', 'rhand': 'de_DGS'},
    }
    
    decoder_input_ids = []
    for s in data_src:
        lang_id = tokenizer.convert_tokens_to_ids(src2id[s][part])
        decoder_input_ids.append(lang_id)
    
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long, device=device).unsqueeze(-1)
    return decoder_input_ids


class MBartT2M(nn.Module):
    """
    mBART-based Text-to-Motion Model.
    
    Extends mBART with motion tokens and multi-head decoding.
    Uses SOKE-style training with motion token strings.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = "mbart_multi",
        stage: str = "lm_pretrain",
        motion_codebook_size: int = 96,
        hand_codebook_size: int = 192,
        rhand_codebook_size: int = 192,
        framerate: float = 20.0,
        down_t: int = 4,
        max_length: int = 256,
        num_heads: int = 3,
        label_smoothing: float = 0.0,
        freeze_encoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        
        assert 'mbart' in model_type
        
        # Parameters
        self.num_heads = num_heads
        self.m_codebook_size = motion_codebook_size
        self.hand_codebook_size = hand_codebook_size
        self.rhand_codebook_size = rhand_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.stage = stage
        self.model_type = model_type
        self.label_smoothing = label_smoothing
        self.freeze_encoder = freeze_encoder
        
        # Load tokenizer and add motion tokens
        self.tokenizer = MBartTokenizer.from_pretrained(model_path, legacy=True)
        
        # Add language tokens for sign languages
        new_lang_token = ['en_ASL', 'zh_CSL', 'de_DGS']
        self.tokenizer.add_tokens(new_lang_token, special_tokens=True)
        
        # Add motion tokens: <motion_id_0> to <motion_id_{codebook_size+2}>
        # +3 for BOS, EOS, PAD
        all_motion_str = [f'<motion_id_{i}>' for i in range(self.m_codebook_size + 3)]
        all_hand_str = [f'<hand_id_{i}>' for i in range(self.hand_codebook_size + 3)] if hand_codebook_size > 0 else []
        all_rhand_str = [f'<rhand_id_{i}>' for i in range(self.rhand_codebook_size + 3)] if rhand_codebook_size > 0 else []
        
        self.tokenizer.add_tokens(all_motion_str + all_hand_str + all_rhand_str)
        
        # Store language token IDs
        self.lang_token_ids = list(map(
            self.tokenizer.convert_tokens_to_ids, 
            ['en_XX', 'zh_CN', 'de_DE', '<mask>'] + new_lang_token
        ))
        
        # Setup token ID to embedding ID mapping
        self._setup_token_mapping(model_path, new_lang_token, all_motion_str, all_hand_str, all_rhand_str)
        
        # Setup vocabulary masks for each head
        self._setup_vocab_masks(model_path, new_lang_token, all_motion_str, all_hand_str, all_rhand_str)
        
        # Instantiate language model
        self.language_model = LMMultiHead(
            model_type=model_type,
            model_path=model_path,
            num_heads=self.num_heads,
            len_token=len(self.tok_id_to_emb_id),
            ids_remove_motion=self.ids_remove_motion,
            ids_remove_hand=self.ids_remove_hand,
            ids_remove_rhand=self.ids_remove_rhand,
            eos_idx=self.eos_idx,
        )
        
        # Replace embeddings with motion-aware embeddings
        self._setup_motion_embeddings()
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            self._freeze_encoder()
        
        # Store special token indices
        self._setup_special_tokens()
        
        # Device tracking
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def _setup_token_mapping(self, model_path, new_lang_token, all_motion_str, all_hand_str, all_rhand_str):
        """Setup token ID to embedding ID mapping (from SOKE)."""
        # Try to load existing mapping
        map_path = os.path.join(model_path, 'map_ids.pkl')
        if os.path.exists(map_path):
            with open(map_path, 'rb') as f:
                self.tok_id_to_emb_id = pickle.load(f)
            print(f"  Loaded map_ids.pkl: {len(self.tok_id_to_emb_id)} entries")
        else:
            # Create new mapping
            self.tok_id_to_emb_id = {i: i for i in range(self.tokenizer.vocab_size)}
        
        # Add new tokens to mapping
        idx = len(self.tok_id_to_emb_id)
        for tok in [*new_lang_token, *all_motion_str, *all_hand_str, *all_rhand_str]:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            if tok_id not in self.tok_id_to_emb_id:
                self.tok_id_to_emb_id[tok_id] = idx
                idx += 1
        
        print(f"  Total token mapping: {len(self.tok_id_to_emb_id)} entries")
        
        # Reverse mapping
        self.emb_id_to_tok_id = {v: k for k, v in self.tok_id_to_emb_id.items()}
        
        # EOS index
        self.eos_idx = self.tok_id_to_emb_id.get(
            self.tokenizer.convert_tokens_to_ids('</s>'), 
            2  # default EOS
        )

    def _setup_vocab_masks(self, model_path, new_lang_token, all_motion_str, all_hand_str, all_rhand_str):
        """Setup vocabulary masks for each prediction head."""
        tokenizer_with_prefix_space = MBartTokenizer.from_pretrained(
            model_path, add_prefix_space=True, legacy=True
        )
        tokenizer_with_prefix_space.add_tokens(new_lang_token, special_tokens=True)
        tokenizer_with_prefix_space.add_tokens(all_motion_str + all_hand_str + all_rhand_str)
        
        all_motion_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_motion_str)
        all_hand_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_hand_str)
        all_rhand_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_rhand_str)
        
        vocab = tokenizer_with_prefix_space.get_vocab()
        all_vocab_ids = set(vocab.values())
        special_ids = {0, 1, 2, 3}
        lang_ids = set(self.lang_token_ids)
        
        # Body: remove all non-motion tokens
        ids_to_remove = all_vocab_ids - set(all_motion_ids) - special_ids - lang_ids
        self.ids_remove_motion = [
            [self.tok_id_to_emb_id[x]] for x in ids_to_remove 
            if x in self.tok_id_to_emb_id
        ]
        
        # Left hand
        ids_to_remove = all_vocab_ids - set(all_hand_ids) - special_ids - lang_ids
        self.ids_remove_hand = [
            [self.tok_id_to_emb_id[x]] for x in ids_to_remove 
            if x in self.tok_id_to_emb_id
        ]
        
        # Right hand
        ids_to_remove = all_vocab_ids - set(all_rhand_ids) - special_ids - lang_ids
        self.ids_remove_rhand = [
            [self.tok_id_to_emb_id[x]] for x in ids_to_remove 
            if x in self.tok_id_to_emb_id
        ]

    def _setup_motion_embeddings(self):
        """Replace embeddings with motion-aware embeddings."""
        old_embeddings = self.language_model.main_lm.get_input_embeddings()
        new_num_tokens = len(self.tok_id_to_emb_id) - old_embeddings.num_embeddings
        
        if new_num_tokens > 0:
            new_token_emb = NewTokenEmb(old_embeddings, new_num_tokens)
            
            self.language_model.main_lm.model.shared = new_token_emb
            self.language_model.main_lm.model.encoder.embed_tokens = new_token_emb
            self.language_model.main_lm.model.decoder.embed_tokens = new_token_emb
            
            self.motion_embeddings = new_token_emb
            
            print(f"  Motion embeddings: {new_num_tokens} new tokens added")

    def _freeze_encoder(self):
        """Freeze encoder parameters except motion embeddings."""
        for param in self.language_model.main_lm.model.encoder.parameters():
            param.requires_grad = False
        
        if hasattr(self, 'motion_embeddings'):
            self.motion_embeddings.motion_embeddings.weight.requires_grad = True
            self.motion_embeddings.word2motionProj.weight.requires_grad = True
            self.motion_embeddings.word2motionProj.bias.requires_grad = True
        
        print("  Encoder frozen (motion embeddings unfrozen)")

    def _setup_special_tokens(self):
        """Setup special token indices for motion sequences."""
        self.body_bos_idx = self.tokenizer.convert_tokens_to_ids(f'<motion_id_{self.m_codebook_size}>')
        self.body_eos_idx = self.tokenizer.convert_tokens_to_ids(f'<motion_id_{self.m_codebook_size + 1}>')
        self.body_pad_idx = self.tokenizer.convert_tokens_to_ids(f'<motion_id_{self.m_codebook_size + 2}>')
        
        if self.hand_codebook_size > 0:
            self.hand_bos_idx = self.tokenizer.convert_tokens_to_ids(f'<hand_id_{self.hand_codebook_size}>')
            self.hand_eos_idx = self.tokenizer.convert_tokens_to_ids(f'<hand_id_{self.hand_codebook_size + 1}>')
            self.hand_pad_idx = self.tokenizer.convert_tokens_to_ids(f'<hand_id_{self.hand_codebook_size + 2}>')
        
        if self.rhand_codebook_size > 0:
            self.rhand_bos_idx = self.tokenizer.convert_tokens_to_ids(f'<rhand_id_{self.rhand_codebook_size}>')
            self.rhand_eos_idx = self.tokenizer.convert_tokens_to_ids(f'<rhand_id_{self.rhand_codebook_size + 1}>')
            self.rhand_pad_idx = self.tokenizer.convert_tokens_to_ids(f'<rhand_id_{self.rhand_codebook_size + 2}>')

    # =========================================================================
    # SOKE-style methods: Motion token <-> String conversion
    # =========================================================================
    
    def motion_token_to_string(
        self, 
        motion_tokens: Tensor, 
        lengths: List[int], 
        pattern: str = 'motion'
    ) -> List[str]:
        """
        Convert motion token tensor to string list.
        From SOKE: mGPT/archs/mgpt_mbart.py
        
        Args:
            motion_tokens: (B, T) tensor of motion code indices
            lengths: List of valid lengths for each sample
            pattern: 'motion', 'hand', or 'rhand'
            
        Returns:
            List of motion token strings
        """
        motion_string = []
        for i in range(len(motion_tokens)):
            if isinstance(motion_tokens[i], list):
                motion_list = motion_tokens[i][:lengths[i]]
            else:
                motion_i = motion_tokens[i].cpu() if motion_tokens[i].device.type == 'cuda' else motion_tokens[i]
                motion_list = motion_i.tolist()[:lengths[i]]
            
            motion_string.append(
                ''.join([f'<{pattern}_id_{int(idx)}>' for idx in motion_list])
            )
        return motion_string

    def motion_string_to_token(
        self, 
        motion_string: List[str], 
        pattern: str = 'motion'
    ) -> Tuple[List[Tensor], List[str]]:
        """
        Convert motion string back to token tensor.
        From SOKE.
        """
        motion_tokens = []
        output_string = []
        
        for i in range(len(motion_string)):
            string = ''.join(motion_string[i].split(' '))
            string_list = string.split('><')
            
            try:
                token_list = [
                    int(s.split('_')[-1].replace('>', ''))
                    for s in string_list
                ]
            except:
                token_list = [0]
            
            if len(token_list) == 0:
                token_list = [0]
            
            token_list_padded = torch.tensor(token_list, dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(string, '<Motion_Placeholder>'))
        
        return motion_tokens, output_string

    def template_fulfill(
        self,
        tasks: List[dict],
        lengths: List[int],
        motion_strings: List[str],
        texts: List[str],
        pattern: str = 'motion'
    ) -> Tuple[List[str], List[str]]:
        """
        Fill template with actual content.
        From SOKE: mGPT/archs/mgpt_mbart.py
        
        Args:
            tasks: List of task dicts with 'input' and 'output' templates
            lengths: Motion lengths
            motion_strings: Motion token strings
            texts: Text captions
            pattern: 'motion', 'hand', or 'rhand'
            
        Returns:
            Tuple of (input_strings, output_strings)
        """
        inputs = []
        outputs = []
        
        for i in range(len(texts)):
            task = tasks[i] if tasks is not None else {
                'input': ['<Caption_Placeholder>'],
                'output': ['<Motion_Placeholder>']
            }
            
            # Get template
            input_template = random.choice(task['input']) if isinstance(task['input'], list) else task['input']
            output_template = random.choice(task['output']) if isinstance(task['output'], list) else task['output']
            
            # Calculate seconds
            seconds = math.floor(lengths[i] / self.framerate) if lengths[i] > 0 else 0
            
            # Fill placeholders
            input_str = input_template.replace('<Caption_Placeholder>', texts[i])
            input_str = input_str.replace('<Frame_Placeholder>', str(lengths[i]))
            input_str = input_str.replace('<Second_Placeholder>', str(seconds))
            
            output_str = output_template.replace(f'<Motion_Placeholder>', motion_strings[i])
            output_str = output_str.replace('<Caption_Placeholder>', texts[i])
            
            inputs.append(input_str)
            outputs.append(output_str)
        
        return inputs, outputs

    def map_ids(self, input_ids: Tensor, direction: str = 'token_to_emb'):
        """
        Map between token IDs and embedding IDs in-place.
        From SOKE: essential for correct embedding lookup.
        
        Args:
            input_ids: Tensor to map (modified in-place)
            direction: 'token_to_emb' or 'emb_to_token'
        """
        if direction == 'token_to_emb':
            mapping = self.tok_id_to_emb_id
        else:
            mapping = self.emb_id_to_tok_id
        
        # Vectorized mapping for efficiency
        flat = input_ids.view(-1)
        for i in range(flat.size(0)):
            val = flat[i].item()
            if val in mapping:
                flat[i] = mapping[val]

    # =========================================================================
    # Forward: SOKE-style training
    # =========================================================================
    
    def forward(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        hand_tokens: Optional[Tensor] = None,
        rhand_tokens: Optional[Tensor] = None,
        lengths: List[int] = None,
        tasks: List[dict] = None,
        data_src: List[str] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for training (SOKE-style).
        
        Args:
            texts: List of text captions
            motion_tokens: (B, T) body motion code indices
            hand_tokens: (B, T) left hand motion code indices
            rhand_tokens: (B, T) right hand motion code indices
            lengths: List of valid motion lengths
            tasks: Task templates (optional)
            data_src: Data source for each sample ('how2sign', 'csl', 'phoenix')
            
        Returns:
            Dict with loss, loss_hand, loss_rhand, logits
        """
        device = self.device
        batch_size = len(texts)
        
        # Default data source
        if data_src is None:
            data_src = ['phoenix'] * batch_size
        
        # Default lengths
        if lengths is None:
            lengths = [motion_tokens.shape[1]] * batch_size
        
        # Default tasks
        if tasks is None:
            tasks = [{'input': ['<Caption_Placeholder>'], 'output': ['<Motion_Placeholder>']}] * batch_size
        
        # =====================================================================
        # Step 1: Convert motion tokens to strings (SOKE-style)
        # =====================================================================
        motion_strings = self.motion_token_to_string(motion_tokens, lengths, pattern='motion')
        
        hand_strings = None
        rhand_strings = None
        if hand_tokens is not None and self.hand_codebook_size > 0:
            hand_strings = self.motion_token_to_string(hand_tokens, lengths, pattern='hand')
        if rhand_tokens is not None and self.rhand_codebook_size > 0:
            rhand_strings = self.motion_token_to_string(rhand_tokens, lengths, pattern='rhand')
        
        # =====================================================================
        # Step 2: Fill templates
        # =====================================================================
        inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts, pattern='motion')
        
        outputs_hand = outputs_rhand = None
        if hand_strings is not None:
            _, outputs_hand = self.template_fulfill(tasks, lengths, hand_strings, texts, pattern='hand')
        if rhand_strings is not None:
            _, outputs_rhand = self.template_fulfill(tasks, lengths, rhand_strings, texts, pattern='rhand')
        
        # =====================================================================
        # Step 3: Tokenize inputs (text)
        # =====================================================================
        source_encoding = self.tokenizer(
            inputs,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_length=True
        )
        
        source_attention_mask = source_encoding.attention_mask.to(device)
        source_input_ids = source_encoding.input_ids.to(device)
        token_len = source_encoding.length.to(device)
        
        # Correct language tokens for source
        correct_lang_token(self.tokenizer, source_input_ids, token_len, data_src, part=None, target=False, model_type=self.model_type)
        
        # Map token IDs to embedding IDs
        self.map_ids(source_input_ids, direction='token_to_emb')
        
        # =====================================================================
        # Step 4: Tokenize outputs (motion tokens) - Body
        # =====================================================================
        target_inputs = self.tokenizer(
            outputs,
            padding='longest',
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_length=True
        )
        
        labels_input_ids = target_inputs.input_ids.to(device)
        labels_attention_mask = target_inputs.attention_mask.to(device)
        token_len = target_inputs.length.to(device)
        
        # Correct language tokens for target
        correct_lang_token(self.tokenizer, labels_input_ids, token_len, data_src, part='body', target=True, model_type=self.model_type)
        
        # Set padding to -100 (ignore in loss)
        labels_input_ids[labels_input_ids == self.tokenizer.pad_token_id] = -100
        
        # Map to embedding IDs
        self.map_ids(labels_input_ids, direction='token_to_emb')
        
        # =====================================================================
        # Step 5: Tokenize outputs - Left Hand
        # =====================================================================
        labels_input_ids_hand = None
        if outputs_hand is not None:
            target_inputs_hand = self.tokenizer(
                outputs_hand,
                padding='longest',
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
                return_length=True
            )
            
            labels_input_ids_hand = target_inputs_hand.input_ids.to(device)
            token_len = target_inputs_hand.length.to(device)
            
            correct_lang_token(self.tokenizer, labels_input_ids_hand, token_len, data_src, part='lhand', target=True, model_type=self.model_type)
            labels_input_ids_hand[labels_input_ids_hand == self.tokenizer.pad_token_id] = -100
            self.map_ids(labels_input_ids_hand, direction='token_to_emb')
        
        # =====================================================================
        # Step 6: Tokenize outputs - Right Hand
        # =====================================================================
        labels_input_ids_rhand = None
        if outputs_rhand is not None:
            target_inputs_rhand = self.tokenizer(
                outputs_rhand,
                padding='longest',
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
                return_length=True
            )
            
            labels_input_ids_rhand = target_inputs_rhand.input_ids.to(device)
            token_len = target_inputs_rhand.length.to(device)
            
            correct_lang_token(self.tokenizer, labels_input_ids_rhand, token_len, data_src, part='rhand', target=True, model_type=self.model_type)
            labels_input_ids_rhand[labels_input_ids_rhand == self.tokenizer.pad_token_id] = -100
            self.map_ids(labels_input_ids_rhand, direction='token_to_emb')
        
        # =====================================================================
        # Step 7: Forward through language model
        # =====================================================================
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            labels=labels_input_ids,
            labels_hand=labels_input_ids_hand,
            labels_rhand=labels_input_ids_rhand,
            decoder_attention_mask=labels_attention_mask,
        )
        
        # Add labels to outputs for accuracy computation
        # (원본 lm_multihead.py는 labels를 반환하지 않으므로 여기서 추가)
        if 'labels' not in outputs:
            outputs['labels'] = labels_input_ids
        if 'labels_hand' not in outputs:
            outputs['labels_hand'] = labels_input_ids_hand
        if 'labels_rhand' not in outputs:
            outputs['labels_rhand'] = labels_input_ids_rhand
        
        return outputs

    # =========================================================================
    # Generation
    # =========================================================================
    
    def generate(
        self,
        texts: List[str],
        data_src: List[str] = None,
        max_length: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        tasks: List[dict] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Generate motion tokens from text.
        """
        device = self.device
        batch_size = len(texts)
        
        if data_src is None:
            data_src = ['phoenix'] * batch_size
        
        if tasks is None:
            tasks = [{'input': ['<Caption_Placeholder>'], 'output': ['']}] * batch_size
        
        # Prepare inputs using template
        motion_strings = [''] * batch_size
        lengths = [0] * batch_size
        inputs, _ = self.template_fulfill(tasks, lengths, motion_strings, texts)
        
        # Tokenize
        source_encoding = self.tokenizer(
            inputs,
            padding='longest',
            max_length=128,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_length=True
        )
        
        source_input_ids = source_encoding.input_ids.to(device)
        source_attention_mask = source_encoding.attention_mask.to(device)
        token_len = source_encoding.length.to(device)
        
        # Correct language tokens
        correct_lang_token(self.tokenizer, source_input_ids, token_len, data_src, part=None, target=False, model_type=self.model_type)
        
        # Map to embedding IDs
        self.map_ids(source_input_ids, direction='token_to_emb')
        
        # Get decoder start tokens
        decoder_start_body = get_decoder_start_token_ids(self.tokenizer, data_src, 'body', device)
        decoder_start_lhand = get_decoder_start_token_ids(self.tokenizer, data_src, 'lhand', device)
        decoder_start_rhand = get_decoder_start_token_ids(self.tokenizer, data_src, 'rhand', device)
        
        # Map decoder start tokens
        self.map_ids(decoder_start_body, direction='token_to_emb')
        self.map_ids(decoder_start_lhand, direction='token_to_emb')
        self.map_ids(decoder_start_rhand, direction='token_to_emb')
        
        # Generate
        outputs = self.language_model.generate(
            inputs=source_input_ids,
            attention_mask=source_attention_mask,
            decoder_start_token_id=decoder_start_body,
            decoder_start_token_id_hand=decoder_start_lhand,
            decoder_start_token_id_rhand=decoder_start_rhand,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        
        # Map back to token IDs for decoding
        self.map_ids(outputs['outputs_re'], direction='emb_to_token')
        if outputs['outputs_hand'] is not None:
            self.map_ids(outputs['outputs_hand'], direction='emb_to_token')
        if outputs['outputs_rhand'] is not None:
            self.map_ids(outputs['outputs_rhand'], direction='emb_to_token')
        
        return outputs

    def decode_to_codes(
        self,
        outputs: Dict[str, Tensor]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Decode generated tokens to motion codes.
        """
        batch_size = outputs['outputs_re'].shape[0]
        
        body_codes_list = []
        lhand_codes_list = []
        rhand_codes_list = []
        
        for i in range(batch_size):
            # Decode body
            body_string = self.tokenizer.decode(outputs['outputs_re'][i], skip_special_tokens=True)
            body_tokens, _ = self.motion_string_to_token([body_string], pattern='motion')
            body_codes = body_tokens[0].cpu().numpy() if len(body_tokens) > 0 else np.array([])
            body_codes_list.append(body_codes)
            
            # Decode left hand
            if outputs['outputs_hand'] is not None:
                lhand_string = self.tokenizer.decode(outputs['outputs_hand'][i], skip_special_tokens=True)
                lhand_tokens, _ = self.motion_string_to_token([lhand_string], pattern='hand')
                lhand_codes = lhand_tokens[0].cpu().numpy() if len(lhand_tokens) > 0 else np.array([])
            else:
                lhand_codes = np.array([])
            lhand_codes_list.append(lhand_codes)
            
            # Decode right hand
            if outputs['outputs_rhand'] is not None:
                rhand_string = self.tokenizer.decode(outputs['outputs_rhand'][i], skip_special_tokens=True)
                rhand_tokens, _ = self.motion_string_to_token([rhand_string], pattern='rhand')
                rhand_codes = rhand_tokens[0].cpu().numpy() if len(rhand_tokens) > 0 else np.array([])
            else:
                rhand_codes = np.array([])
            rhand_codes_list.append(rhand_codes)
        
        return body_codes_list, lhand_codes_list, rhand_codes_list