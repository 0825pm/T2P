"""
Multi-head Language Model for Text-to-Motion (SOKE style)
+ Label Smoothing + Dropout + Freeze Encoder
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional
from transformers import MBartForConditionalGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right


class LMMultiHead(nn.Module):
    """SOKE-style multi-head language model with regularization."""
    
    def __init__(
        self,
        model_type: str,
        model_path: str,
        len_token: int,
        ids_remove_motion: List[List[int]] = None,
        ids_remove_hand: List[List[int]] = None,
        ids_remove_rhand: List[List[int]] = None,
        num_heads: int = 3,
        eos_idx: int = 0,
        alpha_hand: float = 0.4,
        label_smoothing: float = 0.1,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.model_type = model_type
        self.eos_idx = eos_idx
        self.alpha_hand = alpha_hand
        self.len_token = len_token
        self.label_smoothing = label_smoothing
        
        # Load mBART
        self.main_lm = MBartForConditionalGeneration.from_pretrained(model_path)
        self.main_lm.resize_token_embeddings(len_token)
        
        # ========== Freeze Encoder ==========
        if freeze_encoder:
            print("Freezing mBART encoder and shared embeddings...")
            for param in self.main_lm.get_encoder().parameters():
                param.requires_grad = False
            for param in self.main_lm.model.shared.parameters():
                param.requires_grad = False
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store bad word IDs
        self.ids_remove_motion = ids_remove_motion
        self.ids_remove_hand = ids_remove_hand
        self.ids_remove_rhand = ids_remove_rhand
        
        # Create masks
        self.register_buffer('mask_body', torch.zeros(len_token))
        self.register_buffer('mask_lhand', torch.zeros(len_token))
        self.register_buffer('mask_rhand', torch.zeros(len_token))
        
        if ids_remove_motion:
            for ids in ids_remove_motion:
                for idx in ids:
                    if idx < len_token:
                        self.mask_body[idx] = float('-inf')
        
        if ids_remove_hand:
            for ids in ids_remove_hand:
                for idx in ids:
                    if idx < len_token:
                        self.mask_lhand[idx] = float('-inf')
        
        if ids_remove_rhand:
            for ids in ids_remove_rhand:
                for idx in ids:
                    if idx < len_token:
                        self.mask_rhand[idx] = float('-inf')
        
        body_allowed = (self.mask_body == 0).sum().item()
        lhand_allowed = (self.mask_lhand == 0).sum().item()
        rhand_allowed = (self.mask_rhand == 0).sum().item()
        
        print(f"\n=== LMMultiHead Config ===")
        print(f"Label smoothing: {label_smoothing}")
        print(f"Dropout: {dropout}")
        print(f"Freeze encoder: {freeze_encoder}")
        print(f"Body allowed: {body_allowed}, LHand: {lhand_allowed}, RHand: {rhand_allowed}")
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_input_ids_hand: Optional[torch.LongTensor] = None,
        decoder_input_ids_rhand: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs=None,
        labels: Optional[torch.LongTensor] = None,
        labels_hand: Optional[torch.LongTensor] = None,
        labels_rhand: Optional[torch.LongTensor] = None,
        debug: bool = False,
        **kwargs
    ) -> dict:
        """Forward pass with regularization."""
        
        use_cache = labels is None
        
        encoder = self.main_lm.get_encoder()
        decoder = self.main_lm.get_decoder()
        
        # shift_tokens_right
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(labels, pad_token_id=1)
            if labels_hand is not None:
                decoder_input_ids_hand = shift_tokens_right(labels_hand, pad_token_id=1)
            if labels_rhand is not None:
                decoder_input_ids_rhand = shift_tokens_right(labels_rhand, pad_token_id=1)
        
        # Encode
        if encoder_outputs is None:
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        
        hidden_states = encoder_outputs[0]
        token_embeds = decoder.get_input_embeddings()
        
        # Embedding fusion
        if self.model_type == 'mbart_multi' and self.num_heads > 1:
            decoder_embeds_body = token_embeds(decoder_input_ids)
            decoder_embeds_lhand = token_embeds(decoder_input_ids_hand) if decoder_input_ids_hand is not None else decoder_embeds_body
            decoder_embeds_rhand = token_embeds(decoder_input_ids_rhand) if decoder_input_ids_rhand is not None else decoder_embeds_body
            
            decoder_inputs_embeds = (
                (1 - 2 * self.alpha_hand) * decoder_embeds_body +
                self.alpha_hand * decoder_embeds_lhand +
                self.alpha_hand * decoder_embeds_rhand
            )
        else:
            decoder_inputs_embeds = token_embeds(decoder_input_ids)
        
        # Dropout on embeddings
        decoder_inputs_embeds = self.dropout(decoder_inputs_embeds)
        
        # Decode
        decoder_outputs = decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )
        
        sequence_output = decoder_outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # LM head
        lm_logits = self.main_lm.lm_head(sequence_output) + self.main_lm.final_logits_bias
        
        # Apply masks
        lm_logits_body = lm_logits_lhand = lm_logits_rhand = None
        if self.model_type == 'mbart_multi':
            lm_logits_body = lm_logits + self.mask_body
            lm_logits_lhand = lm_logits + self.mask_lhand
            lm_logits_rhand = lm_logits + self.mask_rhand
        else:
            lm_logits_body = lm_logits
        
        # Loss with Label Smoothing
        loss = loss_hand = loss_rhand = None
        loss_fct = CrossEntropyLoss(ignore_index=-100, label_smoothing=self.label_smoothing)
        
        if labels is not None:
            loss = loss_fct(lm_logits_body.view(-1, lm_logits_body.size(-1)), labels.view(-1))
        
        if labels_hand is not None and lm_logits_lhand is not None:
            loss_hand = loss_fct(lm_logits_lhand.view(-1, lm_logits_lhand.size(-1)), labels_hand.view(-1))
        
        if labels_rhand is not None and lm_logits_rhand is not None:
            loss_rhand = loss_fct(lm_logits_rhand.view(-1, lm_logits_rhand.size(-1)), labels_rhand.view(-1))
        
        return {
            'loss': loss,
            'loss_hand': loss_hand,
            'loss_rhand': loss_rhand,
            'logits': lm_logits_body,
            'logits_hand': lm_logits_lhand,
            'logits_rhand': lm_logits_rhand,
            'labels': labels,
            'labels_hand': labels_hand,
            'labels_rhand': labels_rhand,
        }
    
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_start_token_id: int,
        decoder_start_token_id_hand: int = None,
        decoder_start_token_id_rhand: int = None,
        max_length: int = 100,
        num_beams: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> dict:
        """Generate."""
        encoder = self.main_lm.get_encoder()
        
        encoder_outputs = encoder(
            input_ids=inputs,
            attention_mask=attention_mask,
        )
        
        outputs_body = self.main_lm.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            bad_words_ids=self.ids_remove_motion,
            decoder_start_token_id=decoder_start_token_id,
        )
        
        outputs_hand = outputs_rhand = None
        
        if self.num_heads > 1 and decoder_start_token_id_hand is not None:
            outputs_hand = self.main_lm.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=self.ids_remove_hand,
                decoder_start_token_id=decoder_start_token_id_hand,
            )
        
        if self.num_heads > 2 and decoder_start_token_id_rhand is not None:
            outputs_rhand = self.main_lm.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=self.ids_remove_rhand,
                decoder_start_token_id=decoder_start_token_id_rhand,
            )
        
        return {
            'outputs_re': outputs_body,
            'outputs_hand': outputs_hand,
            'outputs_rhand': outputs_rhand,
        }