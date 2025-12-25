"""
Gemma-based Text-to-Motion Model (Interleaved Format)

Architecture:
- Base: Gemma 2B (decoder-only causal LM)
- Motion tokens: <BODY_x> <LHAND_y> <RHAND_z> interleaved per timestep
- Training: Next token prediction with loss only on motion tokens

Input format:
    <bos> {text} <MOTION_START> <BODY_0> <LHAND_5> <RHAND_3> ... <MOTION_END> <eos>

Usage:
    model = GemmaT2M(config)
    outputs = model(input_ids, attention_mask, labels)
    generated = model.generate(texts, max_motion_length=100)
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, Dict, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from dataclasses import dataclass
import warnings


@dataclass
class GemmaT2MOutput:
    """Output class for GemmaT2M model."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    
    # Motion-specific outputs
    motion_loss: Optional[torch.Tensor] = None
    text_loss: Optional[torch.Tensor] = None


class GemmaT2M(nn.Module):
    """
    Gemma-based Text-to-Motion model with interleaved motion tokens.
    
    Motion token format per timestep:
        <BODY_x> <LHAND_y> <RHAND_z>
    
    Full sequence:
        {text} <MOTION_START> <B_0><L_0><R_0> <B_1><L_1><R_1> ... <MOTION_END>
    """
    
    def __init__(
        self,
        base_model: str = "google/gemma-2b",
        body_code_num: int = 96,
        lhand_code_num: int = 192,
        rhand_code_num: int = 192,
        max_text_length: int = 128,
        max_motion_length: int = 150,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        motion_only_loss: bool = True,
        label_smoothing: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.body_code_num = body_code_num
        self.lhand_code_num = lhand_code_num
        self.rhand_code_num = rhand_code_num
        self.max_text_length = max_text_length
        self.max_motion_length = max_motion_length
        self.motion_only_loss = motion_only_loss
        self.label_smoothing = label_smoothing
        self.use_lora = use_lora
        
        # Total motion tokens = body + lhand + rhand + special tokens
        self.total_motion_tokens = body_code_num + lhand_code_num + rhand_code_num + 2  # +2 for start/end
        
        # ========== Load Tokenizer ==========
        print(f"Loading tokenizer from {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            padding_side='left',  # For generation
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # ========== Add Motion Tokens ==========
        self._add_motion_tokens()
        
        # ========== Load Model ==========
        print(f"Loading model from {base_model}...")
        
        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Determine device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            use_device_map = device_map  # "auto" by default
        else:
            self._device = torch.device("cpu")
            use_device_map = None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map=use_device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
        )
        
        # Explicitly move to device if device_map is not used
        if use_device_map is None:
            self.model = self.model.to(self._device)
        
        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # ========== Apply LoRA ==========
        if use_lora:
            self._apply_lora(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
        
        # ========== Store special token IDs ==========
        self.motion_start_id = self.tokenizer.convert_tokens_to_ids("<MOTION_START>")
        self.motion_end_id = self.tokenizer.convert_tokens_to_ids("<MOTION_END>")
        
        # Build motion token ID mappings
        self._build_motion_token_maps()
        
        print(f"Model initialized with {self._count_parameters()} parameters")
        print(f"  - Trainable: {self._count_parameters(trainable_only=True)}")
    
    def _add_motion_tokens(self):
        """Add motion-specific tokens to tokenizer."""
        new_tokens = []
        
        # Special tokens
        new_tokens.extend(["<MOTION_START>", "<MOTION_END>"])
        
        # Body tokens: <BODY_0> to <BODY_95>
        self.body_tokens = [f"<BODY_{i}>" for i in range(self.body_code_num)]
        new_tokens.extend(self.body_tokens)
        
        # Left hand tokens: <LHAND_0> to <LHAND_191>
        self.lhand_tokens = [f"<LHAND_{i}>" for i in range(self.lhand_code_num)]
        new_tokens.extend(self.lhand_tokens)
        
        # Right hand tokens: <RHAND_0> to <RHAND_191>
        self.rhand_tokens = [f"<RHAND_{i}>" for i in range(self.rhand_code_num)]
        new_tokens.extend(self.rhand_tokens)
        
        # Add to tokenizer
        num_added = self.tokenizer.add_tokens(new_tokens, special_tokens=False)
        print(f"Added {num_added} motion tokens to tokenizer")
        print(f"  - Body: {self.body_code_num}, LHand: {self.lhand_code_num}, RHand: {self.rhand_code_num}")
    
    def _build_motion_token_maps(self):
        """Build mappings between code indices and token IDs."""
        # Code index -> Token ID
        self.body_code_to_id = {
            i: self.tokenizer.convert_tokens_to_ids(f"<BODY_{i}>")
            for i in range(self.body_code_num)
        }
        self.lhand_code_to_id = {
            i: self.tokenizer.convert_tokens_to_ids(f"<LHAND_{i}>")
            for i in range(self.lhand_code_num)
        }
        self.rhand_code_to_id = {
            i: self.tokenizer.convert_tokens_to_ids(f"<RHAND_{i}>")
            for i in range(self.rhand_code_num)
        }
        
        # Token ID -> Code index
        self.id_to_body_code = {v: k for k, v in self.body_code_to_id.items()}
        self.id_to_lhand_code = {v: k for k, v in self.lhand_code_to_id.items()}
        self.id_to_rhand_code = {v: k for k, v in self.rhand_code_to_id.items()}
        
        # All motion token IDs (for loss masking)
        self.all_motion_ids = set(
            list(self.body_code_to_id.values()) +
            list(self.lhand_code_to_id.values()) +
            list(self.rhand_code_to_id.values()) +
            [self.motion_start_id, self.motion_end_id]
        )
    
    def _apply_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
    ):
        """Apply LoRA adapters to the model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("Please install peft: pip install peft")
        
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print(f"LoRA applied with r={lora_r}, alpha={lora_alpha}")
        self.model.print_trainable_parameters()
    
    def _count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    @property
    def device(self):
        if hasattr(self, '_device'):
            return self._device
        return next(self.parameters()).device
    
    def codes_to_tokens(
        self,
        body_codes: torch.Tensor,
        lhand_codes: torch.Tensor,
        rhand_codes: torch.Tensor,
    ) -> List[str]:
        """
        Convert motion codes to interleaved token strings.
        
        Args:
            body_codes: (T,) tensor of body code indices
            lhand_codes: (T,) tensor of left hand code indices
            rhand_codes: (T,) tensor of right hand code indices
        
        Returns:
            String of interleaved motion tokens
        """
        tokens = ["<MOTION_START>"]
        
        T = len(body_codes)
        for t in range(T):
            tokens.append(f"<BODY_{int(body_codes[t])}>")
            tokens.append(f"<LHAND_{int(lhand_codes[t])}>")
            tokens.append(f"<RHAND_{int(rhand_codes[t])}>")
        
        tokens.append("<MOTION_END>")
        return " ".join(tokens)
    
    def tokens_to_codes(
        self,
        token_ids: torch.Tensor,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Convert generated token IDs back to motion codes.
        
        Args:
            token_ids: (N,) tensor of generated token IDs
        
        Returns:
            Tuple of (body_codes, lhand_codes, rhand_codes) as lists
        """
        body_codes = []
        lhand_codes = []
        rhand_codes = []
        
        in_motion = False
        
        for tid in token_ids.tolist():
            if tid == self.motion_start_id:
                in_motion = True
                continue
            elif tid == self.motion_end_id:
                break
            
            if in_motion:
                if tid in self.id_to_body_code:
                    body_codes.append(self.id_to_body_code[tid])
                elif tid in self.id_to_lhand_code:
                    lhand_codes.append(self.id_to_lhand_code[tid])
                elif tid in self.id_to_rhand_code:
                    rhand_codes.append(self.id_to_rhand_code[tid])
        
        return body_codes, lhand_codes, rhand_codes
    
    def prepare_inputs(
        self,
        texts: List[str],
        body_codes: Optional[torch.Tensor] = None,
        lhand_codes: Optional[torch.Tensor] = None,
        rhand_codes: Optional[torch.Tensor] = None,
        lengths: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare model inputs from texts and motion codes.
        
        Args:
            texts: List of input text strings
            body_codes: (B, T) tensor of body codes
            lhand_codes: (B, T) tensor of left hand codes
            rhand_codes: (B, T) tensor of right hand codes
            lengths: List of actual motion lengths per sample
        
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        batch_size = len(texts)
        device = self.device
        
        full_texts = []
        
        for i in range(batch_size):
            text = texts[i]
            
            if body_codes is not None:
                # Training: include motion tokens
                L = lengths[i] if lengths else body_codes.shape[1]
                motion_str = self.codes_to_tokens(
                    body_codes[i, :L],
                    lhand_codes[i, :L],
                    rhand_codes[i, :L],
                )
                full_text = f"{text} {motion_str}"
            else:
                # Generation: only text + motion start
                full_text = f"{text} <MOTION_START>"
            
            full_texts.append(full_text)
        
        # Tokenize
        encoding = self.tokenizer(
            full_texts,
            padding="longest",
            truncation=True,
            max_length=self.max_text_length + self.max_motion_length * 3 + 10,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)
        
        # For training, create labels
        labels = None
        if body_codes is not None:
            labels = input_ids.clone()
            
            if self.motion_only_loss:
                # Mask loss for text tokens (only compute loss on motion tokens)
                for i in range(batch_size):
                    # Find motion start position
                    motion_start_pos = (input_ids[i] == self.motion_start_id).nonzero()
                    if len(motion_start_pos) > 0:
                        start_pos = motion_start_pos[0].item()
                        # Mask everything before motion tokens (including <MOTION_START>)
                        labels[i, :start_pos + 1] = -100
            
            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        texts: List[str] = None,
        body_codes: torch.Tensor = None,
        lhand_codes: torch.Tensor = None,
        rhand_codes: torch.Tensor = None,
        lengths: List[int] = None,
        **kwargs
    ) -> GemmaT2MOutput:
        """
        Forward pass for training.
        
        Can be called with either:
        1. Pre-tokenized inputs: input_ids, attention_mask, labels
        2. Raw inputs: texts, body_codes, lhand_codes, rhand_codes, lengths
        """
        # Prepare inputs if raw data provided
        if input_ids is None and texts is not None:
            inputs = self.prepare_inputs(
                texts=texts,
                body_codes=body_codes,
                lhand_codes=lhand_codes,
                rhand_codes=rhand_codes,
                lengths=lengths,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = inputs["labels"]
        
        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        
        return GemmaT2MOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
    
    @torch.no_grad()
    def generate(
        self,
        texts: List[str],
        max_motion_length: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> Dict[str, List]:
        """
        Generate motion codes from text.
        
        Args:
            texts: List of input text strings
            max_motion_length: Maximum motion length in timesteps
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            do_sample: Whether to sample (vs greedy)
            num_beams: Number of beams for beam search
        
        Returns:
            Dictionary with body_codes, lhand_codes, rhand_codes (lists of tensors)
        """
        self.eval()
        device = self.device
        
        # Prepare inputs (text only, no motion codes)
        inputs = self.prepare_inputs(texts=texts)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Max new tokens = motion_length * 3 (interleaved) + 1 (end token)
        max_new_tokens = max_motion_length * 3 + 1
        
        # Generate
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.motion_end_id,
            **kwargs
        )
        
        # Extract only the generated part (not the input)
        batch_size = len(texts)
        results = {
            "body_codes": [],
            "lhand_codes": [],
            "rhand_codes": [],
            "generated_ids": [],
        }
        
        for i in range(batch_size):
            # Get generated tokens (after input)
            input_len = input_ids.shape[1]
            gen_ids = generated[i, input_len:]
            
            # Convert to codes
            body, lhand, rhand = self.tokens_to_codes(gen_ids)
            
            # Ensure equal lengths (take minimum)
            min_len = min(len(body), len(lhand), len(rhand))
            body = body[:min_len]
            lhand = lhand[:min_len]
            rhand = rhand[:min_len]
            
            results["body_codes"].append(torch.tensor(body, dtype=torch.long))
            results["lhand_codes"].append(torch.tensor(lhand, dtype=torch.long))
            results["rhand_codes"].append(torch.tensor(rhand, dtype=torch.long))
            results["generated_ids"].append(gen_ids)
        
        return results
    
    def save_pretrained(self, save_path: str):
        """Save model and tokenizer."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model (LoRA adapters if using LoRA)
        if self.use_lora:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config = {
            "body_code_num": self.body_code_num,
            "lhand_code_num": self.lhand_code_num,
            "rhand_code_num": self.rhand_code_num,
            "max_text_length": self.max_text_length,
            "max_motion_length": self.max_motion_length,
            "motion_only_loss": self.motion_only_loss,
            "use_lora": self.use_lora,
        }
        
        import json
        with open(os.path.join(save_path, "t2m_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, base_model: str = None, **kwargs):
        """Load model from checkpoint."""
        import os
        import json
        
        # Load config
        config_path = os.path.join(load_path, "t2m_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            kwargs.update(config)
        
        # Determine base model
        if base_model is None:
            base_model = kwargs.get("base_model", "google/gemma-2b")
        
        # Create model
        model = cls(base_model=base_model, **kwargs)
        
        # Load weights
        if model.use_lora:
            from peft import PeftModel
            model.model = PeftModel.from_pretrained(
                model.model.base_model,
                load_path,
            )
        else:
            model.model.load_state_dict(
                torch.load(os.path.join(load_path, "pytorch_model.bin")),
                strict=False,
            )
        
        print(f"Model loaded from {load_path}")
        return model


# ========== Utility Functions ==========

def create_model_from_config(config: dict) -> GemmaT2M:
    """Create GemmaT2M model from config dictionary."""
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    
    return GemmaT2M(
        base_model=model_config.get("base_model", "google/gemma-2b"),
        body_code_num=model_config.get("body_code_num", 96),
        lhand_code_num=model_config.get("lhand_code_num", 192),
        rhand_code_num=model_config.get("rhand_code_num", 192),
        max_text_length=model_config.get("max_text_length", 128),
        max_motion_length=model_config.get("max_motion_length", 150),
        use_lora=model_config.get("use_lora", True),
        lora_r=model_config.get("lora_r", 16),
        lora_alpha=model_config.get("lora_alpha", 32),
        lora_dropout=model_config.get("lora_dropout", 0.05),
        lora_target_modules=model_config.get("lora_target_modules", None),
        motion_only_loss=loss_config.get("motion_only_loss", True),
        label_smoothing=loss_config.get("label_smoothing", 0.0),
    )


if __name__ == "__main__":
    # Quick test
    print("Testing GemmaT2M model...")
    
    # Create model (will download Gemma if not cached)
    model = GemmaT2M(
        base_model="google/gemma-2b",
        body_code_num=96,
        lhand_code_num=192,
        rhand_code_num=192,
        use_lora=True,
    )
    
    # Test forward
    texts = ["Hello, how are you?", "This is a test."]
    body_codes = torch.randint(0, 96, (2, 50))
    lhand_codes = torch.randint(0, 192, (2, 50))
    rhand_codes = torch.randint(0, 192, (2, 50))
    lengths = [50, 50]
    
    outputs = model(
        texts=texts,
        body_codes=body_codes,
        lhand_codes=lhand_codes,
        rhand_codes=rhand_codes,
        lengths=lengths,
    )
    
    print(f"Loss: {outputs.loss.item():.4f}")
    
    # Test generation
    gen_outputs = model.generate(texts[:1], max_motion_length=20)
    print(f"Generated body codes length: {len(gen_outputs['body_codes'][0])}")
    
    print("Test passed!")