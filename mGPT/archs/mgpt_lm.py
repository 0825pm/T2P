import os
from typing import List, Union
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import tokenizers
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb
from mGPT.archs.lm_multihead import LMMultiHead


def get_tokens_as_list(tokenizer, word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0][0]
        tokens_list.append(tokenized_word)
    return tokens_list


class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        hand_codebook_size: int = 0,
        rhand_codebook_size: int = 0,
        framerate: float = 20.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 256,
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        num_heads: int = 2,  #number of lm heads
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters
        self.num_heads = num_heads
        self.m_codebook_size = motion_codebook_size
        self.hand_codebook_size = hand_codebook_size
        self.rhand_codebook_size = rhand_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage

        # Add motion tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        all_motion_str = [f'<motion_id_{i}>' for i in range(self.m_codebook_size + 3)]
        self.tokenizer.add_tokens(all_motion_str)
        if model_type == 't5_multi':
            all_hand_str = [f'<hand_id_{i}>' for i in range(self.hand_codebook_size + 3)]
            all_rhand_str = [f'<rhand_id_{i}>' for i in range(self.rhand_codebook_size + 3)]
            self.tokenizer.add_tokens(all_hand_str)
            if self.num_heads > 2:
                self.tokenizer.add_tokens(all_rhand_str)

        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True, legacy=True)
        tokenizer_with_prefix_space.add_tokens(all_motion_str)
        all_motion_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_motion_str)
        ids_remove_motion = list(set(tokenizer_with_prefix_space.get_vocab().values()) - set(all_motion_ids) - set([0,1,2]))
        ids_remove_motion = [[x] for x in ids_remove_motion]
        ids_remove_rhand = None
        if self.num_heads > 2:
            tokenizer_with_prefix_space.add_tokens(all_hand_str)
            all_hand_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_hand_str)
            ids_remove_hand = list(set(tokenizer_with_prefix_space.get_vocab().values()) - set(all_hand_ids) - set([0,1,2]))
            ids_remove_hand = [[x] for x in ids_remove_hand]
            tokenizer_with_prefix_space.add_tokens(all_rhand_str)
            all_rhand_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_rhand_str)
            ids_remove_rhand = list(set(tokenizer_with_prefix_space.get_vocab().values()) - set(all_rhand_ids) - set([0,1,2]))
            ids_remove_rhand = [[x] for x in ids_remove_rhand]

        # Instantiate language model
        self.model_type = model_type
        if model_type == 't5':
            self.language_model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.lm_type = 'encdec'
        elif model_type == 't5_multi':
            self.language_model = LMMultiHead(
                                            model_type,
                                            model_path, 
                                            num_heads=self.num_heads,
                                            codebook_size_hand=self.hand_codebook_size, 
                                            codebook_size_re=self.m_codebook_size, 
                                            rhand_codebook_size=self.rhand_codebook_size,
                                            len_token=len(self.tokenizer),
                                            ids_remove_motion=ids_remove_motion,
                                            ids_remove_hand=ids_remove_hand,
                                            ids_remove_rhand=ids_remove_rhand
                                        )
            self.lm_type = 'encdec'
        elif model_type == "gpt2":
            self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.lm_type = 'dec'
        else:
            raise ValueError("type must be either seq2seq or conditional")

        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def forward(self, texts: List[str], motion_tokens: Tensor,
                lengths: List[int], tasks: dict):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_tokens, lengths, tasks)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_tokens, lengths, tasks)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens[..., 0], lengths, pattern='motion')
        if self.model_type == 't5_multi':
            hand_strings = self.motion_token_to_string(motion_tokens[..., 1], lengths, pattern='hand')
            if self.num_heads > 2:
                rhand_strings = self.motion_token_to_string(motion_tokens[..., 2], lengths, pattern='rhand')

        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        if condition == 'text':
            inputs = texts
            outputs = texts
        elif condition == 'motion':
            inputs = motion_strings
            outputs = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts, pattern='motion')
            outputs_hand = outputs_rhand = None
            if self.model_type == 't5_multi':
                _, outputs_hand = self.template_fulfill(tasks, lengths,
                                                    hand_strings, texts, pattern='hand')
                if self.num_heads > 2:
                    _, outputs_rhand = self.template_fulfill(tasks, lengths,
                                                    rhand_strings, texts, pattern='rhand')
                    
        # print(texts)
        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_attention_mask = source_encoding.attention_mask.to(
            motion_tokens.device)
        source_input_ids = source_encoding.input_ids.to(motion_tokens.device)

        if condition in ['text', 'motion']:
            batch_size, expandend_input_length = source_input_ids.shape
            mask_indices = np.asarray([
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ])
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(
                mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(
                target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids,
                                                     target_sentinel)
            source_input_ids = self.filter_input_ids(source_input_ids,
                                                     input_ids_sentinel)

        else:
            target_inputs = self.tokenizer(outputs,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")

            labels_input_ids = target_inputs.input_ids.to(motion_tokens.device)
            lables_attention_mask = target_inputs.attention_mask.to(motion_tokens.device)
            labels_input_ids[labels_input_ids == 0] = -100
            labels_input_ids_hand = labels_input_ids_rhand = None
            if self.model_type == 't5_multi':
                target_inputs_hand = self.tokenizer(outputs_hand,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")
                labels_input_ids_hand = target_inputs_hand.input_ids.to(motion_tokens.device)
                labels_input_ids_hand[labels_input_ids_hand==0] = -100
                if self.num_heads > 2:
                    target_inputs_rhand = self.tokenizer(outputs_rhand,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")
                    labels_input_ids_rhand = target_inputs_rhand.input_ids.to(motion_tokens.device)
                    labels_input_ids_rhand[labels_input_ids_rhand==0] = -100

        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
            if condition == 'supervised' else None,
            labels=labels_input_ids,
            labels_hand=labels_input_ids_hand,
            labels_rhand=labels_input_ids_rhand,
            decoder_attention_mask=lables_attention_mask
            if condition == 'supervised' else None,
        )

        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):
        self.tokenizer.padding_side = "right"

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        condition = random.choice(
            ['text', 'motion', 'supervised', 'supervised', 'supervised'])

        if condition == 'text':
            labels = texts
        elif condition == 'motion':
            labels = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            labels = []
            for i in range(len(inputs)):
                labels.append(inputs[i] + ' \n ' + outputs[i] +
                              self.tokenizer.eos_token)

        # Tokenize
        inputs = self.tokenizer(labels,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt")

        labels_input_ids = inputs.input_ids.to(motion_tokens.device)
        lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
        outputs = self.language_model(input_ids=labels_input_ids,
                                      attention_mask=lables_attention_mask,
                                      labels=inputs["input_ids"])

        return outputs

    def generate_direct(self,
                        texts: List[str],
                        max_length: int = 256,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None):

        # Device
        try:
            self.device = self.language_model.device
        except:
            self.device = self.language_model.main_lm.device

        # Tokenize
        if self.lm_type == 'dec':
            texts = [text + " \n " for text in texts]

        source_encoding = self.tokenizer(texts,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        if self.lm_type == 'encdec':
            outputs = self.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            )
        elif self.lm_type == 'dec':
            outputs = self.language_model.generate(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                max_new_tokens=max_length)
            self.tokenizer.padding_side = 'left'
        
        outputs_tokens_hand = cleaned_text_hand = outputs_tokens_rhand = cleaned_text_rhand = None
        if 'multi' in self.model_type:
            # print(outputs['outputs_re'])
            outputs_string = self.tokenizer.batch_decode(outputs['outputs_re'], skip_special_tokens=True)
            # print('string re: ', outputs_string)
            outputs_tokens, cleaned_text = self.motion_string_to_token(outputs_string, pattern='motion')
            outputs_string_hand = self.tokenizer.batch_decode(outputs['outputs_hand'], skip_special_tokens=True)
            # print('string hand: ', outputs_string_hand)
            outputs_tokens_hand, cleaned_text_hand = self.motion_string_to_token(outputs_string_hand, pattern='hand')
            if self.num_heads > 2:
                outputs_string_rhand = self.tokenizer.batch_decode(outputs['outputs_rhand'], skip_special_tokens=True)
                outputs_tokens_rhand, cleaned_text_rhand = self.motion_string_to_token(outputs_string_rhand, pattern='rhand')
        else:
            outputs_string = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_tokens, cleaned_text = self.motion_string_to_token(outputs_string, pattern='motion')

        return {'outputs_tokens': outputs_tokens,
                'cleaned_text': cleaned_text,
                'outputs_tokens_hand': outputs_tokens_hand,
                'cleaned_text_hand': cleaned_text_hand,
                'outputs_tokens_rhand': outputs_tokens_rhand,
                'cleaned_text_rhand': cleaned_text_rhand
                }

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             hand_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None,
                             src: List[str] = None):

        try:
            self.device = self.language_model.device
        except:
            self.device = self.language_model.main_lm.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:

            if task == "t2m":
                assert texts is not None
                motion_strings = [''] * len(texts)
                if not with_len:
                    if tasks is None:
                        tasks = [{
                            'input':
                            ['Generate motion: <Caption_Placeholder>'],
                            'output': ['']
                        }] * len(texts)

                    lengths = [0] * len(texts)
                else:
                    tasks = [{
                        'input': [
                            'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(texts)
                    
            elif task == "pred":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': ['Predict motion: <Motion_Placeholder_s1>'],
                    'output': ['']
                }] * len(lengths)

                motion_strings_old = self.motion_token_to_string(
                    motion_tokens, lengths)
                motion_strings = []
                for i, length in enumerate(lengths):
                    split = length // 5
                    motion_strings.append(
                        '>'.join(motion_strings_old[i].split('>')[:split]) +
                        '>')

            elif task == "inbetween":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': [
                        "Complete the masked motion: <Motion_Placeholder_Masked>"
                    ],
                    'output': ['']
                }] * len(lengths)
                motion_strings = self.motion_token_to_string(
                    motion_tokens, lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts,
                                                    stage)
            # print(inputs)

            gen_results = self.generate_direct(inputs,
                                                max_length=400,
                                                num_beams=1,
                                                do_sample=True)

            return gen_results

        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string(
                motion_tokens, lengths, pattern='motion')
            hand_strings = None
            if hand_tokens is not None and len(hand_tokens) > 0:
                hand_strings = self.motion_token_to_string(
                    hand_tokens, lengths, pattern='hand')

            if not with_len:
                tasks = [{
                    'input': ['Generate text: <Motion_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            gen_results = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=False,
                # bad_words_ids=self.bad_words_ids
            )
            return gen_results["cleaned_text"]

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int], pattern: str = 'motion'):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<{pattern}_id_{self.m_codebook_size}>' +
                 ''.join([f'<{pattern}_id_{int(i)}>' for i in motion_list]) +
                 f'<{pattern}_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_token_list_to_string(self, motion_token: Tensor, pattern: str = 'motion'):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (f'<{pattern}_id_{self.m_codebook_size}>' +
                 ''.join([f'<{pattern}_id_{int(i)}>' for i in motion_list]) +
                 f'<{pattern}_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_token(self, motion_string: List[str], pattern: str = 'motion'):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], f'<{pattern}_id_{self.m_codebook_size}>',
                f'<{pattern}_id_{self.m_codebook_size + 1}>', pattern=pattern)
            string_list = string.split('><')
            try:
                token_list = [
                    int(i.split('_')[-1].replace('>', ''))
                    for i in string_list[1:-1]
                ]
            except:
                token_list = [0]
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return motion_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str, text: str, pattern: str = 'motion'):

        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        token_length = length / self.down_t
        predict_head = int(token_length * self.predict_ratio + 1)
        masked_head = int(token_length * self.inbetween_ratio + 1)
        masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]
        ) + f'><{pattern}_id_{self.m_codebook_size+1}>'
        motion_predict_last = f'<{pattern}_id_{self.m_codebook_size}>' + '>'.join(
            motion_splited[predict_head:])

        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<{pattern}_id_{self.m_codebook_size+2}>' * (
            masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        # if random.random() < self.quota_ratio:
            # text = f'\"{text}\"'

        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>',
            motion_string).replace('<Frame_Placeholder>', f'{length}').replace(
                '<Second_Placeholder>', '%.1f' % seconds).replace(
                    '<Motion_Placeholder_s1>', motion_predict_head).replace(
                        '<Motion_Placeholder_s2>',
                        motion_predict_last).replace(
                            '<Motion_Placeholder_Masked>', motion_masked)

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         motion_strings,
                         texts,
                         stage='test',
                         pattern='motion'):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i], pattern=pattern))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i], pattern=pattern))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr, pattern='motion'):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<{pattern}_id_{self.m_codebook_size}><{pattern}_id_0><{pattern}_id_{self.m_codebook_size+1}>'

        return f'<{pattern}_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<{pattern}_id_{self.m_codebook_size+1}>'

    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        input_ids = torch.tensor(input_ids, device=self.device)

        return input_ids
