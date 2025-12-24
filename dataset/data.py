# coding: utf-8
import sys
import os
import io
import os.path
from typing import Optional
import numpy as np
import random

from torchtext import data
from torchtext.data import Dataset, Iterator
import torch

from dataset.constants import UNK_TOKEN, PAD_TOKEN, TARGET_PAD
from dataset.vocabulary import build_vocab, Vocabulary


def adjust_motion_length(motion_frames, min_len=20, max_len=300, window_size=64):
    """
    모션 시퀀스 길이를 조정합니다.
    - min_len보다 짧으면: 선형 보간으로 min_len까지 늘림
    - max_len보다 길면: 선형 보간으로 max_len까지 줄임
    - window_size가 주어지면: 랜덤 윈도우 샘플링
    
    Args:
        motion_frames: list of frames, 각 프레임은 trg_size 길이의 리스트
        min_len: 최소 길이
        max_len: 최대 길이
        window_size: 윈도우 크기 (None이면 윈도우 샘플링 안함)
    
    Returns:
        adjusted motion_frames
    """
    m_length = len(motion_frames)
    
    # numpy array로 변환
    motion_array = np.array(motion_frames)  # (T, D)
    
    # 길이 조정
    if m_length < min_len:
        # 선형 보간으로 늘림
        idx = np.linspace(0, m_length - 1, num=min_len, dtype=int)
        motion_array = motion_array[idx]
    elif m_length > max_len:
        # 선형 보간으로 줄임
        idx = np.linspace(0, m_length - 1, num=max_len, dtype=int)
        motion_array = motion_array[idx]
    
    # 윈도우 샘플링 (window_size가 주어진 경우에만)
    if window_size is not None and len(motion_array) > window_size:
        start_idx = random.randint(0, len(motion_array) - window_size)
        motion_array = motion_array[start_idx:start_idx + window_size]
    
    # 다시 리스트로 변환
    return [list(frame) for frame in motion_array]


def load_data(cfg: dict) -> (Dataset, Dataset, Optional[Dataset], Vocabulary, Vocabulary):
    data_cfg = cfg["data"]
    
    # Source, Target and Files postfixes
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    files_lang = data_cfg.get("files", "files")
    
    # Train, Dev and Test Path
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]

    level = "word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]
    
    # Target size is plus one due to the counter required for the model
    trg_size = cfg["model"]["trg_size"] + 1
    
    # Skip frames is used to skip a set proportion of target frames
    skip_frames = data_cfg.get("skip_frames", 1)
    
    # Window settings for training
    window_size = data_cfg.get("window_size", 64)
    min_motion_len = data_cfg.get("min_motion_length", 20)
    max_motion_len = data_cfg.get("max_motion_length", 300)

    EOS_TOKEN = '</s>'
    tok_fun = lambda s: list(s) if level == "char" else s.split()

    # Source field
    src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
                           pad_token=PAD_TOKEN, tokenize=tok_fun,
                           batch_first=True, lower=lowercase,
                           unk_token=UNK_TOKEN,
                           include_lengths=True)

    # Files field
    files_field = data.RawField()

    def tokenize_features(features):
        features = np.array(features).astype(float)
        features = torch.as_tensor(features)
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    # Target field
    reg_trg_field = data.Field(sequential=True,
                               use_vocab=False,
                               dtype=torch.float32,
                               batch_first=True,
                               include_lengths=False,
                               pad_token=torch.ones((trg_size,)) * TARGET_PAD,
                               preprocessing=tokenize_features,
                               postprocessing=stack_features)

    # ============================================
    # Training Data: 윈도우 64 + min/max 조정 적용
    # ============================================
    train_data = SignProdDataset(path=train_path,
                                 exts=("." + src_lang, "." + trg_lang, "." + files_lang),
                                 fields=(src_field, reg_trg_field, files_field),
                                 trg_size=trg_size,
                                 skip_frames=skip_frames,
                                 # Train용 윈도우 설정
                                 use_window=True,
                                 window_size=window_size,
                                 min_motion_len=min_motion_len,
                                 max_motion_len=max_motion_len,
                                 filter_pred=lambda x: len(vars(x)['src']) <= max_sent_length
                                                       and len(vars(x)['trg']) <= max_sent_length)

    # Build vocab
    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = build_vocab(field="src", min_freq=src_min_freq,
                            max_size=src_max_size,
                            dataset=train_data, vocab_file=src_vocab_file)

    trg_vocab = [None] * trg_size

    # ============================================
    # Dev Data: 윈도우 없이 raw 데이터 사용
    # ============================================
    dev_data = SignProdDataset(path=dev_path,
                               exts=("." + src_lang, "." + trg_lang, "." + files_lang),
                               trg_size=trg_size,
                               fields=(src_field, reg_trg_field, files_field),
                               skip_frames=skip_frames,
                               # Dev는 윈도우 없이 raw
                               use_window=False,
                               window_size=None,
                               min_motion_len=min_motion_len,
                               max_motion_len=max_motion_len)

    # ============================================
    # Test Data: 윈도우 없이 raw 데이터 사용
    # ============================================
    test_data = SignProdDataset(path=test_path,
                                exts=("." + src_lang, "." + trg_lang, "." + files_lang),
                                trg_size=trg_size,
                                fields=(src_field, reg_trg_field, files_field),
                                skip_frames=skip_frames,
                                # Test도 윈도우 없이 raw
                                use_window=False,
                                window_size=None,
                                min_motion_len=min_motion_len,
                                max_motion_len=max_motion_len)

    src_field.vocab = src_vocab

    return train_data, dev_data, test_data, src_vocab, trg_vocab


# pylint: disable=global-at-module-level
global max_src_in_batch, max_tgt_in_batch


def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset, batch_size: int, batch_type: str = "sentence",
                   train: bool = False, shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.
    """
    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


class SignProdDataset(data.Dataset):
    """Sign Language Production Dataset with optional windowing."""

    def __init__(self, path, exts, fields, trg_size, skip_frames=1,
                 use_window=False, window_size=None,
                 min_motion_len=20, max_motion_len=300, **kwargs):
        """
        Arguments:
            path: Common prefix of paths to the data files.
            exts: A tuple containing the extension for each language.
            fields: A tuple containing the fields for data.
            trg_size: Target feature size per frame.
            skip_frames: Skip every N frames.
            use_window: Whether to apply windowing (for training).
            window_size: Size of the window (only used if use_window=True).
            min_motion_len: Minimum motion length.
            max_motion_len: Maximum motion length.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('file_paths', fields[2])]

        src_path, trg_path, file_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
             io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
             io.open(file_path, mode='r', encoding='utf-8') as files_file:

            for src_line, trg_line, files_line in zip(src_file, trg_file, files_file):
                src_line = src_line.strip()
                trg_line = trg_line.strip()
                files_line = files_line.strip()

                # Parse target
                trg_line = trg_line.split(" ")
                if len(trg_line) == 1:
                    continue

                trg_line = [(float(joint) + 1e-8) for joint in trg_line]
                trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size * skip_frames)]

                # ============================================
                # 길이 조정 및 윈도우 적용
                # ============================================
                if use_window:
                    # Train: 윈도우 샘플링 적용
                    trg_frames = adjust_motion_length(
                        trg_frames,
                        min_len=min_motion_len,
                        max_len=max_motion_len,
                        window_size=window_size
                    )
                else:
                    # Dev/Test: 윈도우 없이 min/max만 조정
                    trg_frames = adjust_motion_length(
                        trg_frames,
                        min_len=min_motion_len,
                        max_len=max_motion_len,
                        window_size=None  # 윈도우 없음
                    )

                if src_line != '' and len(trg_frames) > 0:
                    examples.append(data.Example.fromlist([src_line, trg_frames, files_line], fields))

        super(SignProdDataset, self).__init__(examples, fields, **kwargs)