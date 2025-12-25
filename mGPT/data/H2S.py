# coding: utf-8
"""
SOKE H2SDataModule - PyTorch Lightning DataModule
https://github.com/2000ZRL/SOKE

Sign Language Production을 위한 통합 DataModule
- VAE 학습: H2SMotionDatasetVQ
- LM 학습: Text2MotionDatasetCB
- 평가: Text2MotionDatasetEval
"""
import numpy as np
import torch
import os
from os.path import join as pjoin
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .utils import humanml3d_collate, motion_token_collate
from .humanml.dataset_m_vq_sign import H2SMotionDatasetVQ
from .humanml.dataset_t2m import Text2MotionDataset, Text2MotionDatasetEval
from .humanml.dataset_t2m_cb import Text2MotionDatasetCB


class BASEDataModule(pl.LightningDataModule):
    """Base DataModule with common functionality."""
    
    def __init__(self, collate_fn):
        super().__init__()
        
        self.dataloader_options = {"collate_fn": collate_fn}
        self.persistent_workers = True
        self.is_mm = False
        
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
    
    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.DatasetEval(**sample_params)
    
    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = self.Dataset(
                split=self.cfg.TRAIN.SPLIT if hasattr(self.cfg.TRAIN, 'SPLIT') else 'train',
                **self.hparams
            )
        return self._train_dataset
    
    @property
    def val_dataset(self):
        if self._val_dataset is None:
            params = self.hparams.copy()
            params['code_path'] = None
            params['split'] = self.cfg.EVAL.SPLIT if hasattr(self.cfg.EVAL, 'SPLIT') else 'val'
            self._val_dataset = self.DatasetEval(**params)
        return self._val_dataset
    
    @property
    def test_dataset(self):
        if self._test_dataset is None:
            params = self.hparams.copy()
            params['code_path'] = None
            params['split'] = self.cfg.TEST.SPLIT if hasattr(self.cfg.TEST, 'SPLIT') else 'test'
            self._test_dataset = self.DatasetEval(**params)
        return self._test_dataset
    
    def setup(self, stage=None):
        if stage in (None, "fit"):
            _ = self.train_dataset
            _ = self.val_dataset
        if stage in (None, "test"):
            _ = self.test_dataset
    
    def train_dataloader(self):
        options = self.dataloader_options.copy()
        options["batch_size"] = self.cfg.TRAIN.BATCH_SIZE
        options["num_workers"] = self.cfg.TRAIN.NUM_WORKERS
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            persistent_workers=self.persistent_workers,
            **options,
        )
    
    def val_dataloader(self):
        options = self.dataloader_options.copy()
        options["batch_size"] = self.cfg.EVAL.BATCH_SIZE
        options["num_workers"] = self.cfg.EVAL.NUM_WORKERS
        options["shuffle"] = False
        return DataLoader(
            self.val_dataset,
            persistent_workers=self.persistent_workers,
            **options,
        )
    
    def test_dataloader(self):
        options = self.dataloader_options.copy()
        options["batch_size"] = 1 if self.is_mm else self.cfg.TEST.BATCH_SIZE
        options["num_workers"] = self.cfg.TEST.NUM_WORKERS
        options["shuffle"] = False
        return DataLoader(
            self.test_dataset,
            persistent_workers=self.persistent_workers,
            **options,
        )
    
    def predict_dataloader(self):
        return self.test_dataloader()


class H2SDataModule(BASEDataModule):
    """
    Sign Language Production DataModule.
    
    Supports:
        - How2Sign
        - CSL-Daily
        - Phoenix-2014T
    
    Stages:
        - vae: VQ-VAE training (H2SMotionDatasetVQ)
        - lm_pretrain/lm_instruct: LM training (Text2MotionDatasetCB)
        - token: Tokenization stage
    """
    
    def __init__(self, cfg, **kwargs):
        super().__init__(collate_fn=humanml3d_collate)
        
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info
        cfg.DATASET.JOINT_TYPE = 'humanml3d'
        self.name = "humanml3d"
        self.njoints = 22  # Not actually used, SMPL-X based
        
        # Dataset config
        self.hparams.dataset_name = cfg.DATASET.H2S.DATASET_NAME
        self.hparams.csl_root = cfg.DATASET.H2S.CSL_ROOT
        self.hparams.phoenix_root = cfg.DATASET.H2S.get('PHOENIX_ROOT', None)
        self.hparams.pred_data_dir = cfg.DATASET.H2S.get('pred_data_dir', False)
        
        # Paths
        data_root = cfg.DATASET.H2S.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "re_aligned")
        self.hparams.motion_dir = pjoin(data_root, "poses")
        
        # Mean and std
        mean_path = cfg.DATASET.H2S.MEAN_PATH
        std_path = cfg.DATASET.H2S.STD_PATH
        print(f'Loading mean from {mean_path}')
        print(f'Loading std from {std_path}')
        
        self.hparams.mean = torch.load(mean_path)
        self.hparams.std = torch.load(std_path)
        
        # Filter joints: remove lower body (36 dims) and shape (10 dims)
        # Original: 179 -> After filtering: 133
        self.hparams.mean = self.hparams.mean[(3 + 3 * 11):]
        self.hparams.mean = torch.cat([self.hparams.mean[:-20], self.hparams.mean[-10:]], dim=0)
        self.hparams.std = self.hparams.std[(3 + 3 * 11):]
        self.hparams.std = torch.cat([self.hparams.std[:-20], self.hparams.std[-10:]], dim=0)
        
        # For evaluation
        self.hparams.mean_eval = self.hparams.mean
        self.hparams.std_eval = self.hparams.std
        
        # Length settings
        self.hparams.max_motion_length = cfg.DATASET.H2S.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.H2S.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.H2S.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.H2S.UNIT_LEN
        
        # Additional params
        self.hparams.debug = cfg.DEBUG if hasattr(cfg, 'DEBUG') else False
        self.hparams.stage = cfg.TRAIN.STAGE
        
        # Word vectorizer (for evaluation)
        try:
            from .humanml.utils.word_vectorizer import WordVectorizer
            self.hparams.w_vectorizer = WordVectorizer(
                cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab"
            )
        except:
            self.hparams.w_vectorizer = None
        
        # Dataset class selection based on stage
        self.DatasetEval = H2SMotionDatasetVQ if cfg.TRAIN.STAGE == "vae" else Text2MotionDatasetEval
        
        if cfg.TRAIN.STAGE == "vae":
            self.hparams.win_size = 64
            self.Dataset = H2SMotionDatasetVQ
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.get('TASK_PATH', None)
            self.hparams.std_text = cfg.DATASET.H2S.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            from .humanml.dataset_t2m import Text2MotionDataset as Text2MotionDatasetToken
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        else:
            self.Dataset = Text2MotionDataset
        
        # Feature dimension
        self.nfeats = 133
        cfg.DATASET.NFEATS = self.nfeats
    
    def feats2joints(self, features):
        """Convert features back to original scale."""
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        features = features * std + mean
        return features
    
    def normalize(self, features):
        """Normalize features."""
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        features = (features - mean) / std
        return features
    
    def denormalize(self, features):
        """Denormalize features."""
        mean = self.hparams.mean.to(features)
        std = self.hparams.std.to(features)
        features = features * std + mean
        return features
    
    def mm_mode(self, mm_on=True):
        """Toggle multi-modal evaluation mode."""
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list if hasattr(self.test_dataset, 'name_list') else []
            if len(self.name_list) > 0:
                self.mm_list = np.random.choice(
                    self.name_list,
                    min(self.cfg.METRIC.MM_NUM_SAMPLES, len(self.name_list)),
                    replace=False
                )
                self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            if hasattr(self, 'name_list'):
                self.test_dataset.name_list = self.name_list
