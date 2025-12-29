import random
import torch
import pickle, gzip
import os, math
import pandas as pd
import codecs as cs
import numpy as np
from tqdm import tqdm
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .dataset_m import MotionDataset
from .dataset_t2m import Text2MotionDataset
from .load_data import load_h2s_sample, load_csl_sample, load_phoenix_sample, load_iso_sample
import random; random.seed(0)
import json
from copy import deepcopy

# Some how2sign ids are broken, failing in pose fitting.
bad_how2sign_ids = ['0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front', '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front', '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front', 'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front', 'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front', 'g3Cc_1-V31U_12-3-rgb_front']


class H2SMotionDatasetVQ(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        dataset_name='how2sign',
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        # split='train'
        self.dataset_name = dataset_name
        self.root_dir = data_root
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)
        self.mean = mean
        self.std = std
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        assert max_motion_length % unit_length == 0 and min_motion_length % unit_length == 0

        self.all_data = []
        self.h2s_len = self.csl_len = self.phoenix_len = 0

        if 'how2sign' in dataset_name:
            self.data_dir = os.path.join(data_root, split, 'poses')
            self.csv_path = os.path.join(data_root, split, 're_aligned', 'how2sign_realigned_'+split+'_preprocessed_fps.csv')
            self.csv = pd.read_csv(self.csv_path)
            self.fps = self.csv['fps']
            self.csv['DURATION'] = self.csv['END_REALIGNED'] - self.csv['START_REALIGNED']
            self.csv = self.csv[self.csv['DURATION']<30].reset_index(drop=True) # remove sequences longer than 30 seconds
            self.ids = self.csv['SENTENCE_NAME'] #[:200]

            print(f'{split}--loading how2sign annotations...', len(self.ids))
            for idx in tqdm(range(len(self.ids))):
                name = self.ids[idx]
                if name in bad_how2sign_ids:
                    continue
                self.all_data.append({'name': name, 'fps': self.csv[self.csv['SENTENCE_NAME']==name]['fps'].item(), 
                                        'text': self.csv[self.csv['SENTENCE_NAME']==name]['SENTENCE'].item(), 'src': 'how2sign', 'split': split})
            self.h2s_len += len(self.all_data)
        
        if 'csl' in dataset_name:
            if split == 'train':
                ann_path = os.path.join(self.csl_root, 'csl_clean.train')
            else:
                ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print(f'{split}--loading csl annotations...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'csl'
                self.all_data.append(ann)
            self.csl_len += len(self.ann)

        if 'phoenix' in dataset_name:
            if split == 'val':
                ann_path = os.path.join(self.phoenix_root, 'phoenix14t.dev')
            else:
                ann_path = os.path.join(self.phoenix_root, f'phoenix14t.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print(f'{split}--loading phoenix annotations...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'phoenix'
                self.all_data.append(ann)
            self.phoenix_len += len(self.ann)

        print(f'Data loading done. All: {len(self.all_data)}, How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}')
        

    def __len__(self):
        return len(self.all_data)


    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']
        name = sample['name']
        # text = sample['text']
        # print(sample['dataset'])
        if src == 'how2sign':
            clip_poses, text, name, _ = load_h2s_sample(sample, self.root_dir)
        elif src == 'csl':
            clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
        elif src == 'phoenix':
            clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
        elif src == 'asl_iso':
            clip_poses, text, name, _ = load_iso_sample(sample, self.root_dir, dataset='asl_iso')
            src = 'how2sign'
        elif src == 'csl_iso':
            clip_poses, text, name, _ = load_iso_sample(sample, self.csl_root, dataset='csl_iso')
            src = 'csl'
        elif src == 'phoenix_iso':
            clip_poses, text, name, _ = load_iso_sample(sample, self.phoenix_root, dataset='phoenix_iso')
            src = 'phoenix'

        
        if clip_poses is None:
            new_idx = (idx + 1) % len(self.all_data)
            return self.__getitem__(new_idx)
        
        clip_poses = (clip_poses - self.mean.numpy())/(self.std.numpy()+1e-10)
        m_length = clip_poses.shape[0]
        if m_length < self.min_motion_length:
            idx = np.linspace(0, m_length-1, num=self.min_motion_length, dtype=int)
            clip_poses = clip_poses[idx]
        elif m_length > self.max_motion_length:
            idx = np.linspace(0, m_length-1, num=self.max_motion_length, dtype=int)
            clip_poses = clip_poses[idx]
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            idx = (clip_poses.shape[0] - m_length) // 2
            clip_poses = clip_poses[idx:idx + m_length]
        m_length = clip_poses.shape[0]

        return text, torch.from_numpy(clip_poses).float(), m_length, name, None, None, None, None, None, src  #see definitions in utils.py
