import os, math, gzip
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
import torch
import pandas as pd
from tqdm import tqdm
import random; random.seed(0)
from copy import deepcopy
from .load_data import load_csl_sample, load_h2s_sample, load_phoenix_sample

# Some how2sign ids are broken, failing in pose fitting.
bad_how2sign_ids = ['0DU7wWLK-QU_0-8-rgb_front', '0ICZi26jdaQ_28-5-rgb_front', '0vNfEYst_tQ_11-8-rgb_front', '13X0vEMNm7M_8-5-rgb_front', '14weIYQswlE_23-8-rgb_front', '1B56XMJ-j1Q_13-8-rgb_front', '1P0oKY4FNyI_0-8-rgb_front', '1dpRaxOTfZs_0-8-rgb_front', '1ei1kVTw23A_29-8-rgb_front', '1spCnuBmWYk_0-8-rgb_front', '2-vXO7MMLJc_0-5-rgb_front', '21PbS6wnHtY_0-5-rgb_front', '3tyfxL2wO-M_0-8-rgb_front', 'BpYDl3AO4B8_0-1-rgb_front', 'CH7AviIr0-0_14-8-rgb_front', 'CJ8RyW9pzKU_6-8-rgb_front', 'D0T7ho08Q3o_25-2-rgb_front', 'Db5SUQvNsHc_18-1-rgb_front', 'Eh697LCFjTw_0-3-rgb_front', 'F-p1IdedNbg_23-8-rgb_front', 'aUBQCNegrYc_13-1-rgb_front', 'cvn7htBA8Xc_9-8-rgb_front', 'czBrBQgZIuc_19-5-rgb_front', 'dbSAB8F8GYc_11-9-rgb_front', 'doMosV-zfCI_7-2-rgb_front', 'dvBdWGLzayI_10-8-rgb_front', 'eBrlZcccILg_26-3-rgb_front', '39FN42e41r0_17-1-rgb_front', 'a4Nxq0QV_WA_9-3-rgb_front', 'fzrJBu2qsM8_11-8-rgb_front', 'g3Cc_1-V31U_12-3-rgb_front']

class Text2MotionDataset(data.Dataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        dataset_name='how2sign',
        **kwargs,
    ):

        # split = 'train'
        # restrian the length of motion and text
        # self.max_length = 20
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.csl_root = kwargs.get('csl_root', None)
        self.phoenix_root = kwargs.get('phoenix_root', None)

        # Data mean and std
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
            self.ids = self.csv['SENTENCE_NAME'] #[:100]

            print('loading how2sign data...', len(self.ids))
            for idx in tqdm(range(len(self.ids))):
                name = self.ids[idx]
                if name in bad_how2sign_ids:
                    continue
                self.all_data.append({'name': name, 'fps': self.csv[self.csv['SENTENCE_NAME']==name]['fps'].item(), 
                                        'text': self.csv[self.csv['SENTENCE_NAME']==name]['SENTENCE'].item(), 'src': 'how2sign'})
            self.h2s_len = len(self.all_data)
        
        if 'csl' in dataset_name:
            if split == 'train':
                ann_path = os.path.join(self.csl_root, 'csl_clean.train')
            else:
                ann_path = os.path.join(self.csl_root, f'csl_clean.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:800]

            print('loading csl data...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'csl'
                self.all_data.append(ann)
            self.csl_len = len(self.ann)
        
        if 'phoenix' in dataset_name:
            if split == 'val':
                ann_path = os.path.join(self.phoenix_root, 'phoenix14t.dev')
            else:
                ann_path = os.path.join(self.phoenix_root, f'phoenix14t.{split}')
            with gzip.open(ann_path, 'rb') as f:
                self.ann = pickle.load(f) #[:200]

            print(f'{split}--loading phoenix data...', len(self.ann))
            for idx in tqdm(range(len(self.ann))):
                ann = deepcopy(self.ann[idx])
                ann['src'] = 'phoenix'
                self.all_data.append(ann)
            self.phoenix_len += len(self.ann)

        # random.shuffle(self.all_data)
        print(f'Data loading done. All: {len(self.all_data)}, How2Sign: {self.h2s_len}, CSL: {self.csl_len}, Phoenix: {self.phoenix_len}')
        self.nfeats = 133
        # self.reset_max_len(self.max_length)


    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length


    def __len__(self):
        return len(self.all_data)


    def __getitem__(self, idx):
        sample = self.all_data[idx]
        src = sample['src']

        if src == 'how2sign':
            clip_poses, text, name, _ = load_h2s_sample(sample, self.data_dir)
        elif src == 'csl':
            clip_poses, text, name, _ = load_csl_sample(sample, self.csl_root)
        elif src == 'phoenix':
            clip_poses, text, name, _ = load_phoenix_sample(sample, self.phoenix_root)
        
        all_captions = [text]

        clip_poses = (clip_poses - self.mean.numpy())/(self.std.numpy()+1e-10)
        # return torch.from_numpy(clip_poses).float(), basename, clip_text
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

        return text, torch.from_numpy(clip_poses).float(), m_length, name, None, None, None, all_captions, None, src


def sample(input,count):
    ss=float(len(input))/count
    return [ input[int(math.floor(i*ss))] for i in range(count) ]