from typing import List
import os
import torch
from torch import Tensor
from torchmetrics import Metric
from .utils import *
from mGPT.metrics.dtw import dtw, l2_dist, l1_dist, l2_dist_align
from mGPT.metrics.fid import smpl_fid
from functools import partial
from mGPT.utils.human_models import rigid_align, rigid_align_torch_batch, smpl_x
from collections import defaultdict


class TM2TMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname='humanml3d',
                 top_k=3,
                 R_size=32,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "MPJPE, MPVPE DTW"
        # self.top_k = top_k
        # self.R_size = R_size
        # self.text = 'lm' in cfg.TRAIN.STAGE and cfg.model.params.task == 't2m'
        # self.diversity_times = diversity_times

        self.joint_part2idx = smpl_x.joint_part2idx
        self.vertex_part2idx = smpl_x.vertex_part2idx
        self.smplx_part2idx = {'upper_body': list(range(30)), 'lhand': list(range(30, 75)), 'rhand': list(range(75, 120)), 'hand': list(range(30, 120)), 'face': list(range(120, 133))}
        self.J_regressor_idx = smpl_x.J_regressor_idx
        self.name2scores = defaultdict(dict)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("how2sign_count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("csl_count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("phoenix_count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.add_state("how2sign_DTW_MPJPE_PA_lhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_DTW_MPJPE_PA_rhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("how2sign_DTW_MPJPE_PA_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.add_state("csl_DTW_MPJPE_PA_lhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("csl_DTW_MPJPE_PA_rhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("csl_DTW_MPJPE_PA_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        
        self.add_state("phoenix_DTW_MPJPE_PA_lhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("phoenix_DTW_MPJPE_PA_rhand",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        self.add_state("phoenix_DTW_MPJPE_PA_body",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.MR_metrics = ["how2sign_DTW_MPJPE_PA_lhand", "how2sign_DTW_MPJPE_PA_rhand", "how2sign_DTW_MPJPE_PA_body", 
                           "csl_DTW_MPJPE_PA_lhand", "csl_DTW_MPJPE_PA_rhand", "csl_DTW_MPJPE_PA_body",
                           "phoenix_DTW_MPJPE_PA_lhand", "phoenix_DTW_MPJPE_PA_rhand", "phoenix_DTW_MPJPE_PA_body"]

        # All metric
        self.metrics = self.MR_metrics

        
    @torch.no_grad()
    def compute(self, sanity_flag):

        mr_metrics = {}
        for name in self.metrics:
            d = name.split('_')[0]
            mr_metrics[name] = getattr(self, name) / max(getattr(self, f'{d}_count_seq'), 1e-6)

        for name, v in mr_metrics.items():
            print(name, ': ', v)
        
        # Reset
        self.reset()
        
        return mr_metrics


    @torch.no_grad()
    def update(self, 
               feats_rst: Tensor, feats_ref: Tensor,
               joints_rst: Tensor, joints_ref: Tensor,
               vertices_rst: Tensor, vertices_ref: Tensor,
               lengths: List[int], lengths_rst: List[int],
               split: str, src: List[str], name: List[str]):
        # assert joints_rst.shape == joints_ref.shape
        # assert joints_rst.dim() == 4
        # (bs, seq, njoint=22, 3)

        B = len(lengths)
        BT, N = joints_rst.shape[:2]
        joints_rst = joints_rst.reshape(B, BT//B, N, 3)
        BT, N = joints_ref.shape[:2]
        joints_ref = joints_ref.reshape(B, BT//B, N, 3)

        BT, N = vertices_rst.shape[:2]
        vertices_rst = vertices_rst.reshape(B, BT//B, N, 3)
        BT, N = vertices_ref.shape[:2]
        vertices_ref = vertices_ref.reshape(B, BT//B, N, 3)

        # avoid cuda error of DDP in pampjpe
        joints_rst = joints_rst.detach().cpu().numpy()
        joints_ref = joints_ref.detach().cpu().numpy()
        vertices_rst = vertices_rst.detach().cpu()
        vertices_ref = vertices_ref.detach().cpu()

        part_lst = ['body', 'lhand', 'rhand']  #save time for validation during training
        for i in range(len(lengths)):
            cur_len = lengths[i]
            rst_len = lengths_rst[i]
            mesh_gt = vertices_ref[i, :cur_len]
            mesh_out = vertices_rst[i, :rst_len]
            joints_rst_cur = joints_rst[i, :rst_len] 
            joints_ref_cur = joints_ref[i, :cur_len]
            data_src = src[i]
            cur_name = name[i]
            setattr(self, f"{data_src}_count_seq", getattr(self, f"{data_src}_count_seq")+1)
            # print(cur_len, rst_len)

            if split in ['val', 'test']:
                '''
                Note that when align_idx=0, the metric is DTW-JPE; when align_idx=None, the metric is DTW-PA-JPE. But we didn't modify the variable names.
                '''
                joint_idx = self.joint_part2idx['upper_body']
                dist_func = partial(l2_dist_align, wanted=joint_idx, align_idx=0)
                value = dtw(joints_rst_cur, joints_ref_cur, dist_func)[0]
                setattr(self, f'{data_src}_DTW_MPJPE_PA_body', getattr(self, f'{data_src}_DTW_MPJPE_PA_body') + value)
                self.name2scores[cur_name][f'{data_src}_DTW_MPJPE_PA_body'] = value
                # print('body: ', value)

                joint_idx = self.joint_part2idx['lhand']
                joint_gt_lhand = torch.matmul(smpl_x.orig_hand_regressor['left'], mesh_gt).float().numpy()
                joint_out_lhand = torch.matmul(smpl_x.orig_hand_regressor['left'], mesh_out).float().numpy()
                dist_func = partial(l2_dist_align, align_idx=0)
                value = dtw(joint_out_lhand, joint_gt_lhand, dist_func)[0]
                setattr(self, f"{data_src}_DTW_MPJPE_PA_lhand", getattr(self, f"{data_src}_DTW_MPJPE_PA_lhand") + value)
                self.name2scores[cur_name][f"{data_src}_DTW_MPJPE_PA_lhand"] = value
                # print('lhand: ', value)

                joint_idx = self.joint_part2idx['rhand']
                joint_gt_rhand = torch.matmul(smpl_x.orig_hand_regressor['right'], mesh_gt).float().numpy()
                joint_out_rhand = torch.matmul(smpl_x.orig_hand_regressor['right'], mesh_out).float().numpy()
                dist_func = partial(l2_dist_align, align_idx=0)
                value = dtw(joint_out_rhand, joint_gt_rhand, dist_func)[0]
                setattr(self, f"{data_src}_DTW_MPJPE_PA_rhand", getattr(self, f"{data_src}_DTW_MPJPE_PA_rhand") + value)
                self.name2scores[cur_name][f"{data_src}_DTW_MPJPE_PA_rhand"] = value
        