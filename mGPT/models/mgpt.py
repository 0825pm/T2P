import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import time
from mGPT.config import instantiate_from_config
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
import mGPT.render.matplot.plot_3d_global as plot_3d


class MotionGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()

        # Instantiate motion tokenizer
        if motion_vae != None:
            self.vae = instantiate_from_config(motion_vae)
            lm['params']['motion_codebook_size'] = self.vae.code_num
        
        # additional hand vae
        self.hand_vae_cfg = kwargs.get('hand_vae_cfg', None)
        if self.hand_vae_cfg is not None:
            self.hand_vae = instantiate_from_config(self.hand_vae_cfg)
            lm['params']['hand_codebook_size'] = self.hand_vae.code_num
        
        self.rhand_vae_cfg = kwargs.get('rhand_vae_cfg', None)
        if self.rhand_vae_cfg is not None:
            self.rhand_vae = instantiate_from_config(self.rhand_vae_cfg)
            lm['params']['rhand_codebook_size'] = self.rhand_vae.code_num
        
        self.face_vae_cfg = kwargs.get('face_vae_cfg', None)
        if self.face_vae_cfg is not None:
            self.face_vae = instantiate_from_config(self.face_vae_cfg)

        # Freeze the motion tokenizer for lm training
        if 'lm' in self.hparams.stage:
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False
            if self.hand_vae_cfg is not None:
                self.hand_vae.training = False
                for p in self.hand_vae.parameters():
                    p.requires_grad = False
            if self.rhand_vae_cfg is not None:
                self.rhand_vae.training = False
                for p in self.rhand_vae.parameters():
                    p.requires_grad = False
            # Instantiate motion-language model
            self.lm = instantiate_from_config(lm)

        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))

    def forward(self, batch, task="t2m"):
        texts = batch["text"]
        lengths_ref = batch["length"]

        # Forward
        # texts = ['Generate motion: ' + text for text in texts]
        outputs, output_texts = self.lm.generate_direct(texts, do_sample=True)

        # Motion Decode
        feats_rst_lst = []
        lengths = []
        max_len = 0

        for i in range(len(texts)):
            if task == "pred":
                motion = self.vae.decode(
                    torch.cat((batch["motion"][i], outputs[i])))
            elif task in ["t2m", "m2t", "inbetween"]:
                motion = self.vae.decode(outputs[i])
                # motion = self.datamodule.denormalize(motion)
                lengths.append(motion.shape[1])
            else:
                raise NotImplementedError

            if motion.shape[1] > max_len:
                max_len = motion.shape[1]

            if task in ["t2m", "m2t", "pred"]:
                feats_rst_lst.append(motion)

            elif task == "inbetween":
                motion = torch.cat(
                    (batch["motion_heading"][i][None],
                     motion[:, lengths_ref[i] // 4:lengths_ref[i] // 4 * 3,
                            ...], batch["motion_tailing"][i][None]),
                    dim=1)
                feats_rst_lst.append(motion)

        feats_rst = torch.zeros(
            (len(feats_rst_lst), max_len, motion.shape[-1])).to(self.device)

        # padding and concat
        for i in range(len(feats_rst_lst)):
            feats_rst[i, :feats_rst_lst[i].shape[1], ...] = feats_rst_lst[i]

        # Recover joints for evaluation
        joints_rst = self.feats2joints(feats_rst)

        # return set
        outputs = {
            "texts": output_texts,
            "feats": feats_rst,
            "joints": joints_rst,
            "length": lengths
        }

        return outputs

    def train_lm_forward(self, batch):
        tokens_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = batch["tasks"]
        all_captions = batch['all_captions']
        if self.hparams.condition == 'caption':
            texts = [random.choice(all_captions[i]) for i in range(len(texts))]

        # LLM Forward
        outputs = self.lm(texts, tokens_ref, lengths, tasks, src=batch['src'], name=batch['name'])
        # outputs = self.t2m_gpt.generate(texts)
        return {'outputs': outputs}

    @torch.no_grad()
    def val_t2m_forward(self, batch, vis=False):
        feats_ref = batch["motion"]
        # print(feats_ref.shape)
        B, T, C = feats_ref.shape
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        # if self.trainer.datamodule.is_mm:
        #     texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
        #     feats_ref = feats_ref.repeat_interleave(
        #         self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
        #     lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
        #     instructions = pjoin(self.datamodule.hparams.data_root,
        #                          'template_instructions.json')
        #     instructions = json.load(open(instructions, 'r'))
        #     tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        if vis:
            func = max
        else:
            func = max

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)

        rst_len = lengths.copy()
        # Forward
        gen_results = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks,
                                               src=batch['src'],
                                               name=batch['name'])
        outputs_tokens = gen_results['outputs_tokens']
        outputs_tokens_hand = gen_results['outputs_tokens_hand']
        outputs_tokens_rhand = gen_results['outputs_tokens_rhand']

        max_len = max(map(len, outputs_tokens))
        if outputs_tokens_hand is not None:
            max_hand_len = max(map(len, outputs_tokens_hand))
            max_len = max(max_hand_len, max_len)
        if outputs_tokens_rhand is not None:
            max_rhand_len = max(map(len, outputs_tokens_rhand))
            max_len = max(max_rhand_len, max_len)
        max_len *= 4  #upsample factor
        # print('tokens_re: ', len(outputs_tokens), outputs_tokens[0].shape)
        # print('tokens_hand: ', len(outputs_tokens_hand), outputs_tokens_hand[0].shape)

        # Motion Decode
        # feats_rst = torch.zeros_like(feats_ref)
        feats_rst = torch.zeros(B, max_len, C).to(feats_ref)

        for i in range(len(texts)):
            name = batch['name'][i]

            outputs_tokens[i] = torch.clamp(outputs_tokens[i],
                                     0,
                                     self.vae.code_num - 1,
                                     out=None)

            if len(outputs_tokens[i]) > 1:
                motion = self.vae.decode(outputs_tokens[i])
                rst_len[i] = motion.shape[1]
                motion = F.pad(motion, (0, 0, 0, max_len-motion.shape[1]), mode='replicate')
                # rst_len[i] = len(outputs_tokens[i])
                # print('body len: ', len(outputs_tokens[i]))
            else:
                if outputs_tokens_hand is None:
                    motion = torch.zeros(1, max_len, C)
                else:
                    motion = torch.zeros((1, max_len, self.vae.nfeats)).to(feats_ref.device)
                rst_len[i] = 1 #max_len
            feats_rst[i:i + 1, :, :30] = motion[..., :30]
            feats_rst[i:i + 1, :, -13:] = motion[..., 30:43]

            if outputs_tokens_hand is not None:
                outputs_tokens_hand[i] = torch.clamp(outputs_tokens_hand[i],
                                     0,
                                     self.hand_vae.code_num - 1,
                                     out=None)
                if len(outputs_tokens_hand[i]) > 1:
                    motion_hand = self.hand_vae.decode(outputs_tokens_hand[i])
                    rst_len[i] = func(rst_len[i], motion_hand.shape[1])
                    motion_hand = F.pad(motion_hand, (0, 0, 0, max_len-motion_hand.shape[1]), mode='replicate')
                    # rst_len[i] = func(rst_len[i], len(outputs_tokens_hand[i]))
                    # print('lhand len: ', len(outputs_tokens_hand[i]))
                else:
                    motion_hand = torch.zeros((1, max_len, self.hand_vae.nfeats)).to(feats_ref.device)
                    rst_len[i] = min(rst_len[i], max_len)
                feats_rst[i:i + 1, :, 30:30+self.hand_vae.nfeats] = motion_hand
            
            if outputs_tokens_rhand is not None:
                outputs_tokens_rhand[i] = torch.clamp(outputs_tokens_rhand[i],
                                     0,
                                     self.rhand_vae.code_num - 1,
                                     out=None)
                if len(outputs_tokens_rhand[i]) > 1:
                    motion_rhand = self.rhand_vae.decode(outputs_tokens_rhand[i])
                    rst_len[i] = func(rst_len[i], motion_rhand.shape[1])
                    motion_rhand = F.pad(motion_rhand, (0, 0, 0, max_len-motion_rhand.shape[1]), mode='replicate')
                    # rst_len[i] = func(rst_len[i], len(outputs_tokens_rhand[i]))
                    # print('rhand len: ', len(outputs_tokens_rhand[i]))
                else:
                    motion_rhand = torch.zeros((1, max_len, self.rhand_vae.nfeats)).to(feats_ref.device)
                    rst_len[i] = min(rst_len[i], max_len)
                feats_rst[i:i + 1, :, 75:75+self.hand_vae.nfeats] = motion_rhand

        # Recover joints for evaluation
        vertices_ref, joints_ref = self.feats2joints(feats_ref)
        vertices_rst, joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "vertices_ref": vertices_ref,
            "vertices_rst": vertices_rst,
            "length": lengths,
            "lengths_rst": rst_len
            # "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2t_forward(self, batch):
        # self.hparams.metrics_dict = []

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        all_captions = [c[0] for c in batch['all_captions']]

        # Motion Encode
        motion_tokens = []
        hand_tokens = []
        lengths_tokens = []
        feats_ref_hand = feats_ref[..., 30:120]
        feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:]], dim=-1)
        for i in range(len(feats_ref)):
            if self.hand_vae_cfg is None:
                motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
                motion_tokens.append(motion_token[0])
                lengths_tokens.append(motion_token.shape[1])

            else:
                motion_token, _ = self.vae.encode(feats_ref_re[i:i+1])
                motion_tokens.append(motion_token[0])
                lengths_tokens.append(motion_token.shape[1])

                hand_token, _ = self.hand_vae.encode(feats_ref_hand[i:i+1])
                hand_tokens.append(hand_token[0])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               hand_tokens=hand_tokens,
                                               lengths=lengths_tokens,
                                               task="m2t",
                                               stage='test')
        # print(outputs, texts)
        # return set
        rs_set = {
            "m_ref": feats_ref,
            # "t_ref": all_captions,
            "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths,
                                               task=task,
                                               stage='test')

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):
        # batch detach
        feats_ref = batch["motion"]
        joints_ref = None #self.feats2joints(feats_ref)
        feats_rst_hand = feats_rst_re = loss_commit_hand = loss_commit_re = perplexity_re = perplexity_hand = None
        # motion encode & decode
        if self.hand_vae_cfg is None:
            feats_rst, loss_commit, perplexity = self.vae(feats_ref)
        elif self.rhand_vae_cfg is None:
            feats_ref_hand = feats_ref[..., 30:120]
            feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:]], dim=-1)
            feats_rst_hand, loss_commit_hand, perplexity_hand = self.hand_vae(feats_ref_hand)
            feats_rst_re, loss_commit_re, perplexity_re = self.vae(feats_ref_re)
            feats_rst = torch.cat([feats_rst_re[..., :30], feats_rst_hand, feats_rst_re[..., 30:]], dim=-1)
            loss_commit = loss_commit_hand + loss_commit_re
            perplexity = perplexity_hand + perplexity_re
        elif self.face_vae_cfg is None:
            feats_ref_lhand = feats_ref[..., 30:75]
            feats_ref_rhand = feats_ref[..., 75:120]
            feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:]], dim=-1)
            feats_rst_lhand, loss_commit_lhand, perplexity_lhand = self.hand_vae(feats_ref_lhand)
            feats_rst_rhand, loss_commit_rhand, perplexity_rhand = self.rhand_vae(feats_ref_rhand)
            feats_rst_re, loss_commit_re, perplexity_re = self.vae(feats_ref_re)
            feats_rst = torch.cat([feats_rst_re[..., :30], feats_rst_lhand, feats_rst_rhand, feats_rst_re[..., 30:]], dim=-1)
            loss_commit = loss_commit_lhand + loss_commit_rhand + loss_commit_re
            perplexity = perplexity_lhand + perplexity_rhand + perplexity_re
        else:
            feats_ref_lhand = feats_ref[..., 30:75]
            feats_ref_rhand = feats_ref[..., 75:120]
            feats_ref_face = feats_ref[..., 123:]
            feats_ref_re = torch.cat([feats_ref[..., :30], feats_ref[..., 120:123]], dim=-1)
            feats_rst_lhand, loss_commit_lhand, perplexity_lhand = self.hand_vae(feats_ref_lhand)
            feats_rst_rhand, loss_commit_rhand, perplexity_rhand = self.rhand_vae(feats_ref_rhand)
            feats_rst_face, loss_commit_face, perplexity_face = self.face_vae(feats_ref_face)
            feats_rst_re, loss_commit_re, perplexity_re = self.vae(feats_ref_re)
            feats_rst = torch.cat([feats_rst_re[..., :30], feats_rst_lhand, feats_rst_rhand, feats_rst_re[..., 30:], feats_rst_face], dim=-1)
            loss_commit = loss_commit_lhand + loss_commit_rhand + loss_commit_re + loss_commit_face
            perplexity = perplexity_lhand + perplexity_rhand + perplexity_re + perplexity_face
            
        joints_rst = None #self.feats2joints(feats_rst)
        # return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
            "length": batch['length']
        }
        return rs_set


    @torch.no_grad()
    def val_vae_forward(self, batch, split="train", stage=None):
        # Detach batch
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Repeat for multimodal evaluation
        if stage!='demo' and self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        feats_rst = torch.zeros_like(feats_ref)
        code_pred_all = []

        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            if self.hand_vae_cfg is None:
                feats_pred, _, _ = self.vae(feats_ref[i:i + 1, :lengths[i]])
                code_pred, _ = self.vae.encode(feats_ref[i:i + 1, :lengths[i]])
                code_pred_all.append(code_pred[0].tolist())
            elif self.rhand_vae_cfg is None:
                feats_ref_hand = feats_ref[i:i + 1, :lengths[i], 30:120]
                feats_ref_re = torch.cat([feats_ref[i:i + 1, :lengths[i], :30], feats_ref[i:i + 1, :lengths[i], 120:]], dim=-1)

                feats_pred_hand, _, _ = self.hand_vae(feats_ref_hand)
                feats_pred, _, _ = self.vae(feats_ref_re)
                feats_pred = torch.cat([feats_pred[..., :30], feats_pred_hand, feats_pred[..., 30:]], dim=-1)

                code_pred_hand, _ = self.hand_vae.encode(feats_ref_hand)
                code_pred_re, _ = self.vae.encode(feats_ref_re)
                code_pred_all.append([code_pred_hand[0].tolist(), code_pred_re[0].tolist()])
            elif self.face_vae_cfg is None:
                feats_ref_lhand = feats_ref[i:i + 1, :lengths[i], 30:75]
                feats_ref_rhand = feats_ref[i:i + 1, :lengths[i], 75:120]
                feats_ref_re = torch.cat([feats_ref[i:i + 1, :lengths[i], :30], feats_ref[i:i + 1, :lengths[i], 120:]], dim=-1)

                feats_pred_lhand, _, _ = self.hand_vae(feats_ref_lhand)
                feats_pred_rhand, _, _ = self.rhand_vae(feats_ref_rhand)
                feats_pred, _, _ = self.vae(feats_ref_re)
                feats_pred = torch.cat([feats_pred[..., :30], feats_pred_lhand, feats_pred_rhand, feats_pred[..., 30:]], dim=-1)
                # print(feats_pred.shape)

                code_pred_hand, _ = self.hand_vae.encode(feats_ref_lhand)
                code_pred_re, _ = self.vae.encode(feats_ref_re)
                code_pred_all.append([code_pred_hand[0].tolist(), code_pred_re[0].tolist()])
            else:
                feats_ref_lhand = feats_ref[i:i + 1, :lengths[i], 30:75]
                feats_ref_rhand = feats_ref[i:i + 1, :lengths[i], 75:120]
                feats_ref_re = torch.cat([feats_ref[i:i + 1, :lengths[i], :30], feats_ref[i:i + 1, :lengths[i], 120:123]], dim=-1)
                feats_ref_face = feats_ref[i:i + 1, :lengths[i], 123:]

                feats_pred_lhand, _, _ = self.hand_vae(feats_ref_lhand)
                feats_pred_rhand, _, _ = self.rhand_vae(feats_ref_rhand)
                feats_pred_face, _, _ = self.face_vae(feats_ref_face)
                feats_pred, _, _ = self.vae(feats_ref_re)
                feats_pred = torch.cat([feats_pred[..., :30], feats_pred_lhand, feats_pred_rhand, feats_pred[..., 30:], feats_pred_face], dim=-1)
                # print(feats_pred.shape)

                code_pred_hand, _ = self.hand_vae.encode(feats_ref_lhand)
                code_pred_re, _ = self.vae.encode(feats_ref_re)
                code_pred_all.append([code_pred_hand[0].tolist(), code_pred_re[0].tolist()])

            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

            # codeFre_pred = torch.bincount(code_pred[0],
            #                               minlength=self.hparams.codebook_size).to(
            #                                   self.codeFrequency.device)
            # self.codePred.append(code_pred[0])
            # self.codeFrequency += codeFre_pred

        # np.save('../memData/results/codeFrequency.npy',
        #         self.codeFrequency.cpu().numpy())

        # Recover joints for evaluation
        vertices_ref, joints_ref = self.feats2joints(feats_ref)
        vertices_rst, joints_rst = self.feats2joints(feats_rst)
        # print(vertices_ref.shape, vertices_rst.shape)
        # print(joints_ref.shape, joints_rst.shape)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "vertices_ref": vertices_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "vertices_rst": vertices_rst,
            "length": lengths,
            "code_pred": code_pred_all
        }

        return rs_set
    

    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None
        lengths = batch['length']
        src = batch['src']
        name = batch['name']
        # print('task: ', self.hparams.task)
        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_pretrain"] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
                getattr(self.metrics,'MRMetrics').update(
                            feats_rst=rs_set["m_rst"],
                            feats_ref=rs_set["m_ref"],
                            joints_rst=rs_set["joints_rst"],
                            joints_ref=rs_set["joints_ref"],
                            vertices_rst=rs_set["vertices_rst"],
                            vertices_ref=rs_set["vertices_ref"], 
                            lengths=lengths,
                            src=src,
                            name=name
                        )
            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                    getattr(self.metrics, 'TM2TMetrics').update(
                            feats_rst=rs_set["m_rst"],
                            feats_ref=rs_set["m_ref"],
                            joints_rst=rs_set["joints_rst"], 
                            joints_ref=rs_set["joints_ref"],
                            vertices_rst=rs_set["vertices_rst"], 
                            vertices_ref=rs_set["vertices_ref"],
                            lengths=lengths,
                            lengths_rst=rs_set['lengths_rst'],
                            split=split,
                            src=src,
                            name=name
                        )
                elif self.hparams.task == "m2t":
                    rs_set_m2t = self.val_m2t_forward(batch)
                    getattr(self.metrics, 'M2TMetrics').update(
                        pred_texts=rs_set_m2t["t_pred"],
                        gt_texts=rs_set_m2t["t_ref"],
                        lengths=rs_set_m2t['length'],
                        src=src,
                    )
                # elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                #     rs_set = self.val_m2m_forward(batch, self.hparams.task)

            # if self.hparams.task not in ["m2t"]:
            #     # MultiModality evaluation sperately
            #     if self.trainer.datamodule.is_mm:
            #         metrics_dicts = ['MMMetrics']
            #     else:
            #         metrics_dicts = self.hparams.metrics_dict
                    
            #     if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
            #         metrics_dicts.remove('PredMetrics')

            #     for metric in metrics_dicts:
            #         lengths = batch['length']
            #         if metric == "TemosMetric":
            #             getattr(self.metrics,
            #                     metric).update(rs_set["joints_rst"],
            #                                    rs_set["joints_ref"], lengths)
            #         elif metric == "TM2TMetrics":
            #             if self.hparams.stage in [
            #                     "lm_instruct", "lm_pretrain", "lm_rl"
            #             ]:
            #                 getattr(self.metrics, metric).update(
            #                     joints_rst=rs_set["joints_rst"], joints_ref=rs_set["joints_ref"],
            #                     vertices_rst=rs_set["vertices_rst"], vertices_ref=rs_set["vertices_ref"],
            #                     lengths=lengths,
            #                 )
            #         elif metric == "UncondMetrics":
            #             getattr(self.metrics, metric).update(
            #                 recmotion_embeddings=rs_set["lat_rm"],
            #                 gtmotion_embeddings=rs_set["lat_m"],
            #                 lengths=lengths,
            #             )
            #         elif metric == "MRMetrics":
            #             getattr(self.metrics,
            #                     metric).update(rs_set["joints_rst"],
            #                                    rs_set["joints_ref"], lengths)
            #         elif metric == "PredMetrics":
            #             getattr(self.metrics,
            #                     metric).update(rs_set["joints_rst"],
            #                                    rs_set["joints_ref"], lengths)
            #         elif metric == "MMMetrics":
            #             # pass
            #             getattr(self.metrics,
            #                     metric).update(rs_set["m_rst"],
            #                                    rs_set['length'])
            #         else:
            #             raise TypeError(f"Not support this metric {metric}")

            # elif self.hparams.task == "m2t" and self.hparams.stage in [
            #         "lm_instruct", "lm_pretrain", "lm_rl"
            # ]:
            #     self.hparams.metrics_dict = metrics_dicts = ['M2TMetrics']
            #     for metric in metrics_dicts:
            #         if metric == "M2TMetrics":
                        # print(rs_set["t_pred"], batch["all_captions"])

        # return forward output rather than loss during test
        if split in ["test"]:
            if self.hparams.stage == "vae":
                # return rs_set["joints_rst"], rs_set["joints_ref"], rs_set["vertices_rst"], rs_set["vertices_ref"], rs_set["m_ref"], rs_set["m_rst"], batch["length"]
                return {'name': name, 'feats_ref': rs_set["m_ref"], 'feats_rst': rs_set['m_rst'], 'lengths': batch['length'], 'lengths_rst': batch['length'], 'text': batch['text']}
            elif "lm" in self.hparams.stage:
                # return rs_set["joints_rst"], rs_set["joints_ref"], rs_set["vertices_rst"], rs_set["vertices_ref"], rs_set["m_ref"], rs_set["m_rst"], \
                    # rs_set_m2t["t_pred"], rs_set_m2t["t_ref"], batch["length"]
                return {'name': name, 'feats_ref': rs_set["m_ref"], 'feats_rst': rs_set['m_rst'], 'lengths': batch['length'], 'lengths_rst': rs_set['lengths_rst'], 'text': batch['text']}
               
        return loss

    # def on_validation_epoch_end(self):
    #     # Log steps and losses
    #     dico = self.step_log_dict()
    #     # Log losses
    #     dico.update(self.loss_log_dict('train'))
    #     dico.update(self.loss_log_dict('val'))
    #     # Log metrics
    #     dico.update(self.metrics_log_dict())
    #     # print('dico', dico)
    #     # Write to log only if not sanity check
    #     if not self.trainer.sanity_checking:
    #         self.log_dict(dico, sync_dist=True, rank_zero_only=True)
    #         # print('dico', dico)
    #     # dist.barrier()
        
