from typing import List
import os
import torch
from torch import Tensor
from torch.distributed import all_reduce, ReduceOp
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_mean
from .utils import *


class M2TMetrics(Metric):

    def __init__(self,
                 cfg,
                 w_vectorizer,
                 dataname='humanml3d',
                 top_k=3,
                 bleu_k=4,
                 R_size=32,
                 max_text_len=40,
                 diversity_times=300,
                 dist_sync_on_step=True,
                 unit_length=4,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.w_vectorizer = w_vectorizer
        self.name = "matching, fid, and diversity scores"
        # self.text = True if cfg.TRAIN.STAGE in ["diffusion","t2m_gpt"] else False
        self.max_text_len = max_text_len
        self.top_k = top_k
        self.bleu_k = bleu_k
        self.R_size = R_size
        self.diversity_times = diversity_times
        self.unit_length = unit_length
        self.level = 'word' if dataname != 'csl-daily' else 'char'

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")

        self.metrics = []

        # NLG
        for k in range(1, bleu_k + 1):
            self.add_state(
                f"Bleu_{str(k)}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.metrics.append(f"Bleu_{str(k)}")

        self.add_state("ROUGE_L",
                       default=torch.tensor(0.0),
                       dist_reduce_fx="sum")
        self.metrics.append("ROUGE_L")

        # Chached batches
        self.pred_texts = []
        self.gt_texts = []

        # if self.cfg.model.params.task == 'm2t':
        #     from nlgmetricverse import NLGMetricverse, load_metric
        #     metrics = [
        #         load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
        #         load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
        #         load_metric("rouge"),
        #         load_metric("cider"),
        #     ]
        #     self.nlg_evaluator = NLGMetricverse(metrics)


    @torch.no_grad()
    def compute(self, sanity_flag):
        count = self.count.item()
        count_seq = self.count_seq.item()

        # Init metrics dict
        metrics = {metric: getattr(self, metric) for metric in self.metrics}

        # Jump in sanity check stage
        if sanity_flag:
            return metrics

        print("Computing metrics...")

        # NLP metrics
        bleu_scores = bleu(self.gt_texts, self.pred_texts, level=self.level)
        rouge_score = rouge(self.gt_texts, self.pred_texts, level=self.level)
        for k in range(1, self.bleu_k + 1):
            metrics[f"Bleu_{str(k)}"] = torch.tensor(bleu_scores[f'bleu{str(k)}'], device=self.device)
            all_reduce(metrics[f"Bleu_{str(k)}"], op=ReduceOp.AVG)
            print(f"Bleu_{str(k)}: ", metrics[f"Bleu_{str(k)}"])
            
        metrics["ROUGE_L"] = torch.tensor(rouge_score, device=self.device)
        all_reduce(metrics["ROUGE_L"], op=ReduceOp.AVG)
        print('ROUGE_L: ', metrics["ROUGE_L"])

        # Reset
        self.reset()
        self.gt_texts = []
        self.pred_texts = []

        return {**metrics}

    @torch.no_grad()
    def update(self,
               pred_texts: List[str],
               gt_texts: List[str],
               lengths: List[int],
               src: List[str]
               ):
        # only for monolinfual bakc trans!!
        if src[0] == 'csl':
            self.level = 'char'
        else:
            self.level = 'word'

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        # print(pred_texts, gt_texts, self.level)

        self.pred_texts.extend(pred_texts)
        self.gt_texts.extend(gt_texts)
