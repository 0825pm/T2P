import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []
    logger = logger

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))
        
    return callbacks

def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
        "how2sign_DTW_MPJPE_PA_lhand": "Metrics/how2sign_DTW_MPJPE_PA_lhand",
        "how2sign_DTW_MPJPE_PA_rhand": "Metrics/how2sign_DTW_MPJPE_PA_rhand",
        "how2sign_DTW_MPJPE_PA_body": "Metrics/how2sign_DTW_MPJPE_PA_body",
        "csl_DTW_MPJPE_PA_lhand": "Metrics/csl_DTW_MPJPE_PA_lhand",
        "csl_DTW_MPJPE_PA_rhand": "Metrics/csl_DTW_MPJPE_PA_rhand",
        "csl_DTW_MPJPE_PA_body": "Metrics/csl_DTW_MPJPE_PA_body",
        "phoenix_DTW_MPJPE_PA_lhand": "Metrics/phoenix_DTW_MPJPE_PA_lhand",
        "phoenix_DTW_MPJPE_PA_rhand": "Metrics/phoenix_DTW_MPJPE_PA_rhand",
        "phoenix_DTW_MPJPE_PA_body": "Metrics/phoenix_DTW_MPJPE_PA_body",
        "how2sign_MPVPE_PA_all": "Metrics/how2sign_MPVPE_PA_all",
        "how2sign_MPJPE_PA_hand": "Metrics/how2sign_MPJPE_PA_hand",
        "csl_MPVPE_PA_all": "Metrics/csl_MPVPE_PA_all",
        "csl_MPJPE_PA_hand": "Metrics/csl_MPJPE_PA_hand",
        "phoenix_MPVPE_PA_all": "Metrics/phoenix_MPVPE_PA_all",
        "phoenix_MPJPE_PA_hand": "Metrics/phoenix_MPJPE_PA_hand",
        "BLEU_1": "Metrics/Bleu_1",
        "BLEU_2": "Metrics/Bleu_2",
        "BLEU_3": "Metrics/Bleu_3",
        "BLEU_4": "Metrics/Bleu_4",
        "ROUGE_L": "Metrics/ROUGE_L",
    }
    callbacks.append(
        progressLogger(logger,metric_monitor=metric_monitor,log_every_n_steps=1))

    # # Save latest checkpoints
    # checkpointParams = {
    #     'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
    #     'filename': "{epoch}",
    #     'monitor': "step",
    #     'mode': "max",
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 8,
    #     'save_last': True,
    #     'save_on_train_epoch_end': True
    # }
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    # # Save checkpoint every n*10 epochs
    # checkpointParams.update({
    #     'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS * 10,
    #     'save_top_k': -1,
    #     'save_last': False
    # })
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': None,  #cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
        'save_last': True, #None,
        'save_on_train_epoch_end': False
    }
    # callbacks.append(ModelCheckpoint(**checkpointParams))

    metrics = cfg.METRIC.TYPE
    metric_monitor_map = {
        'TemosMetric': {
            'Metrics/APE_root': {
                'abbr': 'APEroot',
                'mode': 'min'
            },
        },
        'TM2TMetrics': {
            'Metrics/how2sign_DTW_MPJPE_PA_lhand': {
                'abbr': 'how2sign_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/how2sign_DTW_MPJPE_PA_body': {
            #     'abbr': 'how2sign_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # },
            'Metrics/csl_DTW_MPJPE_PA_lhand': {
                'abbr': 'csl_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/csl_DTW_MPJPE_PA_body': {
            #     'abbr': 'csl_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # }
            'Metrics/phoenix_DTW_MPJPE_PA_lhand': {
                'abbr': 'phoenix_DTW_MPJPE_PA_lhand',
                'mode': 'min'
            },
            # 'Metrics/phoenix_DTW_MPJPE_PA_body': {
            #     'abbr': 'phoenix_DTW_MPJPE_PA_body',
            #     'mode': 'min'
            # }
        },
        'M2TMetrics': {
            'Metrics/Bleu_4': {
                'abbr': 'BLEU_4',
                'mode': 'max'
            },
            'Metrics/ROUGE_L': {
                'abbr': 'ROUGE_L',
                'mode': 'max'
            },
        },
        'MRMetrics': {
            'Metrics/how2sign_MPJPE_PA_hand': {
                'abbr': 'how2sign_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/how2sign_MPVPE_PA_all': {
            #     'abbr': 'how2sign_MPVPE_PA_all',
            #     'mode': 'min'
            # },
            'Metrics/csl_MPJPE_PA_hand': {
                'abbr': 'csl_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/csl_MPVPE_PA_all': {
            #     'abbr': 'csl_MPVPE_PA_all',
            #     'mode': 'min'
            # },
            'Metrics/phoenix_MPJPE_PA_hand': {
                'abbr': 'phoenix_MPJPE_PA_hand',
                'mode': 'min'
            },
            # 'Metrics/phoenix_MPVPE_PA_all': {
            #     'abbr': 'phoenix_MPVPE_PA_all',
            #     'mode': 'min'
            # },
        },
        'HUMANACTMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UESTCMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UncondMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            }
        }
    }

    # checkpointParams.update({
    #     'every_n_epochs': None,  #cfg.LOGGER.VAL_EVERY_STEPS,
    #     'save_top_k': 1,
    # })

    for metric in metrics:
        if metric in metric_monitor_map.keys():
            metric_monitors = metric_monitor_map[metric]

            # Delete R3 if training VAE
            if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
                del metric_monitors['Metrics/R_precision_top_3']

            for metric_monitor in metric_monitors:
                checkpointParams.update({
                    'filename':
                    metric_monitor_map[metric][metric_monitor]['mode']
                    + "-" +
                    metric_monitor_map[metric][metric_monitor]['abbr']
                    + "{epoch}",
                    'monitor':
                    metric_monitor,
                    'mode':
                    metric_monitor_map[metric][metric_monitor]['mode'],
                })
                callbacks.append(
                    ModelCheckpoint(**checkpointParams))
    return callbacks

class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)
