import torch
from mGPT.utils.misc import neq_load_customized


def load_pretrained(cfg, model, logger=None, phase="train"):    
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
    
    if logger is not None:
        logger.info(f"Loading pretrain model from {ckpt_path}")
        
    # state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    return model


def load_pretrained_vae(cfg, model, logger=None):
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                            map_location="cpu", weights_only=False)['state_dict']

    if logger is not None:
        logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
        
    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    hand_vae_dict = OrderedDict()
    rhand_vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "rhand_vae" in k:
            name = k.replace("rhand_vae.", "")
            rhand_vae_dict[name] = v
        elif "hand_vae" in k:
            name = k.replace("hand_vae.", "")
            hand_vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v
    
    if hasattr(model, 'rhand_vae'):
        print('load rhand vae...')
        neq_load_customized(model.rhand_vae, rhand_vae_dict, verbose=True)
    if hasattr(model, 'hand_vae'):
        print('load hand vae...')
        neq_load_customized(model.hand_vae, hand_vae_dict, verbose=True)
    if hasattr(model, 'vae'):
        print('load vae...')
        # model.vae.load_state_dict(vae_dict, strict=True)
        neq_load_customized(model.vae, vae_dict, verbose=True)
    else:
        # model.motion_vae.load_state_dict(vae_dict, strict=True)
        neq_load_customized(model.motion_vae, vae_dict, verbose=True)
    
    return model
