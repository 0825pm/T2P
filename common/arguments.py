import argparse
import os
import math
import time
import torch
import socket
import sys
import logging

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--channel', default=512, type=int)
    parser.add_argument('--d_hid', default=1024, type=int)
    parser.add_argument('--frames', type=int, default=243)
    parser.add_argument('--max_len', type=int, default=500)
    parser.add_argument('--pad', type=int, default=121) 
    parser.add_argument('--token_num', default=81, type=int)
    parser.add_argument('--layer_index', default=3, type=int)
    parser.add_argument('--dataset', type=str, default='h36m')
    parser.add_argument('--keypoints', default='cpn_ft_h36m_dbb', type=str)
    parser.add_argument('--data_augmentation', type=int, default=1)
    parser.add_argument('--reverse_augmentation', type=bool, default=False)
    parser.add_argument('--test_augmentation', type=bool, default=True)
    parser.add_argument('--crop_uv', type=int, default=0)
    parser.add_argument('--root_path', type=str, default='/home/user/Projects/research/HoT/dataset/')
    parser.add_argument('--actions', default='*', type=str)
    parser.add_argument('--downsample', default=1, type=int)
    parser.add_argument('--subset', default=1, type=float)
    parser.add_argument('--stride', default=243, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--train', default=1, type=int)
    # parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--nepoch', type=int, default=160)
    # parser.add_argument('--nepoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=4e-5)
    # parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay_large', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=int, default=200)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('-lrd', '--lr_decay', default=0.99, type=float)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--previous_dir', type=str, default='')
    parser.add_argument('--n_joints', type=int, default=17)
    parser.add_argument('--out_joints', type=int, default=17)
    parser.add_argument('--out_all', type=int, default=1)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--previous_best', type=float, default= math.inf)
    parser.add_argument('--previous_best_dtw', type=float, default= math.inf)
    parser.add_argument('--previous_best_p2', type=float, default= 0)
    parser.add_argument('--previous_best_dtw_p2', type=float, default= 0)
    parser.add_argument('--previous_name', type=str, default='')

    
    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    # parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
    #                     help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    parser.add_argument('--config', type=str, default='/home/user/Projects/research/HoT/configs/research.yaml')
    parser.add_argument('--exp_name', type=str, default='train')

    
    args = parser.parse_args()

    if args.test:
        args.train = 0

    args.pad = (args.frames-1) // 2

    args.root_joint = 0
    args.subjects_train = 'S1,S5,S6,S7,S8'
    args.subjects_test = 'S9,S11'

    if args.train:
        logtime = time.strftime('%m%d_%H%M_%S_')
        if args.train:
            args.checkpoint = 'checkpoint/' + args.exp_name + '_' + logtime
        os.makedirs(args.checkpoint, exist_ok=True)

        args_write = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))

        file_name = os.path.join(args.checkpoint, 'configs.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(args_write.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)

    return args

