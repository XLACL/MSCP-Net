import os
import os.path as osp
import argparse
import inspect
import sys


def get_config(**kwargs):
    if kwargs.get('exp', None) is None and '--exp' not in sys.argv:
        fn = inspect.currentframe().f_back.f_code.co_filename
        basename = osp.splitext(osp.basename(fn))[0]
        kwargs['exp'] = basename.split('train_')[1] if 'train_' in basename else basename

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--root_path', type=str, default=None)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_iterations', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--patch_size', type=list, default=[256, 256])
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--labeled_bs', type=int, default=12)
    parser.add_argument('--labelnum', type=int, default=10)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=200.0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--round', type=str, default='0')


    # ===== SASNet =====
    parser.add_argument('--use_sab', type=int, default=0,
                        help='1: enable Scale-aware Adaptive Reweighting')
    parser.add_argument('--use_vve', type=int, default=0,
                        help='1: enable View Variance Enhancement')
    parser.add_argument('--use_sdm', type=int, default=0,
                        help='1: enable SDM consistency')
    parser.add_argument('--vve_alpha', type=float, default=0.1,
                        help='VVE: perturbation strength')
    parser.add_argument('--sab_weight', type=float, default=0.1,
                        help='SAB: weight of ensemble dice loss')
    parser.add_argument('--sdm_weight', type=float, default=0.5,
                        help='SDM: weight of SDM consistency loss')
    parser.add_argument('--sdm_warmup', type=int, default=2000,
                        help='SDM: warmup iterations')

    for k, v in kwargs.items():
        if f'--{k}' not in parser._option_string_actions:
            parser.add_argument(f'--{k}', type=type(v), default=v)
        else:
            parser.set_defaults(**{k: v})

    args = parser.parse_args()
    if args.name == "ACDC":
        args.root_path = r'/home/jnu/fcl_data/SSL4MIS/ACDC'
        args.num_classes = 4
    elif args.name == "Synapse":
        args.root_path = r'/home/jnu/fcl_data/SSL4MIS/Synapse'
        args.num_classes = 9
    return args
