import argparse
import importlib
from utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-project', type=str, default="baseline_vit")
    parser.add_argument('-dataset', type=str, default='mini_imagenet',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default="../datasets")
    parser.add_argument('-save_path_prefix', "-prefix", type=str, default="")
    parser.add_argument('-save_all_sessions', type=str2bool, default=False,
                        help='Save checkpoints for every session. Default saves only session 0 and the last session.')

    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-way', type=int, default=None, help='Number of new classes per incremental session')
    parser.add_argument('-shot', type=int, default=None, help='Number of training samples per new class')

    parser.add_argument('-epochs_base', type=int, default=10)
    parser.add_argument('-epochs_joint', type=int, default=30)
    parser.add_argument('-lr_base', type=float, default=0.001)
    parser.add_argument('-lr_new', type=float, default=0.0005) 
    parser.add_argument('-lr_scheduler', type=str, default='none',
                        choices=['none', 'cosine', 'warmup_cosine'])
    parser.add_argument('-min_lr_base', type=float, default=0.0)
    parser.add_argument('-min_lr_new', type=float, default=0.0)
    parser.add_argument('-warmup_epochs_base', type=int, default=0)
    parser.add_argument('-warmup_epochs_new', type=int, default=0)
    parser.add_argument('-warmup_start_lr', type=float, default=3e-5)
    parser.add_argument('-optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'])

    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-decay_new', type=float, default=0) 

    parser.add_argument('-base_mode', type=str, default='ft_dot',
                        choices=['ft_dot', 'ft_cos', 'ft_l2', "ft_dot_freeze"]) 

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_train_base', type=int, default=128)
    parser.add_argument('-batch_size_replay', type=int, default=128, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-batch_size_test', type=int, default=256) 
    
    parser.add_argument('-drop_last_batch', action="store_true", help="Drops the last batch if not equal to the assigned batch size")
    parser.add_argument('-exemplars_count', type=int, default=5)
    
    parser.add_argument('-rand_aug_sup_con', action='store_true', help='')
    parser.add_argument('-prob_color_jitter', type=float, default=0.8)
    parser.add_argument('-min_crop_scale', type=float, default=0.2)

    parser.add_argument('-model', type=str, default='vit_base_patch16_224_dino', help='Name of model to train')
    parser.add_argument('-pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('-encoder_outdim', type=int, default=768)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-proj_hidden_dim', type=int, default=2048)
    parser.add_argument('-proj_output_dim', type=int, default=128)
    parser.add_argument('-feature_extractor', type=str, default='forward_combine',
                        choices=['vit', 'forward_block', 'forward_combine'])
    parser.add_argument('-forward_layers', type=str, default='all',
                        help='ViT block indices for Forward Block, comma-separated or "all"')
    parser.add_argument('-forward_token_nums', type=int, default=1)
    parser.add_argument('-forward_hidden_dim', type=int, default=48)
    parser.add_argument('-forward_active_attn_hidden_dim', type=int, default=8)
    parser.add_argument('-forward_attn_scale', type=float, default=0.1)
    parser.add_argument('-forward_ffn_scale', type=float, default=0.1)
    parser.add_argument('-forward_drop', type=float, default=0.1)
    parser.add_argument('-forward_attn_drop', type=float, default=0.1)
    parser.add_argument('-active_ablation', type=str, default='none',
                        choices=['none', 'no_proj', 'no_self_att', 'no_mlp'])
    parser.add_argument('-prompt_mode', type=str, default='prompt_incremental',
                        choices=['no_prompt', 'prompt_only', 'prompt_incremental'])
    parser.add_argument('-incremental_prompt_context_hidden_dim', type=int, default=48)
    parser.add_argument('-incremental_prompt_context_scale', type=float, default=0.1)
    parser.add_argument('-combine_hidden_dim', type=int, default=48)
    parser.add_argument('-combine_drop', type=float, default=0.1)
    parser.add_argument('-combine_ablation', type=str, default='none',
                        choices=['none', 'no_f_att', 'no_f_mlp', 'no_h'])
    parser.add_argument('-fusion_method', type=str, default='conditional_weights',
                        choices=['simple_average', 'fixed_weights', 'conditional_weights'])
    parser.add_argument('-classifier', type=str, default='orco_fagg_mab_project',
                        choices=['fagg_mab_project', 'orco_fagg_mab_project'])
    parser.add_argument('-mab_num_heads', type=int, default=8)
    parser.add_argument('-mab_hidden_dim', type=int, default=2048)
    parser.add_argument('-mab_drop', type=float, default=0.1)
    parser.add_argument('-mab_attn_drop', type=float, default=0.1)
    parser.add_argument('-mab_res_scale', type=float, default=0.1)
    parser.add_argument('-proto_temperature', type=float, default=1.0)
    parser.add_argument('-proto_context_weight', type=float, default=0.5)
    parser.add_argument('-orco_temperature', type=float, default=1.0)
    parser.add_argument('-orco_reserve_mode', type=str, default='all', choices=['all', 'full'])
    parser.add_argument('-orco_sup_lam', type=float, default=1.0)
    parser.add_argument('-orco_cos_lam', type=float, default=1.0)
    parser.add_argument('-orco_simplex_lam', type=float, default=0.1)
    parser.add_argument('-orco_perturb_epsilon', type=float, default=9e-05)
    parser.add_argument('-orco_perturb_offset', type=float, default=0.5)
    parser.add_argument('-orco_target_epochs', type=int, default=1000)
    parser.add_argument('-orco_target_lr', type=float, default=1.0)
    parser.add_argument('-vit_pretrained_path', type=str, default=None)
    parser.add_argument('-output_dir', type=str)
    parser.add_argument('-simple_aug', action='store_true')
    
    return parser

if __name__ == '__main__':
    # Parse Arguments
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    args.num_gpu = set_gpu(args)
    
    # Trainer initialization
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.test()
