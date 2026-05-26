import os
import re
import json
import click
import torch
import dnnlib
import yaml
from pathlib import Path
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

_CONFIG_PATH_KEYS = {'datadir', 'outdir', 'ref_path'}

#----------------------------------------------------------------------------

@click.command()

# General options.
@click.option('--config_path',      help='Path to YAML config file', metavar='PATH',    type=str)
@click.option('--prompt_path',      help='Path to MS-COCO_val2014_30k_captions.csv', metavar='DIR',    type=str)

# --- Options for Offline Data Loading ---

# Options for solvers

# Additional options for multi-step solvers, 1<=max_order<=4 for iPNDM, 1<=max_order<=3 for DPM-Solver++
@click.option('--max_order',        help='max order for solvers', metavar='INT',                       type=click.IntRange(min=1), default=3)
# Additional options for DPM-Solver++
@click.option('--lower_order_final',help='Lower the order at final stages', metavar='BOOL',            type=bool, default=True)

# Hyperparameters.
@click.option('--batch',            help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
# Performance-related.
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--deterministic',    help='Enable deterministic training settings', metavar='BOOL',     type=bool, default=False, show_default=True)

# I/O-related.
@click.option('--desc',             help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',         help='Do not create a subdirectory for results',                   is_flag=False)
@click.option('--tick',             help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--snap',             help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dump',             help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',             help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('-n', '--dry-run',    help='Print training options and exit',                            is_flag=True)
# Our options
@click.option('--coslr',            help='Cosine Annearling Schedule',                                 is_flag=True)

# multistep choices
@click.option('--mnfe', help='Number of FEs for multistep', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--window_rate', help='Window rate', metavar='FLOAT', type=float, default=0.1, show_default=True)
@click.option('--teacher_schedule_type', help='Teacher schedule type', metavar='STR', type=str, default='logsnr', show_default=True)
@click.option('--teacher_schedule_rho', help='Teacher schedule rho', metavar='FLOAT', type=float, default=7, show_default=True)
@click.option('--tn_training_style', help='TN training style', metavar='STR', type=str, default='full', show_default=True)
@click.option('--frequency', help='Frequency', metavar='INT', type=int, default=10, show_default=True)
@click.option('--use_ema', help='Use EMA', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--ema_decay', help='EMA decay', metavar='FLOAT', type=float, default=0.995, show_default=True)
@click.option('--loss_type', help='Loss type', metavar='STR', type=str, default='lpips', show_default=True)
@click.option('--intermediary_loss', help='Intermediary loss', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--mid_loss_type', help='Mid loss type', metavar='STR', type=str, default='l1', show_default=True)
@click.option('--mid_loss_weight', help='Mid loss weight', metavar='FLOAT', type=float, default=0.2, show_default=True)
@click.option('--learn_bound', help='Learn bound', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--l1_reg_weight', help='L1 regularization weight', metavar='FLOAT', type=float, default=0, show_default=True)
@click.option('--l2_reg_weight', help='L2 regularization weight', metavar='FLOAT', type=float, default=0, show_default=True)
@click.option('--loss_weight', help='Loss weight', metavar='STR', type=str, default='constant', show_default=True)
@click.option('--noise_handler', help='Use noise handler', metavar='STR', type=str, default='none', show_default=True)
@click.option('--noise_handler_bound', help='Noise handler bound', metavar='FLOAT', type=float, default=0.01, show_default=True)
@click.option('--noisenethidden_dim', help='Noise net hidden dim', metavar='INT', type=int, default=128, show_default=True)
@click.option('--noisenet_topk', help='Noise net topk', metavar='INT', type=int, default=10, show_default=True)
@click.option('--alter_training', help='alter training', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--guidance_rate', help='Guidance rate', metavar='FLOAT', type=float, default=0, show_default=True)
@click.option('--outdir', help='Output directory', metavar='DIR', type=str, default='./', show_default=True)
@click.option('--ema_decay_list_kimg', help='EMA decay list kimg', metavar='LIST', type=str, default=None, show_default=True)
@click.option('--run_valid', help='whether to run validation', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--datadir', help='Data directory', metavar='DIR', type=str, default=None, show_default=True)
@click.option('--train_num', help='Train number', metavar='INT', type=int, default=None, show_default=True)
@click.option('--valid_num', help='Validation number', metavar='INT', type=int, default=None, show_default=True)
@click.option('--num_workers', help='Number of DataLoader workers', metavar='INT', type=click.IntRange(min=0), default=4, show_default=True)
@click.option('--sampler_stu', help='Sampler stu', metavar='STR', type=str, default="ipndm", show_default=True)
@click.option('--lr_param', help='Learning rate for schedule parameters', metavar='FLOAT', type=float, default=0.05, show_default=True)
@click.option('--lr_net', help='Learning rate for noise handler parameters', metavar='FLOAT', type=float, default=0.05, show_default=True)
@click.option('--max_grad_norm', help='Gradient clipping norm', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--weight_decay_param', help='Weight decay for schedule parameters', metavar='FLOAT', type=float, default=0.0, show_default=True)
@click.option('--weight_decay_net', help='Weight decay for noise handler parameters', metavar='FLOAT', type=float, default=0.0, show_default=True)
@click.option('--warmup_start_lr', help='Warmup starting LR', metavar='FLOAT', type=float, default=1e-4, show_default=True)
@click.option('--eta_min', help='Minimum LR for cosine schedule', metavar='FLOAT', type=float, default=1e-5, show_default=True)
@click.option('--warmup_ratio', help='Warmup fraction of total steps', metavar='FLOAT', type=float, default=0.1, show_default=True)
@click.option('--lr_scheduler_type', help='LR scheduler type', metavar='STR', type=click.Choice(['cosine', 'plateau'], case_sensitive=False), default='cosine', show_default=True)
@click.option('--plateau_factor', help='ReduceLROnPlateau multiplicative factor', metavar='FLOAT', type=float, default=0.5, show_default=True)
@click.option('--plateau_patience', help='Validation ticks without improvement before LR reduction', metavar='INT', type=int, default=1, show_default=True)
@click.option('--plateau_threshold', help='Minimum validation improvement to avoid plateau LR reduction', metavar='FLOAT', type=float, default=0.0, show_default=True)
@click.option('--plateau_min_lr', help='Minimum LR for ReduceLROnPlateau', metavar='FLOAT', type=float, default=1e-5, show_default=True)
@click.option('--early_stop_patience', help='Validation ticks without improvement before early stop; <=0 disables', metavar='INT', type=int, default=0, show_default=True)
@click.option('--early_stop_min_delta', help='Minimum validation improvement to reset early stopping', metavar='FLOAT', type=float, default=0.0, show_default=True)
@click.option('--debug_visualize', help='Save debug student/teacher images during loss computation', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--init_predictor_path', help='Initialize predictor weights from a saved checkpoint', metavar='PATH', type=str, default=None, show_default=True)
@click.pass_context
def main(ctx, **kwargs):
    opts = dnnlib.EasyDict(kwargs)

    # Load configuration from YAML and merge with command-line args
    opts = load_and_merge_config(opts, ctx)


    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.iLD3_kwargs = dnnlib.EasyDict()
    # conver it to adamw
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.9,0.999], eps=1e-8)
    c.lr_param = opts.lr_param
    c.lr_net = opts.lr_net
    c.cos_lr_schedule = opts.coslr
    c.lr_scheduler_type = opts.lr_scheduler_type
    c.deterministic = opts.deterministic

    c.model_path = opts.model_path
    c.run_valid = opts.run_valid
    c.init_predictor_path = opts.init_predictor_path

    # --- Pass offline data options --- 
    c.datadir = opts.datadir
    c.train_num = opts.train_num
    c.valid_num = opts.valid_num
    c.num_workers = opts.num_workers

    # iLD3 predictor architecture.
    

    c.loss_kwargs.class_name = 'training.loss.iLD3_loss'
    c.loss_kwargs.update(
        num_steps=opts.mnfe+(1 if opts.afs else 0),
        sampler_stu=opts.sampler_stu,
        schedule_type=opts.schedule_type,
        schedule_rho=opts.schedule_rho,
        afs=opts.afs,
        max_order=opts.max_order,
        sigma_min=None,  # Will be set from net in training loop
        sigma_max=None,  # Will be set from net in training loop
        predict_x0=opts.predict_x0,
        lower_order_final=opts.lower_order_final,
        iLD3_predictor=None,  # Will be set in training loop
        teacher_schedule_type=opts.teacher_schedule_type,
        teacher_schedule_rho=opts.teacher_schedule_rho,
        tn_training_style=opts.tn_training_style,
        loss_type=opts.loss_type,
        intermediary_loss=opts.intermediary_loss,
        mid_loss_type=opts.mid_loss_type,
        mid_loss_weight=opts.mid_loss_weight,
        l1_reg_weight=opts.l1_reg_weight,
        l2_reg_weight=opts.l2_reg_weight,
        loss_weight=opts.loss_weight,
        alter_training=opts.alter_training,
        debug_visualize=opts.debug_visualize,
    )

    c.iLD3_kwargs.class_name = 'training.networks.iLD3_predictor'
    c.iLD3_kwargs.update(
        dataset_name=opts.dataset_name,
        img_resolution=None,  # Will be set from net in training loop
        num_steps=opts.mnfe+(1 if opts.afs else 0),
        sampler_stu=opts.sampler_stu,
        guidance_type=opts.guidance_type,
        guidance_rate=opts.guidance_rate,
        schedule_type=opts.schedule_type,
        schedule_rho=opts.schedule_rho,
        afs=opts.afs,
        mode=opts.mode,
        bound_tn=opts.bound_tn,
        bound_tn2=opts.bound_tn2,
        bound_epscaling=opts.bound_epscaling,
        max_order=opts.max_order,
        predict_x0=opts.predict_x0,
        lower_order_final=opts.lower_order_final,
        noise_schedule=opts.noise_schedule,
        window_rate=opts.window_rate,
        use_ema=opts.use_ema,
        ema_decay_list_kimg=opts.ema_decay_list_kimg,
        ema_decay=opts.ema_decay,
        learn_bound=opts.learn_bound,
        noise_handler=opts.noise_handler,
        noise_handler_bound=opts.noise_handler_bound,
        noisenethidden_dim=opts.noisenethidden_dim,
        noisenet_topk=opts.noisenet_topk
    )
    # Training options.
    c.total_kimg = opts.total_kimg      # Train for total_kimg k trajectories
    c.kimg_per_tick = opts.tick
    c.snapshot_ticks = opts.snap
    c.state_dump_ticks = opts.dump
    c.frequency = opts.frequency
    c.update(dataset_name=opts.dataset_name, batch_size=opts.batch, batch_gpu=opts.batch_gpu, gpus=dist.get_world_size(), cudnn_benchmark=opts.bench)
    c.update(guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate, prompt_path=opts.prompt_path)
    c.max_grad_norm = opts.max_grad_norm
    c.weight_decay_param = opts.weight_decay_param
    c.weight_decay_net = opts.weight_decay_net
    c.warmup_start_lr = opts.warmup_start_lr
    c.eta_min = opts.eta_min
    c.warmup_ratio = opts.warmup_ratio
    c.plateau_factor = opts.plateau_factor
    c.plateau_patience = opts.plateau_patience
    c.plateau_threshold = opts.plateau_threshold
    c.plateau_min_lr = opts.plateau_min_lr
    c.early_stop_patience = opts.early_stop_patience
    c.early_stop_min_delta = opts.early_stop_min_delta
    

    c.noise_schedule = opts.noise_schedule

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Description string.
    if opts.schedule_type == 'polynomial':
        schedule_str = 'polynomial' # 'poly' + str(opts.schedule_rho)
    elif opts.schedule_type == 'logsnr':
        schedule_str = 'logsnr'
    elif opts.schedule_type == 'time_uniform':
        schedule_str = 'time_uniform'
    elif opts.schedule_type == 'discrete':
        schedule_str = 'discrete'
    else:
        raise ValueError("Got wrong schedule type: {}".format(opts.schedule_type))
    # Calculate required NFE


    c.num_step = opts.mnfe + (1 if opts.afs else 0)
    nfe = opts.mnfe
    nfe = 2 * nfe if opts.dataset_name == 'ms_coco' else nfe


    if opts.afs == True:
        desc = f'{opts.dataset_name:s}-{opts.mnfe}-{opts.sampler_stu}-{schedule_str}-afs'
    else:
        desc = f'{opts.dataset_name:s}-{opts.mnfe}-{opts.sampler_stu}-{schedule_str}'
    if opts.desc is not None:
        desc += f'{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        print(dist.get_rank(),desc)
        
        c.run_dir = os.path.join(opts.outdir, desc)
        if os.path.exists(c.run_dir):
            import shutil
            shutil.rmtree(c.run_dir)  # Remove existing directory
        os.makedirs(c.run_dir)  # Create fresh directory

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Number of workers:       {c.num_workers}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.

    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

def update_opts(opts, kwargs):
    for key, value in kwargs.items():
        opts[key] = value
    return opts


# Helper function defined AFTER main
def load_and_merge_config(opts, ctx):
    """Loads configuration from YAML file specified in opts.config_path and merges it into opts."""
    if 'config_path' in opts and opts.config_path is not None:
        config_path_value = opts.config_path # Store before potential deletion
        try:
            config_path = Path(config_path_value).resolve()
            with open(config_path, 'r') as f:
                config_from_yaml = yaml.safe_load(f)
            # Merge YAML config into Click defaults while preserving explicit CLI/env values.
            for key, value in config_from_yaml.items():
                if isinstance(value, str):
                    value = os.path.expanduser(os.path.expandvars(value))
                    if key.endswith('_path') or key.endswith('_dir') or key in _CONFIG_PATH_KEYS:
                        value_path = Path(value)
                        if not value_path.is_absolute():
                            value = str((config_path.parent / value_path).resolve())
                if key in opts:
                    source = ctx.get_parameter_source(key)
                    if str(source) in {'ParameterSource.DEFAULT', 'ParameterSource.DEFAULT_MAP'}:
                        opts[key] = value
                else:
                    opts[key] = value
            dist.print0(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            dist.print0(f"Warning: Config file not found at {config_path_value}. Using command-line arguments only.")
        except yaml.YAMLError as exc:
            dist.print0(f"Error parsing YAML file: {exc}. Using command-line arguments only.")

        # Remove config_path from opts after loading, as it's not needed further
        if 'config_path' in opts:
             del opts['config_path'] # remove the key itself

    return opts

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
