import os
import re
import csv
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch import autocast
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key
from samplers.uni_pc import UniPC
from samplers.dpm_solverpp import DPM_SolverPP
from samplers.ipndm import iPNDM
from IPython.display import HTML
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import animation
import torchvision
import torchvision.transforms as transforms
from training.training_loop import get_noise_scheduler
from solver_utils import create_model
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------


@click.command()
# General options
@click.option('--predictor_path',          help='Path to trained iLD3 instructor', metavar='DIR',                   type=str)
@click.option('--model_path',              help='Network filepath', metavar='PATH|URL',                             type=str)
@click.option('--prompt_path',             help='Prompt filepath', metavar='PATH',                                  type=str)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)
@click.option('--use_fp16',                help='Whether to use mixed precision', metavar='BOOL',                   type=bool, default=False)
@click.option('--num_steps',               help='Sampling steps for teacher mode', metavar='INT',                    type=int, default=None)
@click.option('--solver',                  help='Solver for teacher mode', metavar='STR',                            type=str, default=None)
@click.option('--dataset_name',            help='Dataset for teacher mode', metavar='STR',                           type=str, default=None)
@click.option('--guidance_type',           help='Guidance type for teacher mode', metavar='STR',                     type=str, default='none', show_default=True)
@click.option('--guidance_rate',           help='Guidance rate for teacher mode', metavar='FLOAT',                   type=float, default=0.0, show_default=True)
@click.option('--afs',                     help='Use AFS for teacher mode', metavar='BOOL',                          type=bool, default=False, show_default=True)

# Options for sampling
@click.option('--return_inters',           help='Whether to save intermediate outputs', metavar='BOOL',             type=bool, default=False)

# Options for saving
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str)
@click.option('--grid',                    help='Whether to make grid',                                             type=bool, default=False)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         type=bool, default=True, is_flag=True)
@click.option('--noise_schedule', help='Noise schedule', metavar='STR', type=str, default='VE', show_default=True)
@click.option('--no_predictor', help='Whether to use iLD3 predictor', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--no_pred_step', help='num_steps', metavar='INT', type=int, default=30, show_default=True)
@click.option('--schedule_type', help='schedule_type', metavar='STR', type=str, default=None, show_default=True)
@click.option('--schedule_rho', help='schedule_rho', metavar='FLOAT', type=float, default=7, show_default=True)
@click.option('--max_order', help='max_order', metavar='INT', type=int, default=None, show_default=True)
@click.option('--device', help='Device', metavar='STR', type=str, default='cuda', show_default=True)
def main(predictor_path, max_batch_size, seeds, grid, outdir, subdirs, device=torch.device('cuda'), **solver_kwargs):

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Load models.
    if dist.get_rank() != 0:
        torch.distributed.barrier()     # rank 0 goes first


    iLD3_predictor = None
    prompt = solver_kwargs['prompt']
    solver_kwargs = {key: value for key, value in solver_kwargs.items()}

    use_predictor = predictor_path is not None
    if use_predictor:
        dist.print0(f'Loading iLD3 predictor from "{predictor_path}"...')
        with dnnlib.util.open_url(predictor_path, verbose=(dist.get_rank() == 0)) as f:
            iLD3_predictor = pickle.load(f)['model'].to(device)

    if use_predictor:
        if iLD3_predictor.sampler_stu == 'iLD3' :
            solver_kwargs['iLD3_predictor'] = iLD3_predictor
        else:
            solver_kwargs['iLD3_predictor'] = iLD3_predictor
        solver_kwargs['solver'] = solver = iLD3_predictor.sampler_stu
        solver_kwargs['num_steps'] = iLD3_predictor.num_steps
        if solver_kwargs['no_predictor']:
            solver_kwargs['num_steps'] = solver_kwargs['no_pred_step']
        solver_kwargs['guidance_type'] = iLD3_predictor.guidance_type
        solver_kwargs['guidance_rate'] = iLD3_predictor.guidance_rate
        solver_kwargs['afs'] = iLD3_predictor.afs
        solver_kwargs['denoise_to_zero'] = False


        if solver_kwargs['no_predictor']:
            solver_kwargs['max_order'] = solver_kwargs['max_order']
            solver_kwargs['order'] = solver_kwargs['max_order']
        else:
            solver_kwargs['max_order'] = iLD3_predictor.max_order
            solver_kwargs['order'] = iLD3_predictor.max_order

        solver_kwargs['predict_x0'] = iLD3_predictor.predict_x0
        if solver_kwargs['schedule_type'] is not None:
            pass
        else:
            solver_kwargs['schedule_type'] = iLD3_predictor.schedule_type


        solver_kwargs['schedule_rho'] = iLD3_predictor.schedule_rho
        solver_kwargs['prompt'] = prompt
        solver_kwargs['dataset_name'] = dataset_name = iLD3_predictor.dataset_name
    else:
        if solver_kwargs.get('dataset_name') is None:
            raise click.UsageError('Missing --dataset_name when --predictor_path is not provided.')
        if solver_kwargs.get('solver') is None:
            raise click.UsageError('Missing --solver when --predictor_path is not provided.')
        if solver_kwargs.get('model_path') is None:
            raise click.UsageError('Missing --model_path when --predictor_path is not provided.')
        solver_kwargs['dataset_name'] = dataset_name = solver_kwargs['dataset_name']
        solver_kwargs['num_steps'] = solver_kwargs['num_steps'] if solver_kwargs.get('num_steps') is not None else solver_kwargs['no_pred_step']
        solver_kwargs['order'] = solver_kwargs['max_order']
        solver_kwargs['predict_x0'] = False if dataset_name == 'lsun_bedroom_ldm' else True
        if solver_kwargs['schedule_type'] is None:
            solver_kwargs['schedule_type'] = 'logsnr'
        solver_kwargs['schedule_rho'] = solver_kwargs.get('schedule_rho', 7)
        solver_kwargs['prompt'] = prompt
        solver_kwargs['denoise_to_zero'] = False
        solver = solver_kwargs['solver']

    
    # Load pre-trained diffusion models.

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{local_rank}')
    print("loading model on device", device)

    net, solver_kwargs['model_source'] = create_model(dataset_name, solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device, solver_kwargs['model_path'])
    # TODO: support mixed precision 
    # net.use_fp16 = solver_kwargs['use_fp16']

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    if hasattr(net, 'sigma_min'):
        solver_kwargs['sigma_min'] = net.sigma_min
        solver_kwargs['sigma_max'] = net.sigma_max
        img_channels = net.img_channels
        img_resolution = net.img_resolution
    elif dataset_name == 'lsun_bedroom_ldm': # lsun_bedroom_ldm
        solver_kwargs['sigma_min'] = 0.002
        solver_kwargs['sigma_max'] = 80.0
        img_channels = 3
        img_resolution = 64 # [3, 64, 64]
    elif dataset_name == 'ms_coco': # ms_coco
        solver_kwargs['sigma_min'] = 0.002
        solver_kwargs['sigma_max'] = 80.0
        img_channels = 16
        img_resolution = 64 # [16, 128, 128]
    # Align NFE definition with training (`mnfe`): predictor stores
    # `num_steps = mnfe + 1` when AFS is enabled, otherwise `num_steps = mnfe`.
    nfe = solver_kwargs['num_steps'] - (1 if solver_kwargs["afs"] else 0)
    # Keep both values explicit in logs:
    solver_kwargs['nfe'] = nfe
    
    # Initialize selected prompts for MS-COCO dataset
    sample_captions = None
    
    # Load the prompts
    if dataset_name in ['ms_coco'] and solver_kwargs['prompt'] is None:
        # Loading MS-COCO captions for FID-30k evaluaion
        # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
        prompt_path = solver_kwargs['prompt_path']
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)
        
        # No need to pre-select captions - instead we'll use a deterministic mapping
        # based directly on the seed value
        num_captions = len(sample_captions)
        dist.print0(f"Loaded {num_captions} captions. Using deterministic seed-to-prompt mapping.")
    
    noise_schedule = get_noise_scheduler(solver_kwargs['noise_schedule'], net)
    # get noise schedule

    # Construct solver, 5 solvers are provided
    if solver == 'uni_pc':
        sampler_fn = UniPC(noise_schedule)
    elif solver in ['dpm_solverpp', 'dpmpp']:
        sampler_fn = DPM_SolverPP(noise_schedule)   
    elif solver == 'ipndm':
        sampler_fn = iPNDM(noise_schedule)
    elif solver == 'ipndm_flux':
        from samplers.ipndm_flux import iPNDM_Flux
        sampler_fn = iPNDM_Flux(noise_schedule)
    else:
        raise ValueError(f"Unsupported solver: {solver}")
    # Print solver settings.
    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        if value is None:
            continue
        elif key == 'iLD3_predictor':
            continue
        elif key == 'max_order' and solver in ['euler', 'dpm']:
            continue
        elif key in ['predict_x0', 'lower_order_final'] and solver not in ['dpmpp']:
            continue
        elif key in ['prompt'] and dataset_name not in ['ms_coco']:
            continue
        dist.print0(f"\t{key}: {value}")

    # Loop over batches.
    if outdir is None:
        if grid:
            outdir = os.path.join(f"./samples/grids/{dataset_name}", f"{solver}_nfe{nfe}")
        else:
            outdir = os.path.join(f"./samples/{dataset_name}", f"{solver}_nfe{nfe}")
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    if solver_kwargs['no_predictor'] or (not use_predictor): # pure samplers / teacher mode.
        iLD3_predictor = None
    for batch_id, batch_seeds in enumerate(tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0))):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)
        latents = noise_schedule.prior_transformation(latents)

        class_labels = None
        if (hasattr(net, 'label_dim') and net.label_dim) or (dataset_name == 'ms_coco'):
            if solver_kwargs['model_source'] == 'adm':                                              # ADM models
                class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)
            elif solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                assert sample_captions is not None
                
                # Create a deterministic mapping from seeds to prompts
                # This ensures the same seed always gets the same prompt across different runs
                prompts = []
                for seed in batch_seeds:
                    # Use a deterministic function of the seed to select the prompt
                    # Simple modulo ensures consistent mapping
                    prompt_idx = int(seed.item()) % len(sample_captions)
                    prompts.append(sample_captions[prompt_idx])
                
            else: # imagenet64
                class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]

        # Generate images.
        with torch.no_grad():
            if solver_kwargs['model_source'] == 'ldm' and dataset_name == 'lsun_bedroom_ldm':
                with autocast("cuda"):
                    with net.ema_scope():
                        images = sampler_fn(
                            net = net,
                            latents = latents,
                            hypernets = iLD3_predictor,
                            class_labels = class_labels,
                            **solver_kwargs
                        )
                        if isinstance(images, tuple): images = images[0]                                               
                        images = net.decode_first_stage(images)
            elif solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                context = autocast("cuda") if solver_kwargs.get('use_fp16', False) else torch.no_grad()
                with context:
                    inters = sampler_fn(
                        net = net,
                        latents = latents,
                        prompts = prompts,
                        hypernets = iLD3_predictor,
                        guidance = solver_kwargs['guidance_rate'],
                        **solver_kwargs
                    )
                    images = sampler_fn.decode(inters, net['ae'], 512, 512)
            else:
                inters = sampler_fn(
                    net = net,
                    hypernets = iLD3_predictor,
                    latents = latents,
                    class_labels = class_labels,
                    nums_steps = solver_kwargs['num_steps'],
                    **solver_kwargs
                )
                images = inters          
        
        # Save images.
        if grid:
            images = torch.clamp(images / 2 + 0.5, 0, 1)
            os.makedirs(outdir, exist_ok=True)
            nrows = int(images.shape[0] ** 0.5)
            image_grid = make_grid(images, nrows, padding=0)
            save_image(image_grid, os.path.join(outdir, "grid.png"))
        else:
            # if image is not a single tensor
            if isinstance(images, tuple):
                images = images[0]
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

        # plot.visualize_3d_trajectories(net, inters, inters_1, device='cuda', output_dir='./plot')


    torch.distributed.barrier()
    dist.print0('Done.')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
