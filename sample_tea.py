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
        # Ensure seeds are integers and within the valid range for torch.Generator
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 31)) for seed in seeds] # Use 1<<31 for safety

    def randn(self, size, **kwargs):
        if len(self.generators) == 0:
             return torch.empty(0, *size[1:], **kwargs) # Handle empty batch
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        if len(self.generators) == 0:
            # Handle empty batch case for randint if necessary, depends on usage
            # Returning an empty tensor of appropriate shape might be needed
             return torch.empty(0, *size[1:], dtype=kwargs.get('dtype', torch.long))
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
# --- This function seems unused now, but kept for potential future use ---
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
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion)
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)



#----------------------------------------------------------------------------



#----------------------------------------------------------------------------


@click.command()
# General options
@click.option('--model_path',              help='Network filepath', metavar='PATH|URL',                             type=str)
@click.option('--prompt_path',             help='Prompt filepath', metavar='PATH',                                  type=str)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)
@click.option('--use_fp16',                help='Whether to use mixed precision', metavar='BOOL',                   type=bool, default=False) # TODO: Support fp16

# Options for sampling
@click.option('--seeds',                   help='Fixed random seeds (e.g. 1,2,5-10)', metavar='LIST',                type=parse_int_list, default=None)
@click.option('--return_inters',           help='Whether to save intermediate outputs', metavar='BOOL',             type=bool, default=False)

# Options for saving
@click.option('--outdir',                  help='Where to save the output images and conditions', metavar='DIR',    type=str, required=True) # Made outdir required
@click.option('--noise_schedule', help='Noise schedule', metavar='STR', type=str, default='VE', show_default=True)

# Options for data generation
@click.option('--data_num', help='Number of training data images to generate', metavar='INT', type=int, default=1000, show_default=True)
@click.option('--validation_num', help='Number of validation data images to generate', metavar='INT', type=int, default=500, show_default=True)

# Options for solver
@click.option('--num_steps', help='Number of steps', metavar='INT', type=int, default=30, show_default=True)
@click.option('--solver', help='Solver', metavar='STR', type=str, default='ipndm', show_default=True)
@click.option('--dataset_name', help='Dataset name', metavar='STR', type=str, default='cifar10', show_default=True)
@click.option('--guidance_type', help='Guidance type', metavar='STR', type=str, default='uncond', show_default=True)
@click.option('--guidance_rate', help='Guidance rate', metavar='FLOAT', type=float, default=4.0, show_default=True)
@click.option('--schedule_type', help='Schedule type', metavar='STR', type=str, default='logsnr', show_default=True)
@click.option('--schedule_rho', help='Schedule rho', metavar='FLOAT', type=float, default=7, show_default=True)
@click.option('--max_order', help='Max order', metavar='INT', type=int, default=3, show_default=True)
@click.option('--afs', help='Whether to use AFS', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--device', help='Device', metavar='STR', type=str, default='cuda', show_default=True)
def main(max_batch_size, outdir, data_num, validation_num, device=torch.device('cuda'), **solver_kwargs):

    dist.init()
    #--------------------------------------------------------------------------
    # Generate unique seeds for training and validation on rank 0
    #--------------------------------------------------------------------------
    train_seeds_list = []
    valid_seeds_list = []
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device(f'cuda:{local_rank}')
    if dist.get_rank() == 0:
        total_seeds_needed = data_num + validation_num
        max_seed_val = (1 << 31)
        if total_seeds_needed > max_seed_val:
             raise ValueError(f"Requested {total_seeds_needed} seeds, but max is {max_seed_val}")

        provided_seeds = solver_kwargs.get('seeds', None)
        if provided_seeds is not None:
            if len(provided_seeds) != total_seeds_needed:
                raise ValueError(
                    f"Expected {total_seeds_needed} seeds, but got {len(provided_seeds)} from --seeds."
                )
            all_seeds_np = np.asarray(provided_seeds, dtype=np.int64)
            train_seeds_np = all_seeds_np[:data_num]
            valid_seeds_np = all_seeds_np[data_num:]
            dist.print0(
                f"Using fixed seeds: {train_seeds_np[0]}-{train_seeds_np[-1] if len(train_seeds_np) > 0 else 'N/A'} "
                f"for train and {len(valid_seeds_np)} validation seeds."
            )
        else:
            dist.print0(f"Generating {data_num} training seeds and {validation_num} validation seeds...")
            rng = np.random.default_rng()
            all_seeds_np = rng.choice(max_seed_val, size=total_seeds_needed, replace=False)
            train_seeds_np = all_seeds_np[:data_num]
            valid_seeds_np = all_seeds_np[data_num:]
            dist.print0(f"Generated {len(train_seeds_np)} unique training seeds and {len(valid_seeds_np)} unique validation seeds.")

        train_seeds_list = train_seeds_np.tolist()
        valid_seeds_list = valid_seeds_np.tolist()
        seeds_obj = [train_seeds_list, valid_seeds_list]
        torch.distributed.broadcast_object_list(seeds_obj, src=0, device=device) # Use appropriate device

    else:
        seeds_obj = [None, None]
        torch.distributed.broadcast_object_list(seeds_obj, src=0, device=device) # Use appropriate device
        train_seeds_list = seeds_obj[0]
        valid_seeds_list = seeds_obj[1]

    # Combine seeds for processing and create a lookup set for train seeds
    all_seeds_to_process = train_seeds_list + valid_seeds_list
    train_seed_set = set(train_seeds_list)

    if not all_seeds_to_process:
        dist.print0("No seeds to process. Exiting.")
        torch.distributed.barrier()
        return

    #--------------------------------------------------------------------------
    # Prepare batches for distributed processing
    #--------------------------------------------------------------------------
    all_seeds_tensor = torch.tensor(all_seeds_to_process, dtype=torch.long)
    num_batches = ((len(all_seeds_tensor) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    if num_batches == 0 and len(all_seeds_tensor) > 0:
        num_batches = dist.get_world_size()
    elif len(all_seeds_tensor) == 0:
        num_batches = 0

    if num_batches > 0:
        all_batches = all_seeds_tensor.tensor_split(num_batches)
        rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    else:
        rank_batches = []


    #--------------------------------------------------------------------------
    # Load models (rank 0 first)
    #--------------------------------------------------------------------------
    if dist.get_rank() != 0:
        torch.distributed.barrier()     # rank 0 goes first

    # Update settings from solver_kwargs
    prompt_arg = solver_kwargs.get('prompt', None) # Specific prompt from argument
    solver_kwargs = {key: value for key, value in solver_kwargs.items() if value is not None}
    solver = solver_kwargs['solver']
    solver_kwargs['denoise_to_zero'] = False
    dataset_name = solver_kwargs['dataset_name']
    solver_kwargs['predict_x0'] = False if dataset_name == 'lsun_bedroom_ldm' else True
    solver_kwargs['lower_order_final'] = True
    solver_kwargs['prompt'] = prompt_arg # Keep the specific prompt if provided
    
    guidance_type = solver_kwargs['guidance_type'] # Get guidance type for logic below

    # Load pre-trained diffusion models.

    print("loading model on device", device)
    net, model_source = create_model(dataset_name, guidance_type, solver_kwargs['guidance_rate'], device, solver_kwargs['model_path'])
    solver_kwargs['model_source'] = model_source # Store model source for later use

    if dist.get_rank() == 0:
        torch.distributed.barrier() # Other ranks follow.

    # Update solver settings based on loaded model
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
        
    solver_kwargs['nfe'] = solver_kwargs['num_steps'] # for multistep sampling, nfe is the number of steps

    # Load prompts if needed for MS-COCO and no specific prompt was given
    sample_captions = None # Initialize
    if model_source == 'ldm' and dataset_name == 'ms_coco' and prompt_arg is None:
        prompt_path = solver_kwargs['prompt_path']
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)
        dist.print0(f"Loaded {len(sample_captions)} captions for MS-COCO.")
        # Note: Using captions based on seed index might be complex with train/valid split.
        # The current logic might need adjustment if specific captions per seed are required.
        # Simplification: If prompt is None, maybe use a default generic prompt or require a specific prompt for MS-COCO.


    noise_schedule = get_noise_scheduler(solver_kwargs['noise_schedule'], net)


    # Construct solver
    if solver == 'uni_pc':
        sampler_fn = UniPC(noise_schedule)
    elif solver == 'dpmpp':
        sampler_fn = DPM_SolverPP(noise_schedule)
    elif solver == 'ipndm':
        sampler_fn = iPNDM(noise_schedule)
    elif solver == 'ipndm_flux':
        from samplers.ipndm_flux import iPNDM_Flux
        sampler_fn = iPNDM_Flux(noise_schedule)
    else:
        raise ValueError(f"Unsupported solver: {solver}")

    # --- Determine Condition Type and Header ---
    condition_type = 'none'
    csv_header = ['seed']
    if (hasattr(net, 'label_dim') and net.label_dim) or (dataset_name == 'ms_coco'): # Check if the loaded model uses any form of conditioning labels
        if model_source == 'adm': # ADM uses class index directly
            condition_type = 'class_label'
            csv_header = ['seed', 'class_label']
        elif model_source == 'ldm' and dataset_name == 'ms_coco': # for flux ldm
            condition_type = 'prompt'
            csv_header = ['seed', 'prompt']
        # Add elif conditions for other specific models if they use labels differently
        else: # Fallback for other conditional models (assume class index)
            condition_type = 'class_label'
            csv_header = ['seed', 'class_label']
            dist.print0(f"Warning: Assuming 'class_label' index conditioning for model source '{model_source}' with label_dim={net.label_dim}. Verify if correct.")

    dist.print0(f"Condition type detected: {condition_type}")

    # Print solver settings.
    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        # --- Cleaned up print logic slightly ---
        if value is None: continue
        if key == 'seeds':
            if len(value) == 0:
                dist.print0("\tseeds: []")
            else:
                dist.print0(f"\tseeds: {value[0]}-{value[-1]} (count={len(value)})")
            continue
        # Example conditions for skipping irrelevant options
        if key == 'max_order' and solver not in ['dpmpp', 'uni_pc', 'ipndm']: continue # Adjust relevant solvers
        # if key in ['predict_x0', 'lower_order_final'] and solver not in ['dpmpp']: continue # Adjust relevant solvers
        if key in ['prompt'] and condition_type != 'prompt': continue # Only show prompt if relevant
        dist.print0(f"\t{key}: {value}")


    # Create output directories
    train_dir = os.path.join(outdir, 'train')
    valid_dir = os.path.join(outdir, 'valid')
    if dist.get_rank() == 0:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        dist.print0(f'Generating images...')
        dist.print0(f' - Training images ({data_num}) will be saved to "{train_dir}"')
        dist.print0(f' - Validation images ({validation_num}) will be saved to "{valid_dir}"')
        if condition_type != 'none':
            dist.print0(f' - Conditioning info ({condition_type}) will be saved to conditions.csv in respective folders.')


    #--------------------------------------------------------------------------
    # Loop over batches and generate images / collect conditions
    #--------------------------------------------------------------------------
    processed_seeds = 0
    local_train_conditions = [] # List to store {'seed': s, 'condition': cond} for train
    local_valid_conditions = [] # List to store {'seed': s, 'condition': cond} for valid

    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier() # Sync before each batch
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents.
        rnd = StackedRandomGenerator(device, batch_seeds.tolist())
        latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)
        
        latents = noise_schedule.prior_transformation(latents)
        
        # Determine conditions for the batch
        class_labels = None # This will hold the tensor needed by the sampler_fn
        prompts = None      # Store text prompts if used
        conditions_for_csv = [None] * batch_size # Store the condition value (label index or prompt string) for the CSV

        if condition_type == 'class_label':
            # Generate integer class labels first
            label_indices = rnd.randint(net.label_dim, size=(batch_size,), device=device)
            conditions_for_csv = label_indices.cpu().tolist() # Store the integer labels for CSV
            # Prepare the format needed by the specific model
            if model_source == 'adm':
                class_labels = label_indices # ADM uses indices directly
            elif model_source == 'edm': # EDM (CIFAR10, ImageNet64) uses one-hot
                 class_labels = torch.eye(net.label_dim, device=device)[label_indices]
            else: # Fallback or other models that need indices
                 class_labels = label_indices

        elif condition_type == 'prompt':
            # Handle prompts for LDM/Stable Diffusion
            assert sample_captions is not None
            prompt_indices = torch.randint(len(sample_captions), size=(batch_size,), device=device) # Sample indices
            prompts = [sample_captions[i] for i in prompt_indices.cpu().tolist()]
            conditions_for_csv = prompts # Store the sampled prompts for CSV

        # Generate images.
        images = None
        inters = None
        with torch.no_grad():
            # --- Image Generation Logic ---
            # Pass the correctly formatted condition (class_labels, c, uc)
            if model_source == 'ldm' and dataset_name == 'lsun_bedroom_ldm':
                context = autocast("cuda") if solver_kwargs.get('use_fp16', False) else torch.no_grad()
                with context:
                    with net.ema_scope():
                        inters = sampler_fn(net = net, 
                                            latents = latents, 
                                            class_labels = class_labels, 
                                            order = solver_kwargs['max_order'],
                                            **solver_kwargs)
                        images = net.decode_first_stage(inters)

            elif model_source == 'ldm' and dataset_name == 'ms_coco': # for flux
                context = autocast("cuda") if solver_kwargs.get('use_fp16', False) else torch.no_grad()
                with context:
                    inters = sampler_fn(
                        net = net,
                        latents = latents,
                        prompts = prompts,
                        guidance = solver_kwargs['guidance_rate'],
                        order = solver_kwargs['max_order'],
                        **solver_kwargs
                    )
                    images = sampler_fn.decode(inters, net['ae'], 512, 512)
                    
            else:
                 inters = sampler_fn(
                    net = net,
                    latents = latents,
                    class_labels = class_labels, # Pass class labels (indices or one-hot)
                    verbose = (dist.get_rank() == 0 and processed_seeds == 0),
                    **solver_kwargs
                 )
                 images = inters

        if images is None:
            dist.print0(f"Warning: Image generation failed for batch with seeds starting {batch_seeds[0]}. Skipping saving.")
            continue

        # Post-process and save images and collect conditions
        images_np = (images.clamp(-1, 1) * 127.5 + 127.5).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        batch_seeds_list = batch_seeds.tolist()
        for i, seed_val in enumerate(batch_seeds_list):
            image_np = images_np[i]
            condition_value = conditions_for_csv[i] # Get condition for this seed (label index or prompt string)

            # Determine output directory and condition list
            if seed_val in train_seed_set:
                current_out_dir = train_dir
                condition_list_to_append = local_train_conditions
            else:
                current_out_dir = valid_dir
                condition_list_to_append = local_valid_conditions

            # Save the image
            image_path = os.path.join(current_out_dir, f'{seed_val:06d}.png')
            try:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
                # Append condition info if applicable
                if condition_type != 'none':
                    # Use the second header column name (e.g., 'class_label' or 'prompt') as the key
                     condition_list_to_append.append({'seed': seed_val, csv_header[1]: condition_value})
            except Exception as e:
                 dist.print0(f"Error saving image or collecting condition for {image_path}: {e}")

        processed_seeds += batch_size * dist.get_world_size() # Estimate total processed count


    #--------------------------------------------------------------------------
    # Gather condition information and write CSVs on Rank 0
    #--------------------------------------------------------------------------
    torch.distributed.barrier() # Ensure all ranks finished processing

    all_train_conditions = []
    all_valid_conditions = []

    if condition_type != 'none':
        if dist.get_rank() == 0:
            # Create lists to gather objects from all ranks
            gathered_train_conditions = [None for _ in range(dist.get_world_size())]
            gathered_valid_conditions = [None for _ in range(dist.get_world_size())]
            # Gather
            torch.distributed.gather_object(local_train_conditions, gathered_train_conditions if dist.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(local_valid_conditions, gathered_valid_conditions if dist.get_rank() == 0 else None, dst=0)

            # Process gathered data on rank 0
            dist.print0("Gathered condition information. Writing CSVs...")
            # Flatten the lists of lists
            for rank_list in gathered_train_conditions:
                 if rank_list: all_train_conditions.extend(rank_list)
            for rank_list in gathered_valid_conditions:
                 if rank_list: all_valid_conditions.extend(rank_list)

            # Sort by seed for consistent order (optional but good practice)
            all_train_conditions.sort(key=lambda x: x['seed'])
            all_valid_conditions.sort(key=lambda x: x['seed'])

            # Write Train CSV
            train_csv_path = os.path.join(train_dir, 'conditions.csv')
            try:
                with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_header)
                    writer.writeheader()
                    writer.writerows(all_train_conditions)
                dist.print0(f"Saved training conditions to {train_csv_path}")
            except Exception as e:
                 dist.print0(f"Error writing training conditions CSV: {e}")

            # Write Valid CSV
            valid_csv_path = os.path.join(valid_dir, 'conditions.csv')
            try:
                with open(valid_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_header)
                    writer.writeheader()
                    writer.writerows(all_valid_conditions)
                dist.print0(f"Saved validation conditions to {valid_csv_path}")
            except Exception as e:
                 dist.print0(f"Error writing validation conditions CSV: {e}")

        else:
            # Participate in gathering
            torch.distributed.gather_object(local_train_conditions, None, dst=0)
            torch.distributed.gather_object(local_valid_conditions, None, dst=0)

    # Final barrier
    torch.distributed.barrier()
    dist.print0('Done generating images and conditions.')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
