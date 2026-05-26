"""Main training loop."""
import os
import csv
import time
import copy
import json
import pickle
import numpy as np
import torch
import dnnlib
import random
import glob
from PIL import Image
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from models.ldm.util import instantiate_from_config
from torch_utils.download_util import check_file_by_key
import matplotlib.pyplot as plt
import torch.nn as nn
from training.lr_scheduler import LinearWarmupCosineAnnealingLR
from noise_schedulers import NoiseScheduleVE, NoiseScheduleVP
import torchvision.transforms as transforms
from solver_utils import create_model

#----------------------------------------------------------------------------
# Helper function to extract seed from filename
def get_seed_from_filename(filename):
    """Extracts the seed (assumed to be the number before .png) from a filename."""
    try:
        basename = os.path.basename(filename)
        seed_str = os.path.splitext(basename)[0]
        return int(seed_str)
    except Exception as e:
        print(f"Warning: Could not parse seed from filename {filename}: {e}")
        return None

#----------------------------------------------------------------------------
def normalize_class_label_batch(batch_conditions):
    """Convert collated class labels into a list of Python ints."""
    if batch_conditions is None:
        return None

    if isinstance(batch_conditions, torch.Tensor):
        if batch_conditions.ndim == 0:
            return [int(batch_conditions.item())]
        return [int(x) for x in batch_conditions.detach().cpu().tolist()]

    if isinstance(batch_conditions, np.ndarray):
        if batch_conditions.ndim == 0:
            return [int(batch_conditions.item())]
        return [int(x) for x in batch_conditions.tolist()]

    if isinstance(batch_conditions, (list, tuple)):
        normalized = []
        for condition in batch_conditions:
            if condition is None or condition == '':
                return None
            if isinstance(condition, torch.Tensor):
                if condition.numel() != 1:
                    return None
                normalized.append(int(condition.item()))
            elif isinstance(condition, np.generic):
                normalized.append(int(condition.item()))
            else:
                try:
                    normalized.append(int(condition))
                except (TypeError, ValueError):
                    return None
        return normalized

    try:
        return [int(batch_conditions)]
    except (TypeError, ValueError):
        return None

#----------------------------------------------------------------------------
# Dataset class for loading pre-generated images and their seeds
class PreGeneratedDataset(Dataset):
    def __init__(self, data_dir, num_expected=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, '**', '*.png'), recursive=True)) # Assuming PNG format

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.data_dir}")

        if num_expected is not None and len(self.image_paths) != num_expected:
            print(f"Warning: Found {len(self.image_paths)} images in {self.data_dir}, but expected {num_expected}.")
            # Decide how to handle this: error, truncate, or use found images?
            # Using found images for now.
            self.image_paths = self.image_paths[:num_expected] # Option: Truncate

        self.seeds = [get_seed_from_filename(p) for p in self.image_paths]
        # Filter out paths where seed extraction failed
        # Also keep track of the original index to filter conditions later
        valid_indices_map = {i: seed for i, seed in enumerate(self.seeds) if seed is not None}
        valid_indices = list(valid_indices_map.keys())

        self.image_paths = [self.image_paths[i] for i in valid_indices]
        self.seeds = [valid_indices_map[i] for i in valid_indices]

        if not self.image_paths:
             raise ValueError(f"No valid image files with extractable seeds found in {self.data_dir}")

        # --- Load conditions if available --- 
        self.conditions = {} # Dictionary mapping seed -> condition
        self.condition_type = 'none' # Type detected from CSV header
        conditions_csv_path = os.path.join(self.data_dir, 'conditions.csv')
        if os.path.exists(conditions_csv_path):
            print(f"Loading conditions from {conditions_csv_path}...")
            try:
                with open(conditions_csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        print(f"Warning: Empty header in {conditions_csv_path}")
                        return
                    # Determine condition type from header (expecting 'seed' and one other column)
                    if len(reader.fieldnames) >= 2:
                        self.condition_type = reader.fieldnames[1] # e.g., 'class_label' or 'prompt'
                        print(f"Detected condition type: {self.condition_type}")
                    else:
                        print(f"Warning: Unexpected header format in {conditions_csv_path}: {reader.fieldnames}")
                        self.condition_type = 'unknown'

                    for row in reader:
                        try:
                            seed = int(row['seed'])
                            # Store the value from the second column
                            condition_value = row[self.condition_type]
                            # Convert class_label back to int if needed
                            if self.condition_type == 'class_label':
                                condition_value = int(condition_value)
                            self.conditions[seed] = condition_value
                        except (KeyError, ValueError) as e:
                             print(f"Warning: Skipping row in {conditions_csv_path} due to parsing error: {row} - {e}")
            except Exception as e:
                 print(f"Error loading conditions CSV {conditions_csv_path}: {e}")
                 # Continue without conditions if loading fails
                 self.condition_type = 'none'
                 self.conditions = {}
            print(f"Loaded {len(self.conditions)} conditions.")
        else:
            print(f"Conditions file not found: {conditions_csv_path}. Proceeding without conditions.")

        print(f"Initialized dataset from {self.data_dir} with {len(self.image_paths)} images.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        seed = self.seeds[idx]
        condition = self.conditions.get(seed, None) # Get condition for this seed, default to None
        try:
            # Load image, ensure it's RGB
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
             print(f"Error loading image {img_path}: {e}")
             # Return a dummy image or skip? For simplicity, re-raising might be best during debugging.
             raise IOError(f"Could not load image {img_path}") from e

        if self.transform:
            image = self.transform(image) # Should transform PIL Image to Tensor [-1, 1]

        # Ensure seed is an integer
        seed = int(seed)

        return image, seed, condition

#----------------------------------------------------------------------------
# Wrapper for torch.Generator needed for generating initial latents from seeds
# (Copied from sample_tea.py - ensure consistency or centralize this)
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
        # This might not be needed if class labels are not used or handled differently
        if len(self.generators) == 0:
             return torch.empty(0, *size[1:], dtype=kwargs.get('dtype', torch.long))
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


from noise_schedulers import NoiseScheduleVE, NoiseScheduleVP, NoiseScheduleOT
def get_noise_scheduler(noise_schedule, net=None):
    if noise_schedule == 'VE':
        return NoiseScheduleVE(schedule='edm')
    elif noise_schedule == 'VP':
        noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=net.alphas_cumprod)
        noise_schedule.lambda_min = noise_schedule.marginal_lambda(noise_schedule.T).item()
        noise_schedule.lambda_max = noise_schedule.marginal_lambda(1.0 / noise_schedule.total_N).item()
        return noise_schedule
    elif noise_schedule == 'OT':
        return NoiseScheduleOT(schedule='time_uniform')
    else:
        raise ValueError("Got wrong noise scheduler!")


#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    datadir             = None,     # Root directory for pre-generated data. Required.
    train_num           = None,     # Expected number of training images. Required.
    valid_num           = None,     # Expected number of validation images. Required.
    iLD3_kwargs         = {},       # Options for iLD3 predictor.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    seed                = 0,        # Global random seed.
    batch_size          = None,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 20,       # Training duration, measured in thousands of training images.
    kimg_per_tick       = 1,        # Interval of progress prints.
    snapshot_ticks      = 1,        # How often to save network snapshots, None = disable.
    state_dump_ticks    = 20,       # How often to dump training state, None = disable.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    deterministic       = False,    # Enable deterministic training settings?
    num_workers         = 4,        # Number of workers for DataLoader.
    model_path          = None,     
    noise_schedule     = 'VE',     # Noise schedule used during data generation (needed for predictor init?) - CHECK
    num_step                = 0,        # Max NFE (needed for loss_fn setup?) - CHECK
    order               = 3,        # Order (needed for loss_fn setup?) - CHECK
    device              = torch.device('cuda'),
    frequency           = 10,       # How often to run validation (measured in ticks). Example: validate every 10 ticks.
    run_valid           = None,
    offload             = True,
    max_grad_norm       = 1.0,
    weight_decay_param  = 0.0,
    weight_decay_net    = 0.0,
    warmup_start_lr     = 1e-4,
    eta_min             = 1e-5,
    warmup_ratio        = 0.1,
    lr_scheduler_type   = 'cosine',
    plateau_factor      = 0.5,
    plateau_patience    = 1,
    plateau_threshold   = 0.0,
    plateau_min_lr      = 1e-5,
    early_stop_patience = 0,
    early_stop_min_delta= 0.0,
    init_predictor_path = None,
    **kwargs,
):
    # Initialize.
    start_time = time.time()
    local_seed = (seed * dist.get_world_size() + dist.get_rank()) % (1 << 31)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as err:
            dist.print0(f'Warning: could not enable deterministic algorithms: {err}')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate required arguments
    if datadir is None or train_num is None or valid_num is None:
        raise ValueError("`datadir`, `train_num`, and `valid_num` must be specified when using offline data.")

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    if batch_size != batch_gpu * num_accumulation_rounds * dist.get_world_size():
         raise ValueError(f"batch_size ({batch_size}) must be divisible by batch_gpu ({batch_gpu}) * num_gpus ({dist.get_world_size()})")
    dist.print0(f"Using batch_gpu={batch_gpu}, accumulation_rounds={num_accumulation_rounds}")

    if model_path is not None:
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first

        dataset_name = kwargs.get('dataset_name', None) # Get from kwargs if passed
        guidance_type = kwargs.get('guidance_type', None)
        guidance_rate = kwargs.get('guidance_rate', 0.0)
        # print("device", device)

        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device = torch.device(f'cuda:{local_rank}')
        print("loading model on device", device)
        net, model_source = create_model(dataset_name, guidance_type, guidance_rate, device, model_path)

    if hasattr(net, 'sigma_min'):
        sigma_min = net.sigma_min
        sigma_max = net.sigma_max
        img_channels = net.img_channels
        img_resolution = net.img_resolution
    elif dataset_name == 'lsun_bedroom_ldm': # lsun_bedroom_ldm
        sigma_min = 0.002
        sigma_max = 80.0
        img_channels = 3
        img_resolution = 64 # [3, 64, 64]
    elif dataset_name == 'ms_coco': # ms_coco
        sigma_min = 0.002
        sigma_max = 80.0
        img_channels = 16
        img_resolution = 64 # [16, 128, 128]
        # import memsave_torch
        # net['flow'] = memsave_torch.nn.convert_to_memory_saving(net['flow'])
        # net['ae'] = memsave_torch.nn.convert_to_memory_saving(net['ae'])
        # net['flow'].gradient_checkpointing_enable()
        # net['ae'].gradient_checkpointing_enable()

    if dist.get_rank() == 0:
        torch.distributed.barrier() # other ranks follow


    # Construct iLD3 predictor.
    dist.print0('Constructing predictor...')
    # Ensure img_size is correctly passed, potentially derived from loaded data or teacher net
    iLD3_kwargs.update(img_size=(img_channels, img_resolution, img_resolution))
    iLD3_predictor = dnnlib.util.construct_class_by_name(**iLD3_kwargs) # subclass of torch.nn.Module
    iLD3_predictor.train().requires_grad_(True).to(device)

    # Initialize EMA decay list if specified in kwargs
    if iLD3_predictor.use_ema:
        dist.print0(f'Initializing multiple EMAs with half-lives (kimg): {iLD3_predictor.ema_decay_list_kimg}')
        iLD3_predictor.get_ema_decay_list(batch_size=batch_size)
        dist.print0(f'EMA decay rates: {iLD3_predictor.ema_decay_list}')
    elif iLD3_predictor.use_ema:
        dist.print0(f'Using single EMA with decay rate: {iLD3_predictor.ema_decay}')

    # Setup optimizer and DDP.
    dist.print0('Setting up optimizer...')
    # Ensure find_unused_parameters is set correctly based on whether predictor has unused params during training
    ddp = torch.nn.parallel.DistributedDataParallel(iLD3_predictor, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True) # May need find_unused_parameters=True


    
    expected_condition_type = 'none'
    if hasattr(net, 'label_dim') and net.label_dim:
        if model_source == 'adm':
            expected_condition_type = 'class_label'
        elif model_source == 'ldm' and dataset_name == 'ms_coco':
            expected_condition_type = 'prompt'
        elif model_source == 'edm': # Assume EDM conditional models use class labels
            expected_condition_type = 'class_label'
        else: # Fallback
            expected_condition_type = 'class_label'
            dist.print0(f"Warning: Assuming 'class_label' conditioning for training based on model source '{model_source}\' and label_dim={net.label_dim}. Verify.")
    elif dataset_name == 'ms_coco':
        expected_condition_type = 'prompt'



    dist.print0(f"Training expects condition type: {expected_condition_type}")

    # --- Assign noise schedule to predictor BEFORE initializing loss --- 
    iLD3_predictor.noise_schedule = get_noise_scheduler(noise_schedule, net) # Set noise schedule for predictor/loss

    # Setup loss function.
    loss_kwargs.update(num_steps=num_step, sampler_stu=iLD3_kwargs.sampler_stu, \
                       schedule_type=iLD3_kwargs.schedule_type, schedule_rho=iLD3_kwargs.schedule_rho, \
                       afs=iLD3_kwargs.afs, order=order, max_order=iLD3_kwargs.max_order, sigma_min=sigma_min, sigma_max=sigma_max, \
                       predict_x0=iLD3_kwargs.predict_x0, lower_order_final=iLD3_kwargs.lower_order_final, iLD3_predictor=iLD3_predictor)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # Need to verify if this loss expects online teacher generation

    # Setup optimizer parameters groups and scheduler.
    param_groups = [
        {
            'params': [p for n, p in iLD3_predictor.named_parameters() if not isinstance(getattr(iLD3_predictor, n.split('.')[0], None), nn.Module)],
            'lr': kwargs.get('lr_param', optimizer_kwargs.get('lr')),
            'weight_decay': weight_decay_param,
        },
        {
            'params': [p for n, p in iLD3_predictor.named_parameters() if isinstance(getattr(iLD3_predictor, n.split('.')[0], None), nn.Module)],
            'lr': kwargs.get('lr_net', optimizer_kwargs.get('lr')),
            'weight_decay': weight_decay_net,
        }
    ]
    # Filter out empty groups
    param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
    if not param_groups:
         param_groups = [{'params': iLD3_predictor.parameters()}] # Fallback if grouping heuristic fails

    optimizer = dnnlib.util.construct_class_by_name(params=param_groups, **optimizer_kwargs)

    # Initialize predictor parameters (might depend on teacher net details).
    # Pass `net` here, which is either the loaded teacher or a dummy.
    ref_schedule = loss_fn.solver_stu.get_time_steps_wrapped(iLD3_kwargs.schedule_type, device, num_step)
    # print(ref_schedule)
    iLD3_predictor.initialize_parameters(ref_schedule, iLD3_kwargs.schedule_type) # Pass teacher/dummy net

    if init_predictor_path is not None:
        dist.print0(f'Loading predictor weights from "{init_predictor_path}" after initialization...')
        with dnnlib.util.open_url(init_predictor_path, verbose=(dist.get_rank() == 0)) as f:
            init_data = pickle.load(f)
        init_model = init_data['model'].to(device)
        missing_keys, unexpected_keys = iLD3_predictor.load_state_dict(init_model.state_dict(), strict=False)
        if missing_keys:
            dist.print0(f'Checkpoint load missing keys: {missing_keys}')
        if unexpected_keys:
            dist.print0(f'Checkpoint load unexpected keys: {unexpected_keys}')

        # When continuing from a saved checkpoint, the EMA trackers created
        # during predictor construction still contain the pre-load random
        # initialization unless we explicitly resync them here.
        if iLD3_predictor.use_ema and hasattr(iLD3_predictor, 'ema_params_list'):
            dist.print0('Resyncing EMA trackers from loaded predictor weights...')
            for ema_params in iLD3_predictor.ema_params_list:
                for name, param in iLD3_predictor.named_parameters():
                    if param.requires_grad and name in ema_params:
                        ema_params[name] = param.detach().clone()
            if hasattr(iLD3_predictor, 'num_updates'):
                iLD3_predictor.num_updates.zero_()
        del init_model, init_data

    # Learning rate scheduler
    scheduler = None
    frequency_num = max(batch_size, int(np.ceil((frequency * 1000) / batch_size)) * batch_size)
    print(frequency_num)
    if kwargs.get('cos_lr_schedule', False) and lr_scheduler_type == 'cosine':
        dist.print0('Using Cosine Annealing Learning Rate Schedule')
        # Calculate total steps based on number of training images and batch size
        steps_per_epoch = int(np.ceil(train_num / batch_size))
        total_epochs = int(np.ceil(total_kimg * 1000 / train_num))
        total_steps = total_epochs * steps_per_epoch
        dist.print0(f"Total training steps estimated: {total_steps} ({total_epochs} epochs)")

        # Adjust warmup and max epochs based on total steps
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=warmup_steps, # Name is epochs, but pass steps
                                                  max_epochs=total_steps,     # Name is epochs, but pass steps
                                                  warmup_start_lr=warmup_start_lr,
                                                  eta_min=eta_min)
    elif lr_scheduler_type == 'plateau':
        if not run_valid:
            raise ValueError('ReduceLROnPlateau requires run_valid=True.')
        dist.print0('Using ReduceLROnPlateau Learning Rate Schedule')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            threshold_mode='abs',
            min_lr=plateau_min_lr,
        )


    # Setup Datasets and DataLoaders
    dist.print0("Setting up Datasets and DataLoaders...")
    # Define image transform: Convert PIL to Tensor and normalize to [-1, 1]
    # Normalization depends on how teacher images were saved. Assuming they are [0, 255] PNGs.
    
    transform = transforms.Compose([
        transforms.ToTensor(),                # Converts PIL [0, 255] HWC to Tensor [0, 1] CHW
        transforms.Normalize([0.5], [0.5])    # Normalizes [0, 1] to [-1, 1] (assuming 3 channels have same mean/std)
    ])

    train_dataset = PreGeneratedDataset(os.path.join(datadir, 'train'), num_expected=train_num, transform=transform)
    valid_dataset = PreGeneratedDataset(os.path.join(datadir, 'valid'), num_expected=valid_num, transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, seed=seed)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False) # No shuffle for validation

    train_loader = DataLoader(train_dataset, batch_size=batch_gpu, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_gpu * 4, sampler=valid_sampler, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)

    # Prepare for training.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    min_valid_loss = float('inf') # Initialize minimum validation loss
    bad_validation_ticks = 0
    train_loss_list = [] # For potential plotting

    # --- Main Training Loop ---
    epoch = 0
    while True:
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        for batch_data in train_loader:
            # Move data to device
            images_teacher = batch_data['image'].to(device)
            seeds_tensor = batch_data['seed'] # Seeds are already a tensor
            conditions = batch_data['condition']
            seeds = seeds_tensor.tolist() # Convert tensor back to list if needed by StackedRandomGenerator

            # Check for termination condition
            if cur_nimg >= total_kimg * 1000:
                break

            rnd = StackedRandomGenerator(device, seeds)
            latents_shape = [len(seeds), img_channels, img_resolution, img_resolution]
            latents = rnd.randn(latents_shape, device=device)
            latents = iLD3_predictor.noise_schedule.prior_transformation(latents)
            
            # Accumulate gradients.
            optimizer.zero_grad(set_to_none=True)
            for round_idx in range(num_accumulation_rounds):
                # Determine batch slice for accumulation
                start_idx = round_idx * batch_gpu
                end_idx = start_idx + batch_gpu
                batch_latents_accum = latents[start_idx:end_idx]
                batch_teacher_accum = images_teacher[start_idx:end_idx]

                # Check if batch slice is empty (can happen with uneven data)
                if batch_latents_accum.shape[0] == 0:
                    continue

                with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                    batch_class_labels = None # For ADM/EDM
                    batch_prompts = None
                    batch_conditions = conditions[start_idx:end_idx]
                    if expected_condition_type == 'class_label':
                        normalized_labels = normalize_class_label_batch(batch_conditions)
                        if normalized_labels is not None:
                            label_indices = torch.tensor(normalized_labels, dtype=torch.long, device=device)
                            # Format for the specific model
                            if model_source == 'adm':
                                batch_class_labels = label_indices
                            elif dataset_name == 'imagenet64': # Assume EDM needs one-hot
                                batch_class_labels = torch.eye(net.label_dim, device=device)[label_indices]
                            else: # Fallback
                                batch_class_labels = label_indices
                        elif batch_conditions is not None:
                            dist.print0("Warning: Could not normalize class_label conditions. Skipping condition.")

                    elif expected_condition_type == 'prompt':
                        # Ensure conditions are strings (prompts)
                        if batch_conditions is not None and all(isinstance(c, str) for c in batch_conditions):
                            batch_prompts = list(batch_conditions)
                        elif batch_conditions is not None:
                            dist.print0("Warning: Loaded conditions are not strings for prompt type. Skipping condition.")
              
                    loss, str2print, stu_out = loss_fn(
                        iLD3_predictor_ddp=ddp,
                        start_step=0, # Likely not needed if target is given
                        end_step=num_step,     # Likely not needed
                        net=net,             # Teacher net might not be needed here
                        tensor_in=batch_latents_accum, # Initial noise latents
                        teacher_out=batch_teacher_accum, # The loaded teacher image
                        class_labels=batch_class_labels, # Pass formatted class labels
                        prompts=batch_prompts,
                        guidance=guidance_rate,
                        model_source=model_source,
                        dataset_name=dataset_name,
                        # rounds=cur_tick // frequency # Example - pass training progress if needed
                    )
                    # Scale loss for accumulation
                    loss_scaled = loss / num_accumulation_rounds


                    loss_scaled.backward()
            torch.nn.utils.clip_grad_norm_(ddp.module.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            if dataset_name == 'ms_coco':
                clear_grads_and_cache(net['flow'])
            # reset_peak_memory_stats()
            # print(f"Memory after step: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            # input("stop here5")


            # Update EMA if used
            if iLD3_predictor.use_ema:
                 # Call update_ema on the original module, not the DDP wrapper
                 ddp.module.update_ema()

            # Update LR scheduler
            if scheduler and lr_scheduler_type != 'plateau':
                scheduler.step()
            # Log training loss
            training_stats.report('Loss/loss', loss) # Report non-scaled loss for tracking
            train_loss_list.append(loss.item())
            # Optional: Print learning rates
            lr_names = ['lr_param', 'lr_net']
            lr_str = ""
            for idx, group in enumerate(optimizer.param_groups):
                label = lr_names[idx] if idx < len(lr_names) else f'lr_group{idx}'
                lr_str += f"| {label}: {group['lr']:.2e} "
            dist.print0(str2print + lr_str) # Assuming str2print comes from loss_fn

            # --- Maintenance Tasks ---
            cur_nimg += batch_size # Increment by global batch size
            done = (cur_nimg >= total_kimg * 1000)

 

            # --- Run Validation ---
            
            if run_valid and frequency > 0 and (done or (cur_nimg % frequency_num == 0)):
                dist.print0(f"Running validation at tick {cur_tick}, kimg {cur_nimg/1000:.1f}...")
                total_val_loss = 0.0
                num_val_batches = 0
                ddp.eval() # Set predictor to evaluation mode

                with torch.no_grad():
                    for val_batch_data in valid_loader:
                        val_images_teacher = val_batch_data['image'].to(device)
                        val_seeds_tensor = val_batch_data['seed']
                        val_conditions = val_batch_data['condition']
                        val_seeds = val_seeds_tensor.tolist()

                        if not val_seeds: continue # Skip empty batches

                        # Generate initial latents for validation
                        val_rnd = StackedRandomGenerator(device, val_seeds)
                        val_latents_shape = [len(val_seeds), img_channels, img_resolution, img_resolution]
                        val_latents = val_rnd.randn(val_latents_shape, device=device)
                        val_latents = iLD3_predictor.noise_schedule.prior_transformation(val_latents)
                        
                        # --- Prepare validation conditions --- 
                        val_batch_class_labels = None
                        val_batch_prompts = None
                        if expected_condition_type == 'class_label':
                            normalized_val_labels = normalize_class_label_batch(val_conditions)
                            if normalized_val_labels is not None:
                                val_label_indices = torch.tensor(normalized_val_labels, dtype=torch.long, device=device)
                                if model_source == 'adm': val_batch_class_labels = val_label_indices
                                elif dataset_name == 'imagenet64': val_batch_class_labels = torch.eye(net.label_dim, device=device)[val_label_indices]
                                else: val_batch_class_labels = val_label_indices
                            elif val_conditions is not None:
                                dist.print0("Warning: Could not normalize validation class_label conditions. Skipping condition.")
                        elif expected_condition_type == 'prompt':
                            if val_conditions is not None and all(isinstance(c, str) for c in val_conditions):
                                val_batch_prompts = list(val_conditions)


                        val_loss, _, _ = loss_fn( # Assuming loss_fn returns loss, print_str, student_out
                            start_step=0,
                            end_step=num_step,
                            net=net,
                            iLD3_predictor_ddp=ddp,
                            tensor_in=val_latents,
                            teacher_out=val_images_teacher,
                            class_labels=val_batch_class_labels, # Pass formatted validation labels
                            prompts=val_batch_prompts,
                            guidance=guidance_rate,
                            model_source=model_source,
                            dataset_name=dataset_name,
                            # Pass other args if needed by loss_fn in eval mode
                        )
                        # Handle potential reduction within loss_fn. If it's sum, divide by batch size. If mean, use as is.
                        # Assuming loss_fn returns mean loss per batch element:
                        dist.print0(f"Validation loss: {val_loss.item()}")
                        total_val_loss += val_loss.item()
                        num_val_batches += len(val_seeds)                  # Accumulate number of samples

                ddp.train() # Set back to training mode

                # Synchronize validation loss across all GPUs
                if dist.get_world_size() > 1:
                    val_stats_tensor = torch.tensor([total_val_loss, num_val_batches], dtype=torch.float64, device=device)
                    torch.distributed.all_reduce(val_stats_tensor, op=torch.distributed.ReduceOp.SUM)
                    total_val_loss = val_stats_tensor[0].item()
                    num_val_batches = val_stats_tensor[1].item()

                avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
                training_stats.report('Loss/validation', avg_val_loss)
                dist.print0(f"Validation Loss: {avg_val_loss:.6f}")

                if scheduler and lr_scheduler_type == 'plateau':
                    prev_lrs = [group['lr'] for group in optimizer.param_groups]
                    scheduler.step(avg_val_loss)
                    new_lrs = [group['lr'] for group in optimizer.param_groups]
                    if any(new_lr < old_lr for old_lr, new_lr in zip(prev_lrs, new_lrs)):
                        lr_summary = ', '.join(f'{lr:.2e}' for lr in new_lrs)
                        dist.print0(f"Reduced learning rates after plateau: {lr_summary}")

                # --- Save Best Model based on Validation Loss ---
                if avg_val_loss < (min_valid_loss - early_stop_min_delta):
                    min_valid_loss = avg_val_loss
                    bad_validation_ticks = 0
                    dist.print0(f"*** New best validation loss: {min_valid_loss:.6f}. Saving model... ***")
                    # Save on rank 0
                    if dist.get_rank() == 0:
                        try:
                            save_model_and_ema(ddp, run_dir, prefix='network-best')
                        except Exception as e:
                            dist.print0(f"Error saving best model: {e}")
                    # Barrier to ensure rank 0 finishes saving before others proceed
                    torch.distributed.barrier()
                else:
                    bad_validation_ticks += 1

                if early_stop_patience > 0 and bad_validation_ticks >= early_stop_patience:
                    dist.print0(f"Early stopping triggered after {bad_validation_ticks} validation ticks without improvement.")
                    done = True
                # --- End Save Best Model ---
            # --- End Validation ---
            if (done or (cur_nimg % frequency_num == 0)):
                # Print status line, accumulating the same information in training_stats.
                tick_end_time = time.time()
                fields = []
                fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
                fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
                fields += [f"loss {training_stats.report0('Loss/loss', loss.item()):<6.4f}"] # Report last training loss
                # Check if validation loss exists in the collected stats dictionary
                if 'Loss/validation' in training_stats.default_collector.as_dict(): # Only report if validation ran
                     fields += [f"val_loss {training_stats.report0('Loss/validation', avg_val_loss):<6.4f}"]
                fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
                fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
                fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3 if cur_nimg > tick_start_nimg else 0):<7.2f}"]
                fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
                fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
                # fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"] # Optional
                torch.cuda.reset_peak_memory_stats()
                dist.print0(' '.join(fields))

                # Check for abort.
                if (not done) and dist.should_stop():
                    done = True
                    dist.print0()
                    dist.print0('Aborting...')

                # Save periodic snapshots so checkpoint selection can be driven
                # by downstream proxy FID, not just validation loss.
                snapshot_due = snapshot_ticks is not None and snapshot_ticks > 0 and ((cur_tick + 1) % snapshot_ticks == 0)
                if (snapshot_due or done) and dist.get_rank() == 0:
                    snapshot_prefix = f'network-snapshot-{cur_nimg:07d}'
                    try:
                        save_model_and_ema(ddp, run_dir, prefix=snapshot_prefix)
                    except Exception as e:
                        dist.print0(f"Error saving snapshot {snapshot_prefix}: {e}")
                if snapshot_due or done:
                    torch.distributed.barrier()

                # Update logs.
                training_stats.default_collector.update()
                if dist.get_rank() == 0:
                    if stats_jsonl is None:
                        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                    stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
                    stats_jsonl.flush()
                dist.update_progress(cur_nimg // 1000, total_kimg)

                # Update state.
                cur_tick += 1
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()
                maintenance_time = tick_start_time - tick_end_time
                if done:
                    break # Exit inner loop if done

            # Check again if done after potentially completing a tick/validation
            if done:
                break # Exit outer (epoch) loop if done

        # Check again if done after completing an epoch
        if done:
            break # Exit while loop if done

        epoch += 1 # Increment epoch counter


    # --- End of Training ---
    if dist.get_rank() == 0:
        try:
            save_model_and_ema(ddp, run_dir, prefix='network-final')
            save_model_and_ema(ddp, run_dir, prefix='network-last')
        except Exception as e:
            dist.print0(f"Error saving final models: {e}")

    torch.distributed.barrier()
    
    # Save final training loss plot
    if dist.get_rank() == 0 and train_loss_list:
        try:
            plt.figure() # Create a new figure
            plt.plot(train_loss_list)
            plt.title('Training Loss Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(run_dir, 'train_loss_plot.png'))
            plt.close() # Close the figure
            dist.print0(f"Saved training loss plot to {os.path.join(run_dir, 'train_loss_plot.png')}")
        except Exception as e:
            dist.print0(f"Error saving training loss plot: {e}")

    dist.print0()
    dist.print0('Exiting...')
    if stats_jsonl is not None:
         stats_jsonl.close()

#----------------------------------------------------------------------------


def save_model_and_ema(ddp, run_dir, prefix='network-final'):
    # Save main model (without EMA storage for consistency)
    model_to_save = ddp.module
    clean_model = copy.deepcopy(model_to_save)
    clean_model = remove_ema_attributes(clean_model)
    data = dict(model=clean_model.eval().requires_grad_(False).cpu())
    final_snapshot_path = os.path.join(run_dir, f'{prefix}.pkl')
    with open(final_snapshot_path, 'wb') as f:
        pickle.dump(data, f)
    dist.print0(f"Saved final model to {final_snapshot_path}")
    del data, clean_model  # conserve memory
    
    # Save all EMA models if available
    if model_to_save.use_ema and hasattr(model_to_save, 'ema_params_list'):
        dist.print0(f"Saving {len(model_to_save.ema_params_list)} EMA models...")
        
        # If we have the decay list in kimgs, include that in the filename
        has_kimg_list = hasattr(model_to_save, 'ema_decay_list_kimg')
        
        for ema_idx, _ in enumerate(model_to_save.ema_params_list):
            # Get the kimg value for better filename if available
            kimg_str = ""
            if has_kimg_list:
                kimg_value = model_to_save.ema_decay_list_kimg[ema_idx]
                kimg_str = f"{kimg_value}k"
            
            # Create a fresh copy of the model without EMA storage
            clean_model = copy.deepcopy(model_to_save)
            
            # Temporarily apply EMA weights
            ema_params = model_to_save.ema_params_list[ema_idx]
            for name, param in clean_model.named_parameters():
                if name in ema_params:
                    param.data.copy_(ema_params[name].to(param.device))
            
            # Remove EMA storage to save space
            clean_model = remove_ema_attributes(clean_model)
            
            # Create filename with EMA index and kimg
            ema_prefix = 'network-ema' if prefix == 'network-final' else f'{prefix}-ema'
            final_ema_path = os.path.join(run_dir, f'{ema_prefix}-{kimg_str}.pkl')
            
            # Save the clean model
            data_ema = dict(model=clean_model.eval().requires_grad_(False).cpu())
            with open(final_ema_path, 'wb') as f:
                pickle.dump(data_ema, f)
                
            del data_ema, clean_model  # conserve memory
            dist.print0(f"Saved EMA model {ema_idx} ({kimg_str}) to {final_ema_path}")
            



# Custom collate function to handle potentially missing conditions
def custom_collate_fn(batch):
    # batch is a list of tuples: [(img1, seed1, cond1), (img2, seed2, cond2), ...]
    images = [item[0] for item in batch]
    seeds = [item[1] for item in batch]

    processed_conditions = [item[2] for item in batch]

    # Stack images and convert seeds to tensor
    images_collated = torch.stack(images, 0)
    seeds_collated = torch.tensor(seeds, dtype=torch.long)

    return {
        'image': images_collated,
        'seed': seeds_collated,
        'condition': processed_conditions  # Processed conditions with no None values
    }

#----------------------------------------------------------------------------
# remove the redundant attributes of the model.
def remove_ema_attributes(model):
    """Remove all EMA-related attributes from a model to reduce file size when saving."""
    # List of known EMA-related attributes to remove
    ema_attributes = [
        'ema_params_list',
        'ema_decay_list',
        'ema_decay_list_kimg',
        'ema_decay_buffer',
        'num_updates',
        '_original_requires_grad',
        '_tn_params_mask',
        '_fixed_tn_params'
    ]
    
    # Remove all EMA attributes
    for attr in ema_attributes:
        if hasattr(model, attr):
            delattr(model, attr)
    
    # Also check for buffer attributes that might have been registered
    for name, _ in list(model._buffers.items()):
        if 'ema' in name.lower():
            model._buffers.pop(name)
    
    return model

#----------------------------------------------------------------------------
def reset_peak_memory_stats():
    """Reset GPU memory stats and cached tensors."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    # Only in PyTorch 2.0+
    if hasattr(torch, '_C'):
        torch._C._cuda_resetAccumulatedMemoryStats()


def clear_grads_and_cache(net):
    # Clear gradients after backward pass is complete
    for param in net.parameters():
        param.grad = None  # More efficient than zero_grad
    
    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()
