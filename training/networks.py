import numpy as np
import torch
from torch_utils import persistence
from training.noisenet import NoiseHandler_fullsvd, NoiseHandler_values, NoiseHandler_golden, NoiseHandler_condition
import torch.nn as nn
from torch.nn.functional import silu
import torch.nn.functional as F
from solver_utils import get_schedule
from contextlib import contextmanager
#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x



@persistence.persistent_class
class iLD3_predictor(torch.nn.Module):

    def __init__(
        self,
        dataset_name            = None,
        img_size                = None,
        num_steps               = None,
        sampler_stu             = None, 
        M                       = None,
        guidance_type           = None,      
        guidance_rate           = None,
        schedule_type           = None,
        schedule_rho            = None,
        afs                     = False,
        mode                    = 'time',
        bound_tn                = 0,
        bound_tn2               = 0,
        bound_epscaling         = 0,
        max_order               = None,
        predict_x0              = True,
        window_rate             = 0.5,
        use_ema                 = False,
        ema_decay               = 0.995,
        ema_decay_list_kimg     = None,  # Add parameter for specifying EMA decay list in kimg
        noise_handler           = 'null',
        noisenethidden_dim      = 128,
        noisenet_topk           = 10,
        **kwargs
    ):
        super().__init__()
        assert sampler_stu in ['ipndm','ipndm_flux', 'uni_pc', 'dpm_solverpp']
        assert bound_tn >= 0
        assert bound_tn2 >= 0
        assert bound_epscaling >= 0
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.num_steps = num_steps
        self.sampler_stu = sampler_stu
        self.M = M
        self.guidance_type = guidance_type
        self.guidance_rate = guidance_rate
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.afs = afs
        self.bound_tn = bound_tn
        self.bound_tn2 = bound_tn2
        self.bound_epscaling = bound_epscaling
        self.max_order = max_order
        self.predict_x0 = predict_x0
        self.mode = mode
        self.window_rate = window_rate
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_decay_list_kimg = ema_decay_list_kimg  # Store the kimg values
        self.noisenethidden_dim = noisenethidden_dim
        self.noisenet_topk = noisenet_topk
        self.condition_type = noise_handler

        
        noise_handler_class = NoiseHandler_condition
        # print(img_size)
        self.noise_handler = noise_handler_class(
            output_dim = (num_steps+1)*3, 
            num_channels = img_size[0], 
            height = img_size[1], 
            width = img_size[2], 
            bound_tn = self.bound_tn,
            bound_tn2 = self.bound_tn2,
            bound_epscaling = self.bound_epscaling,
            hidden_dim = self.noisenethidden_dim,
            top_k = self.noisenet_topk,
            condition_type = self.condition_type
        )
        
        
        self.tn_params = nn.Parameter(torch.ones(num_steps+1)) 
        self.tn2_params = nn.Parameter(torch.zeros(num_steps+1)) 

        self.epscaling_params = nn.Parameter(torch.randn(num_steps+1)) 

        self.sigmoid = torch.nn.Sigmoid()

    
    def initialize_ema(self):
        """Initialize multiple EMA parameters and buffers."""
        # Get length of decay list if it exists
        ema_count = len(self.ema_decay_list_kimg) if hasattr(self, 'ema_decay_list_kimg') else 1
        
        # Create a list of dictionaries to store EMA parameters
        self.ema_params_list = [{} for _ in range(ema_count)]
        
        # Initialize parameter copies for each EMA
        for name, param in self.named_parameters():
            if param.requires_grad:
                for i in range(ema_count):
                    # Store parameter on the same device as the original parameter
                    self.ema_params_list[i][name] = param.detach().clone()
        
        # Register buffers for tracking updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))
        
        # If no decay list exists yet, register the default decay
        if hasattr(self, 'ema_decay_list'):
            self.register_buffer('ema_decay_buffer', torch.tensor(self.ema_decay_list, dtype=torch.float32))
        else:
            self.register_buffer('ema_decay_buffer', torch.tensor(self.ema_decay, dtype=torch.float32))
    
    def update_ema(self):
        """Update all EMA parameters."""
        if not self.use_ema or not hasattr(self, 'ema_params_list'):
            return
            
        # Increase update counter
        self.num_updates += 1
        
        # Update each set of EMA parameters with its corresponding decay rate
        with torch.no_grad():
            for i, ema_params in enumerate(self.ema_params_list):
                # Get the decay rate for this EMA
                if hasattr(self, 'ema_decay_list_kimg'):
                    # decay = min(self.ema_decay_buffer[i], (1 + self.num_updates) / (10 + self.num_updates))
                    decay = self.ema_decay_buffer[i]
                else:
                    # Fallback to single decay if list not available
                    decay = min(self.ema_decay_buffer, (1 + self.num_updates) / (10 + self.num_updates))
                
                one_minus_decay = 1.0 - decay
                
                # Update parameters for this EMA
                for name, param in self.named_parameters():
                    if param.requires_grad and name in ema_params:
                        # Ensure both tensors are on the same device
                        if ema_params[name].device != param.device:
                            ema_params[name] = ema_params[name].to(param.device)
                        
                        ema_params[name].mul_(decay).add_(param.data, alpha=one_minus_decay)
    
    @contextmanager
    def ema_scope(self, ema_index=0):
        """
        Context manager to temporarily switch to EMA parameters.
        
        Args:
            ema_index (int): Index of the specific EMA to use. Defaults to 0 (first EMA).
        """
        if not self.use_ema or not hasattr(self, 'ema_params_list') or ema_index >= len(self.ema_params_list):
            yield
            return
            
        # Store current parameters
        current_params = {}
        ema_params = self.ema_params_list[ema_index]
        
        for name, param in self.named_parameters():
            if param.requires_grad and name in ema_params:
                current_params[name] = param.data.clone()
                # Ensure EMA parameters are on the same device as model parameters
                if ema_params[name].device != param.device:
                    ema_params[name] = ema_params[name].to(param.device)
                param.data.copy_(ema_params[name])               
        try:
            yield
        finally:
            # Restore original parameters
            for name, param in self.named_parameters():
                if param.requires_grad and name in current_params:
                    param.data.copy_(current_params[name])


    def get_ema_decay_list(self, batch_size):
        """
        Calculates a list of EMA decay rates based on specified half-lives in kimg.

        Args:
            num_steps: Total number of training steps (currently unused here).
            ema_decay_list_kimg: A list of half-life values specified in kimg (thousands of images).
            batch_size: The batch size used during training.

        Returns:
            A list of EMA decay rates corresponding to each kimg half-life.
        """
        decay_list = []
        self.ema_decay_list_kimg = [float(kimg) for kimg in self.ema_decay_list_kimg.split(',')]
        for kimg in self.ema_decay_list_kimg:
            if kimg <= 0:
                decay = 0.0
            else:
                decay = 0.5**(batch_size / (kimg * 1000.0)) # Use float division
            decay_list.append(decay)
        self.ema_decay_list = decay_list
        self.initialize_ema()
            
        return decay_list

    def discretize_model_wrapper(self, input1, input2, lambda_max, lambda_min, noise_schedule, mode, window_rate=0.5, noise = None, condition = None):
        '''
        checked!
        '''
        # to be changed
        def model_time_fn():
            time1, time2 = input1, input2
            epscaling = self.epscaling_params
            if self.noise_handler is not None and noise is not None:
                tn1_scale, tn2_scale, epscaling_scale = self.noise_handler(noise, condition)
                # print(time1, '111')
                # print(time2, '222')
                time1 = time1 * tn1_scale[...,] # multi, since init as 1
                time2 = time2 + tn2_scale[...,] # add, since init as 0
                epscaling = epscaling * epscaling_scale[...,] # multi, since init as 1
            t_max, t_min = noise_schedule.inverse_lambda(lambda_min).to(time1.device), noise_schedule.inverse_lambda(lambda_max).to(time1.device)
            

            dim = len(time1.shape)-1
            time_plus = torch.nn.functional.softmax(time1, dim=dim)
            time_md = torch.cumsum(time_plus, dim=dim).flip(dim)
            
            if len(time_md.shape) == 2:
                min_val = time_md.min(dim=1, keepdim=True)[0]  # [batch, 1]
                max_val = time_md.max(dim=1, keepdim=True)[0]  # [batch, 1]
                normed = (time_md - min_val) / (max_val - min_val)  # Added small epsilon for numerical stability
            else:
                normed = (time_md - time_md.min()) / (time_md.max() - time_md.min())
            

            time_steps = normed * (t_max - t_min) + t_min

            cloned_time_steps = time_steps.clone().detach()
            if len(cloned_time_steps.shape) == 2:
                diffs = (cloned_time_steps[:, 1:] - cloned_time_steps[:, :-1]).abs()  # [batch, step-1]
                max_move = diffs.min(dim=1).values * window_rate  # [batch]
            else:
                max_move = (time_steps[1:] - time_steps[:-1]).abs().min().item() * window_rate


            if len(time_steps.shape) == 2:
                max_move = max_move.view(-1, 1)
                clipped_time2 = torch.clamp(time2, min=-max_move, max=max_move)
            else:
                clipped_time2 = torch.clamp(time2, min=-max_move, max=max_move)
            mask = torch.ones_like(normed)
            mask[..., 0] = 0.
            mask[..., -1] = 0.
            # print(time_steps[0])
            return time_steps, time_steps + (clipped_time2 * mask), epscaling
        # changing
        def model_lambda_fn():
            lambda1, lambda2 = input1, input2

            epscaling = self.epscaling_params

            if self.noise_handler and noise is not None:
                # DEBUG: Add comprehensive debugging for noise handler outputs
                with torch.no_grad():
                    # Check input noise statistics
                    noise_mean = noise.mean().item()
                    noise_std = noise.std().item()
                    noise_min = noise.min().item()
                    noise_max = noise.max().item()
                    
            
                
                tn1_scale, tn2_scale, epscaling_scale = self.noise_handler(noise, condition)
                
              
                # print(tn1_scale[0])
                # print(tn1_scale[1])
                
                lambda1 = lambda1 * tn1_scale[...,] # multi, since init as 1
                lambda2 = lambda2 + tn2_scale[...,] # add, since init as 0
                epscaling = epscaling * epscaling_scale[...,] # multi, since init as 1
            dim = len(lambda1.shape)-1
            lamb_plus = F.softmax(lambda1, dim=dim)
            lamb_md = torch.cumsum(lamb_plus, dim=dim)

            if len(lamb_md.shape) == 2:
                min_val = lamb_md.min(dim=1, keepdim=True)[0]  # [batch, 1]
                max_val = lamb_md.max(dim=1, keepdim=True)[0]  # [batch, 1]
                normed = (lamb_md - min_val) / (max_val - min_val)  # Added small epsilon for numerical stability
            else:
                normed = (lamb_md - lamb_md.min()) / (lamb_md.max() - lamb_md.min())

            lamb_steps1 = normed * (lambda_max - lambda_min) + lambda_min

            mask = torch.ones_like(lamb_steps1)
            
            cloned_lamb1 = lambda1.clone().detach()
            if len(cloned_lamb1.shape) == 2:  # [batch, step]
                diffs = (cloned_lamb1[:, 1:] - cloned_lamb1[:, :-1]).abs()  # [batch, step-1]
                max_move = diffs.min(dim=1).values * window_rate  # [batch]
            else:  # [step]
                max_move = (cloned_lamb1[1:] - cloned_lamb1[:-1]).abs().min().item() * window_rate
            # print(lambda2[0], '111')
            if len(lambda2.shape) == 2:  # [batch, step]
                max_move = max_move.view(-1, 1)
                clipped_lamb2 = torch.clamp(lambda2, min=-max_move, max=max_move)
            else:  # [step]
                clipped_lamb2 = torch.clamp(lambda2, min=-max_move, max=max_move)
            mask[..., 0] = 0.
            mask[..., -1] = 0.
            
            lamb_steps2 = lamb_steps1 + clipped_lamb2 * mask
            time1 = noise_schedule.inverse_lambda(lamb_steps1)
            time2 = noise_schedule.inverse_lambda(lamb_steps2)

            # for debug
            # print(time1[0])
            # print(time1[1])

            return time1, time2, epscaling

        return model_time_fn if mode == 'time' else model_lambda_fn


    def get_params(self, noise_schedule, mode, window_rate=0.5, noise = None, condition = None):
        lambda_max = noise_schedule.lambda_max
        lambda_min = noise_schedule.lambda_min
        
        
        model_fn = self.discretize_model_wrapper(
            self.tn_params,
            self.tn2_params,
            lambda_max,
            lambda_min,
            noise_schedule,
            mode,
            window_rate,
            noise,
            condition
        )
        
        # Call the function to get the actual values
        
        tn1_wrapped, tn2_wrapped, epscaling = model_fn()
        
        return tn1_wrapped, tn2_wrapped, epscaling


    def initialize_parameters(self, ref_schedule, schedule_type):
        print(f"initializing parameters to schedule {schedule_type}")
        
        # Initialize optimizer with higher learning rate and add scheduler
        optimizer = torch.optim.Adam([self.tn_params], lr=0.2)  # Increased from 0.001 to 0.1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=50,
        )
        max_iters = 2000
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=0.001)
        error = float('inf')
        best_error = float('inf')
        patience_counter = 0
         
        for iter_count in range(max_iters):
            optimizer.zero_grad()
            tn1, tn2, epscaling = self.get_params(self.noise_schedule, self.mode, self.window_rate)
            error = torch.norm(tn1 - ref_schedule)
            
            # Early stopping check
            if error < best_error:
                best_error = error
                patience_counter = 0
                # Save best parameters
                best_params = self.tn_params.data.clone()
            else:
                patience_counter += 1
            
            if error < 1e-5:  # Convergence achieved
                break
                
                
            error.backward()
            optimizer.step()
            scheduler.step(error)
            
            if iter_count % 50 == 0:
                print(f'Iteration {iter_count}, Deviation: {error.item():.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        print(f'Final error: {error.item():.6f}')
        print(f'Time steps after initialization:')
        print(tn1)

    def forward(self, latents, condition):
        # Get time steps
        tn1, tn2, epscaling = self.get_params(self.noise_schedule, self.mode, self.window_rate, latents, condition)
        # Calculate epscaling
        epscaling = (2 * self.sigmoid(0.5 * epscaling) -1) * self.bound_epscaling + 1

        return tn1, tn2, epscaling

