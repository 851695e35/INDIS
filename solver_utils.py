import torch
import numpy as np
from models.latent_diff import load_model_from_config
from torch_utils.download_util import check_file_by_key
from torch_utils import distributed as dist
import pickle
import dnnlib
#----------------------------------------------------------------------------
def load_ldm_model(config, ckpt, verbose=False):
    from models.ldm.util import instantiate_from_config
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        dist.print0("missing keys:")
        dist.print0(m)
    if len(u) > 0 and verbose:
        dist.print0("unexpected keys:")
        dist.print0(u)
    return model
def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None, model_path=None):
    model_path, classifier_path = check_file_by_key(dataset_name, model_path)
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')
    net = None
    model_source = None
    if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64']:         # models from EDM
        try:
            with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
                net = pickle.load(f)['ema'].to(device)
        except Exception as e:
            dist.print0(f"Error loading model: {e}")
            dist.print0("Attempting to load with pickle protocol 4...")
            try:
                # Try with explicit pickle protocol version
                with open(model_path, 'rb') as f:
                    net = pickle.load(f, fix_imports=True, encoding='latin1')['ema'].to(device)
            except Exception as inner_e:
                dist.print0(f"Secondary loading attempt failed: {inner_e}")
                dist.print0("Please check if the model file is corrupted or try downloading it again.")
                raise RuntimeError("Failed to load model after multiple attempts")
                
        net.sigma_min = 0.002
        net.sigma_max = 80.0
        model_source = 'edm'
    elif dataset_name in ['lsun_bedroom']:                                  # models from Consistency Models
        from models.cm.cm_model_loader import load_cm_model
        from models.networks_edm import CMPrecond
        net = load_cm_model(model_path)
        net = CMPrecond(net).to(device)
        model_source = 'cm'
    else:
        if guidance_type == 'cg':            # clssifier guidance           # models from ADM
            assert classifier_path is not None
            from models.guided_diffusion.cg_model_loader import load_cg_model
            from models.networks_edm import CGPrecond
            net, classifier = load_cg_model(model_path, classifier_path)
            net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
            model_source = 'adm'
        elif guidance_type in ['uncond', 'cfg']:                            # models from LDM

            from omegaconf import OmegaConf
            from models.networks_edm import CFGPrecond
            
            if dataset_name in ['lsun_bedroom_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
                net = load_model_from_config(config, model_path)
                # net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
                
            elif dataset_name in ['ms_coco']: # for flux-dev model
                from flux.util import load_t5, load_clip, load_flow_model, load_ae
                name = 'flux-dev'
                assert guidance_type == 'cfg'
                t5 = load_t5(device, max_length=256 if name == "flux-schnell" else 512, local_path=model_path)
                clip = load_clip(device, local_path=model_path)

                model_name = 'flux1-dev' if name == "flux-dev" else 'flux-schnell'
                flow_model_path = f'{model_path}/{model_name}.safetensors'
                model = load_flow_model(name, device=device, local_path=flow_model_path)
                ae_local_path = f'{model_path}/ae.safetensors'
                ae = load_ae(name, device=device, local_path=ae_local_path)
                net = {'t5': t5, 'clip': clip, 'flow': model, 'ae': ae}

            model_source = 'ldm'
    if net is None:
        raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    
    if dataset_name in ['ms_coco']:
        net['flow'].eval()
        net['ae'].eval()
        net['t5'].eval()
        net['clip'].eval()
        # set all requires_grad to False
        for param in net['flow'].parameters():
            param.requires_grad = False
        for param in net['ae'].parameters():
            param.requires_grad = False
        for param in net['t5'].parameters():
            param.requires_grad = False
        for param in net['clip'].parameters():
            param.requires_grad = False
    else:
        net.eval()

    return net, model_source


def get_schedule(num_steps, sigma_min, sigma_max, device=None, schedule_type='polynomial', schedule_rho=7, net=None):
    """
    Get the time schedule for sampling.

    Args:
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        device: A torch device.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_type: A `float`. Time step exponent.
        net: A pre-trained diffusion model. Required when schedule_type == 'discrete'.
    Returns:
        a PyTorch tensor with shape [num_steps].
    """
    if schedule_type == 'polynomial':
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / schedule_rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / schedule_rho) - sigma_max ** (1 / schedule_rho))) ** schedule_rho
    elif schedule_type == 'logsnr':
        logsnr_max = -1 * torch.log(torch.tensor(sigma_min))
        logsnr_min = -1 * torch.log(torch.tensor(sigma_max))
        t_steps = torch.linspace(logsnr_min.item(), logsnr_max.item(), steps=num_steps, device=device)
        t_steps = (-t_steps).exp()
    elif schedule_type == 'time_uniform':
        epsilon_s = 1e-3
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        step_indices = torch.arange(num_steps, device=device)
        vp_beta_d = 2 * (np.log(torch.tensor(sigma_min).cpu() ** 2 + 1) / epsilon_s - np.log(torch.tensor(sigma_max).cpu() ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(torch.tensor(sigma_max).cpu() ** 2 + 1) - 0.5 * vp_beta_d
        t_steps_temp = (1 + step_indices / (num_steps - 1) * (epsilon_s ** (1 / schedule_rho) - 1)) ** schedule_rho
        t_steps = vp_sigma(vp_beta_d.clone().detach().cpu(), vp_beta_min.clone().detach().cpu())(t_steps_temp.clone().detach().cpu())
    elif schedule_type == 'discrete':
        assert net is not None
        t_steps_min = net.sigma_inv(torch.tensor(sigma_min, device=device))
        t_steps_max = net.sigma_inv(torch.tensor(sigma_max, device=device))
        step_indices = torch.arange(num_steps, device=device)
        t_steps_temp = (t_steps_max + step_indices / (num_steps - 1) * (t_steps_min ** (1 / schedule_rho) - t_steps_max)) ** schedule_rho
        t_steps = net.sigma(t_steps_temp)
    else:
        raise ValueError("Got wrong schedule type {}".format(schedule_type))
    return t_steps.to(device)


# Copied from the DPM-Solver codebase (https://github.com/LuChengTHU/dpm-solver).
# Different from the original codebase, we use the VE-SDE formulation for simplicity
# while the official implementation uses the equivalent VP-SDE formulation. 
##############################
### Utils for DPM-Solver++ ###
##############################
#----------------------------------------------------------------------------

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        v: a PyTorch tensor with shape [N].
        dim: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]
    
#----------------------------------------------------------------------------

def dynamic_thresholding_fn(x0):
    """
    The dynamic thresholding method
    """
    dims = x0.dim()
    p = 0.995
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    s = expand_dims(torch.maximum(s, 1. * torch.ones_like(s).to(s.device)), dims)
    x0 = torch.clamp(x0, -s, s) / s
    return x0

#----------------------------------------------------------------------------

def dpm_pp_update(x, model_prev_list, t_prev_list, t, order, predict_x0=True, scale=1):
    if order == 1:
        return dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1], predict_x0=predict_x0, scale=scale)
    elif order == 2:
        return multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0, scale=scale)
    elif order == 3:
        return multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=predict_x0, scale=scale)
    else:
        raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

#----------------------------------------------------------------------------

def dpm_solver_first_update(x, s, t, model_s=None, predict_x0=True, scale=1):
    s, t = s.reshape(-1, 1, 1, 1), t.reshape(-1, 1, 1, 1)
    lambda_s, lambda_t = -1 * s.log(), -1 * t.log()
    h = lambda_t - lambda_s

    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    if predict_x0:
        x_t = (t / s) * x - scale * phi_1 * model_s
    else:
        x_t = x - scale * t * phi_1 * model_s
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
    t = t.reshape(-1, 1, 1, 1)
    model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
    t_prev_1, t_prev_0 = t_prev_list[-2].reshape(-1, 1, 1, 1), t_prev_list[-1].reshape(-1, 1, 1, 1)
    lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0 = h_0 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    if predict_x0:
        x_t = (t / t_prev_0) * x - scale * (phi_1 * model_prev_0 + 0.5 * phi_1 * D1_0)
    else:
        x_t = x - scale * (t * phi_1 * model_prev_0 + 0.5 * t * phi_1 * D1_0)
    return x_t

#----------------------------------------------------------------------------

def multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, predict_x0=True, scale=1):
    
    t = t.reshape(-1, 1, 1, 1)
    model_prev_2, model_prev_1, model_prev_0 = model_prev_list[-3], model_prev_list[-2], model_prev_list[-1]
    
    t_prev_2, t_prev_1, t_prev_0 = t_prev_list[-3], t_prev_list[-2], t_prev_list[-1]
    t_prev_2, t_prev_1, t_prev_0 = t_prev_2.reshape(-1, 1, 1, 1), t_prev_1.reshape(-1, 1, 1, 1), t_prev_0.reshape(-1, 1, 1, 1)
    lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = -1 * t_prev_2.log(), -1 * t_prev_1.log(), -1 * t_prev_0.log(), -1 * t.log()

    h_1 = lambda_prev_1 - lambda_prev_2
    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0, r1 = h_0 / h, h_1 / h
    D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
    D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
    D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
    D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
    
    phi_1 = torch.expm1(-h) if predict_x0 else torch.expm1(h)
    phi_2 = phi_1 / h + 1. if predict_x0 else phi_1 / h - 1.
    phi_3 = phi_2 / h - 0.5
    if predict_x0:
        x_t = (t / t_prev_0) * x - scale * (phi_1 * model_prev_0 - phi_2 * D1 + phi_3 * D2)
    else:
        x_t =  x - scale * (t * phi_1 * model_prev_0 + t * phi_2 * D1 + t * phi_3 * D2)
    return x_t


