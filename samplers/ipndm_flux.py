import torch
from samplers.general_solver import ODESolver
from solver_utils import get_schedule
from einops import rearrange, repeat
from flux.modules.conditioner import HFEmbedder
from torch import Tensor
from torch.utils.checkpoint import checkpoint
def einsum_float_double(string, a, b):
    """
    Compute einsum(a, b) with float64 precision.
    """
    return torch.einsum(string, a.double(), b.double())

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)
    # [1, 16, 64, 64] -> [1, 1024, 64]
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    if isinstance(prompt, str):
        prompt = [prompt]

    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    img = img.to(txt.dtype)

    return img, img_ids.to(img.device), txt.to(img.device), txt_ids.to(img.device), vec.to(img.device)
    
import math
def unpack(x: Tensor, height: int, width: int) -> Tensor:
    # First determine the output shape
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    
    # Use checkpoint to save memory during reshaping

    def reshape_fn(x):
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h,
            w=w,
            ph=2,
            pw=2,
        )
    
    return reshape_fn(x)

from typing import Callable
def get_flux_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
    device: torch.device = None,
) -> Tensor:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

    def time_shift(mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def get_lin_function(
        x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps


class FlowModelWrapper(torch.nn.Module):
    def __init__(self, flow_model, img_ids, txt, txt_ids, vec, guidance_vec):
        super().__init__()
        self.flow_model = flow_model
        self.img_ids = img_ids
        self.txt = txt
        self.txt_ids = txt_ids
        self.vec = vec
        self.guidance_vec = guidance_vec
        
    def forward(self, x, t):
        return self.flow_model(
            img=x,
            timesteps=t,
            img_ids=self.img_ids,
            txt=self.txt,
            txt_ids=self.txt_ids,
            y=self.vec,
            guidance=self.guidance_vec
        )

# when applied to flux, ipdnm directly reduce to Adams-Bashforth method.
class iPNDM_Flux(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="velocity_prediction",
    ):
        super().__init__(noise_schedule, algorithm_type)
        assert algorithm_type == "velocity_prediction" # need to be velocity prediction

    def get_time_steps_wrapped(self, schedule_type, device, num_steps, is_dev=True):

        H = W = 64
        t_steps = get_flux_schedule(
            num_steps=num_steps, 
            image_seq_len=H * W // 4,
            shift=is_dev,
            device=device,
            )

        return t_steps

    def decode(self, x, ae, width, height, train=False):
        # decode latents to pixel space
        batch_x = unpack(x.float(), width, height)
        
        # Use checkpointing for the entire batch processing
        def decode_single_sample(single_x, ae):
            single_x = single_x.unsqueeze(0)
            with torch.autocast(device_type=single_x.device.type, dtype=torch.bfloat16):
                decoded = ae.decode(single_x)
            
            decoded = decoded.clamp(-1, 1)
            # Just return the tensor without squeezing to avoid copy
            return decoded[0]
        
        # Process batches with checkpointing and collect results
        return_batch = []
        
        for i, x_item in enumerate(batch_x):
            # Use checkpoint to save memory during decoding
            if not train:
                decoded = decode_single_sample(x_item, ae)
            else:
                decoded = checkpoint(
                    decode_single_sample,
                    x_item,
                    ae,
                    preserve_rng_state=False,
                    use_reentrant=False
                )
            
            return_batch.append(decoded)
            
        
        # Stack with careful memory management
        result = torch.stack(return_batch)
        
        # Clear references to free memory
        del batch_x, return_batch
        
        return result

    def __call__(self, 
                latents,
                net,
                prompts,
                guidance,
                hypernets = None,
                num_steps = None,
                afs = False,
                schedule_type='polynomial',
                schedule_rho=7,
                return_inters=False,
                order=2,
                train=False,
                cpu_offload=False,
                **kwargs):
        student_out = self.forward(
            latents=latents,
            net=net,
            prompts=prompts,
            guidance=guidance,
            hypernets=hypernets,
            num_steps=num_steps,
            afs=afs,
            schedule_type=schedule_type,
            schedule_rho=schedule_rho,
            return_inters=return_inters,
            order=order,
            cpu_offload=cpu_offload,
        )
        

        return student_out

    def forward(self, 
                latents,
                net,
                prompts,
                guidance,
                hypernets = None,
                num_steps = None,
                afs = False,
                schedule_type='polynomial',
                return_inters=False,
                order=2,
                cpu_offload=False,
                **kwargs):


        if cpu_offload:
            net['t5'] = net['t5'].to(latents.device)
            net['clip'] = net['clip'].to(latents.device)

        transformed_latents, img_ids, txt, txt_ids, vec = prepare(net['t5'], net['clip'], latents, prompts)

        if cpu_offload:
            net['t5'] = net['t5'].cpu()
            net['clip'] = net['clip'].cpu()


        guidance_vec = torch.full((latents.shape[0],), guidance, device=latents.device, dtype=txt.dtype)

        self.model = net['flow']
        
        inputs = (img_ids, txt, txt_ids, vec, guidance_vec)
        if hypernets is not None:
            def model_fn(x, t):
                # return self.model(x, t, *inputs)
                return checkpoint(self.model, x, t, *inputs, use_reentrant=False)
            self.model_fn = model_fn
        else:
            self.model_fn = lambda x, t: self.model(x, t.expand((x.shape[0])), *inputs)
        
        if hypernets is None:
            is_dev = True
            timesteps = self.get_time_steps_wrapped(latents, latents.device, num_steps)
            timesteps2 = timesteps
            cn = [1.0] * len(timesteps)
            cn = torch.tensor(cn)
        else:
            hypernets_fn = hypernets
            condition = (txt, vec)
            hypernets = hypernets_fn(latents, condition)
            timesteps = hypernets[0]
            timesteps2 = hypernets[1]
            cn = hypernets[2]
        x = transformed_latents.to(txt.dtype)
        timesteps = timesteps.to(txt.dtype)
        timesteps2 = timesteps2.to(txt.dtype)
        cn = cn.to(txt.dtype)
    
        vel_buffer = list()
        x_next = x
        
        if return_inters:
            x_list = [x]

        steps = num_steps
        for step in range(steps):
            step_order = min(order, step + 1)
            
            c_cur1 = cn[..., step]
            t_cur1, t_next1 = timesteps[..., step], timesteps[..., step + 1]
            t_cur2, t_next2 = timesteps2[..., step], timesteps2[..., step + 1]

            # for flux, the latent dim is only 3 including batch.
            t_cur1 = t_cur1.view(-1, 1, 1)
            t_next1 = t_next1.view(-1, 1, 1)
            c_cur1 = c_cur1.view(-1, 1, 1).to(x.device)
            
            x_cur = x_next 
            vel_cur = self.model_fn(x_cur, t_cur2) * c_cur1
            
            vel_buffer.append(vel_cur)
                
            if step_order == 1:
                x_next = x_cur + vel_buffer[-1] * (t_next1 - t_cur1)
            elif step_order == 2:
                x_next = x_cur + (3/2 * vel_cur - 1/2 * vel_buffer[-2]) * (t_next1 - t_cur1)
            elif step_order == 3:
                x_next = x_cur + (23/12 * vel_cur - 16/12 * vel_buffer[-2] + 5/12 * vel_buffer[-3]) * (t_next1 - t_cur1)
            elif step_order == 4:
                x_next = x_cur + (55/24 * vel_cur - 59/24 * vel_buffer[-2] + 37/24 * vel_buffer[-3] - 9/24 * vel_buffer[-4]) * (t_next1 - t_cur1)
        
            # Only use checkpoint during training
            # print("x_next", step, x_next.shape)
        
            # Update velocity buffer
            if len(vel_buffer) == order - 1:
                for k in range(order - 2):
                    vel_buffer[k] = vel_buffer[k + 1]
                vel_buffer[-1] = vel_cur
            else:
                vel_buffer.append(vel_cur)
            
            if return_inters:
                x_list.append(x_next)
        
        if return_inters:
            x_list = torch.stack(x_list)
            return x_list
        
        vel_buffer.clear()
        return x_next



