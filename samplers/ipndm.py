import torch
from samplers.general_solver import ODESolver
from solver_utils import get_schedule
# import checkpoints
from torch.utils.checkpoint import checkpoint
def einsum_float_double(string, a, b):
    """
    Compute einsum(a, b) with float64 precision.
    """
    return torch.einsum(string, a.double(), b.double())

class iPNDM(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="noise_prediction",
    ):
        super().__init__(noise_schedule, algorithm_type)
        self.noise_schedule = noise_schedule # noiseScheduleVP
        assert algorithm_type == "noise_prediction" # need to be noise prediction!
        self.predict_x0 = algorithm_type == "data_prediction" # false


    def noise_pred_fn(self, x, t, net, class_labels=None, condition=None, unconditional_condition=None):
        t_input = t.expand((x.shape[0]))
        if hasattr(net, 'guidance_type'):
            output = net(x, t_input, condition=condition, unconditional_condition=unconditional_condition)
        else:
            output = net(x, t_input, class_labels=class_labels)
        
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if isinstance(alpha_t, torch.Tensor):
            alpha_t = alpha_t.view(-1, 1, 1, 1)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t = sigma_t.view(-1, 1, 1, 1)
        return (x - alpha_t * output) / sigma_t

    def data_pred_fn(self, x, t, net, class_labels=None, condition=None, unconditional_condition=None):
        
        model = lambda x, t, c: net.apply_model(x, t, c)
        if self.noise_schedule.schedule == "discrete":
            t_input = (t - 1.0 / self.noise_schedule.total_N) * 1000.0
            t_input = t_input.expand((x.shape[0]))
        else:
            t_input = t.expand((x.shape[0]))
        if class_labels is None and condition is None:
            # output = checkpoint(model, x, t_input, None, use_reentrant=False)
            output = model(x, t_input, None)
        elif hasattr(net, 'guidance_type'):
            output = net(x, t_input, condition=condition, unconditional_condition=unconditional_condition)
        else:
            output = net(x, t_input, class_labels=class_labels)
        return output
    

    def get_time_steps_wrapped(self, schedule_type, device, num_steps):
        t_0 = self.noise_schedule.eps 
        t_T = self.noise_schedule.T 
        t_steps = self.get_time_steps(skip_type=schedule_type, t_T=t_T, t_0=t_0, N=num_steps, device=device)
        return t_steps

    def __call__(self, 
                latents,
                net,
                class_labels,
                condition = None,
                unconditional_condition = None,
                hypernets = None,
                num_steps = None,
                afs = False,
                sigma_min=0.002, 
                sigma_max=80, 
                schedule_type='polynomial',
                schedule_rho=7, 
                denoise_to_zero=False,
                return_inters=False,
                predict_x0=False,
                train=False,
                order=2,
                lower_order_final=True,
                **kwargs):
        student_out = self.forward(
            latents=latents,
            net=net,
            class_labels=class_labels,
            condition=condition,
            unconditional_condition=unconditional_condition,
            hypernets=hypernets,
            num_steps=num_steps,
            afs=afs,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            schedule_type=schedule_type,
            schedule_rho=schedule_rho,
            denoise_to_zero=denoise_to_zero,
            return_inters=return_inters,
            predict_x0=predict_x0,
            train=train,
            order=order,
            lower_order_final=lower_order_final,
        )
        
        if return_inters and train is False:
            return student_out

        return student_out

    def forward(self, 
                latents,
                net,
                class_labels,
                condition,
                unconditional_condition,
                hypernets = None,
                num_steps = None,
                afs = False,
                schedule_type='polynomial',
                return_inters=False,
                predict_x0=False,
                order=2,
                **kwargs):
        if not predict_x0:
            self.model = lambda x, t: self.data_pred_fn(x, t, net, class_labels, condition, unconditional_condition)
        else:          
            self.model = lambda x, t: self.noise_pred_fn(x, t, net, class_labels, condition, unconditional_condition)

        if hypernets is None:
            timesteps = self.get_time_steps_wrapped(schedule_type, latents.device, num_steps)
            timesteps2 = timesteps
            cn = [1.0] * len(timesteps)
            cn = torch.tensor(cn)
        else:
            hypernets_fn = hypernets
            hypernets = hypernets_fn(latents, class_labels)
            timesteps = hypernets[0]
            timesteps2 = hypernets[1]
            cn = hypernets[2]
        x = latents

        # print(timesteps[0])
        # print(timesteps[1])

        # print(cn[0])
        # print(cn[1])
    
        epsilon_buffer = list()
        x_next = x
        
        if return_inters:
            x_list = [x]

        steps = num_steps
        for step in range(steps):
            step_order = min(order, step + 1)
            
            c_cur1 = cn[..., step]
            t_cur1, t_next1 = timesteps[..., step], timesteps[..., step + 1]
            t_cur2, t_next2 = timesteps2[..., step], timesteps2[..., step + 1]

            t_cur1 = t_cur1.view(-1, 1, 1, 1)
            t_next1 = t_next1.view(-1, 1, 1, 1)
            c_cur1 = c_cur1.view(-1, 1, 1, 1).to(x.device)
            
            x_cur = x_next 
            if step == 0 and afs: 
                if predict_x0:
                    epsilon_cur = x_cur / ((1+t_cur1**2).sqrt()) * c_cur1
                else:
                    epsilon_cur = x_cur
            else:
                epsilon_cur = self.model_fn(x_cur, t_cur2) * c_cur1
                
            
            lambda_s, lambda_t = self.noise_schedule.marginal_lambda(t_cur1), self.noise_schedule.marginal_lambda(t_next1)
            h = lambda_t - lambda_s
            log_alpha_s, log_alpha_t = self.noise_schedule.marginal_log_mean_coeff(t_cur1), self.noise_schedule.marginal_log_mean_coeff(t_next1)
            sigma_t = self.noise_schedule.marginal_std(t_next1)
            phi_1 = torch.expm1(h)
            if step_order == 1:
                x_next = (
                    torch.exp(log_alpha_t - log_alpha_s) * x_cur 
                    - (sigma_t * phi_1) * epsilon_cur
                )
            elif step_order == 2:
                x_next = (
                    torch.exp(log_alpha_t - log_alpha_s) * x_cur 
                    - (sigma_t * phi_1) * (3 * epsilon_cur - 1 * epsilon_buffer[-1]) / 2
                )
            elif step_order == 3:
                x_next = (
                    torch.exp(log_alpha_t - log_alpha_s) * x_cur 
                    - (sigma_t * phi_1) * (23 * epsilon_cur - 16 * epsilon_buffer[-1] + 5 * epsilon_buffer[-2]) / 12
                )
            elif step_order == 4:
                x_next = (
                    torch.exp(log_alpha_t - log_alpha_s) * x_cur 
                    - (sigma_t * phi_1) * (55 * epsilon_cur - 59 * epsilon_buffer[-1] + 37 * epsilon_buffer[-2] - 9 * epsilon_buffer[-3]) / 24
                )
            
            if len(epsilon_buffer) == order - 1:
                for k in range(order - 2):
                    epsilon_buffer[k] = epsilon_buffer[k + 1]
                epsilon_buffer[-1] = epsilon_cur
            else:
                epsilon_buffer.append(epsilon_cur)
            
            if return_inters:  
                x_list.append(x_next)
            
        if return_inters:
            x_list = torch.stack(x_list)
            return x_list
        return x_next



