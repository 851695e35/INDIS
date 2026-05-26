import torch
from samplers.general_solver import ODESolver
from samplers.general_solver import update_lists
from solver_utils import get_schedule


class DPM_SolverPP(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="data_prediction",
    ):
        super().__init__(noise_schedule, algorithm_type)
        self.noise_schedule = noise_schedule
        self.predict_x0 = algorithm_type == "data_prediction"

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
        # Call forward method
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

        # Return in the same format as other solvers
        return student_out



    def dpm_solver_first_update(self, x, s, t, model_s=None):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = self.model_fn(x, s)
        x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
        return x_t


    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        
        phi_1 = torch.expm1(-h)
        phi_2 = phi_1 / h + 1.0
        phi_3 = phi_2 / h - 0.5
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            + (alpha_t * phi_2) * D1
            - (alpha_t * phi_3) * D2
        )
        return x_t


    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order):
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))


    def get_time_steps_wrapped(self, schedule_type, device, num_steps):
        t_0 = self.noise_schedule.eps 
        t_T = self.noise_schedule.T 
        t_steps = self.get_time_steps(skip_type=schedule_type, t_T=t_T, t_0=t_0, N=num_steps, device=device)
        return t_steps

    def one_step(self, t1, t2, c1, t_prev_list, model_prev_list, step, x_next, order, first=True, run_pred=True):
        
        x_next = self.multistep_dpm_solver_update(x_next, model_prev_list, t_prev_list, t1, step)
        if run_pred:
            model_x_next = self.model_fn(x_next, t2) * c1
            update_lists(t_prev_list, model_prev_list, t1, model_x_next, order, first=first)
        return x_next

    def sample(
        self,
        model_fn,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="uniform",
        lower_order_final=True,
        flags=None,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = self.noise_schedule.eps if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        
        timesteps, timesteps2 = self.prepare_timesteps(steps=steps, t_start=t_T, t_end=t_0, skip_type=skip_type, device=device, load_from=flags.load_from)

        with torch.no_grad():
            return self.sample_simple(model_fn, x, order, lower_order_final, timesteps, timesteps2)
        

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
            
        step = 0
        x = latents
        t1 = timesteps[...,step]
        t2 = timesteps2[..., step]
        c1 = cn[...,step]

        t1 = t1.view(-1, 1, 1, 1)
        c1 = c1.view(-1, 1, 1, 1)
        
        steps = num_steps
        t_prev_list = [t1]
        
        if afs:
            d_cur = x / ((1+t1**2).sqrt()) * c1
            model_prev_list = [d_cur*c1]
        else:
            model_prev_list = [self.model_fn(x, t2)*c1]
        
        if return_inters:
            x_list = [x]
            
        for step in range(1, order):
            t1 = timesteps[...,step]
            t2 = timesteps2[...,step]
            c1 = cn[...,step]
            t1 = t1.view(-1, 1, 1, 1)
            c1 = c1.view(-1, 1, 1, 1)
            x = self.one_step(t1, t2, c1, t_prev_list, model_prev_list, step, x, order, first=True, run_pred=True)
            if return_inters:
                x_list.append(x)
        
        for step in range(order, steps + 1):
            t1 = timesteps[...,step]
            t2 = timesteps2[...,step]
            c1 = cn[...,step]
            t1 = t1.view(-1, 1, 1, 1)
            c1 = c1.view(-1, 1, 1, 1)
            step_order = min(order, steps + 1 - step)
            run_pred = True
            if step == steps:
                run_pred = False
            x = self.one_step(t1, t2, c1, t_prev_list, model_prev_list, step_order, x, order, first=False, run_pred=run_pred)
            if return_inters:
                x_list.append(x)
                
        if return_inters:
            # convert x_list to a tensor
            x_list = torch.stack(x_list)
            return x_list
        return x

