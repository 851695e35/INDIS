import torch
from samplers.general_solver import ODESolver
from solver_utils import get_schedule

def einsum_float_double(string, a, b):
    """
    Compute einsum(a, b) with float64 precision.
    """
    return torch.einsum(string, a.double(), b.double()).float()

class UniPC(ODESolver):
    def __init__(
        self,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
        variant='bh1',
    ):
        super().__init__(noise_schedule, algorithm_type)
        self.noise_schedule = noise_schedule # noiseScheduleVP
        assert algorithm_type in ["data_prediction", "noise_prediction"]
        self.correcting_xt_fn = correcting_xt_fn # None
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio # 0.995
        self.thresholding_max_val = thresholding_max_val # 1.0
        
        self.variant = variant # bh1
        self.predict_x0 = algorithm_type == "data_prediction" # true

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

        return student_out

    def multistep_uni_pc_bh_update(self, x, model_prev_list, t_prev_list, t, c1, t2, order, x_t=None, use_corrector=True):
        if len(t.shape) == 0:
            t = t.view(-1)
            t2 = t2.view(-1)
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        # print(t.shape, "error4")
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)
        # print(lambda_t.shape, lambda_prev_0.shape, "error2")
        h = lambda_t - lambda_prev_0
        # print(h.shape)
        if len(h.shape) == 4:
            h = h.squeeze(-1).squeeze(-1).squeeze(-1)

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            # print(lambda_prev_i.shape, lambda_prev_0.shape, h.shape)
            rk = (lambda_prev_i - lambda_prev_0) / h.view(-1, 1, 1, 1)

            
            # print(model_prev_i.shape, model_prev_0.shape, rk.shape)
            D1s.append((model_prev_i - model_prev_0) / rk)
            rk = rk.squeeze(-1).squeeze(-1).squeeze(-1)
            rks.append(rk)
        # append a tensor of [batch] of ones
        one = torch.tensor([1.0], device=x.device)
        
        if len(rks) > 0:
            one = one.expand(rks[-1].shape)

        rks.append(one)
        # print(rks[0].shape, '2')
        # conver the list of tensor to a tensor

        rks = torch.stack(rks)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh) # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()
        # print(rks.shape, 2)
        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i 
        # print(R[0].shape, b[0].shape, '1')
        R = torch.stack(R)
        # if (len(R.shape) == 3):
        #     R = R.permute(2,1,0)
        # if b[0] is [32] b is [32, 2], else if b[0] is [], then b is [2]
        if b[0].shape == torch.Size([1]):
            b = torch.cat(b)
        else: # a list of [32] tensor, make it a [2, 32] tensor
            b = torch.stack(b)
            # swap the first and second dimension
            # b = b.permute(1, 0)
        
        # print(R.shape, b.shape)
        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(R.shape) ==3 :
            R = R.squeeze(-1)
        if len(D1s) > 0:
            # print(D1s[0].shape)
            D1s = torch.stack(D1s, dim=1) # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                # print(R.shape, b.shape, "error4")
                if len(R.shape) == 3:
                    R = R.permute(2,1,0)
                    b = b.permute(1,0)
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[...,:-1, :-1], b[...,:-1])
        else:
            D1s = None

        if use_corrector:
            # for order 1, we use a simplified version
            # print(R.shape, b.shape, "error")
            # if len(R.shape) == 3:   
            #     R = R.permute(2,1,0)
            #     b = b.permute(1,0)
            # print(R.shape, b.shape, "error_single")
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)
                # print(rhos_c.shape, "error")

        model_t = None


        h_phi_1 = h_phi_1.view(-1, 1, 1, 1)
        x_t_ = (
            sigma_t / sigma_prev_0 * x
            - alpha_t * h_phi_1 * model_prev_0
        )
        if x_t is None:
            if use_predictor:
                if len(rhos_c.shape) == 2:
                    pred_res = einsum_float_double('bk,bkchw->bchw', rhos_c[..., :-1], D1s)
                else:
                    pred_res = einsum_float_double('k,bkchw->bchw', rhos_p, D1s) # D1s float64, rhos_p float32
            else:
                pred_res = 0
            B_h = B_h.view(-1, 1, 1, 1)
            # print(x_t_.shape, alpha_t.shape, B_h.shape)
            x_t = x_t_ - alpha_t * B_h * pred_res

        if use_corrector:
            # model_t = self.model_fn(x_t, t2)
            # print(t2.shape)
            model_t = self.model_fn(x_t, t2) * c1
            # print(c1,t)
            if D1s is not None:
                # print(rhos_c.shape, D1s.shape, "error")
                if len(rhos_c.shape) == 2:
                    corr_res = einsum_float_double('bk,bkchw->bchw', rhos_c[..., :-1], D1s)
                else:
                    corr_res = einsum_float_double('k,bkchw->bchw', rhos_c[:-1], D1s)

                
            else:
                corr_res = 0
            D1_t = (model_t - model_prev_0)
            # print(x_t_.shape, alpha_t.shape, B_h.shape, rhos_c[..., -1].shape, D1_t.shape)
            if rhos_c[..., -1].shape != torch.Size([]):
                tmp = rhos_c[..., -1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                tmp = rhos_c[..., -1]
            x_t = x_t_ - alpha_t * B_h * (corr_res + tmp * D1_t)
            # print(x_t.shape, "error")
        return x_t, model_t

    
    def one_step(self, t1, t2, c1, t_prev_list, model_prev_list, step, x_next, order, first=True, use_corrector=True):
        x_next, model_x_next = self.multistep_uni_pc_bh_update(x_next, model_prev_list, t_prev_list, t1, c1, t2, step, use_corrector=use_corrector)
        # print(model_x_next[0][0][0][0])
        # print(t1, first)
        if model_x_next is None:
            model_x_next = self.model_fn(x_next, t2) * c1
        if model_x_next is not None:
            self.update_lists(t_prev_list, model_prev_list, t1, model_x_next, order, first=first)
        return x_next


    def update_lists(self, t_list, model_list, t_, model_x, order, first=False):
        if first:
            t_list.append(t_)
            model_list.append(model_x)
            # print(model_x[0][0][0][0])
            return
        for m in range(order - 1):
            t_list[m] = t_list[m + 1]
            model_list[m] = model_list[m + 1]
        t_list[-1] = t_
        model_list[-1] = model_x
        # print(model_x[0][0][0][0],"11")

    
    def sample(self, model_fn, x, steps=20, t_start=None, t_end=None, order=2, \
                skip_type='uniform', lower_order_final=True, flags=None, return_intermediates=False
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        t_0 = self.noise_schedule.eps if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        timesteps, timesteps2 = self.prepare_timesteps(steps=steps, t_start=t_T, t_end=t_0, skip_type=skip_type, device=device, load_from=flags.load_from)
        
        with torch.no_grad():
            return self.forward(model_fn, x, order, lower_order_final, timesteps, timesteps2)
        
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

        device = latents.device
        if hypernets is None:
            timesteps = self.get_time_steps_wrapped(schedule_type, device, num_steps)
            timesteps2 = timesteps
            cn = [1.0] * len(timesteps)
            cn = torch.tensor(cn, device=device)
        else:
            hypernets_fn = hypernets
            hypernets = hypernets_fn(latents, class_labels)
            timesteps = hypernets[0].to(device)
            timesteps2 = hypernets[1].to(device)
            cn = hypernets[2].to(device)

        step = 0
        x = latents
        t1 = timesteps[...,step]
        t1 = t1.view(-1, 1, 1, 1)
        t2 = timesteps2[step]
        c1 = cn[..., step]
        c1 = c1.view(-1, 1, 1, 1)
        # if train:
        #     print(timesteps)
        # print(timesteps.shape, "error5")
        steps = num_steps
        t_prev_list = [t1]
        
        if afs:
            d_cur = x / ((1+t1**2).sqrt())
            model_prev_list = [d_cur*c1]
        else:
            model_prev_list = [self.model_fn(x, t2)*c1]
        # print(timesteps)
        if return_inters:
            x_list = [x]
            
        for step in range(1, order):
            t1 = timesteps[..., step]
            t2 = timesteps2[..., step]
            c1 = cn[..., step]
            t1 = t1.view(-1, 1, 1, 1)
            c1 = c1.view(-1, 1, 1, 1)
            x = self.one_step(t1, t2, c1, t_prev_list, model_prev_list, step, x, order, first=True)
            if return_inters:
                x_list.append(x)
        
        for step in range(order, steps + 1):
            t1 = timesteps[...,step]
            t2 = timesteps2[..., step]
            c1 = cn[..., step]
            t1 = t1.view(-1, 1, 1, 1)
            c1 = c1.view(-1, 1, 1, 1)
            step_order = min(order, steps + 1 - step)
            if step == steps:
                use_corrector = False
            else:
                use_corrector = True
            x = self.one_step(t1, t2, c1, t_prev_list, model_prev_list, step_order, x, order, first=False, use_corrector=use_corrector)
            if return_inters:
                x_list.append(x)
                
        if return_inters:
            # convert x_list to a tensor
            x_list = torch.stack(x_list)
            return x_list
        return x

