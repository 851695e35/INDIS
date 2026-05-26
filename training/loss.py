import torch
from torch_utils import persistence
from torch_utils import distributed as dist
from samplers.uni_pc import UniPC
from samplers.dpm_solverpp import DPM_SolverPP
from samplers.ipndm import iPNDM
from solver_utils import get_schedule
from piq import LPIPS
from inception import compute_inception_mse_loss
from inception import InceptionFeatureExtractor
import lpips
from torch import autocast
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

#----------------------------------------------------------------------------

def get_solver_fn(solver_name, noise_schedule=None):
    if solver_name == 'uni_pc':
        solver_fn = UniPC(noise_schedule)
    elif solver_name == 'dpm_solverpp':
        solver_fn = DPM_SolverPP(noise_schedule)
    elif solver_name == 'ipndm':
        solver_fn = iPNDM(noise_schedule)
    elif solver_name == 'ipndm_flux':
        from samplers.ipndm_flux import iPNDM_Flux
        solver_fn = iPNDM_Flux(noise_schedule)
    else:
        raise ValueError("Got wrong solver name {}".format(solver_name))
    return solver_fn

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# compute the distance between two images
def compute_distance_between_two(x, y, n_channels=3, resolution=256):
    '''
    x: bs x 3 x 256 x 256
    y: bs x 3 x 256 x 256
    '''
    square_distance = (x - y) ** 2
    distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance

def compute_distance_between_two_L1(x, y, n_channels=3, resolution=256):
    '''
    x: bs x 3 x 256 x 256
    y: bs x 3 x 256 x 256
    '''
    square_distance = torch.abs(x - y)
    distance = square_distance.sum(dim=(1, 2, 3)) / (n_channels * resolution * resolution)
    return distance


@persistence.persistent_class
class iLD3_loss:
    def __init__(
        self, num_steps=None, sampler_stu=None, sampler_tea=None, M=None, 
        schedule_type=None, schedule_rho=None, afs=False, order=3, max_order=None, 
        sigma_min=None, sigma_max=None, predict_x0=True, lower_order_final=True, iLD3_predictor=None,
        teacher_schedule_type=None, teacher_schedule_rho=None, tn_training_style=None, loss_type=None,
        intermediary_loss=False, mid_loss_type=None, mid_loss_weight=None,
        l1_reg_weight=0, l2_reg_weight=0, loss_weight=None, alter_training=False,
        debug_visualize=False,
    ):
        """
        Initialize iLD3 loss with regularization.
        
        Args:
            ... (existing args) ...
            l1_reg_weight (float): Weight for L1 regularization. Default: 1e-5
                - L1 regularization encourages sparsity
                - Common range: 1e-6 to 1e-4
                - Start small and increase if needed
            l2_reg_weight (float): Weight for L2 regularization. Default: 1e-4
                - L2 regularization prevents large parameter values
                - Common range: 1e-5 to 1e-3
                - Start small and increase if needed
        """
        self.num_steps = num_steps
        self.noise_schedule = iLD3_predictor.noise_schedule
        self.solver_stu = get_solver_fn(sampler_stu, self.noise_schedule)
        self.schedule_type = schedule_type
        self.schedule_rho = schedule_rho
        self.teacher_schedule_type = teacher_schedule_type
        self.teacher_schedule_rho = teacher_schedule_rho
        self.afs = afs
        self.order = order
        self.max_order = max_order
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.predict_x0 = predict_x0
        self.lower_order_final = lower_order_final
        self.tn_training_style = tn_training_style
        self.num_steps_teacher = None
        self.tea_slice = None           # a list to extract the intermediate outputs of teacher sampling trajectory
        self.t_steps = None
        self.loss_type = loss_type
        self.mid_loss_type = mid_loss_type
        self.mid_loss_weight = mid_loss_weight
        self.iLD3_predictor = iLD3_predictor
        self.intermediary_loss = intermediary_loss
        self.mask = None
        self.device = self.iLD3_predictor.tn_params.device
        self.channels = self.iLD3_predictor.img_size[0] # C
        self.resolution = self.iLD3_predictor.img_size[1] # H, W
        self.l1_reg_weight = l1_reg_weight  # New L1 regularization weight
        self.l2_reg_weight = l2_reg_weight  # New L2 regularization weight
        self.loss_weight = loss_weight
        self.alter_training = alter_training
        self.debug_visualize = debug_visualize
        self.feature_extractor = InceptionFeatureExtractor(device=self.device)
        if self.intermediary_loss:
            self.loss_mid = self._initialize_loss_fn(self.mid_loss_type)

        self.loss_fn = self._initialize_loss_fn(self.loss_type)

    def compute_l2_loss(self, x, y):
        return compute_distance_between_two(x, y, self.channels, self.resolution)

    def compute_l1_loss(self, x, y):
        return compute_distance_between_two_L1(x, y, self.channels, self.resolution)

    def compute_inception_loss(self, x, y):
        x = (x * 127.5 + 128).clip(0, 255)
        y = (y * 127.5 + 128).clip(0, 255)
        return compute_inception_mse_loss(x, y, self.feature_extractor)

    def _initialize_loss_fn(self, loss_type):
        if loss_type == 'lpips':
            return lpips.LPIPS(net='vgg').to(self.device)
        elif loss_type == 'l2':
            return self.compute_l2_loss
        elif loss_type == 'l1':
            return self.compute_l1_loss
        elif loss_type == 'inception':
            return self.compute_inception_loss
        else:
            raise NotImplementedError



    def __call__(self, iLD3_predictor_ddp, start_step, end_step, net, tensor_in, guidance=0.0, class_labels=None, prompts=None, teacher_out=None, rounds=0, model_source='edm', dataset_name='lsun_bedroom_ldm'):
        # Student trajectory
        if self.intermediary_loss:
            self.student_slice = [i for i in range(start_step, end_step+1) if self.mask[i]]
        else:
            self.student_slice = None
        if model_source == 'ldm' and dataset_name == 'lsun_bedroom_ldm':
            with net.ema_scope():
                student_out = self.solver_stu(
                    latents=tensor_in,
                    net=net,
                    class_labels=class_labels,
                    hypernets=iLD3_predictor_ddp,
                    num_steps=end_step-start_step,
                    afs=self.afs,
                    schedule_type=self.schedule_type,
                    schedule_rho=self.schedule_rho,
                    denoise_to_zero=False,
                    return_inters=self.intermediary_loss,
                    predict_x0=self.predict_x0,
                    train=True,
                    order=self.max_order,
                    lower_order_final=self.lower_order_final,
                    mask=self.mask,
                )
                # raw decode first stage will make the gradient fail to backprop

                student_out = net.differentiable_decode_first_stage(student_out)
        elif model_source == 'ldm' and dataset_name == 'ms_coco':

 
            student_out = self.solver_stu(
                latents=tensor_in,
                net=net,
                hypernets=iLD3_predictor_ddp,
                num_steps=end_step-start_step,
                afs=self.afs,
                schedule_type=self.schedule_type,
                schedule_rho=self.schedule_rho,
                denoise_to_zero=False,
                guidance=guidance,
                return_inters=self.intermediary_loss,
                prompts=prompts,
                order=self.max_order,
                cpu_offload=False, # 
            )
            student_out = self.solver_stu.decode(student_out, net['ae'], 512, 512, train=True)
            torch.cuda.empty_cache()
        else:
            student_out = self.solver_stu(
                latents=tensor_in,
                net=net,
                class_labels=class_labels,
                hypernets=iLD3_predictor_ddp,
                num_steps=end_step-start_step,
                afs=self.afs,
                sigma_min=None,
                sigma_max=None,
                schedule_type=self.schedule_type,
                schedule_rho=self.schedule_rho,
                denoise_to_zero=False,
                return_inters=self.intermediary_loss,
                predict_x0=self.predict_x0,
                train=True,
                order=self.max_order,
                lower_order_final=self.lower_order_final,
                mask=self.mask,
            )            

        if self.intermediary_loss:
            student_out = student_out[self.student_slice]

        if self.intermediary_loss:
            # Compute loss for the final output
            loss = self.loss_fn(student_out[-1], teacher_out[-1])
            for i in range(len(student_out)-1):
                loss_mid = self.loss_mid(student_out[i], teacher_out[i]) * self.mid_loss_weight
                loss = loss + loss_mid
        else:
            loss = self.loss_fn(student_out, teacher_out)

        # Add regularization terms
        if self.l1_reg_weight > 0:
            l1_reg = torch.norm(iLD3_predictor_ddp.module.tn_params, p=1)
            loss = loss + self.l1_reg_weight * l1_reg

        if self.l2_reg_weight > 0:
            l2_reg = torch.norm(iLD3_predictor_ddp.module.tn_params, p=2)
            loss = loss + self.l2_reg_weight * l2_reg

        str2print = f"Step: {start_step} | Loss: {loss.mean().item():8.4f} "
        if self.l1_reg_weight > 0:
            str2print += f"| L1_reg: {l1_reg.item():5.4f} "
        if self.l2_reg_weight > 0:
            str2print += f"| L2_reg: {l2_reg.item():5.4f} " 



        if self.debug_visualize:
            # Optional batch-level visualization for debugging only.
            import matplotlib.pyplot as plt
            import numpy as np

            # Convert tensors to numpy arrays
            # convert the type of student_out and teacher_out first to float32
            student_out = student_out.to(torch.float32)
            teacher_out = teacher_out.to(torch.float32)
            student_out_np = student_out.detach().cpu().numpy()
            teacher_out_np = teacher_out.detach().cpu().numpy()
            # convert to uint8
            student_out_np = (student_out_np * 127.5 + 128).clip(0, 255)
            teacher_out_np = (teacher_out_np * 127.5 + 128).clip(0, 255)
            student_out_np = student_out_np.astype(np.uint8)
            teacher_out_np = teacher_out_np.astype(np.uint8)

            # Plot the first image of student and teacher
            # plot two figs in one row
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # Transpose the arrays to get (H, W, C) format
            student_img = student_out_np[0].transpose(1, 2, 0)
            teacher_img = teacher_out_np[0].transpose(1, 2, 0)
            axs[0].imshow(student_img)
            axs[1].imshow(teacher_img)
            plt.savefig(f"student_teacher_first_image.png")
            plt.close()


        latents_norm = tensor_in.norm(dim=[1, 2, 3], p=2)
        if loss.ndim == 4:
            loss = loss.sum(dim=[1,2,3])
        elif loss.ndim == 2:
            loss = loss.sum(dim=1)
        if self.loss_weight == 'constant':
            pass
        elif self.loss_weight == 'div_norm':
            loss = loss / latents_norm
        elif self.loss_weight == 'mul_norm':
            loss = loss * latents_norm
        loss = loss.sum()    

        return loss, str2print, student_out
    
