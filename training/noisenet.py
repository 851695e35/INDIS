import torch
import torch.nn as nn
from torch_utils import persistence

# Add a custom identity module that can replace BatchNorm1d
class IdentityNorm(nn.Module):
    """Identity normalization layer that works with any batch size"""
    def __init__(self, num_features=None):
        super().__init__()
        self.num_features = num_features
        
    def forward(self, x):
        return x

@persistence.persistent_class
class NoiseHandler_condition(torch.nn.Module):

    def __init__(self, output_dim, num_channels, height, width, bound_tn=0, bound_tn2=0, bound_epscaling=0, hidden_dim=128, top_k=32, condition_type='uncond'):
        super().__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.bound_tn = bound_tn
        self.bound_tn2 = bound_tn2
        self.bound_epscaling = bound_epscaling
        self.segment = self.output_dim // 3
        self.top_k = min(top_k, num_channels * height)  # Ensure top_k doesn't exceed matrix dimensions
        self.condition_type = condition_type
        # Configure conditioning dimensions and networks based on condition type
        if self.condition_type == 'uncond' or self.condition_type is None:
            self.use_condition = False
        else:
            self.use_condition = True
            if self.condition_type == 'class_idx':
                self.condition_dim = 1000
                # Standard condition network for class indices
                tailored_hidden_dim = 8
                self.condition_net = nn.Sequential(
                    nn.Linear(self.condition_dim, tailored_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(tailored_hidden_dim, output_dim),
                )
            elif self.condition_type == 'text_embedding':
                
                self.t5_reduction = nn.Sequential(
                    nn.Linear(512, hidden_dim),  # Reduce T5 feature dimension
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),  # Process all tokens
                )
                
                self.clip_reduction = nn.Sequential(
                    nn.Linear(768, hidden_dim),  # Reduce CLIP feature dimension
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),  # Process all tokens
                )
                
                self.text_fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
            else:
                raise ValueError(f"Invalid condition type: {self.condition_type}")
        
        self.bn = IdentityNorm(hidden_dim) if self.condition_type == 'text_embedding' else nn.BatchNorm1d(hidden_dim)
        
        self.values_net = nn.Sequential(
            nn.Linear(num_channels * self.top_k, hidden_dim),
            self.bn,
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.u_net = nn.Sequential(
            nn.Linear(num_channels * height * self.top_k, hidden_dim),
            self.bn,
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.v_net = nn.Sequential(
            nn.Linear(num_channels * width * self.top_k, hidden_dim),
            self.bn,
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.svd_fusion = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            self.bn,
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        if self.use_condition:
            self.final_fusion = nn.Sequential(
                nn.Linear(output_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        self.sigmoid = torch.nn.Sigmoid()

    def process_text_embedding(self, condition):
        txt, vec = condition
        txt = txt.to(self.t5_reduction[0].weight.dtype)
        txt = txt.mean(dim=-1)
        txt_features = self.t5_reduction(txt)
        
        vec = vec.to(self.clip_reduction[0].weight.dtype)
        vec_features = self.clip_reduction(vec)
        
        combined = torch.cat([txt_features, vec_features], dim=-1)
        return self.text_fusion(combined)

    def feature_extractor(self, u, s, v):
        u_v = u.mean(dim=(1, 2, 3)) + v.mean(dim=(1, 2, 3))
        s_d = s.mean(dim=(1, 2))
        ret = u_v * s_d
        return ret
    
    def joint_feature(self, joint, delta, ratio=0.5):
        joint = joint / joint.norm(dim=1, keepdim=True)
        return joint * ratio + delta[..., None] * (1 - ratio)

    def forward(self, noise, condition=None):
        noise = noise / noise.norm(dim=1, keepdim=True)

        u, s, v = torch.linalg.svd(noise)
        delta = self.feature_extractor(u, s, v)

        v = v.transpose(-2, -1)
        
        s = s[:, :, :self.top_k]
        u = u[:, :, :self.top_k]
        v = v[:, :, :self.top_k]
        
        # Reshape components
        s = s.reshape(s.shape[0], -1)  # [batch_size, channels * top_k]
        u = u.reshape(u.shape[0], -1)  # [batch_size, num_channels * height * top_k]
        v = v.reshape(v.shape[0], -1)  # [batch_size, num_channels * width * top_k]

        # Process SVD components
        s_features = self.values_net(s)
        u_features = self.u_net(u)
        v_features = self.v_net(v)
        
        # Combine SVD features
        svd_features = torch.cat([u_features, v_features, s_features], dim=-1)
        svd_features = self.svd_fusion(svd_features)
        
        # Process condition if needed
        if self.use_condition and condition is not None:
            if self.condition_type == 'class_idx':
                condition = condition / torch.sqrt(torch.tensor(condition.shape[1], dtype=condition.dtype))
                condition_features = self.condition_net(condition)
            elif self.condition_type == 'text_embedding':
                condition_features = self.process_text_embedding(condition)
                
            # Combine SVD features with condition features
            joint = torch.cat([svd_features, condition_features], dim=-1)
            joint = self.final_fusion(joint)
        else:
            # Use only SVD features if no condition or condition_type is 'uncond'
            joint = svd_features
        
        joint = self.joint_feature(joint, delta)
        

        # print(joint.shape)
        # print(joint[0].min(), joint[0].max())
        # print(joint[1].min(), joint[1].max())

        # Generate outputs
        tns = joint[..., :self.segment]

        # print(tns[0])
        # print(tns[1])
        
        
        tns = (self.sigmoid(tns) * 2 - 1) * self.bound_tn + 1
        
        tn2s = joint[..., self.segment:2*self.segment]
        tn2s = (self.sigmoid(tn2s) * 2 - 1) * self.bound_tn2
        
        epscalings = joint[..., 2*self.segment:]
        epscalings = (self.sigmoid(epscalings) * 2 - 1) * self.bound_epscaling + 1
        
        return tns, tn2s, epscalings



@persistence.persistent_class
class NoiseHandler_fullsvd(torch.nn.Module):

    def __init__(self, output_dim, num_channels, height, width, bound_tn = 0, bound_tn2 = 0, bound_epscaling = 0, hidden_dim = 128, top_k = 32):
        super().__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.bound_tn = bound_tn
        self.bound_tn2 = bound_tn2
        self.bound_epscaling = bound_epscaling
        self.segment = self.output_dim // 3
        self.top_k = min(top_k, num_channels * height)  # Ensure top_k doesn't exceed matrix dimensions
        
        # Create identity norm for batch-size-safe normalization
        self.bn = IdentityNorm(hidden_dim)
        
        # Networks for top-k singular values
        self.values_net = nn.Sequential(
            nn.Linear(num_channels * self.top_k, hidden_dim),
            self.bn,
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Networks for top-k left singular vectors
        self.u_net = nn.Sequential(
            nn.Linear(num_channels * height * self.top_k, hidden_dim),
            # Use comment instead of BatchNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Networks for top-k right singular vectors
        self.v_net = nn.Sequential(
            nn.Linear(num_channels * width * self.top_k, hidden_dim),
            # Use comment instead of BatchNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.joint_net = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            # Use comment instead of BatchNorm
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, noise):
        # Compute SVD
        u, s, v = torch.linalg.svd(noise)
        # transpose v to [batch, num_channels, top_k, width]
        v = v.transpose(-2, -1)
        
        # Select top-k components
        s = s[:, :, :self.top_k]
        u = u[:, :, : self.top_k]
        v = v[:, :, :self.top_k]
        # Reshape components
        s = s.reshape(s.shape[0], -1)  # [batch_size, top_k]
        u = u.reshape(u.shape[0], -1)  # [batch_size, num_channels * height * top_k]
        v = v.reshape(v.shape[0], -1)  # [batch_size, num_channels * width * top_k]

        s = self.values_net(s)
        u = self.u_net(u)
        v = self.v_net(v)
        
        # Combine features
        joint = torch.cat([u, v, s], dim=-1)
        joint = self.joint_net(joint)
        
        # Generate outputs
        tns = joint[..., :self.segment]
        tns = (self.sigmoid(tns) * 2 - 1) * self.bound_tn + 1
        
        tn2s = joint[..., self.segment:2*self.segment]
        tn2s = (self.sigmoid(tn2s) * 2 - 1) * self.bound_tn2
        
        epscalings = joint[..., 2*self.segment:]
        epscalings = (self.sigmoid(epscalings) * 2 - 1) * self.bound_epscaling + 1
        
        return tns, tn2s, epscalings

class NoiseHandler_golden(torch.nn.Module):
    
    def __init__(self, output_dim, num_channels, height, width, bound_tn = 0, bound_tn2 = 0, bound_epscaling = 0, hidden_dim = 128, top_k = 32):
        super().__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.bound_tn = bound_tn
        self.bound_tn2 = bound_tn2
        self.bound_epscaling = bound_epscaling
        self.segment = self.output_dim // 3
        self.top_k = min(top_k, num_channels * height)  # Ensure top_k doesn't exceed matrix dimensions
        
        # Networks for top-k singular values
        self.values_net = nn.Sequential(
            nn.Linear(num_channels * self.top_k, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), # inplace=True when mem is a problem
            nn.Linear(hidden_dim, output_dim),
        )

        # Networks for top-k left singular vectors
        self.u_net = nn.Sequential(
            nn.Linear(num_channels * height * self.top_k, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Networks for top-k right singular vectors
        self.v_net = nn.Sequential(
            nn.Linear(num_channels * width * self.top_k, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.hidden_dim_out = hidden_dim
        self.output_net = nn.Sequential(
            nn.Linear(output_dim, self.hidden_dim_out),
            nn.BatchNorm1d(self.hidden_dim_out),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_out, output_dim),
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, noise):
        # Compute SVD
        u, s, v = torch.linalg.svd(noise)
        # transpose v to [batch, num_channels, top_k, width]
        v = v.transpose(-2, -1)
        
        # Select top-k components
        s = s[:, :, :self.top_k]
        u = u[:, :, : self.top_k]
        v = v[:, :, :self.top_k]
        # Reshape components
        s = s.reshape(s.shape[0], -1)  # [batch_size, top_k]
        u = u.reshape(u.shape[0], -1)  # [batch_size, num_channels * height * top_k]
        v = v.reshape(v.shape[0], -1)  # [batch_size, num_channels * width * top_k]

        s = self.values_net(s)
        u = self.u_net(u)
        v = self.v_net(v)
        # print(s.shape, u.shape, v.shape)
        out = s + u + v

        out = self.output_net(out)

        
        # Generate outputs
        tns = out[..., :self.segment]
        tns = (self.sigmoid(tns) * 2 - 1) * self.bound_tn + 1
        
        tn2s = out[..., self.segment:2*self.segment]
        tn2s = (self.sigmoid(tn2s) * 2 - 1) * self.bound_tn2
        
        epscalings = out[..., 2*self.segment:]
        epscalings = (self.sigmoid(epscalings) * 2 - 1) * self.bound_epscaling + 1
        
        return tns, tn2s, epscalings

@persistence.persistent_class
class NoiseHandler_values(torch.nn.Module):

    def __init__(self, output_dim, num_channels, height, width, bound_tn = 0, bound_tn2 = 0, bound_epscaling = 0, hidden_dim = 1024):
        super().__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.bound_tn = bound_tn
        self.bound_tn2 = bound_tn2
        self.bound_epscaling = bound_epscaling
        self.segment = self.output_dim // 3
        
        # Networks for top-k singular values
        self.values_net = nn.Sequential(
            nn.Linear(num_channels * height, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # init the last weights
        # self.values_net[-1].weight.data.fill_(0)
        # self.values_net[-1].bias.data.fill_(0)


        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, noise):
        # Compute SVD
        u, s, v = torch.svd(noise)
               
        s = s.reshape(s.shape[0], -1)  # [batch_size, top_k]
        

        s = self.values_net(s)

        # Generate outputs
        tns = s[..., :self.segment]
        tns = (self.sigmoid(tns) * 2 - 1) * self.bound_tn + 1
        
        tn2s = s[..., self.segment:2*self.segment]
        tn2s = (self.sigmoid(tn2s) * 2 - 1) * self.bound_tn2
        
        epscalings = s[..., 2*self.segment:]
        epscalings = (self.sigmoid(epscalings) * 2 - 1) * self.bound_epscaling + 1

        # print(tns[0], tn2s[0], epscalings[0], '111')
        # print(tns[1], tn2s[1], epscalings[1], '222')
        
        
        return tns, tn2s, epscalings
        return tns, tn2s, epscalings