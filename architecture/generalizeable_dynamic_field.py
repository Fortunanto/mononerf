import torch
from torch import nn, optim
import torchdiffeq
from util import positional_encoding
from architecture.residual_mlp import ImplicitNet
from architecture.building_blocks import Swish, ResidualBlock
from einops import rearrange, repeat
from util.utils_3d import *


class VelocityField(nn.Module):
    def __init__(self, feature_dim,time_encoding_dim):
        super(VelocityField, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim+time_encoding_dim, 512),
            Swish(),
            ResidualBlock(512),
            nn.Linear(512, 256),
            Swish(),
            ResidualBlock(256),
            nn.Linear(256, 128),
            Swish(),
            ResidualBlock(128),
            nn.Linear(128, 3)
        )
        self.features = None
        self.time_encoding_dim = time_encoding_dim
        
    def forward(self, x):
        """
        Performs a forward pass on the network.

        Args:
            t (torch.Tensor): The time variable, not directly used in the forward computation but needed for the ODE solver. Shape: ().
            features (torch.Tensor): The input features. Shape: (batch_size, feature_dim).

        Returns:
            torch.Tensor: The output velocity. Shape: (batch_size, 3).
        """
        x = torch.cat([x,self.features], dim=-1)
        return self.network(x)
        # except:
            # assert False, f"features.shape {features.shape} t_enc.shape {t_enc.shape}"

class Dynamics(nn.Module):
    def __init__(self, velocity_field, time_indices,encoding_dim):
        super(Dynamics, self).__init__()
        self.velocity_field = velocity_field
        self.time_indices = time_indices
        self.time_encoding = encoding_dim

    def forward(self, t, x):
        t = (self.time_indices + t).unsqueeze(1)
        t_enc = positional_encoding(t,self.time_encoding)
        x_reshaped = rearrange(x,'a b -> (a b) 1')
        x_enc = positional_encoding(x_reshaped,10)
        x_enc = rearrange(x_enc,'(batch xy) pos_enc -> batch (xy pos_enc)',xy=3)
        x = torch.cat([x_enc, t_enc], dim=-1)
        velocity = self.velocity_field(x)
        return velocity


class PointTrajectory(nn.Module):
    """
    A Pytorch module that represents the trajectory of points in 3D space.

    The PointTrajectory uses an ImplicitNet for encoding features and a VelocityField for determining the dynamics of the trajectory.

    Args:
        point_dim (int): The dimension of the points.
        feature_dim (int): The dimension of the input features.
        out_dim (int): The output dimension of the ImplicitNet.
    """

    def __init__(self,point_dim, feature_dim,time_encoding_dim = 3):
        super(PointTrajectory, self).__init__()
        self.velocity_field = VelocityField(feature_dim+point_dim*20,time_encoding_dim*2)
        self.time_encoding_dim = time_encoding_dim
        self.time_indices = None
        self.dynamics = Dynamics(self.velocity_field, self.time_indices,time_encoding_dim)

    def forward(self, initial_point,features,intrinsics,poses,time_indices,time_span=100):
        """
        Calculates the trajectory of the points over time and projects them into 2D space.

        Args:
            initial_point (torch.Tensor): The initial points for the trajectory. Shape: (batch_size, point_dim).
            features (torch.Tensor): The input features. Shape: (batch_size, feature_dim).
            intrinsics (torch.Tensor): The camera intrinsics. Shape: (batch_size, 3, 3).
            poses (torch.Tensor): The camera poses. Shape: (batch_size, 4, 4).
            time_span (int): The time span for the trajectory calculation.

        Returns:
            torch.Tensor: The 3D trajectory. Shape: (batch_size, time_span, 3).
            torch.Tensor: The 2D projection of the trajectory. Shape: (batch_size, time_span, 2).
        """
        n_samples = initial_point.shape[1]
        initial_shape = initial_point.shape
        features_encoded_expanded = repeat(features,'features -> (batch n_samples) features',n_samples = n_samples,batch=initial_point.shape[0])
        self.dynamics.velocity_field.features = features_encoded_expanded
        timespan = torch.linspace(-1, 1, steps=time_span).to(initial_point.device)
        initial_point = rearrange(initial_point, 'batch n_samples xyz -> (batch n_samples) xyz')
        time_indices = rearrange(repeat(time_indices, 'batch  -> batch n_samples', n_samples=n_samples),'batch n_samples -> (batch n_samples)')
        self.dynamics.time_indices = time_indices
        trajectory = torchdiffeq.odeint_adjoint(self.dynamics, initial_point, timespan,rtol=1e-3,atol=1e-3,method='dopri5')
        trajectory = rearrange(trajectory, 'time (batch n_samples) xyz -> time batch n_samples xyz',n_samples = n_samples)
        time = torch.tensor(0.0)
        velocity = self.dynamics(time, initial_point)
        velocity = rearrange(velocity, '(batch n_samples) xyz -> batch n_samples xyz',n_samples = n_samples)

        return trajectory,velocity


if __name__ == '__main__':
    point_trajectory = PointTrajectory(3, 2048, 256).to(device="cuda")
    initial_point = torch.randn(10, 3).to(device="cuda")
    initial_point_indices = torch.randint(1, 150, (10,)).to(device="cuda")
    ones = torch.ones((10, 1), device="cuda")
    initial_point_indices = initial_point_indices.unsqueeze(1)  # reshape from (n,) to (n, 1)
    initial_point_indices_2d = torch.cat((initial_point_indices, ones), dim=1)
    # assert False, f"initial_point_indices_2d: {initial_point_indices_2d}"
    matrix_indicer = torch.Tensor([[1,1,1],[-1,0,1]]).to(device="cuda")
    indices = torch.matmul(initial_point_indices_2d, matrix_indicer).long()
    indices = rearrange(indices, 'batch xyz -> (batch xyz)')
    intrinsics = torch.load("data/ball/data/intrinsics.pt")
    poses = torch.load("data/ball/data/pose.pt")
    intr= intrinsics[indices]
    posse = poses[indices]
    # assert False, f"intr.shape: {intr.shape} posse.shape: {posse.shape}"
    # Concatenate a and ones along dimension 1
    
    # assert False, f"out {out}"
    features = torch.randn(2048).to(device="cuda")
    
    # assert False, f"poses.shape: {poses.shape} intrinsics.shape: {intrinsics.shape}"
    output,projection = point_trajectory(initial_point, features,intr,posse, time_span=3)
    print(projection[0])