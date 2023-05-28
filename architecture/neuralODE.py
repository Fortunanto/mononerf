from torchdiffeq import odeint
from architecture.residual_mlp import ImplicitNet
import torch.nn as nn
import torch

class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64, 64, 64, 64]):
        super(NeuralODE, self).__init__()
        self.ode_func = ImplicitNet(input_dim, hidden_dim, d_out=input_dim)
    
    def forward(self, input_traj, timestamps):
        """
        Inputs:
            - input_traj: Input trajectory (tensor of shape [batch_size, input_dim])
            - timestamps: Time stamps of the trajectory (tensor of shape [batch_size])
        
        Returns:
            - output_traj: Output trajectory (tensor of shape [batch_size, input_dim])
        """
        # Define the ODE function
        def ode_func_wrapper(x,t):
            assert False, f"x.shape: {x.shape}, t.shape: {t.shape}"
            return self.ode_func(x)
        # assert False, f"input_traj.shape: {input_traj}, timestamps.shape: {timestamps}"        
        # Solve the ODE using odeint
        output_traj = odeint(ode_func_wrapper, input_traj, timestamps)
        
        return output_traj

if __name__ == '__main__':
    ode = NeuralODE(2, [64])
    x = torch.randn(10,2)
    assert False, f"x.shape: {x.shape}"
    output = ode(torch.randn(10,2), torch.linspace(-1, 1, 10))
    print(output.shape)