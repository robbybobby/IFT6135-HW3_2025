# %%
import torch
import torch.utils.data
import torchvision
from torch import nn
from typing import Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
import os 

from cfg_utils.args import * 


class CFGDiffusion():
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.n_steps = n_steps
        
        self.lambda_min = -20
        self.lambda_max = 20



    ### UTILS
    def get_exp_ratio(self, l: torch.Tensor, l_prim: torch.Tensor):
        return torch.exp(l-l_prim)
    
    def get_lambda(self, t: torch.Tensor): 
        # TODO: Write function that returns lambda_t for a specific time t. Do not forget that in the paper, lambda is built using u in [0,1]
        # Note: lambda_t must be of shape (batch_size, 1, 1, 1)
        u = t.float() / (self.n_steps)
        lambda_t = self.lambda_max + (u)*(self.lambda_min - self.lambda_max)

        return lambda_t.view(-1, 1, 1, 1)
    
    def alpha_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Alpha(lambda_t) for a specific time t according to (1)
        var = 1 / (1 + torch.exp(-lambda_t))
        var = torch.sqrt(var)

        return var
    
    def sigma_lambda(self, lambda_t: torch.Tensor): 
        #TODO: Write function that returns Sigma(lambda_t) for a specific time t according to (1)
        var = 1 - (1 / (1 + torch.exp(-lambda_t)))
        var = torch.sqrt(var)

        return var
    
    ## Forward sampling
    def q_sample(self, x: torch.Tensor, lambda_t: torch.Tensor, noise: torch.Tensor):
        #TODO: Write function that returns z_lambda of the forward process, for a specific: x, lambda l and N(0,1) noise  according to (1)
        alphat = self.alpha_lambda(lambda_t)
        sigmat = self.sigma_lambda(lambda_t)
        z_lambda_t = alphat * x + sigmat * noise

        return z_lambda_t
               
    def sigma_q(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l) according to (2)
        sigmat = self.sigma_lambda(lambda_t)
        sigmatp = self.sigma_lambda(lambda_t_prim)

        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        var_q = sigmatp**2 - exp_ratio**2 * sigmat**2
    
        return var_q.sqrt()
    
    def sigma_q_x(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns variance of the forward process transition distribution q(•|z_l, x) according to (3)
        alphat = self.alpha_lambda(lambda_t)
        alphatp = self.alpha_lambda(lambda_t_prim)

        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        var_q_x = alphatp**2 - exp_ratio**2 * alphat**2
    
        return var_q_x.sqrt()

    ### REVERSE SAMPLING
    def mu_p_theta(self, z_lambda_t: torch.Tensor, x: torch.Tensor, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor):
        #TODO: Write function that returns mean of the forward process transition distribution according to (4)
        alphatp = self.alpha_lambda(lambda_t_prim)
        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)

        first = exp_ratio * z_lambda_t
        second = alphatp * (1 - exp_ratio**2) * x
        mu = first + second
    
        return mu

    def var_p_theta(self, lambda_t: torch.Tensor, lambda_t_prim: torch.Tensor, v: float=0.3):
        #TODO: Write function that returns var of the forward process transition distribution according to (4)
        sigmat = self.sigma_lambda(lambda_t)
        sigmatp = self.sigma_lambda(lambda_t_prim)
        exp_ratio = self.get_exp_ratio(lambda_t, lambda_t_prim)
        first = (1-v) * sigmatp**2
        second = v * (sigmatp**2 - exp_ratio**2 * sigmat**2)
        var =  first + second

        return var
    
    def p_sample(self, z_lambda_t: torch.Tensor, lambda_t : torch.Tensor, lambda_t_prim: torch.Tensor,  x_t: torch.Tensor, set_seed=False):
        # TODO: Write a function that sample z_{lambda_t_prim} from p_theta(•|z_lambda_t) according to (4) 
        # Note that x_t correspond to x_theta(z_lambda_t)
        if set_seed:
            torch.manual_seed(42)
        mu = self.mu_p_theta(z_lambda_t, x_t, lambda_t, lambda_t_prim)
        var = self.var_p_theta(lambda_t, lambda_t_prim)
        eps = torch.randn_like(z_lambda_t)
        sample = mu + torch.sqrt(var) * eps
    
        return sample 

    ### LOSS
    def loss(self, x0: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None, set_seed=False):
        if set_seed:
            torch.manual_seed(42)
        batch_size = x0.shape[0]
        dim = list(range(1, x0.ndim))
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )
        if noise is None:
            noise = torch.randn_like(x0)
        #TODO: q_sample z
        lambdat = self.get_lambda(t)
        zt = self.q_sample(x = x0, lambda_t = lambdat, noise = noise)

        #TODO: compute loss
        pred_noise = self.eps_model(zt, t, labels)
        loss = F.mse_loss(pred_noise, noise, reduction = 'mean')

    
        return loss



    