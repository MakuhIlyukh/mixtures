# %%
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize
from geotorch import positive_semidefinite
from tqdm import tqdm

from src.initializers import DirichletInitializer, init_factory


class SoftmaxParametrization(torch.nn.Module):
    def __init__(self, c=0):
        super().__init__()
        self.c = c

    def forward(self, X):
        return nn.functional.softmax(X, dim=-1)
    
    def right_inverse(self, Y):
        return torch.log(Y) + self.c


class GM(torch.nn.Module):
    def __init__(self,
                 k, d, c=0):
        super().__init__()

        # means
        self.m_w = torch.nn.Parameter(
            torch.randn((k, d), requires_grad=True, dtype=torch.float64))
        
        # inversed covs
        self.l_w = torch.nn.Parameter(
            torch.empty((k, d, d), requires_grad=True, dtype=torch.float64))
        positive_semidefinite(self, "l_w")
        
        # mix probs
        self.p_w = torch.nn.Parameter(
            torch.empty(k, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'p_w', SoftmaxParametrization(c))
        self.p_w = torch.distributions.Dirichlet(
                concentration=torch.full((k,), 1.0, dtype=torch.float64)
            ).sample()
        
        # constants
        self._mult_const = (torch.pi * 2)**(-d / 2)
        self.k = k
        self.d = d

    def forward(self, x):
        s = torch.zeros((x.shape[0], 1), dtype=torch.float64)
        with parametrize.cached():
        # TODO: можно ли не пробегаться по всем k?
            for k in range(self.k):
                x_cent = (x - self.m_w[k])
                # TODO: можно избавиться от суммирования, если изменить порядок операций?
                s += (
                    self.p_w[k]
                    * self._mult_const
                    * torch.sqrt(torch.det(self.l_w[k]))
                    * torch.exp(
                        - torch.sum(
                            x_cent * (x_cent @ self.l_w[k]),
                            dim=1,
                            keepdim=True)
                        / 2
                    ))
        return s

    def s_w(self, no_grad=True):
        if no_grad:
            with torch.no_grad():
                return torch.linalg.inv(self.l_w)
        else:
            return torch.linalg.inv(self.l_w)
