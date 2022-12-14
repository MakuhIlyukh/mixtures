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
                 k, d, c=0,
                 m_init="random",
                 p_init="dirichlet"):
        super().__init__()

        # means
        self.m_w = torch.nn.Parameter(
            torch.empty((k, d), requires_grad=True))
        
        with torch.no_grad():
            if isinstance(m_init, str) and m_init == "random":
                self.m_w.copy_(torch.randn(
                    (k, d), dtype=torch.float64))
            elif callable(m_init):
                self.m_w.copy_(m_init())
            else:
                raise ValueError("m_init must be callable or 'random'")
        
        # inversed covs
        self.l_w = torch.nn.Parameter(
            torch.empty((k, d, d), requires_grad=True, dtype=torch.float64))
        positive_semidefinite(self, "l_w")
        
        # mix probs
        self.p_w = torch.nn.Parameter(
            torch.empty(k, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'p_w', SoftmaxParametrization(c))
        
        with torch.no_grad():
            if isinstance(p_init, str):
                if p_init == 'dirichlet':
                    self.p_w = torch.distributions.Dirichlet(
                            concentration=torch.full(
                                (k,), 1.0, dtype=torch.float64)
                        ).sample()
                elif p_init == "1/k":
                    self.p_w = torch.full(
                        (k,), 1/k, dtype=torch.float64)
                else:
                    raise ValueError("p_init must be 'dirchlet', '1/k' or callable")
            elif callable(p_init):
                self.p_w = p_init()
            else:
                raise ValueError("p_init must be 'dirchlet', '1/k' or callable")

        # constants
        self._mult_const = (torch.pi * 2)**(-d / 2)
        self.k = k
        self.d = d

    def forward(self, x):
        s = torch.zeros((x.shape[0], 1), dtype=torch.float64)
        with parametrize.cached():
        # TODO: ?????????? ???? ???? ?????????????????????? ???? ???????? k?
            for k in range(self.k):
                x_cent = (x - self.m_w[k])
                # TODO: ?????????? ???????????????????? ???? ????????????????????????, ???????? ???????????????? ?????????????? ?????????????????
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
