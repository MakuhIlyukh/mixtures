# %%
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize
from geotorch import positive_semidefinite
from tqdm import tqdm
from torchviz import make_dot

from src.initializers import DirichletInitializer, init_factory
from src.datasets import load_dataset
from config import DATASETS_ARTIFACTS_PATH
from src.losses import nll


class SoftmaxParametrization(torch.nn.Module):
    def __init__(self, c=0):
        super().__init__()
        self.c = c

    def forward(self, X):
        return nn.functional.softmax(X, dim=-1)
    
    def right_inverse(self, Y):
        self.inversed = torch.log(Y) + self.c
        return self.inversed


torch.manual_seed(131)


class SimpleModule(torch.nn.Module):
    def __init__(self, k, c=0):
        super().__init__()
        
        self.p_w = torch.nn.Parameter(
            torch.empty(k, dtype=torch.float64, requires_grad=True))
        sp = SoftmaxParametrization(c)
        parametrize.register_parametrization(
            self, 'p_w', sp)
        p_w = torch.distributions.Dirichlet(
                concentration=torch.full((k,), 1.0, dtype=torch.float64)
            ).sample()
        # self.p_w = p_w
        with torch.no_grad():
            t = torch.full((k,), 7.0, dtype=torch.float64)  # dtype is important for set_!!!
            self.parametrizations.p_w.original.set_(t) 
        print(self.parametrizations.p_w.original.data_ptr())
        print(t.data_ptr())
        # print(sp.inversed.data_ptr())
        print(self.p_w)
        # print(p_w)
        # t = torch.full((k,), 1.0)
        # with torch.no_grad():
        #     self.parametrizations.p_w.original.copy_(t)
        # print(self.parametrizations.p_w.original)
        # t.fill_(5.2)
        # print(self.parametrizations.p_w.original)
        self.k = k
    
    def forward(self, X):
        with parametrize.cached():
            s = torch.tensor(0.0, dtype=torch.float64)
            for i in range(self.k):
                s += self.p_w[i]
        return s


sm = SimpleModule(5)
# make_dot(y, dict(sm.named_parameters()))