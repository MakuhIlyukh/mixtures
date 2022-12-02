from typing import Iterable

import torch
from sklearn.cluster import KMeans


def tensor_init_factory(rval_tensor):
    def tensor_func():
        return rval_tensor
    return tensor_func


def init_factory(init, named_inits, name):
    """

    Example of usage:
    class my_module(torch.nn.Module):
        def __init__(self, w_init="default_key"):
            w = torch.Tensor(1.0, dtype=torch.float64)
            w_init_callable = init_factory(
                w_init,
                {
                    "norm": torch.nn.init.norm_,
                    "uniform": torch.nn.init.uniform_
                },
                "w_init")
            w.copy_(w_init_callable())

    """
    if isinstance(init, str):
        try:
            res = named_inits[init]
        except KeyError:
            raise ValueError(f"Unknown name for {name}={init}")
    elif callable(init):
        res = init
    elif isinstance(init, torch.Tensor):
        res = tensor_init_factory(init)
    else:
        raise ValueError(f"{name} must be str, callable or Tensor")


class DirichletInitializer:
    def __init__(self, k, concentration):
        if (    isinstance(concentration, float)
                or isinstance(concentration, Iterable)
                or isinstance(concentration, int)):
            concentration = torch.tensor(concentration, dtype=torch.float64)
        
        if isinstance(concentration, torch.Tensor):  # not elif
            if concentration.size() == torch.Size([]):
                concentration = torch.full(
                    (k,), concentration.item(), dtype=torch.float64)
            elif concentration.size() != torch.Size([k]):
                raise ValueError("size of concentrations != k")
        else: # not (int, float or tensor)
            raise ValueError("concentrations must be int, float or tensor")

        self.concentration = concentration
        self.distr = torch.distributions.Dirichlet(self.concentration)

    def __call__(self, tensor):
        return self.distr.sample()


class KMeansInitializer:
    def __init__(self, k, X, n_init=1):
        km = KMeans(n_clusters=k, n_init=n_init)
        km.fit(X)
        self.centers = km.cluster_centers_.copy()
    
    def __call__(self):
        return torch.tensor(self.centers, dtype=torch.float64)
