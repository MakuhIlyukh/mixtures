from typing import Iterable

import torch


def tensor_init_factory(rval_tensor):
    def assign(lval_tensor):
        lval_tensor.data = rval_tensor.detach().clone()
    return assign


def init_factory(init, named_inits, name):
    """

    Example of usage:
    class my_module(torch.nn.Module):
        def __init__(self, w_init="default_key"):
            w = torch.torch.Tensor(1.0, dtype=torch.float64)
            w_init_callable = init_factory(
                w_init,
                {
                    "norm": torch.nn.init.norm_,
                    "uniform": torch.nn.init.uniform_
                },
                "w_init")
            w_init_callable(w)

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
        tensor.data = self.distr.sample()


