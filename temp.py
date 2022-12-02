# %%
import torch
from torch import nn
import numpy as np


# %%
k = torch.tensor(0.5, requires_grad=True)
x = torch.distributions.uniform.Uniform(0, k).sample(). \
    clone().detach().requires_grad_(True)*k
loss = x
loss.backward()
k.grad



# %%
g = torch.Generator()
g = g.manual_seed(123)
concentration = torch.tensor([1, 1, 1], dtype=torch.float64)
distr = torch.distributions.Dirichlet(concentration, )
distr.sample()


# %%
torch.linalg.inv(torch.tensor(
    [
        [[1, 2],
         [3, 4]],

        [[4, 5],
         [6, 7]]
    ], dtype=torch.float64))


# %%
x = torch.tensor(1.0)
y = x.detach()
y.fill_(5)
y.requires_grad


# %%
x = torch.tensor(5.0, requires_grad=True)
y = x.detach().requires_grad_(True)
z = x*x
t = torch.log(y)

z.backward()
print(1, 'x.grad', x.grad)
t.backward()
print(2, 'x.grad', x.grad)
print(3, 'y.grad', y.grad)


# %%
x = torch.tensor(1.0, requires_grad=True)
z = x*x
with torch.no_grad():
    x.fill_(5.0)
z.backward()

print(1, 'x.grad', x.grad)
print(2, 'x', x)


# %%
from src.initializers import DirichletInitializer


dr = DirichletInitializer(2, [1, 1])
dr()


# %%
import torch


type(torch.nn.init.uniform_)


# %%
torch.distributions.Dirichlet().


# %%
import torch
from src.initializers import DirichletInitializer


dirichlet = DirichletInitializer(4, [1.0] * 4)
x = torch.tensor([1, 2, 3, 4], dtype=torch.float64)
dirichlet(x)
x


# %%
x = torch.tensor(2.0, requires_grad=False)
y = torch.tensor(1.5, requires_grad=True)
x.copy_(y)
x.requires_grad


# %%
import torch

from src.models import GM


gm = GM(5, 2)


# %%
x = torch.tensor(X_proc[:1,])
s = torch.zeros((x.shape[0], 1), dtype=torch.float64)
for k in range(gm.k):
    x_cent = (x - gm.m_w[k])
    # TODO: можно избавиться от суммирования, если изменить порядок операций?
    sk =  (
        gm.p_w[k]
        * gm._mult_const
        * torch.sqrt(torch.det(gm.l_w[k]))
        * torch.exp(
            - torch.sum(
                x_cent * (x_cent @ gm.l_w[k]),
                dim=1,
                keepdim=True)
            / 2
        ))
    s += sk
    print(sk)
s


# %%
k = 4
x_cent = (x - gm.m_w[k])
print(gm.p_w[k])
print(gm._mult_const)
print(torch.sqrt(torch.det(gm.l_w[k])))
print(torch.exp(
    - torch.sum(
        x_cent * (x_cent @ gm.l_w[k]),
        dim=1,
        keepdim=True)
    / 2
))
sk =  (
    gm.p_w[k]
    * gm._mult_const
    * torch.sqrt(torch.det(gm.l_w[k]))
    * torch.exp(
        - torch.sum(
            x_cent * (x_cent @ gm.l_w[k]),
            dim=1,
            keepdim=True)
        / 2
    ))
s += sk
print(sk)


# %%
np.exp(
    - (
        x_cent.detach().numpy()
        @ gm.l_w[k].detach().numpy()
        @ x_cent.detach().numpy().T
    ) / 2
)

# %%
1 / torch.det(gm.s_w()[k])


# %%
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

mn_dist = multivariate_normal(
    gm.m_w[k].detach().numpy(),
    gm.s_w()[k].detach().numpy())
mn_dist.pdf(X_proc[:1])


# %%
xs = np.linspace(-2, 2, 100)
ys = np.linspace(-2, 2, 100)
XX, YY = np.meshgrid(xs, ys)
ZZ = mn_dist.pdf(np.vstack([XX.ravel(), YY.ravel()]).T).T.reshape(100, 100)
plt.contour(XX, YY, ZZ)

XSA = np.random.multivariate_normal(
    gm.m_w[k].detach().numpy(),
    gm.s_w()[k].detach().numpy(),
    1000)
scatter(X_proc)
scatter(X_proc[:1], 'red')


# %%
np.vstack([XX.ravel(), YY.ravel()]).T.reshape(100, 100, 2)


# %%
XSA = mn_dist?


# %%
def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)
    
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))


pdf_multivariate_gauss(
    X_proc[:1].T,
    gm.m_w[4].detach().numpy().reshape(-1, 1),
    gm.s_w()[4].detach().numpy())


# %%
import numpy as np
import matplotlib.pyplot as plt


xs = np.linspace(-0.5, 2, 100)
ys = np.linspace(-0.5, 2, 100)
XX, YY = np.meshgrid(xs, ys)
ZZ = gm(
        torch.tensor(np.vstack([XX.ravel(), YY.ravel()]).T)
    ).detach().numpy().T.reshape(100, 100)
plt.contourf(XX, YY, ZZ, level=200)

scatter(X_proc)
scatter(gm.m_w.detach().numpy(), 'black')
# scatter(gm_sampler.m, 'red')


# %%
xs = np.linspace(-0.5, 2, 100)
ys = np.linspace(-0.5, 2, 100)
XX, YY = np.meshgrid(xs, ys)
ZZ = mn_dist.pdf(np.vstack([XX.ravel(), YY.ravel()]).T)


.T.reshape(100, 100)
plt.contour(XX, YY, ZZ)

XSA = np.random.multivariate_normal(
    gm.m_w[k].detach().numpy(),
    gm.s_w()[k].detach().numpy(),
    1000)
scatter(X_proc)
scatter(X_proc[:1], 'red')


# %%
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

k = gm_sampler.k
d = gm_sampler.d
mn_dist = []
s = np.empty((k, d, d), dtype=np.float64)
m = np.empty((k, d))
max_x = np.max(X, axis=0)
min_x = np.min(X, axis=0)
scaler = MinMaxScaler()
X_proc = scaler.fit_transform(X)
for i in range(k):
    m[i] = scaler.transform(gm_sampler.m[i : i + 1]).squeeze(0)
    XX, YY = np.meshgrid(max_x, max_x)
    s[i] = (gm_sampler.s[i] / XX / YY)
    mn_dist.append(
        multivariate_normal(
            m[i],
            s[i]
        )
    )

xs = np.linspace(-0.5, 4, 100)
ys = np.linspace(-0.5, 5, 100)
XX, YY = np.meshgrid(xs, ys)

ZZ = np.zeros((100, 100), dtype=np.float64)
for i in range(k):
    ZZ += gm_sampler.p[i] * mn_dist[i].pdf(np.vstack([XX.ravel(), YY.ravel()]).T).T.reshape(100, 100)
# plt.contour(XX, YY, ZZ, levels=100)
# scatter(X_proc)
scatter(m, 'red')
scatter(scaler.transform(gm_sampler.m))
# %%
gm_sampler.m[i]
gm_sampler.s[i]


# %%
from os.path import join as joinp

from src.datasets import (
    GaussianMixtureSampler, load_dataset)
from config import DATASETS_ARTIFACTS_PATH as DAP
from src.plotting import gm_plot


with open(joinp(DAP, "gms.pkl"), 'rb') as f:
    gm_sampler = GaussianMixtureSampler.load(f)
with open(joinp(DAP, "Xy.pkl"), 'rb') as f:
    X, y = load_dataset(f)
k = gm_sampler.k
d = gm_sampler.d


# %%
from src.plotting import (
    gm_plot, scatter)


scatter(X, y)
gm_plot(
    gm_sampler.m,
    gm_sampler.s)


# %%
import torch.nn.utils.parametrize as parametrize
import torch

class SimpleParametrization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X):
        return X*2
    
    def right_inverse(self, X):
        return X/2


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([3.0, 5.0], requires_grad=True))
        parametrize.register_parametrization(
            self, "w", SimpleParametrization())
        self.b = torch.nn.Parameter(torch.tensor(7.0, requires_grad=True))
    
    def forward(self, X):
        s = torch.tensor(0.0)
        with parametrize.cached():
            for i in range(2):
                s += (self.w[i]
                        *self.w[i]
                        *self.w[i]
                        + 7*self.w[i]
                        + 1/self.w[i])
        return s


module = SimpleModule()
y = module(torch.tensor(1.0))
y.backward()
module.parametrizations.w.original.grad


# %%
y = module.b*17
y.backward()
module.b.grad, t.grad


# %%
x = torch.tensor(5.0, requires_grad=True)
module.parametrizations.w.original.data = 7*x
y = 3*module.w
y.backward()
module.parametrizations.w.original.grad

# %%
# with torch.no_grad():
module.w = module.w * 13
module.w, module.parametrizations.w.original

# %%
with torch.no_grad():
    module.b.data = torch.tensor(7.0)
module.b


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
        with parametrize.cached():
            s = torch.zeros((x.shape[0], 1), dtype=torch.float64)
            # l_w = self.l_w
            # print(l_w)
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


# torch.manual_seed(123)
# gm = GM(5, 2)
with open(DATASETS_ARTIFACTS_PATH + "/Xy.pkl", "rb") as f:
    X, y = load_dataset(f)
# pdfs = gm(torch.from_numpy(X))
# loss = nll(pdfs)
# # loss.backward()

# make_dot(loss, params=dict(gm.named_parameters()))
# # gm.parametrizations.l_w.original.grad


# %%
import torch
from torch.nn.utils import parametrize


torch.manual_seed(131)


class SimpleModule(torch.nn.Module):
    def __init__(self, k, c=0):
        super().__init__()
        
        self.p_w = torch.nn.Parameter(
            torch.empty(k, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'p_w', SoftmaxParametrization(c))
        p_w = torch.distributions.Dirichlet(
                concentration=torch.full((k,), 1.0, dtype=torch.float64)
            ).sample()
        self.p_w = p_w
        t = torch.full((k,), 1.0)
        with torch.no_grad():
            self.parametrizations.p_w.original.copy_(t)
        print(self.parametrizations.p_w.original)
        t.fill_(5.2)
        print(self.parametrizations.p_w.original)
        self.k = k
    
    def forward(self, X):
        with parametrize.cached():
            s = torch.tensor(0.0, dtype=torch.float64)
            for i in range(self.k):
                s += self.p_w[i]
        return s


sm = SimpleModule(5)
y = sm(X)
# make_dot(y, dict(sm.named_parameters()))
# %%
