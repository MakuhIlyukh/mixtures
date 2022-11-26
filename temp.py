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