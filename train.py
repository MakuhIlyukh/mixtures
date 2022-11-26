# %%
import torch
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from src.datasets import (
    load_dataset, GaussianMixtureSampler, scatter)
from src.models import GM
from src.losses import nll


def train(X, gm, n_epochs, optimizer, loss_fn):
    X = torch.from_numpy(X)
    tqdm_bar = tqdm(range(n_epochs))
    for epoch in tqdm(tqdm_bar):
        optimizer.zero_grad()
        dens = gm(X)
        loss = loss_fn(dens)
        loss.backward()
        optimizer.step()
        tqdm_bar.set_postfix({"loss": loss.item()})
        

if __name__ == '__main__':
    torch.manual_seed(106)

    with open("data/gms.pkl", 'rb') as f:
        gm_sampler = GaussianMixtureSampler.load(f)
    with open("data/Xy.pkl", 'rb') as f:
        X, y = load_dataset(f)
    k = gm_sampler.k
    d = gm_sampler.d

    gm = GM(k, d)
    optimizer = torch.optim.Adam(
        params=gm.parameters(),
        lr=10**(-3),
        weight_decay=0.0)
    loss_fn = nll
    n_epochs = 8*10**3
    # X_proc = MinMaxScaler().fit_transform(X)
    X_proc = X.copy()

    train(X_proc, gm, n_epochs, optimizer, loss_fn)


# %%
# optimizer = torch.optim.Adam(
#     params=gm.parameters(),
#     lr=10**(-1),
#     weight_decay=0.0)
n_epochs = 10**3

train(X_proc, gm, n_epochs, optimizer, loss_fn)