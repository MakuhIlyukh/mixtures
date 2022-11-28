# %%
from os.path import join as joinp

import torch
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import mlflow
import matplotlib.pyplot as plt

from src.datasets import (
    load_dataset, GaussianMixtureSampler)
from src.plotting import (
    scatter, gm_plot)
from src.models import GM
from src.losses import nll
from config import (
    DATASETS_ARTIFACTS_PATH as DAP,
    TRAINED_MODELS_PATH as TRMP,
    TRAIN_PLOTS_PATH as TPP)


TRAIN_SEED = 106
LR = 10**(-5)
MIN_MAX_SCALING = False
N_EPOCHS = 8*10**3
WEIGHT_DECAY = 0.0
OPTIMIZER = "SGD"
LOSS_PREFIX = "NLL"
PLOT_EVERY = 100


def train(X, gm, n_epochs, optimizer, loss_fn,
          loss_prefix="", plot_every=100):
    # loss_name is used for logging
    loss_name = loss_prefix + "_loss"
    X_numpy = X
    X = torch.from_numpy(X)
    # TODO: add batch splitting
    tqdm_bar = tqdm(range(n_epochs))
    for epoch in tqdm_bar:
        optimizer.zero_grad()
        dens = gm(X)
        loss = loss_fn(dens)
        loss.backward()
        optimizer.step()

        tqdm_bar.set_postfix({
            loss_name: loss.item()})

        if epoch % plot_every == 0:
            fig, ax = plt.subplots()
            scatter(X_numpy, y, axis=ax)
            gm_plot(
                gm.m_w.detach().numpy(),
                gm.s_w(no_grad=True).numpy(),
                axis=ax)
            fig.savefig(joinp(TPP, f"gm_plot_{epoch}.png"))
            plt.close(fig)  # ???: может лучше очищать фигуру и создавать не в цикле?
        
        mlflow.log_metric(loss_name, loss.item(), step=epoch)


if __name__ == '__main__':
    # mlflow starting, params logging
    train_run = mlflow.start_run()
    mlflow.log_params({
        "TRAIN_SEED": TRAIN_SEED,
        "LR": LR,
        "MIN_MAX_SCALING": MIN_MAX_SCALING,
        "N_EPOCHS": N_EPOCHS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "OPTIMIZER": OPTIMIZER,
        "LOSS": LOSS_PREFIX})

    # setting seed of torch
    torch.manual_seed(TRAIN_SEED)

    # loading dataset
    with open(joinp(DAP, "gms.pkl"), 'rb') as f:
        gm_sampler = GaussianMixtureSampler.load(f)
    with open(joinp(DAP, "Xy.pkl"), 'rb') as f:
        X, y = load_dataset(f)
    k = gm_sampler.k
    d = gm_sampler.d

    # creating a model
    gm = GM(k, d)

    # choosing an optimizer
    optimizer = torch.optim.SGD(
        params=gm.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY)
    
    # preprocessing steps
    if MIN_MAX_SCALING:
        scaler = MinMaxScaler()
        X_proc = scaler.fit_transform(X)
        # TODO: apply scaler to gm.m_w and gm.s_w (gm.l_w)
        # TODO: добавь константую инициализацию пресижона 
        #       (EYE * alpha, константа определяется из датасета)
    else:
        X_proc = X

    # training
    loss_fn = nll
    train(X_proc, gm, N_EPOCHS, optimizer, loss_fn,
          loss_prefix=LOSS_PREFIX, plot_every=PLOT_EVERY)

    # TODO: artifacts logging
    # TODO: Обрати внимание на то, чтобы папка plots очищалась
    #       перед запуском, ибо иначе будут логироваться данные с прошлых
    #       запусков. А может должна очищаться не только папка plots?
    #       А может вообще нужно, чтобы файлы не сохранялись в папки,
    #       а сразу заносились в mlflow runs?
    mlflow.log_artifact(TPP)
    # mlflow.pytorch.log_model(gm, TRMP)