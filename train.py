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
from src.utils import (
    set_commit_tag, del_folder_content)
from src.models import GM
from src.losses import nll
from config import (
    DATASETS_ARTIFACTS_PATH,
    TRAINED_MODELS_PATH,
    TRAIN_PLOTS_PATH,
    MODELS_TAG_KEY,
    TRAINING_TAG_VALUE)


TRAIN_SEED = 106
LR = 10**(-2)
MIN_MAX_SCALING = False
N_EPOCHS = 100
WEIGHT_DECAY = 0.0
LOSS_PREFIX = "NLL"
PLOT_EVERY = 1


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
            fig.savefig(joinp(TRAIN_PLOTS_PATH, f"gm_plot_{epoch}.png"))
            plt.close(fig)  # ???: может лучше очищать фигуру и создавать не в цикле?
        
        mlflow.log_metric(loss_name, loss.item(), step=epoch)


if __name__ == '__main__':
    # starting mlflow
    train_run = mlflow.start_run()
    
    # mlflow logging
    # params
    mlflow.log_params({
        "TRAIN_SEED": TRAIN_SEED,
        "LR": LR,
        "MIN_MAX_SCALING": MIN_MAX_SCALING,
        "N_EPOCHS": N_EPOCHS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LOSS": LOSS_PREFIX})
    # commit
    set_commit_tag()
    # tag
    mlflow.set_tag(MODELS_TAG_KEY, TRAINING_TAG_VALUE)

    # clearing folders
    del_folder_content(TRAIN_PLOTS_PATH)

    # setting seed of torch
    torch.manual_seed(TRAIN_SEED)

    # loading dataset
    with open(joinp(DATASETS_ARTIFACTS_PATH, "gms.pkl"), 'rb') as f:
        gm_sampler = GaussianMixtureSampler.load(f)
    with open(joinp(DATASETS_ARTIFACTS_PATH, "Xy.pkl"), 'rb') as f:
        X, y = load_dataset(f)
    k = gm_sampler.k
    d = gm_sampler.d

    # creating a model
    gm = GM(k, d)

    # choosing an optimizer
    optimizer = torch.optim.Adam(
        params=gm.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY)
    mlflow.log_param("OPTIMIZER", optimizer.__class__)
    
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
    mlflow.log_artifact(TRAIN_PLOTS_PATH)
    # mlflow.pytorch.log_model(gm, TRMP)