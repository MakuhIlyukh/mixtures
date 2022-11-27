# %%
from os.path import join as joinp
import json

import numpy as np
import matplotlib.pyplot as plt
import mlflow

from src.datasets import GaussianMixtureSampler, scatter, save_dataset
from src.utils import set_commit_tag
from config import (
    DATASETS_ARTIFACTS_PATH as DAP,
    DATASETS_TAG_KEY,
    GENERATION_TAG_VALUE)


K = 5
D = 2
N = 2000
SEED = 20


if __name__ == '__main__':
    with mlflow.start_run() as data_gen_run:
        # log params
        mlflow.log_params({
            "K": K,
            "D": D,
            "N": N,
            "SEED": SEED})
        # adding tags
        mlflow.set_tag(DATASETS_TAG_KEY, GENERATION_TAG_VALUE)
        set_commit_tag()
        
        # dataset generation
        rnd = np.random.RandomState(SEED)
        gms = GaussianMixtureSampler(K, D, rnd)
        X, y = gms.sample(N)
        fig, axis = plt.subplots()
        scatter(X, y, axis=axis)
        plt.show()

        # saving data and log
        # dataset
        with open(joinp(DAP, 'Xy.pkl'), 'wb') as f:
            save_dataset(X, y, f)
        # sampler
        with open(joinp(DAP, 'gms.pkl'), 'wb') as f:
            gms.save(f)
        # run_id
        run_id = data_gen_run.info.run_id
        with open(joinp(DAP, 'run_id.json'), 'w') as f:
            json.dump(run_id, f)
        # figure
        fig.savefig(joinp(DAP, "figure.png"))
        # log data
        mlflow.log_artifact(DAP)
