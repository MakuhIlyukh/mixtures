# %%
import numpy as np

from src.datasets import GaussianMixtureSampler, scatter, save_dataset
import matplotlib.pyplot as plt


K = 5
D = 2
N = 2000
SEED = 21


if __name__ == '__main__':
    rnd = np.random.RandomState(SEED)
    gms = GaussianMixtureSampler(K, D, rnd)
    X, y = gms.sample(N)
    scatter(X, y)
    plt.show()
    with open('data/Xy.pkl', 'wb') as f:
        save_dataset(X, y, f)
    with open('data/gms.pkl', 'wb') as f:
        gms.save(f)
