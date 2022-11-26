""" Mixtures Generation """


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
import pickle
from functools import partial


class GaussianMixtureSampler:
    def __init__(self, k, d, rnd_state, m_init="10xnorm",
                 p_init="dirichlet", s_init="sklearn"):
        """

        :param k: number of gaussians.
        :param d: dimension of multivariate distribution.
        """
        if m_init == "10xnorm":
            m_init = lambda rnd_state: rnd_state.randn(k, d) * 10
        elif callable(m_init):
            pass
        else:
            raise ValueError("unkown m_init")

        if p_init == "dirichlet":
            p_init = lambda rnd_state: rnd_state.dirichlet(alpha=[1]*k)
        elif callable(p_init):
            pass
        else:
            raise ValueError("unkown p_init")

        if s_init == "sklearn":
            s_init = lambda rnd_state: make_spd_matrix(
                d, random_state=rnd_state)
        elif callable(s_init):
            pass
        else:
            raise ValueError("unknown s_init")

        self._m = m_init(rnd_state)
        self._p = p_init(rnd_state)
        self._s = np.empty((k, d, d), dtype=np.float64)
        for i in range(k):
            self._s[i] = s_init(rnd_state)
        self._k = k
        self._d = d
        self._l = np.linalg.inv(self._s)
        self.rnd_state = rnd_state
        self._MULT_CONST = (np.pi * 2)**(-self._d / 2)

    def sample(self, n):
        """ Generates samples. """
        X = np.empty((n, self._d))
        y = self.rnd_state.choice(self._k, p=self._p, size=n)
        for i, c in enumerate(y):
            X[i] = self.rnd_state.multivariate_normal(self._m[c],
                                                             self._s[c])
        return X, y

    def save(self, file_like):
        """ Saves mixture to file.

        :param file_like: for example open(filename, "wb")
        """
        pickle.dump(self, file_like)

    @staticmethod
    def load(file_like):
        """ Loads mixture from file.

        :param file_like: for example open(filename, "wb")
        """
        return pickle.load(file_like)

    def pdf(self, x):
        s = np.zeros((x.shape[0], 1), dtype=np.float64)
        for k in range(self._k):
            x_cent = (x - self._m[k])
            s += self._p[k] * self._MULT_CONST * np.linalg.det(self._l[k]) \
                * np.exp(-np.sum((x_cent * (x_cent @ self._l[k])),
                                 axis=1, keepdims=True) / 2)
        return s

    @property
    def m(self):
        return self._m

    @property
    def p(self):
        return self._p

    @property
    def s(self):
        return self._s

    @property
    def l(self):
        return self._l
    
    @property
    def k(self):
        return self._k
    
    @property
    def d(self):
        return self._d


def save_dataset(X, y, file_like):
    """ Saves (X, y) to file. """
    pickle.dump((X, y), file_like)


def load_dataset(file_like):
    """ Loads (X, y) from file. """
    return pickle.load(file_like)


def scatter(X, y=None, axis=None):
    if axis is None:
        axis = plt
    axis.scatter(X[:, 0], X[:, 1], c=y)
