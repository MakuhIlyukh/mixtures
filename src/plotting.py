import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg


def scatter(X, y=None, axis=None):
    if axis is None:
        axis = plt
    return axis.scatter(X[:, 0], X[:, 1], c=y)


def gm_plot(means, covariances,
            color_iter=None, axis=None,
            alpha=0.7, lw=2):
    if color_iter is None:
        color_iter = (plt.cm.Set1(i) for i in range(means.shape[0]))
    for i, (mean, covar, color) in enumerate(zip(means,
                                                 covariances,
                                                 color_iter)):
        if axis is None:
            axis = plt.gca()
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1],
                                  180.0 + angle,
                                  color=color,
                                  lw=lw)
        # ell.set_clip_box(plt.bbox)
        ell.set_alpha(alpha)
        axis.add_artist(ell)

