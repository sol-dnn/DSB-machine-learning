import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state


def rand_checkers(n_samples=200, sigma=0.1, random_state=0):
    rng = check_random_state(random_state)
    nbp = n_samples // 16
    nbn = n_samples // 16
    xapp = rng.rand((nbp + nbn) * 16).reshape((nbp + nbn) * 8, 2)
    yapp = np.ones((nbp + nbn) * 8)
    idx = 0
    for i in range(-2, 2):
        for j in range(-2, 2):
            if ((i + j) % 2) == 0:
                nb = nbp
            else:
                nb = nbn
                yapp[idx:(idx + nb)] = [(i + j) % 3 + 1] * nb

            xapp[idx:(idx + nb), 0] = rng.rand(nb)
            xapp[idx:(idx + nb), 0] += i + sigma * rng.randn(nb)
            xapp[idx:(idx + nb), 1] = rng.rand(nb)
            xapp[idx:(idx + nb), 1] += j + sigma * rng.randn(nb)
            idx += nb

    ind = np.arange(xapp.shape[0])
    rng.shuffle(ind)
    res = np.hstack([xapp, yapp[:, np.newaxis]])
    return np.array(res[ind, :2]), np.array(res[ind, 2])
