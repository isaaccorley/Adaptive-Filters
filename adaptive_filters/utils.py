import numpy as np


def noiseChannel(
    x,
    sigma_n=25,
    seed=np.random.randint(low=1, high=1E4, size=1)
):

    # Set random seed
    np.random.seed(seed)

    # Real arrays
    if not np.iscomplexobj(x):
        n = np.sqrt(sigma_n) * np.random.randn(*x.shape)
        np.random.seed(seed + 75)
        n0 = np.sqrt(sigma_n) * np.random.randn(*x.shape)

    # Complex arrays
    else:
        n = np.sqrt(sigma_n/2) * (np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape))
        np.random.seed(seed + 75)
        n0 = np.sqrt(sigma_n/2) * (np.random.randn(*x.shape) + 1j*np.random.randn(*x.shape))

    # Add noise to signal
    xn = x + n

    return xn, n0, n

def to_history_matrix(a, n, bias=False):
    """ Convert a 1-D vector a 2-D historical matrix """
    h = np.array([a[i:i+n] for i in range(len(a)-n+1)]) 
    
    if bias:
        h = np.vstack((h.T, np.ones(len(h)))).T

    return h