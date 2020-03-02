import numpy as np
from scipy import signal

from .base import AdaptiveFilter


class WienerFilter(AdaptiveFilter):

    def __init__(self, *args, **kwargs):
        super(WienerFilter, self).__init__(*args, **kwargs)

    def predict(self, x):
        # Estimate the local mean and variance
        mean = signal.correlate(x, np.ones(x.shape), 'same') / np.product(x.shape, axis=0)
        var = (signal.correlate(x ** 2, np.ones(x.shape), 'same') / np.product(x.shape, axis=0)
                - mean ** 2)

        # Estimate the noise power
        noise = mean(np.ravel(var), axis=0)

        res = (x - mean)
        res *= (1 - noise / var)
        res += mean
        y_hat = np.where(var < noise, mean, res)
        return y_hat

    def step(self, xi, yi, wi):
        # Predict
        yi_hat = self.predict(xi)

        # Compute error
        ei = yi - yi_hat

        # Wiener filter doesn't use weights
        w = None

        return yi_hat, ei, w