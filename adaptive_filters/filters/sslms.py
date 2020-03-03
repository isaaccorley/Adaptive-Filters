import numpy as np

from .base import AdaptiveFilter


class SSLMSFilter(AdaptiveFilter):

    def __init__(self, *args, **kwargs):
        super(SSLMSFilter, self).__init__(*args, **kwargs)

    def step(self, xi, yi, wi):
        # Predict
        yi_hat = self.predict(xi)

        # Compute error
        ei = yi - yi_hat

        # Compute gradient dw = mu * sign(e(k)) * sign(x(k))
        dw = self.mu * np.sign(ei) * np.sign(xi)

        # Update weights
        w = wi + dw

        return yi_hat, ei, w