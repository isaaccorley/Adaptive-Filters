import numpy as np
from .base import AdaptiveFilter


class NLMSFilter(AdaptiveFilter):

    def __init__(self, *args, **kwargs):
        super(NLMSFilter, self).__init__(*args, **kwargs)

    def step(self, xi, yi, wi):
        # Predict
        yi_hat = self.predict(xi)

        # Compute error
        ei = yi - yi_hat

        # Update learning rate eta = mu / (eps + l2_norm(x))
        nu = self.mu / (self.eps + np.linalg.norm(xi))

        # Compute gradient dw = nu * e(k) * x(k)
        dw = nu * ei * xi

        # Update weights
        w = wi + dw

        return yi_hat, ei, w