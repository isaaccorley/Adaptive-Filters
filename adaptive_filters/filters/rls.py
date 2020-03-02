import numpy as np

from .base import AdaptiveFilter


class RLSFilter(AdaptiveFilter):

    def __init__(self, *args, **kwargs):
        super(RLSFilter, self).__init__(*args, **kwargs)

        self.eps = eps
        self.R = 1/self.eps * np.identity(self.n)


    def step(self, xi, yi, wi):
        # Predict
        yi_hat = self.predict(xi)

        # Compute error
        ei = yi - yi_hat

        # Compute gradient
        dw = self.mu * ei * xi

        # Update weights
        R1 = np.dot(np.dot(np.dot(self.R, xi), xi.T), self.R)
        R2 = self.mu + np.dot(np.dot(xi, self.R), xi.T)
        self.R = 1/self.mu * (self.R - R1/R2)
        dw = np.dot(self.R, xi.T) * ei
        w = wi + dw

        return yi_hat, ei, w