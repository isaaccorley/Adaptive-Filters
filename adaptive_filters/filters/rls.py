import numpy as np

from .base import AdaptiveFilter


class RLSFilter(AdaptiveFilter):

    def __init__(self, *args, **kwargs):
        super(RLSFilter, self).__init__(*args, **kwargs)

        # Initialize R(0) - inverse of autocorrelation matrix
        self.R = 1/self.eps * np.identity(self.n)


    def step(self, xi, yi, wi):
        # Predict
        yi_hat = self.predict(xi)

        # Compute error
        ei = yi - yi_hat

        # Compute gradient

        # R_num = R(k−1)x(k)x(k)'R(k−1)
        R_num = np.dot(np.dot(np.dot(self.R, xi), xi.T), self.R)

        # R_den = mu + x(k)'R(k−1)x(k)
        R_den = self.mu + np.dot(np.dot(xi, self.R), xi.T)

        # R(k) = 1/mu * (R(k-1) - R_num/R_den)
        self.R = 1/self.mu * (self.R - R_num/R_den)

        # dw = R(k)
        dw = np.dot(self.R, xi.T) * ei

        # Update weights
        w = wi + dw

        return yi_hat, ei, w