from .base import AdaptiveFilter


class LMSFilter(AdaptiveFilter):

    def __init__(self, *args, **kwargs):
        super(LMSFilter, self).__init__(*args, **kwargs)

    def step(self, xi, yi, wi):
        # Predict
        yi_hat = self.predict(xi)

        # Compute error
        ei = yi - yi_hat

        # Compute gradient
        dw = self.mu * ei * xi

        # Update weights
        w = wi + dw

        return yi_hat, ei, w