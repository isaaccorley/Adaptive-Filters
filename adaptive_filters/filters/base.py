import numpy as np
from tqdm import tqdm


class AdaptiveFilter(object):

    def __init__(
        self,
        n=4,
        mu=1E-2,
        w="random",
        sigma=0.5
    ):
        self.n = n
        self.mu = mu
        self.w = self.init_weights(w, sigma)


    def init_weights(self, w, sigma=0.5):
        """ Initialize filter coefficients """
        if isinstance(w, str):
            if w == "random":
                w = np.random.normal(0, sigma, self.n)
            elif w == "zeros":
                w = np.zeros(self.n)
            elif w == "ones":
                w = np.ones(self.n)
            else:
                raise ValueError("Uknown weight initialization type")
        
        return w

    def predict(self, x):
        """ Predict output y given input x and weights w """
        return np.dot(self.w, x)

    def fit(self, x, y):
        """ Fit the adaptive filter """

        x, y = np.array(x), np.array(y)
        assert x.shape[0] == y.shape[0], "x and y must have the same length"

        N = x.shape[0]
        y_hat, e = np.zeros(N), np.zeros(N)

        print("Fitting LMS Filter")
        for i in tqdm(range(N)):
            # Perform an update step
            y_hat[i], e[i], self.w = self.step(x[i], y[i], self.w)

        return y_hat, e, self.w