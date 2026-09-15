import numpy as np
import scipy.stats as sps
from cuqi.distribution import Distribution

class Poisson(Distribution):
    """ Poisson distribution. 

    Defines a Poisson distribution.

    """

    def __init__(self, rate=None, **kwargs):

        # Init from abstract distribution class
        super().__init__(**kwargs)

        self.rate = rate
  
    def logpdf(self, x):
        if isinstance(x, (float,int)):
            x = np.array([x])
        return np.sum(sps.poisson.logpmf(x, mu=self.rate))

    def _sample(self,N=1,rng=None):
        if rng is not None:
            s = rng.poisson(lam=self.rate, size=(N, self.dim)).T
        else:
            s = np.random.poisson(lam=self.rate, size=(N, self.dim)).T
        return s
