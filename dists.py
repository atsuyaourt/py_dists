import warnings

import numpy as np
from scipy import stats as st


class dists:
    """
    Class to fit distribution to data
    """
    def __init__(self, data, dist_name='best', params=None, bins=200, test_dists=None):
        self.data = data
        self.bins = bins
        if dist_name == 'best':
            if test_dists is None:
                self.test_dists = ['norm', 'exponweib', 'weibull_max', 'weibull_min', 'pareto', 'genextreme']
            else:
                self.test_dists = test_dists
            self.best_fit_distribution()
        else:
            self.dist_name = dist_name
            if params is not None:
                self.params = params
            self.generate_distribution()
            

    def generate_distribution(self):
        """Generate distribution"""
        _data = self.data[~np.isnan(self.data)] 

        self.dist = None
        self.params = None

        if self.dist_name=='hist':
            h = np.histogram(_data, bins=self.bins, density=True)
            self.dist = st.rv_histogram(h)
        else:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    # fit dist to data
                    self.dist = getattr(st, self.dist_name)
                    if self.params is not None:
                        # Separate parts of parameters
                        arg = self.params[:-2]
                        loc = self.params[-2]
                        scale = self.params[-1]
                        self.params = self.dist.fit(_data, loc=loc, scale=scale, *arg)
                    else:
                        self.params = self.dist.fit(_data)
                        # Separate parts of parameters
                        arg = self.params[:-2]
                        loc = self.params[-2]
                        scale = self.params[-1]
                    self.dist = self.dist(loc=loc, scale=scale, *arg)
            except Exception as e:
                print(e)

    def best_fit_distribution(self):
        """
        Find best fit distribution to data
        Adapted from: https://stackoverflow.com/a/37616966/6367063
        """
        # Get histogram of original data
        _data = self.data[~np.isnan(self.data)]
        y, x = np.histogram(_data, bins=self.bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Best holders
        best_dist_name = 'norm'
        best_dist = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        for dist_name in self.test_dists:
            # Try to fit the distribution
            try:
                # fit dist to data
                self.dist_name = dist_name
                self.generate_distribution()
                # Calculate fitted PDF and error with fit in distribution
                pdf = self.dist.pdf(x)
                sse = np.sum(np.power(y - pdf, 2.0))
                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_dist_name = dist_name
                    best_dist = self.dist
                    best_params = self.params
                    best_sse = sse
            except Exception as e:
                print(e)
                continue

        self.dist_name = best_dist_name
        self.dist = best_dist
        self.params = best_params
