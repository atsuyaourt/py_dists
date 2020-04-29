import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dists import dists
from scipy.stats import gamma, rv_histogram

# sns.set_context('talk')
sns.set_context('paper')
sns.set_style('ticks')


N = 1000
dat = gamma.rvs(8.15, scale=3.68, size=N)


# default: best fit, prescribed test distribution
best_fit1 = dists(dat)
print('best fit: {} with params {}'.format(best_fit1.dist_name, best_fit1.params))

# best fit, given a list of distribution
best_fit2 = dists(dat, test_dists=['norm', 'lognorm', 'gamma'])
print('best fit: {} with params {}'.format(best_fit2.dist_name, best_fit2.params))

# fit a gamma distribution
gamma_fit = dists(dat, dist_name='gamma')
print('gamma params {}'.format(gamma_fit.dist_name, gamma_fit.params))

# fit a histogram
hist_fit = dists(dat, dist_name='hist', bins=100)


# x = np.linspace(hist_fit.dist.ppf(0.01), hist_fit.dist.ppf(0.99), 100)
x = np.linspace(dat.min(), dat.max(), 100)
fig, ax = plt.subplots()
ax.plot(x, hist_fit.dist.pdf(x), 'r-', alpha=0.8, label='hist fit')
ax.plot(x, best_fit1.dist.pdf(x), 'b-', lw=3, alpha=0.8, label='best fit ({})'.format(best_fit1.dist_name))
ax.plot(x, gamma_fit.dist.pdf(x), 'g-', lw=3, alpha=0.8, label=gamma_fit.dist_name)
ax.hist(dat, density=True, bins=100, color='yellow')
ax.legend()
