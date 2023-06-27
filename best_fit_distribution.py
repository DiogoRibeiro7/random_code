import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

_distn_names = [dist_name for dist_name in dir(st) if isinstance(getattr(st, dist_name), st.rv_continuous)]


def best_fit_distribution(data: np.ndarray, bins: int = 200, ax: object = None) -> list:
    """
    Model data by finding the best fit distribution to the data.

    Args:
        data (np.ndarray): The input data.
        bins (int, optional): The number of bins for the histogram. Defaults to 200.
        ax (object, optional): The axis object to plot the fitted PDF. Defaults to None.

    Returns:
        list: A list of tuples containing the best fit distribution, its parameters, and the sum of squared errors (SSE).

    """
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Best holders
    best_distributions = []

    # Estimate distribution parameters from data
    for distribution_name in _distn_names:
        if distribution_name not in ['levy_stable', 'studentized_range']:
            distribution = getattr(st, distribution_name)

            # Try to fit the distribution
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                try:
                    # Fit dist to data
                    params = distribution.fit(data)
                    arg, loc, scale = params[:-2], params[-2], params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # Plot if axis passed in
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)

                    # Identify if this distribution is better
                    best_distributions.append((distribution, params, sse))
                except Exception:
                    pass

    return sorted(best_distributions, key=lambda x: x[2])


def make_pdf(dist: object, params: tuple, size: int = 10000) -> pd.Series:
    """
    Generate the distribution's Probability Distribution Function (PDF).

    Args:
        dist (object): The distribution object.
        params (tuple): The distribution parameters.
        size (int, optional): The number of points to generate in the PDF. Defaults to 10000.

    Returns:
        pd.Series: The PDF as a pandas Series.

    """
    # Separate parts of parameters
    arg, loc, scale = params[:-2], params[-2], params[-1]

    # Get sane start and end points of the distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn it into a pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)
    return pdf



# Load data from statsmodels datasets
data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])

# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_distibutions = best_fit_distribution(data, 200, ax)
best_dist = best_distibutions[0]

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(u'El Niño sea temp.\n All Fitted Distributions')
ax.set_xlabel(u'Temp (°C)')
ax.set_ylabel('Frequency')

# Make PDF with best params 
pdf = make_pdf(best_dist[0], best_dist[1])

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist[0].shapes + ', loc, scale').split(', ') if best_dist[0].shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_dist[1])])
dist_str = '{}({})'.format(best_dist[0].name, param_str)

ax.set_title(u'El Niño sea temp. with best fit distribution \n' + dist_str)
ax.set_xlabel(u'Temp. (°C)')
ax.set_ylabel('Frequency')
plt.show()