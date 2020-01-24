import numpy as np
import numpy.random as npr
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from sklearn.linear_model import Lasso

def generate_data(sample_size, dimension, seed):
    """generate simulated data with controlled theta_true
    """
    npr.seed(seed)
    # Set up parameters
    intercept = 0 # I set the intercept to zero for simplicity. Otherwise,
                  # you'd have to include an intercept in your models.
    sigma_noise = 1
    proportion_of_nonzero_coefficients = 0.1
    signal_absolute_value = 10

    # Prepare containers to return
    theta_true = np.zeros(dimension)
    support = np.zeros(dimension)

    for i in range(dimension):
        if bernoulli(proportion_of_nonzero_coefficients).rvs():
            if bernoulli(0.5).rvs():
                theta_true[i] = np.random.normal(signal_absolute_value, 1)
            else:
                theta_true[i] = np.random.normal(-signal_absolute_value, 1)
            support[i] = 1

        else:
            theta_true[i] = np.random.normal(0, 0.25)

    X = np.random.normal(0, 1, (sample_size, dimension))
    y = np.random.normal(X.dot(theta_true) + intercept, sigma_noise)

    return X, y, theta_true, sigma_noise, np.where(support)[0]


def plot_coefficients(axes, theta_true, indices_support, theta_hat, lower_bound=None, upper_bound=None, color='b', label=""):
    """plot theta_true and estimated theta, with filled in error bars. Then plot residuals.
    """
    indices = range(len(theta_true))

    # Plot credible intervals
    axes[0].plot(indices, theta_true, 'g', alpha=0.5, linewidth=3, label="true values")
    axes[0].plot(indices, theta_hat, color=color, alpha=0.5, linewidth=3, label=label+" estimate")

    if lower_bound is not None:
        axes[0].fill_between(indices, upper_bound, lower_bound, color=color, alpha=0.3)

    ymin, ymax = axes[0].get_ylim()
    delta = ymax-ymin
    axes[0].vlines(indices_support, ymin-.1*delta, ymax+.1*delta, linestyle='--', color='g')
    axes[0].set_ylim([ymin, ymax])

    # Plot residuals
    axes[1].plot(indices, theta_hat-theta_true, color=color, alpha=0.5, linewidth=3, label=label+" residual")

    if lower_bound is not None:
        axes[1].fill_between(indices, upper_bound-theta_true, lower_bound-theta_true, color=color, alpha=0.3)

    ymin, ymax = axes[1].get_ylim()
    delta = ymax-ymin
    axes[1].vlines(indices_support, ymin-.1*delta, ymax+.1*delta, linestyle='--', color='g')
    axes[1].set_ylim([ymin, ymax])
    plt.legend()

    return # axes

def get_sklearn_lasso_estimate(X, y):
    """apply scikit-learn lasso. This should return an estimated theta_hat.
    """
    # here should go your code. Meanwhile, I'll just output a constant zero vector.
    ls = Lasso(fit_intercept=True)
    ls.fit(X, y)
    return ls.coef_

def get_mcmc_sample_for_laplace_prior(X, y):
    # This should return a pymc3 Trace object

    lasso = pm.Model()
    with lasso:
        prior_location = 0
        prior_scale = 1
        theta = pm.Laplace('theta', mu=prior_location, b=prior_scale, shape=X.shape[1])
        y_noiseless = tt.dot(X, theta)
        likelihood = pm.Normal('likelihood', y_noiseless, observed=y)
        step = pm.Metropolis(tune_interval=1)
        trace = pm.sample(1000)

    return trace

def get_mcmc_sample_for_horseshoe_prior(X, y):
    # This should return a pymc3 Trace object
    return

def get_mcmc_sample_for_finnish_horseshoe_prior(X, y):
    # This should return a pymc3 Trace object
    return

if __name__ == '__main__':

    X, y, theta, support = generate_data(100, 200, 1)
