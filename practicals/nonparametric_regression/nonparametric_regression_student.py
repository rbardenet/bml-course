import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy.stats as sps
import scipy.optimize as spo
import cma
import theano.tensor as tt
import sklearn.gaussian_process as sklgp
from sklearn.gaussian_process.kernels import (Matern, WhiteKernel)

def generate_data(noisy_regressed_function, sample_size, dimension, seed):
    """generate simulated data
    """
    npr.seed(seed)
    # scale is 1
    X = npr.rand(sample_size*dimension).reshape((sample_size, dimension))
    X = np.sort(X, axis=0) # This is just for plotting purposes in dimension 1
    y = np.array([noisy_regressed_function(x) for x in X])

    return X, y

class BayesianOptimization:

    def __init__(self, X, y):
        self.X, self.y = X.copy(), y.copy() # Initial training set
        self.sample_size = X.shape[0]
        self.dimension = X.shape[1]

        # Uncomment the following two lines and specify a kernel to specify the GP
        # kernel = ???
        # self.gp = sklgp.GaussianProcessRegressor(kernel = kernel, normalize_y=True)

    def fit(self):
        """fit the kernel's parameters using the marginal likelihood.
        sklearn does it for you, look it up. Return nothing.
        """
        # Note that when creating the GP, we passed a flag to sklearn
        # to force centering at each fit
        return

    def predict(self, X_test):
        """give the mean and standard deviations of the posterior GP
        at the locations in X_test. Use sklearn. Return two (1,d)-arrays.
        """
        return

    def sample_y(self, X_test, random_state):
        """sample realizations for the y values at the locations
        in X_test. Use sklearn. Return a (1,len(X_test))-array.
        """
        return

    def acquisition_criterion(self, X_test):
        """implement your choice of acquisition criterion, say EI.
        Return a (1,len(X_test))-array.
        """
        return

    def find_next_point(self, number_of_restarts):
        """maximize acquisition criterion. Return a (1,1)-array
        containing the optimal value of x.
        """
        return

    def update(self, x, y):
        """add the new point and its noisy target value
        to the training set. Nothing to change here.
        """
        self.X = np.concatenate((self.X, x.reshape(1,self.dimension)), 0)
        self.y = np.concatenate((self.y, y.reshape(1,1)), 0)
        return


#    def optimize_acquisition_criterion(self):
