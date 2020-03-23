import numpy as np
import numpy.random as npr
import scipy.special as sps
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pymc3 as pm
import theano.tensor as tt
import theano

def perform_and_visualize_PCA(X, y):
    return

def get_sklearn_results(X_train, y_train, X_test):
    """
    use sklearn's LogisticRegression to predict the test labels.
    Output should be a (N_test, 1) numpy array.
    """
    return

def get_logistic_results(X_train, y_train, X_test):
    """
    perform MCMC on a multinomial model.
    This should return two pymc3 Trace objects: the first one is the usual
    sample of a chain on the parameters, targeting the posterior. The second
    one is detailed at the end of the notebook. It is a chain on the labels,
    targeting the posterior predictive.
    """
    return

def predict(ppc):
    """
    extract predictions from the posterior predictive chain
    """
    return

if __name__ == '__main__':
    print("hello M1DS students!")
