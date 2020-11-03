import pandas as pd
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import scipy.stats as sps
import scipy.optimize as spo
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns

class PracticalMaterial():

    def __init__(self, data_url="https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"):
        """initialize class
        """
        self.data_url = data_url
        self.dimension = 3 # theta = (b, a, log_sigma)

    def fetch_data(self):
        """fetch data on GDP and ruggedness index
        """
        data = pd.read_csv(self.data_url, encoding="ISO-8859-1")
        df = data[["cont_africa", "rugged", "rgdppc_2000"]] # keep only 3 features
        df = df[np.isfinite(df.rgdppc_2000)]
        df["rgdppc_2000"] = np.log(df["rgdppc_2000"]) # take the log GDP
        self.df = df # usual pandas dataframe
        self.X = df["rugged"].values # sklearn format (np.arrays)
        self.y = df["rgdppc_2000"].values
        return self.X, self.y

    def get_log_target(self, theta):
        """return unnormalized log posterior
        """
        return

    def find_map(self):
        """find MAP to locate proposal there and scale it using the inverse Hessian
        """
        return

class ImportanceSampling():

    def __init__(self, dimension, log_target, num_samples):
        """initialize class
        """
        self.dimension = dimension
        self.log_target = log_target
        self.num_samples = num_samples
        return

    def find_map(self, verbose=False):
        """find MAP to locate proposal there and scale it using the inverse Hessian
        """
        return theta_map, inverse_hessian

    def propose(self):
        """sample quadrature nodes and compute self-normalized weights
        """
        return

    def get_estimate(self, f):
        """estimate the integral of function f wrt the target
        """
        return estimate

    def get_sample(self, num_resamples):
        """resample from the IS approximation
        """
        return list_of_samples

    def get_ess_per_sample(self):
        """return the effetive sample size per sample
        """
        return 1.

def run_tests():
    """run a few tests of ImportanceSampling
    """
    return

if __name__ == '__main__':
    print("hello M2DS students!")
