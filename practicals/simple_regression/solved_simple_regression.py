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
        X = self.X
        y = self.y
        prior_mean = np.array([np.mean(y), 0, 0])
        nu = 2.0
        log_prior = -((self.dimension+nu)/2)*np.log(1+npl.norm(theta-prior_mean)**2/nu), # say, a Student
        log_likelihood = - theta[2] - npl.norm(y - X*theta[1]-theta[0])**2 / (2*np.exp(theta[2])**2 )
        return log_prior + log_likelihood

    def find_map(self):
        """find MAP to locate proposal there and scale it using the inverse Hessian
        """
        dimension = 3 # so far, I keep this hardcoded
        res = spo.minimize(lambda theta: -self.get_log_target(theta), np.zeros((dimension,)))
        self.map = res['x']
        self.hess_inv = res['hess_inv']
        return self.map, self.hess_inv

class ImportanceSampling():

    def __init__(self, dimension, log_target, num_samples):
        """initialize class
        """
        return

    def find_map(self, verbose=False):
        """find MAP to locate proposal there and scale it using the inverse Hessian
        """
        return

    def propose(self):
        return

    def get_estimate(self, f):
        return self.estimate

    def get_sample(self, num_resamples):
        return

    def get_ess_per_sample(self):
        return


def run_tests():
    """run a few tests of ImportanceSampling
    """
    return

if __name__ == '__main__':
    print("hello M2DS students!")
