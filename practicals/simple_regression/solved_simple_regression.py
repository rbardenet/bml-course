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
        prior_mean = np.array([np.mean(y), -0, 0])
        #log_prior =  -npl.norm(theta-prior_median)**2 # Gaussian
        nu=2.0
        log_prior = -((self.dimension+nu)/2)*np.log(1+npl.norm(theta-prior_mean)**2/nu), # Student
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
        self.dimension = dimension
        self.log_target = log_target
        self.num_samples = num_samples
        return

    def find_map(self, verbose=False):
        """find MAP to locate proposal there and scale it using the inverse Hessian
        """
        res = spo.minimize(lambda theta: -self.log_target(theta), np.zeros((self.dimension,)))
        self.map = res['x']
        self.hess_inv = res['hess_inv']
        if verbose:
            print("MAP is", self.map)
            print("Inv. Hessian is", self.hess_inv)
        return

    def propose(self):
        proposal = sps.multivariate_normal(mean=self.map, cov=3*self.hess_inv)
        self.nodes = proposal.rvs(size=self.num_samples)
        unnormalized_log_weights = np.array([self.log_target(node) - proposal.logpdf(node) for node in self.nodes])
        self.log_weights = unnormalized_log_weights - logsumexp(unnormalized_log_weights)
        self.weights = np.exp(self.log_weights) # weights are normalized
        return

    def get_estimate(self, f):
        """estimate the integral of function f wrt the target
        """
        self.function_values = np.array([f(node) for node in self.nodes])
        self.estimate = np.sum([w*f for w, f in zip(self.weights, self.function_values)], 0)
        return self.estimate

    def get_sample(self, num_resamples):
        """sample from the IS approximation
        """
        indices = npr.choice(np.arange(self.num_samples), size=num_resamples, p=self.weights.flatten())
        return [self.nodes[i] for i in indices]

    def get_ess_per_sample(self):
        """compute and return effetive sample size per sample
        """
        N = self.num_samples
        self.ess_per_sample =  1./N/np.sum(self.weights**2)
        return self.ess_per_sample

def run_tests():
    """run a few tests of ImportanceSampling
    """
    mean = np.array([1,2]) # this is what we should recover
    nu = 2 # parameter for the Student
    # Our proposal is Gaussian, so it should be easy to integrate a Gaussian, and
    # harder to integrate either heavy-tailed or multimodal distributions.
    log_targets = [
        lambda x: -npl.norm(x-mean)**2/2-np.log(2*np.pi), # Gaussian
        lambda x: -(1+nu/2)*np.log(1+npl.norm(x-mean)**2/nu), # Student
        lambda x: -np.log(.5) - npl.norm(x-27*mean)**2/2-np.log(2*np.pi)
                -np.log(.5) - npl.norm(x+25*mean)**2/2-np.log(2*np.pi) # mixture of Gaussians
    ]

    for log_pi in log_targets:
        # For instance, we can try to estimate the mean of a known Gaussian
        IS = ImportanceSampling(2, log_target=log_pi, num_samples=1000)
        IS.find_map(verbose=False) # this is used to center the proposal
        IS.propose()
        sns.boxplot(IS.weights)
        plt.title("IS weights")
        plt.show()
        res = IS.get_estimate(lambda theta:theta)
        print("Estimate is", res, "with ESS/sample", IS.get_ess_per_sample())

    return

if __name__ == '__main__':
    print("hello M2DS students!")
