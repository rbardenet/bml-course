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
        df_african = df[df["cont_africa"] == 1]
        df_rest = df[df["cont_africa"] == 0]
        self.X = dict()
        self.y = dict()
        self.X["African"] = df_african["rugged"].values # sklearn format (np.arrays)
        self.y["African"] = df_african["rgdppc_2000"].values
        self.X["rest"] = df_rest["rugged"].values # sklearn format (np.arrays)
        self.y["rest"] = df_rest["rgdppc_2000"].values
        return self.X, self.y

class GibbsSampler():

    def __init__(self, num_full_sweeps=10):
        """initialize class. You have nothing to modify here unless you want
        to play with the prior parameters or the starting point of the MCMC chain.
        """
        # initialize variables
        self.num_full_sweeps = num_full_sweeps
        self.varnames = [
            "beta_bar", "beta_African", "beta_rest",
            "sigma2_African", "sigma2_rest"
        ]
        self.dimension = len(self.varnames)
        self.state = dict( zip(self.varnames, [0, 0, 0, 1, 1]) ) # initialize Markov chain
        self.a, self.b = 1, 1 # Parameters for the prior over the sigma's.
        self.sigma2_beta = 1 # parameter for the prior over the global beta.

        # you will keep all MCMC samples in the history variable. It is a dictionary of
        # lists. By the end of the MCMC run, each list should be a one-dimensional time-series
        # of length num_full_sweeps, corresponding to the MCMC trace of the variable
        # specified by the corresponding key in the dictionary.
        self.history = {key: [] for key in self.varnames}

    def load_and_normalize_data(self):
        """
        normalize data: y because every time y is used, it
        gets subtracted by its average, and X to make integration
        wrt the improper prior over the intercept easier. You have
        nothing to modify in this method.
        """
        # Load data
        PM = PracticalMaterial()
        self.X, self.y = PM.fetch_data()
        for key in self.X:
            self.X[key] -= np.mean(self.X[key])
            self.y[key] -= np.mean(self.y[key])
        return

    def plot_data(self):
        """
        plot data and return axes, so that we can later
        superimpose posteriors. You have nothing to modify here.
        """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
        sns.scatterplot(self.X["rest"], self.y["rest"], ax=ax[1])
        ax[0].set(xlabel="Ruggedness index", ylabel="log GDP", title="Non-African nations")
        sns.scatterplot(self.X["African"], self.y["African"], ax=ax[0])
        ax[1].set(xlabel="Ruggedness index", ylabel="log GDP", title="African nations")
        return fig, ax

    def sample_local_variables(self):
        """sample the "within-class" conditionals. This method should modify
        self.state[k] for k in ["beta_African", "beta_rest",
        "sigma2_African", "sigma2_rest"].
        """
        return

    def sample_global_variables(self):
        """sample from conditional of beta_bar given the rest. This method should
        modify self.state["beta_bar"].
        """
        return

    def run(self):
        """Run num_full_sweeps epochs of Gibbs sampling. You have nothing to
        modify here.
        """
        for t in range(self.num_full_sweeps):
            self.sample_local_variables()
            self.sample_global_variables()
            for variable in self.varnames:
                self.history[variable].append(self.state[variable])
        return

    def plot_traces(self):
        """plot the history of the chain versus iteration number. You have nothing to
        modify here.
        """
        df = pd.DataFrame.from_dict(self.history)
        df.plot(figsize=((12,8)))
        plt.show()
        return

    def plot_pairwise_marginals(self):
        """plot the approximate 2-marginals of the posterior. You have nothing to
        modify here.
        """
        df = pd.DataFrame.from_dict(self.history)
        sns.pairplot(df, diag_kind="kde")
        plt.show()
        return

if __name__ == '__main__':
    print("hello M2DS students!")
