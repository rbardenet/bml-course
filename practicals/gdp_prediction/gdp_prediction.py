import numpy as np
import numpy.random as npr
import pymc3 as pm
import theano.tensor as tt
import pandas as pd


class GDPPredictionUsingLinearRegression:
    def __init__(
        self, data_url="https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
    ):
        self.data_url = data_url
        self.dimension = 3  # theta = (b, a, log_sigma)

    def fetch_data(self):
        """fetch data on GDP and ruggedness index"""
        data = pd.read_csv(self.data_url, encoding="ISO-8859-1")
        df = data[["cont_africa", "rugged", "rgdppc_2000"]]  # keep only 3 features
        df = df[np.isfinite(df.rgdppc_2000)]
        df["rgdppc_2000"] = np.log(df["rgdppc_2000"])  # take the log GDP
        self.df = df  # usual pandas dataframe
        self.X = df["rugged"].values  # sklearn format (np.arrays)
        self.y = df["rgdppc_2000"].values
        self.y -= np.mean(self.y)
        self.y /= np.std(self.y)
        return self.X, self.y

    def get_mcmc_sample(self):
        """This should return a pymc3 Trace object"""


class GDPPredictionUsingHierarchicalRegression:
    def __init__(
        self, data_url="https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
    ):
        self.data_url = data_url
        self.dimension = 3  # theta = (b, a, log_sigma)

    def fetch_data(self):
        """fetch data on GDP and ruggedness index"""
        data = pd.read_csv(self.data_url, encoding="ISO-8859-1")
        df = data[["cont_africa", "rugged", "rgdppc_2000"]]  # keep only 3 features
        df = df[np.isfinite(df.rgdppc_2000)]
        df["rgdppc_2000"] = np.log(df["rgdppc_2000"])  # take the log GDP
        df["rgdppc_2000"] = (df["rgdppc_2000"] - df["rgdppc_2000"].mean()) / df[
            "rgdppc_2000"
        ].std()
        df_african = df[df["cont_africa"] == 1]
        df_rest = df[df["cont_africa"] == 0]
        self.X = dict()
        self.y = dict()
        self.X["African"] = df_african["rugged"].values  # sklearn format (np.arrays)
        self.y["African"] = df_african["rgdppc_2000"].values
        self.X["rest"] = df_rest["rugged"].values  # sklearn format (np.arrays)
        self.y["rest"] = df_rest["rgdppc_2000"].values
        return self.X, self.y

    def get_mcmc_sample(self, num_iterations):
        """This should return a pymc3 Trace object"""


if __name__ == "__main__":
    print("Hello M2DS students!")
