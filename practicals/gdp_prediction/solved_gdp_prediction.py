import numpy as np
import numpy.random as npr
import pymc3 as pm
import theano.tensor as tt
import pandas as pd

class GDPPredictionUsingLinearRegression():

    def __init__(self, data_url="https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"):
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
        self.y -= np.mean(self.y)
        self.y /= np.std(self.y)
        return self.X, self.y
    
    def get_mcmc_sample(self):
        """This should return a pymc3 Trace object
        """
        X, y = self.X, self.y
        regression = pm.Model()
        with regression:
            prior_location = 0
            prior_scale = 1
            alpha = pm.StudentT('alpha', mu=prior_location, sigma=prior_scale, nu=1)
            beta = pm.StudentT('beta', mu=prior_location, sigma=prior_scale, nu=1)
            log_sigma = pm.StudentT('log_sigma', mu=prior_location, sigma=prior_scale, nu=1)
            y_noiseless = alpha + tt.dot(X, beta)
            likelihood = pm.Normal('likelihood', mu=y_noiseless, sigma=np.exp(log_sigma), observed = y)
            trace = pm.sample(2000)
            
        return trace
    
class GDPPredictionUsingHierarchicalRegression():
    
    def __init__(self, data_url="https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"):
        self.data_url = data_url
        self.dimension = 3 # theta = (b, a, log_sigma)

    def fetch_data(self):
        """fetch data on GDP and ruggedness index
        """
        data = pd.read_csv(self.data_url, encoding="ISO-8859-1")
        df = data[["cont_africa", "rugged", "rgdppc_2000"]] # keep only 3 features
        df = df[np.isfinite(df.rgdppc_2000)]
        df["rgdppc_2000"] = np.log(df["rgdppc_2000"]) # take the log GDP
        df["rgdppc_2000"] = ( df["rgdppc_2000"] - df["rgdppc_2000"].mean() ) / df["rgdppc_2000"].std()
        df_african = df[df["cont_africa"] == 1]
        df_rest = df[df["cont_africa"] == 0]
        self.X = dict()
        self.y = dict()
        self.X["African"] = df_african["rugged"].values # sklearn format (np.arrays)
        self.y["African"] = df_african["rgdppc_2000"].values
        self.X["rest"] = df_rest["rugged"].values # sklearn format (np.arrays)
        self.y["rest"] = df_rest["rgdppc_2000"].values
        return self.X, self.y
    
    def get_mcmc_sample(self, num_iterations):
        """This should return a pymc3 Trace object
        """
        X, y = self.X, self.y
        hierarchical_regression = pm.Model()

        # Normalize

        with hierarchical_regression:
            prior_location = 0
            prior_scale = 1
            
            alpha = pm.Normal('alpha', mu=prior_location, sigma=prior_scale)
            beta = pm.Normal('beta', mu=prior_location, sigma=prior_scale)
            log_sigma = pm.Normal('log_sigma', mu=prior_location, sigma=prior_scale)

            alpha_rest = pm.Normal('alpha_rest', mu=alpha, sigma=prior_scale)
            beta_rest = pm.Normal('beta_rest', mu=beta, sigma=prior_scale)
            log_sigma_rest = pm.Normal('log_sigma_rest', mu=log_sigma, sigma=prior_scale)

            alpha_African = pm.Normal('alpha_African', mu=alpha, sigma=prior_scale)
            beta_African = pm.Normal('beta_African', mu=beta, sigma=prior_scale)
            log_sigma_African = pm.Normal('log_sigma_African', mu=log_sigma, sigma=prior_scale)
            
            y_noiseless_African = alpha_African + tt.dot(X["African"], beta_African)
            likelihood_African = pm.Normal(
                'likelihood_African', mu=y_noiseless_African, sigma=np.exp(log_sigma_African), observed = y["African"]
            )
            y_noiseless_rest = alpha_rest + tt.dot(X["rest"], beta_rest)
            likelihood_rest = pm.Normal(
                'likelihood_rest', mu=y_noiseless_rest, sigma=np.exp(log_sigma_rest), observed = y["rest"]
            )

            trace = pm.sample(num_iterations)
            
        return trace


if __name__ == '__main__':
    print("Hello M2DS students!")
