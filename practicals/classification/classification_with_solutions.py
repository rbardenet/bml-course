import numpy as np
import numpy.random as npr
import scipy.special as sps
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pymc3 as pm
import theano.tensor as tt
import theano
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def perform_and_visualize_PCA(X, y):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_transformed = pca.transform(X)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(X_transformed[:,0], X_transformed[:,1], c=y, edgecolors='k')
    return ax

def get_sklearn_results(X_train, y_train, X_test):
    """
    use sklearn's LogisticRegression to predict the test labels.
    Output should be a (N_test, 1) numpy array.
    """
    logReg = LogisticRegression(solver='lbfgs', multi_class="multinomial")
    # Note how by default skl has an l2 regularization, making the solution the MAP for Gaussian priors
    logReg.fit(X_train, y_train)
    return logReg.intercept_, logReg.coef_, logReg.predict(X_test)

def get_logistic_results(X_train, y_train, X_test):
    """
    perform MCMC on a multinomial results.
    This should return two pymc3 Trace objects: the first one is the usual
    sample of a chain on the parameters, targeting the posterior. The second
    one is detailed at the end of the notebook. It is a chain on the labels,
    targeting the posterior predictive.
    """
    multinomial_regression = pm.Model()
    X_shared = theano.shared(X_train) # this is useful later on to compute predictions on X_test
    y_shared = theano.shared(y_train)
    #print("y_shared", y_shared)
    #y_shared_print = tt.printing.Print('y_shared')(y_shared)
    num_classes = 3
    num_samples = X_train.shape[0]
    dimension = X_train.shape[1]

    with multinomial_regression:

        # set up priors
        prior_location = 0
        prior_scale = 1
        #b = pm.MvNormal('b', mu = np.array([1,2,3]), cov = np.eye(3), shape=(1, 3))
        b = pm.Normal('b', mu=prior_location, sd=prior_scale, shape=num_classes)
        theta_list = []
        for k in range(num_classes):
            theta_list.append(
                pm.Normal(name='theta'+str(k), mu=prior_location, sd=prior_scale, shape=dimension)
            )
        theta = tt.stack(theta_list, axis=1)
        #theta_print = tt.printing.Print('theta')(theta)
        # set up likelihood
        #log_weights = pm.Deterministic('log_weights', b + tt.dot(X_shared, theta) )
        unnormalized_weights = tt.exp(b + tt.dot(X_shared, theta))
        weights = pm.Deterministic('weights', unnormalized_weights/unnormalized_weights.sum(axis=-1).reshape((num_samples, 1)))
        #X_print = tt.printing.Print('X')(X_shared[:5,:])

        y_list = []
        #y_list = [pm.Categorical('y'+str(i), p=weights[i,:], observed=y_shared[i]) for i, x in enumerate(X_shared)]
        #weights_print = tt.printing.Print('weights')(weights[5,:])
        #y_print = tt.printing.Print('y')(y_shared[5])
        for i in range(tt.shape(X_shared).eval()[0]):
            y_list.append(pm.Categorical('y'+str(i), p=weights[i,:], observed=y_shared[i]))
        y = tt.stack(y_list)
        #y = pm.Categorical('y', p=weights, observed=y_shared, total_size=num_samples) # Categorical normalizes for you
        #logpval = pm.Deterministic('logpval', multinomial_regression.logpt)

        # get posterior sample
        trace = pm.sample(2000, tune=500, target_accept=.8)

        # get MAP for comparison with sklearn
        map_estimate = pm.find_MAP()

        # get predictions using the posterior predictive
        #X_shared.set_value(X_test) # shared tensors allow replacing values
        #y_shared.set_value(np.zeros((len(X_test),), dtype='int')) # dummy values
        #ppc = pm.sample_posterior_predictive(trace, samples=1000)

    return map_estimate, trace#, ppc

def predict(trace, X_test):
    """
    extract predictions from the posterior predictive chain. It'd
    be neater to do this using the pymc3 model defined earlier using
    theano's shared arrays, but I've run into technical issues with
    shared arrays and pymc3's categorical distribution. I ended up
    coding the following function by hand. I've learnt that pymc4 will
    use tensorflow instead of theano, btw.
    """
    varnames = trace.varnames
    num_classes = 3
    num_samples = trace[varnames[0]].shape[0]
    dimension = X_test.shape[1]
    posterior_predictive = np.zeros((X_test.shape[0], num_classes))

    for i in range(num_samples):
        b = trace['b'][i]
        theta0 = trace['theta0'][i].reshape((dimension,1))
        theta1 = trace['theta1'][i].reshape((dimension,1))
        theta2 = trace['theta2'][i].reshape((dimension,1))
        theta = np.concatenate([theta0, theta1, theta2], axis=1)
        unnormalized_weights = np.exp(b + np.dot(X_test, theta))
        weights = unnormalized_weights/unnormalized_weights.sum(axis=-1).reshape((X_test.shape[0], 1))
        posterior_predictive += weights

    preds = [np.argmax(w) for w in weights]
    return preds

if __name__ == '__main__':
    print("hello M1DS students!")
