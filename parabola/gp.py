import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jit, vmap, lax, random
from jax.scipy.stats.norm import cdf, pdf
from functools import partial

from scipy.optimize import minimize

# def kernel(xn, xm, params):
#     p0, p1, p2, p3 = params
#     # eqn 6.63 in PRML
#     return p0*jnp.exp(-p1/2 * jnp.dot(xn-xm, xn-xm)) + p2 + p3*jnp.dot(xn, xm)

# def kernel(xn, xm, params):
#     p0, p1 = params
#     return jnp.tanh(p0*jnp.dot(xn, xm) + p1)

def kernel(xn, xm, variance):
    # eqn 6.23 in PRML
    return jnp.exp(-variance[0] * jnp.dot(xn - xm, xn - xm))

class GP:

    def __init__(self, kernel, params, beta):
        self.kernel = jit(kernel)
        self.params = params
        self.beta = beta

        # jit compile function to compute covariance
        self.k = jit(vmap(kernel, (0, 0, None)))
        self.K = jit(vmap(vmap(kernel, (0, None, None)), (None, 0, None)))

        # gradient of covariance matrix w.r.t. kernel parameters
        self.Cgrad = jit(jacfwd(self.K, 2))

    def fit(self, X, y, tol=1e-3):
        # objective is to maximize the log-likelihood w.r.t. kernel parameters

        # X is an n x m matrix with n observations each with m features
        # y is an n x 1 matrix with n observations of single variable response
        self.X = X
        self.y = y.ravel()
        n, m = self.X.shape

        # define function to return log-likelihood and its gradient w.r.t. kernel parameters
        @jit
        def objective(params):
            # covariance
            C = jnp.eye(n) / self.beta + self.K(self.X, self.X, params)
            C = (C + C.T) / 2.

            # precision
            Cinv = jnp.linalg.inv(C)
            Cinv = (Cinv + Cinv.T) / 2.

            # gradient of covariance matrix
            Cgrad = self.Cgrad(self.X, self.X, params)

            # negative log likelihood of data given parameters
            NLL = jnp.nansum(jnp.log(jnp.linalg.eigvalsh(C))) + self.y @ Cinv @ self.y

            # gradient of negative log likelihood
            gradNLL = jnp.trace(Cinv @ Cgrad) - jnp.einsum('n,nm,mol,op,p->l', self.y, Cinv, Cgrad, Cinv, self.y)

            return NLL, gradNLL

        # use Scipy's minimize to find optimal parameters
        res = minimize(objective, self.params, jac=True, tol=tol)
        # print(res)
        self.params = res.x

        # compute inverse covariance matrix using optimal parameters
        self.Cinv = np.linalg.inv(jnp.eye(n) / self.beta + self.K(X, X, self.params))

    def predict(self, Xtest):
        # Xtest has dimensions [l samples x d features]

        # k_ij = kernel(X_test[i], X_train[j])
        k = self.K(self.X, Xtest, self.params)

        # predict mean
        m = jnp.einsum('lm,mn,n->l', k, self.Cinv, self.y)

        # measurement variance
        v = 1. / self.beta

        # epistemic variance
        v += np.clip(self.k(Xtest, Xtest, self.params) - jnp.einsum('ln,nm,lm->l', k, self.Cinv, k), 0, np.inf)

        return m, jnp.sqrt(v)

    def conditioned_stdv(self, Xtest, X, Cinv):
        # k_ij = kernel(X_test[i], X_train[j])
        k = self.K(X, Xtest, self.params)

        # measurement variance
        v = 1. / self.beta

        # epistemic variance
        v += np.clip(self.k(Xtest, Xtest, self.params) - jnp.einsum('ln,nm,lm->l', k, Cinv, k), 0, np.inf)

        return jnp.sqrt(v)

    # return indeces of optimal samples
    def search(self, data, objective, N, max_reps=1, batch_size=512, exploit=True):

        # initialize X matrix to condition on
        X = self.X.copy()
        Cinv = self.Cinv.copy()

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # make predictions once
        all_preds = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples // batch_size):
            # make predictions on data
            all_preds.append(self.predict(data[batch_inds])[0])

        # compute objective (f: R^[n_t, n_o, w_exp] -> R) in batches
        objective_batch = jit(vmap(lambda pred, stdv: objective(pred, stdv), (0, 0)))

        # search for new experiments until find N
        best_experiments = []
        while len(best_experiments) < N:

            # compute utilities in batches to avoid memory problems
            utilities = []
            for preds, batch_inds in zip(all_preds, np.array_split(np.arange(n_samples), n_samples // batch_size)):
                stdvs = self.conditioned_stdv(data[batch_inds], X, Cinv)
                if exploit:
                    utilities.append(objective_batch(preds, stdvs))
                else:
                    utilities.append(stdvs)
            utilities = jnp.concatenate(utilities)
            # print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # plt.plot(np.array(utilities).ravel())

            # pick an experiment
            # print(f"Picked experiment {len(best_experiments)} out of {N}")
            exp = np.argmax(utilities)

            # append datapoint to X
            X = np.concatenate((X, np.expand_dims(data[exp], 0)))

            # update inverse
            Cinv = np.linalg.inv(jnp.eye(X.shape[0]) / self.beta + self.K(X, X, self.params))

            # add experiment to the list
            if sum(np.in1d(best_experiments, exp)) < max_reps:
                best_experiments += [exp.item()]
            else:
                if exploit:
                    print("Max replicates exceeded, switching to pure exploration")
                    exploit = False
                else:
                    print("Max exploration replicates exceeded, terminating")
                    return np.sort(best_experiments)

        return np.sort(best_experiments)