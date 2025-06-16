import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jacrev, jit, vmap, lax, random
from jax.scipy.stats.norm import cdf, pdf
from functools import partial
import time

# import scipy's optimizer
from scipy.optimize import minimize

# import matrix math functions
from .linalg import *


class FFNN():

    def __init__(self, n_inputs, n_hidden, n_outputs, rng_key=123):

        # set rng key
        self.rng_key = random.PRNGKey(rng_key)

        # store dimensions
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # determine shapes of weights/biases = [Wih, bih, Who, bho]
        self.shapes = [[n_hidden, n_inputs], [n_hidden], [n_outputs, n_hidden], [n_outputs]]
        self.k_params = []
        self.n_params = 0
        for shape in self.shapes:
            self.k_params.append(self.n_params)
            self.n_params += np.prod(shape)
        self.k_params.append(self.n_params)
            
        # initialize parameters
        self.params = np.zeros(self.n_params)
        for k1,k2,shape in zip(self.k_params, self.k_params[1:], self.shapes):
            if len(shape)>1:
                stdv = 1/np.sqrt(shape[-1])
            # self.params[k1:k2] = random.uniform(self.rng_key, shape=(k2 - k1,),
            #                                     minval=-self.param_0, maxval=self.param_0)
            self.params[k1:k2] = stdv*random.normal(self.rng_key, shape=(k2-k1,))

        # initialize hyper-parameters
        self.a = 1e-4
        self.b = 1e-4

        # initialize covariance
        self.Ainv = None

        ### define jit compiled functions ###

        # batch prediction
        self.forward_batch = jit(vmap(self.forward, (None, 0)))

        # jit compile gradient w.r.t. params
        self.Gi = jit(jacfwd(self.forward))
        self.G = jit(jacfwd(self.forward_batch))

        # jit compile function to compute gradient of loss w.r.t. parameters
        self.compute_grad_NLL = jit(jacrev(self.compute_NLL))

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        # params is a vector = [Wih, bih, Who, bho]
        return [np.reshape(params[k1:k2], shape) for k1, k2, shape in
                zip(self.k_params, self.k_params[1:], self.shapes)]

    # per-sample prediction
    @partial(jit, static_argnums=0)
    def forward(self, params, sample):
        # reshape params
        Wih, bih, Who, bho = self.reshape(params)

        # hidden layer
        h = nn.tanh(Wih @ sample + bih)

        # looks super bizarre on cosines data
        # h = nn.leaky_relu(Wih @ sample + bih)

        # output
        out = Who @ h + bho

        return out

    # estimate posterior parameter distribution
    def fit(self, X, Y, evd_tol=1e-3, nlp_tol=None, alpha_0=1e-3, alpha_1=1., patience=1, max_fails=3):

        # estimate parameters using gradient descent
        self.itr = 0
        passes = 0
        fails = 0
        convergence = np.inf
        previdence = -np.inf

        # init convergence status
        converged = False

        # initialize hyper parameters
        self.init_hypers(X, Y, alpha_0)

        while not converged:
            # update Alpha and Beta hyper-parameters
            if self.itr > 0: self.update_hypers(X, Y)

            # fit using updated Alpha and Beta
            self.res = minimize(fun=self.objective,
                                jac=self.jacobian,
                                hess=self.hessian,
                                x0=self.params,
                                args=(X, Y,),
                                tol=nlp_tol,
                                method='Newton-CG') # callback=self.callback)
            self.params = self.res.x
            self.loss = self.res.fun

            # update parameter precision matrix (Hessian)
            # print("Updating precision...")
            '''if self.itr == 0:
                self.alpha = alpha_1 * jnp.ones_like(self.params)'''
            self.update_precision(X, Y)

            # update evidence
            self.update_evidence()
            # print("Evidence {:.3f}".format(self.evidence))

            # check convergence
            convergence = np.abs(previdence - self.evidence) / np.max([1., np.abs(self.evidence)])

            # update pass count
            if convergence < evd_tol:
                passes += 1
                # print("Pass count ", passes)
            else:
                passes = 0

            # increment fails if convergence is negative
            if self.evidence < previdence:
                fails += 1
                # print("Fail count ", fails)

            # finally compute covariance (Hessian inverse)
            self.update_covariance(X, Y)

            # determine whether algorithm has converged
            if passes >= patience:
                converged = True

            # update evidence
            previdence = np.copy(self.evidence)
            self.itr += 1

    def callback(self, xk, res=None):
        print("Loss: {:.3f}, Residuals: {:.3f}".format(self.loss, self.residuals))
        return True

    # function to compute NLL loss function
    # @partial(jit, static_argnums=(0,))
    def compute_NLL(self, params, X, Y, Beta):
        outputs = self.forward_batch(params, X)
        error = jnp.nan_to_num(outputs - Y)
        self.residuals = jnp.sum(error)/X.shape[0]
        return jnp.einsum('nk,kl,nl->', error, Beta, error) / 2.

    # define objective function
    def objective(self, params, X, Y):
        # init loss with parameter penalty
        self.loss = jnp.dot(self.alpha * params, params) / 2.

        # forward pass
        self.loss += self.compute_NLL(params, X, Y, self.Beta)

        return self.loss

    # define function to compute gradient of loss w.r.t. parameters
    def jacobian(self, params, X, Y):

        # gradient of -log prior
        g = self.alpha * params

        # gradient of -log likelihood
        g += self.compute_grad_NLL(params, X, Y, self.Beta)

        # return gradient of -log posterior
        return g

    # define function to compute approximate Hessian
    def hessian(self, params, X, Y):
        # init w/ hessian of -log(prior)
        A = jnp.diag(self.alpha)

        # outer product approximation of Hessian:

        # Compute gradient of model output w.r.t. parameters
        G = self.G(params, X)

        # update Hessian
        A += A_next(G, self.Beta)

        return (A + A.T)/2.

    # update hyper-parameters alpha and Beta
    def init_hypers(self, X, Y, alpha_0):
        # compute number of independent samples in the data
        self.N = np.sum(~np.isnan(Y), 0)

        # init alpha
        self.alpha = alpha_0 * jnp.ones_like(self.params)

        # update Beta
        self.Beta = jnp.eye(self.n_outputs)
        self.BetaInv = jnp.eye(self.n_outputs)

    # update hyper-parameters alpha and Beta
    def update_hypers(self, X, Y):

        # forward
        outputs = self.forward_batch(self.params, X)
        error = jnp.nan_to_num(outputs - Y)

        # backward
        G = self.G(self.params, X)

        # sum of measurement covariance update
        yCOV = np.sum(error**2, 0) + trace_GGM(G, self.Ainv)
        
        # update alpha
        # self.alpha = 1. / (self.params ** 2 + jnp.diag(self.Ainv) + 2. * self.a)
        alpha = self.n_params / (jnp.sum(self.params**2) + jnp.trace(self.Ainv) + 2.*self.a)
        self.alpha = alpha*jnp.ones_like(self.params)

        # divide by number of observations
        yCOV = yCOV / self.N

        # update beta
        self.Beta = jnp.diag(1./(yCOV + self.b))
        self.BetaInv = jnp.diag(yCOV)

    # compute precision matrix
    def update_precision(self, X, Y):

        # compute inverse precision (covariance Matrix)
        A = np.diag(self.alpha)

        # update A
        G = self.G(self.params, X)
        A += A_next(G, self.Beta)

        # make sure that matrices are symmetric and positive definite
        self.A, _ = make_pos_def((A + A.T)/2., self.alpha)

    # compute covariance matrix
    def update_covariance(self, X, Y):

        ### fast / approximate method: ###
        # self.Ainv, _ = make_pos_def(compute_Ainv(self.A), jnp.ones(self.n_params))

        # compute inverse precision (covariance Matrix)
        self.Ainv = np.diag(1./self.alpha)

        # update Ainv
        G = self.G(self.params, X)
        for Gi in G:
            self.Ainv -= Ainv_next(Gi, self.Ainv, self.BetaInv) 

        # make sure Ainv is positive definite
        self.Ainv, _ = make_pos_def((self.Ainv + self.Ainv.T)/2., jnp.ones_like(self.alpha))

    # compute the log marginal likelihood
    def update_evidence(self):
        # compute evidence
        self.evidence = 1 / 2 * np.sum(self.N*np.log(np.diag(self.Beta))) + \
                        1 / 2 * np.nansum(np.log(self.alpha)) - \
                        1 / 2 * log_det(self.A) - self.loss

    # function to predict mean of outcomes
    def predict_point(self, X):
        # make point predictions
        preds = self.forward_batch(self.params, X)

        return preds
        
    # function to predict mean of outcomes
    def predict_point_params(self, X, params):
        # make point predictions
        preds = self.forward_batch(params, X)

        return preds

    # function to predict mean and stdv of outcomes
    def predict(self, X):

        # function to get diagonal of a tensor
        get_diag = vmap(jnp.diag, (0,))

        # point estimates
        preds = np.array(self.predict_point(X))

        # compute sensitivities
        G = self.G(self.params, X)

        # compute covariances
        COV = np.array(compute_predCOV(self.BetaInv, G, self.Ainv))

        # pull out standard deviations
        stdvs = np.sqrt(get_diag(COV))

        return preds, stdvs

    # function to predict mean and stdv of outcomes given updated covariance
    def conditional_predict(self, X, Ainv):

        # function to get diagonal of a tensor
        get_diag = vmap(jnp.diag, (0,))

        # point estimates
        preds = self.predict_point(X)

        # compute sensitivities
        G = self.G(self.params, X)

        # compute covariances
        COV = compute_epistemic_COV(G, Ainv)

        # pull out standard deviations
        stdvs = jnp.sqrt(get_diag(COV))

        return preds, stdvs
    
    # function to predict variance at X given precision A
    def conditioned_stdv(self, X, Ainv):

        # compute sensitivities
        G = self.G(self.params, X)

        # compute updated *epistemic* prediction covariance
        COV = np.einsum("nki,ij,nlj->nkl", G, Ainv, G) + self.BetaInv

        # pull out standard deviations
        get_diag = vmap(jnp.diag, (0,))
        stdvs = np.sqrt(get_diag(COV))

        return stdvs    

    # return indeces of optimal samples
    def search_EI(self, data, objective, N, max_reps=1, batch_size=512, exploit=True):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # make predictions once
        all_preds  = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
            # make predictions on data
            all_preds.append(self.forward_batch(self.params, data[batch_inds]))
        
        # compute objective (f: R^[n_t, n_o, w_exp] -> R) in batches
        objective_batch = jit(vmap(lambda pred, stdv: objective(pred, stdv), (0,0)))

        # initialize conditioned parameter covariance
        Ainv_q = jnp.copy(self.Ainv)

        # search for new experiments until find N
        best_experiments = []
        while len(best_experiments) < N:

            # compute utilities in batches to avoid memory problems
            utilities = []
            for preds, batch_inds in zip(all_preds, np.array_split(np.arange(n_samples), n_samples//batch_size)):
                stdvs = self.conditioned_stdv(data[batch_inds], Ainv_q)
                if exploit:
                    utilities.append(objective_batch(preds, stdvs))
                else:
                    utilities.append(stdvs)
            utilities = jnp.concatenate(utilities)
            # print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])
            
            # pick an experiment
            # print(f"Picked experiment {len(best_experiments)} out of {N}")
            exp = np.argmax(utilities)

            # condition posterior on selected sample
            Gi = self.Gi(self.params, data[exp])
            Ainv_q -= Ainv_next(Gi, Ainv_q, self.BetaInv)

            # switch to pure exploration if same samples keep getting picked
            if sum(np.isin(best_experiments, exp)) < max_reps:
                # add experiment to the list
                best_experiments += [exp.item()]
            else:
                if exploit:
                    print("Max replicates exceeded, switching to pure exploration")
                    exploit=False
                else:
                    print("Max exploration replicates exceeded, terminating")
                    return np.sort(best_experiments)

        return np.sort(best_experiments)

    # search for next best experiment
    def get_next_experiment(self, f_P, f_I, best_experiments, explore, max_explore):

        # init with previous selected experiment
        next_experiment = best_experiments[-1]
        w = np.copy(explore)
        while next_experiment in best_experiments and w < max_explore:

            # evaluate utility of each experimental condition
            utilities = f_P + w * f_I

            # select next best condition
            next_experiment = np.argmax(utilities).item()

            # increase exploration rate
            w *= 1.1  # = w + explore

        return next_experiment, w

    # return indeces of optimal samples
    def search_EIG(self, data, objective, N, explore=.001, max_explore=1000, max_reps=1):

        # compute objective (f: R^n_out -> R) in batches
        objective_batch = jit(vmap(objective))

        # make predictions once
        f_P = objective_batch(self.forward_batch(self.params, data)).ravel()

        # init experiments with max predicted objective
        best_experiments = [np.argmax(f_P).item()]

        # initialize conditioned parameter covariance
        Ainv_q = jnp.copy(self.Ainv)

        # compute sensitivities for all samples
        G = self.G(self.params, data)

        # search for new experiments until find N
        while len(best_experiments) < N:

            # condition posterior on selected sample
            Ainv_q -= Ainv_next(G[best_experiments[-1]], Ainv_q, self.BetaInv)

            # compute covariances
            COV = compute_predCOV(self.BetaInv, G, Ainv_q)

            # computed EIG for each condition
            f_I = batch_log_det(COV)

            # get next experiment
            exp, w = self.get_next_experiment(f_P, f_I, best_experiments, explore, max_explore)
            print("Explore rate: ", w)

            # switch to pure exploration if same samples keep getting picked
            if sum(np.in1d(best_experiments, exp)) < max_reps:
                # add experiment to the list
                best_experiments += [exp]
            else:
                print("Max exploration replicates exceeded, terminating")
                return np.sort(best_experiments)

        return np.sort(best_experiments)

    # return indeces of optimal samples
    def search_explore(self, data, N, max_reps=1):

        # initialize conditioned parameter covariance
        Ainv_q = jnp.copy(self.Ainv)

        # compute sensitivities for all samples
        G = self.G(self.params, data)

        # search for new experiments until find N
        best_experiments = []
        while len(best_experiments) < N:

            # condition posterior on selected sample
            if len(best_experiments) > 0:
                Ainv_q -= Ainv_next(G[best_experiments[-1]], Ainv_q, self.BetaInv)

            # compute covariances
            COV = np.array(compute_predCOV(self.BetaInv, G, Ainv_q))

            # computed EIG for each condition
            f_I = batch_log_det(COV)

            # get next experiment
            exp = np.argmax(f_I).item()

            # switch to pure exploration if same samples keep getting picked
            if sum(np.in1d(best_experiments, exp)) < max_reps:
                # add experiment to the list
                best_experiments += [exp]
            else:
                print("Max exploration replicates exceeded, terminating")
                return np.sort(best_experiments)

        return np.sort(best_experiments)
    
    # return indeces of optimal samples
    def search_Thompson(self, data, objective, N, max_reps=1):

        # compute objective (f: R^n_out -> R) in batches
        objective_batch = jit(vmap(objective))

        # make predictions once
        pred_mean, pred_stdv = self.predict(data)

        # evaluate best guess of objectives
        f_P = objective_batch(pred_mean).ravel()

        # init experiments with max predicted objective
        best_experiments = [np.argmax(f_P).item()]

        # search for new experiments until find N
        while len(best_experiments) < N:

            # sample from posterior predictive
            pred_sample = pred_mean + np.random.randn(*pred_stdv.shape) * pred_stdv

            # evaluate objectives
            f_P = objective_batch(pred_sample).ravel()

            # best sample
            exp = np.argmax(f_P).item()

            # switch to pure exploration if same samples keep getting picked
            if sum(np.in1d(best_experiments, exp)) < max_reps:
                # add experiment to the list
                best_experiments += [exp]
            else:
                print("Max exploration replicates exceeded, terminating")
                return np.sort(best_experiments)

        return np.sort(best_experiments)
