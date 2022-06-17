import numpy as np
import jax.numpy as jnp
from jax import nn, jacfwd, jit, vmap, lax, random
from functools import partial
import time

# import MCMC library
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC

class FFNN():

    def __init__(self, n_inputs, n_hidden, n_outputs, param_0=.2):

        # store dimensions
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # bounds on initial parameter guess
        self.param_0 = param_0

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
        for k1,k2,shape in zip(self.k_params[:-1], self.k_params[1:-1], self.shapes[:-1]):
            self.params[k1:k2] = np.random.uniform(-self.param_0, self.param_0, k2-k1)

        # initialize hyper-parameters
        self.a = 1e-4
        self.b = 1e-4

        # initialize covariance
        self.Ainv = None

        ### define jit compiled functions ###

        # batch prediction
        self.forward_batch = jit(vmap(self.forward, in_axes=(None, 0)))

        # jit compile gradient w.r.t. params
        self.G  = jit(jacfwd(self.forward_batch))
        self.Gi = jit(jacfwd(self.forward))

        # jit compile Newton update direction computation
        def NewtonStep(G, g, alpha, Beta):
            # compute hessian
            A = jnp.diag(alpha) + jnp.einsum('nki,kl,nlj->ij', G, Beta, G)
            # solve for Newton step direction
            d = jnp.linalg.solve(A, g)
            return d
        self.NewtonStep = jit(NewtonStep)

        # jit compile inverse Hessian computation step
        def Ainv_next(G, Ainv, BetaInv):
            GAinv = G@Ainv
            Ainv_step = GAinv.T@jnp.linalg.inv(BetaInv + GAinv@G.T)@GAinv
            Ainv_step = (Ainv_step + Ainv_step.T)/2.
            return Ainv_step
        self.Ainv_next = Ainv_next

        # jit compile measurement covariance computation
        def compute_yCOV(errors, G, Ainv):
            return jnp.einsum('nk,nl->kl', errors, errors) + jnp.einsum('nki,ij,nlj->kl', G, Ainv, G)
        self.compute_yCOV = jit(compute_yCOV)

    # reshape parameters into weight matrices and bias vectors
    def reshape(self, params):
        # params is a vector = [Wih, bih, Who, bho]
        return [np.reshape(params[k1:k2], shape) for k1,k2,shape in zip(self.k_params, self.k_params[1:], self.shapes)]

    # per-sample prediction
    def forward(self, params, sample):
        # reshape params
        Wih, bih, Who, bho = self.reshape(params)

        # hidden layer
        h = nn.tanh(Wih@sample + bih)

        # output
        out = Who@h + bho

        return out

    # fit to data
    def fit(self, X, Y, lr=1e-2, map_tol=1e-3, evd_tol=1e-3):
        # fit until convergence of evidence
        previdence = -np.inf
        evidence_converged = False
        epoch = 0
        best_evidence_params = np.copy(self.params)
        best_params = np.copy(self.params)

        while not evidence_converged:

            # update hyper-parameters
            self.update_hypers(X, Y)

            # use Newton descent to determine parameters
            prev_loss = np.inf

            # fit until convergence of NLP
            converged = False
            while not converged:
                # forward passs
                outputs = self.forward_batch(self.params, X)
                errors  = np.nan_to_num(outputs - Y)
                residuals = np.sum(errors)

                # compute convergence of loss function
                loss = self.compute_loss(errors)
                convergence = (prev_loss - loss) / max([1., loss])
                if epoch%10==0:
                    print("Epoch: {}, Loss: {:.5f}, Residuals: {:.5f}, Convergence: {:5f}".format(epoch, loss, residuals, convergence))

                # stop if less than tol
                if abs(convergence) <= map_tol:
                    # set converged to true to break from loop
                    converged = True
                else:
                    # lower learning rate if convergence is negative
                    if convergence < 0:
                        lr /= 2.
                        # re-try with the smaller step
                        self.params = best_params - lr*d
                    else:
                        # update best params
                        best_params = np.copy(self.params)

                        # update previous loss
                        prev_loss = loss

                        # compute gradients
                        G = self.G(self.params, X)
                        g = np.einsum('nk,kl,nli->i', errors, self.Beta, G) + self.alpha*self.params

                        # determine Newton update direction
                        d = self.NewtonStep(G, g, self.alpha, self.Beta)

                        # update parameters
                        self.params -= lr*d

                        # update epoch counter
                        epoch += 1

            # Update Hessian estimation
            G = self.G(self.params, X)
            self.A, self.Ainv = self.compute_precision(G)

            # compute evidence
            evidence = self.compute_evidence(X, loss)

            # determine whether evidence is converged
            evidence_convergence = (evidence - previdence) / max([1., abs(evidence)])
            print("\nEpoch: {}, Evidence: {:.5f}, Convergence: {:5f}".format(epoch, evidence, evidence_convergence))

            # stop if less than tol
            if abs(evidence_convergence) <= evd_tol:
                evidence_converged = True
            else:
                if evidence_convergence < 0:
                    # reset :(
                    self.params = np.copy(best_evidence_params)
                    # Update Hessian estimation
                    G = self.G(self.params, X)
                    self.A, self.Ainv = self.compute_precision(G)
                    # reset evidence back to what it was
                    evidence = previdence
                    # lower learning rate
                    lr /= 2.
                else:
                    # otherwise, update previous evidence value
                    previdence = evidence
                    # update measurement covariance
                    self.yCOV = self.compute_yCOV(errors, G, self.Ainv)
                    # update best evidence parameters
                    best_evidence_params = np.copy(self.params)

    # update hyper-parameters alpha and Beta
    def update_hypers(self, X, Y):
        if self.Ainv is None:
            self.yCOV = np.einsum('nk,nl->kl', np.nan_to_num(Y), np.nan_to_num(Y))
            self.yCOV = (self.yCOV + self.yCOV.T)/2.
            # update alpha
            self.alpha = np.ones(self.n_params)
            # update Beta
            self.Beta = X.shape[0]*np.linalg.inv(self.yCOV + 2.*self.b*np.eye(self.n_outputs))
            self.Beta = (self.Beta + self.Beta.T)/2.
            self.BetaInv = np.linalg.inv(self.Beta)
        else:
            # update alpha
            self.alpha = 1. / (self.params**2 + np.diag(self.Ainv) + 2.*self.a)
            # update beta
            self.Beta = X.shape[0]*np.linalg.inv(self.yCOV + 2.*self.b*np.eye(self.n_outputs))
            self.Beta = (self.Beta + self.Beta.T)/2.
            self.BetaInv = np.linalg.inv(self.Beta)

    # compute loss
    def compute_loss(self, errors):
        return 1/2*(np.einsum('nk,kl,nl->', errors, self.Beta, errors) + np.dot(self.alpha*self.params, self.params))

    # compute Precision and Covariance matrices
    def compute_precision(self, G):
        # compute Hessian (precision Matrix)
        A = jnp.diag(self.alpha) + jnp.einsum('nki,kl,nlj->ij', G, self.Beta, G)
        A = (A + A.T)/2.

        # compute inverse precision (covariance Matrix)
        Ainv = jnp.diag(1./self.alpha)
        for Gn in G:
            Ainv -= self.Ainv_next(Gn, Ainv, self.BetaInv)
        # Ainv = jnp.linalg.inv(A) # <-- faster but less accurate than above
        return A, Ainv

    # compute the log marginal likelihood
    def compute_evidence(self, X, loss):
        # compute evidence
        Hessian_eigs = np.linalg.eigvalsh(self.A)
        evidence = X.shape[0]/2*np.nansum(np.log(np.linalg.eigvalsh(self.Beta))) + \
                   1/2*np.nansum(np.log(self.alpha)) - \
                   1/2*np.nansum(np.log(Hessian_eigs[Hessian_eigs>0])) - loss
        return evidence

    def fit_MCMC(self, X, Y, num_warmup=1000, num_samples=4000, rng_key=0):

        # define probabilistic model
        def pyro_model():

            # sample from Laplace approximated posterior
            w = numpyro.sample("w",
                               dist.MultivariateNormal(loc=self.params,
                                                       covariance_matrix=self.Ainv))

            # sample from zero-mean Gaussian prior with independent precision priors
            '''with numpyro.plate("params", self.n_params):
                alpha = numpyro.sample("alpha", dist.Exponential(rate=1e-4))
                w = numpyro.sample("w", dist.Normal(loc=0., scale=(1./alpha)**.5))'''

            # sample from zero-mean Gaussian prior with single precision prior
            '''alpha = numpyro.sample("alpha", dist.Exponential(rate=1e-4))
            w = numpyro.sample("w", dist.MultivariateNormal(loc=np.zeros(self.n_params),
                                                            precision_matrix=alpha*np.eye(self.n_params)))'''

            # output of neural network:
            preds = self.forward_batch(w, X)

            # sample model likelihood with max evidence precision matrix
            numpyro.sample("Y",
                           dist.MultivariateNormal(loc=preds, precision_matrix=self.Beta),
                           obs = Y)

        # init MCMC object with NUTS kernel
        kernel = NUTS(pyro_model, step_size=1.)
        self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)

        # warmup
        self.mcmc.warmup(random.PRNGKey(rng_key), init_params=self.params)

        # run MCMC
        self.mcmc.run(random.PRNGKey(rng_key), init_params=self.params)

        # save posterior params
        self.posterior_params = np.array(self.mcmc.get_samples()['w'])

    # function to predict metabolites and variance
    def predict(self, X):
        # make point predictions
        preds = self.forward_batch(self.params, X)

        # compute sensitivities
        G = self.G(self.params, X)

        # compute covariances
        COV = self.BetaInv + np.einsum("nki,ij,nlj->nkl", G, self.Ainv, G)

        # pull out standard deviations
        get_diag = vmap(jnp.diag, (0,))
        stdvs = np.sqrt(get_diag(COV))

        return preds, stdvs, COV

    # function to predict from posterior samples
    def predict_MCMC(self, X):
        # make point predictions
        preds = jit(vmap(lambda params: self.forward_batch(params, X), (0,)))(self.posterior_params)

        # take mean and standard deviation
        stdvs = np.sqrt(np.diag(self.BetaInv) + np.var(preds, 0))
        preds = np.mean(preds, 0)

        return preds, stdvs

    # return indeces of optimal samples
    def search(self, data, objective, scaler, N,
               batch_size=512, explore = .5, max_explore = 1e3):

        # determine number of samples to search over
        n_samples = data.shape[0]
        batch_size = min([n_samples, batch_size])

        # compute objective (f: R^[n_t, n_o, w_exp] -> R) in batches
        objective_batch = jit(vmap(lambda pred, stdv, explore: objective(scaler.inverse_transform(pred),
                                                                         scaler.inverse_transform(stdv),
                                                                         explore), (0,0,None)))

        # initialize search with pure exploitation
        objectives = []
        all_preds  = []
        for batch_inds in np.array_split(np.arange(n_samples), n_samples//batch_size):
            # make predictions on data
            preds = self.predict_point(data[batch_inds])
            # evaluate objectives with zero uncertainty
            objectives.append(objective_batch(preds, 0.*preds, 1.))
            # save predictions so that they're only evaluated once
            all_preds.append(preds)
        objectives = jnp.concatenate(objectives)
        print("Top 5 utilities: ", jnp.sort(objectives)[::-1][:5])

        if explore <= 0.:
            print("Pure exploitation, returning N max objective experiments")
            return np.array(jnp.argsort(objectives)[::-1][:N])

        # initialize with sample that maximizes objective
        best_experiments = [np.argmax(objectives).item()]
        print(f"Picked experiment {len(best_experiments)} out of {N}")

        # initialize conditioned parameter covariance
        Ainv_q = jnp.copy(self.Ainv)
        Gi = self.Gi(self.params, data[np.argmax(objectives)])
        # update conditioned parameter covariance
        for Gt in Gi:
            Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)

        # search for new experiments until find N
        eval_utilities = True
        while len(best_experiments) < N:

            # compute utilities in batches to avoid memory problems
            utilities = []
            for preds, batch_inds in zip(all_preds, np.array_split(np.arange(n_samples), n_samples//batch_size)):
                stdvs = self.conditioned_stdv(data[batch_inds], Ainv_q)
                utilities.append(objective_batch(preds, stdvs, explore))
            utilities = jnp.concatenate(utilities)
            print("Top 5 utilities: ", jnp.sort(utilities)[::-1][:5])

            # sort utilities from best to worst
            exp_sorted = jnp.argsort(utilities)[::-1]
            for exp in exp_sorted:
                # accept if unique
                if exp not in best_experiments:
                    best_experiments += [exp.item()]
                    # compute sensitivity to sample
                    Gi = self.Gi(self.params, data[exp])
                    # update conditioned parameter covariance
                    for Gt in Gi:
                        Ainv_q -= self.Ainv_next(Gt, Ainv_q, self.BetaInv)
                    print(f"Picked experiment {len(best_experiments)} out of {N}")
                    if eval_utilities:
                        break
                    # IF already exceed max exploration
                    # AND have enough selected experiments, return
                    if len(best_experiments) == N:
                        return best_experiments

                # increase exploration if not unique
                elif explore < max_explore:
                    explore *= 2.
                    print("Increased exploration rate to {:.3f}".format(explore))
                    break
                else:
                    # if the same experiment was picked twice at the max
                    # exploration rate, do not re-evaluate utilities
                    eval_utilities = False

        return best_experiments
