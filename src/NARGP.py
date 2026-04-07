"""
NARGP: Non-linear Autoregressive Gaussian Process

Implements the NARGP model where:
    y_high(x) ~ GP(mu(x, f_low(x)), k(x, f_low(x), x', f_low(x')))

The kernel structure allows non-linear transformation of low-fidelity predictions.
"""

import torch
import numpy as np
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class BaseGP(gpytorch.models.ExactGP):
    """Standard GP for the lowest fidelity."""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NARGPModel(gpytorch.models.ExactGP):
    """
    NARGP Model with composite kernel structure.

    Kernel: k_rho(x, x') * k_f(f, f') + k_delta(x, x')

    where:
    - k_rho acts on raw inputs (correlation between fidelities)
    - k_f acts on low-fidelity predictions (non-linear transformation)
    - k_delta acts on raw inputs (bias correction)
    """

    def __init__(self, train_x, train_y, likelihood, input_dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # k_rho(x, x') acting on raw inputs
        self.k_rho = ScaleKernel(RBFKernel(active_dims=list(range(input_dim))))

        # k_f(f, f') acting on the low-fidelity prediction (last dim)
        self.k_f = ScaleKernel(RBFKernel(active_dims=[input_dim]))

        # k_delta(x, x') acting on raw inputs (bias correction)
        self.k_delta = ScaleKernel(RBFKernel(active_dims=list(range(input_dim))))

        # Composite kernel: k_f * k_rho + k_delta
        self.covar_module = self.k_f * self.k_rho + self.k_delta

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NARGP:
    """
    Non-linear Autoregressive Gaussian Process for 2-fidelity regression.

    Two-stage approach:
    1. Train standard GP on low-fidelity data
    2. Train NARGP on high-fidelity data with augmented inputs [X, f_low(X)]
    """

    def __init__(self, train_X_low, train_y_low, train_X_high, train_y_high,
                 device='cuda:0', seed=42, train_iter=200):
        """
        Args:
            train_X_low: Low-fidelity training inputs (N_low, D)
            train_y_low: Low-fidelity training targets (N_low,)
            train_X_high: High-fidelity training inputs (N_high, D)
            train_y_high: High-fidelity training targets (N_high,)
            device: Device for computation
            seed: Random seed
            train_iter: Number of training iterations
        """
        self.seed = seed
        self.device = device
        self.train_iter = train_iter

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        train_X_low = train_X_low.to(self.device).to(torch.double)
        train_y_low = train_y_low.to(self.device).to(torch.double)
        train_X_high = train_X_high.to(self.device).to(torch.double)
        train_y_high = train_y_high.to(self.device).to(torch.double)

        # Normalize targets
        self.y_low_mean = train_y_low.mean()
        self.y_low_std = train_y_low.std() + 1e-6
        train_y_low_n = (train_y_low - self.y_low_mean) / self.y_low_std

        self.y_high_mean = train_y_high.mean()
        self.y_high_std = train_y_high.std() + 1e-6
        train_y_high_n = (train_y_high - self.y_high_mean) / self.y_high_std

        # 1. Train Low Fidelity Model
        self.likelihood_low = GaussianLikelihood().to(self.device)
        self.model_low = BaseGP(train_X_low, train_y_low_n, self.likelihood_low).to(self.device)
        self._train_model(self.model_low, self.likelihood_low, train_X_low, train_y_low_n)

        # 2. Augment High Fidelity Inputs with LF predictions
        self.model_low.eval()
        with torch.no_grad():
            low_pred_at_high = self.model_low(train_X_high).mean.unsqueeze(-1)

        train_X_aug = torch.cat([train_X_high, low_pred_at_high], dim=1)

        # 3. Train High Fidelity NARGP Model
        self.likelihood_high = GaussianLikelihood().to(self.device)
        self.model_high = NARGPModel(train_X_aug, train_y_high_n,
                                     self.likelihood_high,
                                     input_dim=train_X_high.shape[1]).to(self.device)
        self._train_model(self.model_high, self.likelihood_high, train_X_aug, train_y_high_n)

    def _train_model(self, model, likelihood, x, y):
        """Train a GP model."""
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self.train_iter):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

    def predict(self, test_X):
        """
        Predict at high fidelity for test inputs.

        Args:
            test_X: Test inputs (N_test, D)

        Returns:
            mean: Predicted mean (N_test,)
            var: Predicted variance (N_test,)
        """
        test_X = test_X.to(self.device).to(torch.double)

        self.model_low.eval()
        self.model_high.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            low_post = self.likelihood_low(self.model_low(test_X))
            test_X_aug = torch.cat([test_X, low_post.mean.unsqueeze(-1)], dim=1)

            high_post = self.likelihood_high(self.model_high(test_X_aug))
            final_mean = high_post.mean
            final_var = high_post.variance

        # Unnormalize
        final_mean = final_mean * self.y_high_std + self.y_high_mean
        final_var = final_var * (self.y_high_std ** 2)

        return final_mean, final_var

    def predict_mc(self, test_X, nsamples=100):
        """
        Monte Carlo prediction accounting for LF uncertainty.

        Args:
            test_X: Test inputs (N_test, D)
            nsamples: Number of MC samples

        Returns:
            mean: Predicted mean (N_test,)
            var: Predicted variance (N_test,)
        """
        self.model_low.eval()
        self.model_high.eval()
        test_X = test_X.to(self.device).to(torch.double)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            low_post = self.likelihood_low(self.model_low(test_X))

            # Sample low outputs
            Z = low_post.rsample(torch.Size([nsamples]))  # [S, N]

            # Build augmented inputs for all samples
            x_rep = test_X.unsqueeze(0).expand(nsamples, *test_X.shape)
            x_aug = torch.cat([x_rep, Z.unsqueeze(-1)], dim=-1).reshape(-1, test_X.shape[1] + 1)

            high_post = self.likelihood_high(self.model_high(x_aug))
            mu = high_post.mean.reshape(nsamples, -1)
            v = high_post.variance.reshape(nsamples, -1)

            mean = mu.mean(dim=0)
            var = v.mean(dim=0) + mu.var(dim=0, unbiased=False)

        # Unnormalize
        mean = mean * self.y_high_std + self.y_high_mean
        var = var * (self.y_high_std ** 2)

        return mean, var
