"""
ResGP: Residual Gaussian Process for Multi-fidelity Regression

Implements residual learning where:
    y_high(x) = y_low(x) + delta(x)

The residual delta(x) is modeled by a separate GP.
"""

import torch
import numpy as np
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class BaseGP(gpytorch.models.ExactGP):
    """Standard GP model."""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ResGP:
    """
    Residual Gaussian Process for 2-fidelity regression.

    Two-stage approach:
    1. Train GP on low-fidelity data
    2. Train GP on residuals (y_high - y_low) at high-fidelity points
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

        train_X_low = train_X_low.to(self.device)
        train_y_low = train_y_low.to(self.device)
        train_X_high = train_X_high.to(self.device)
        train_y_high = train_y_high.to(self.device)

        # Center target data
        y0 = train_y_low.mean()
        train_y_low_c = train_y_low - y0
        train_y_high_c = train_y_high - y0
        self.y0 = y0

        # 1. Train Low Fidelity Model
        self.likelihood_low = GaussianLikelihood().to(self.device)
        self.model_low = BaseGP(train_X_low, train_y_low_c, self.likelihood_low).to(self.device)
        self._train_model(self.model_low, self.likelihood_low, train_X_low, train_y_low_c)

        # 2. Calculate Residuals
        self.model_low.eval()
        with torch.no_grad():
            # Check for nested design (exact match)
            eq = (train_X_low.unsqueeze(1) == train_X_high.unsqueeze(0)).all(dim=2)
            matched = eq.any(dim=0)

            if matched.all():
                idx_low = eq.float().argmax(dim=0)
                y_low_at_high = train_y_low_c[idx_low]
                residuals = train_y_high_c - y_low_at_high
            else:
                # Fallback to posterior mean
                low_pred_at_high = self.model_low(train_X_high).mean
                residuals = train_y_high_c - low_pred_at_high

        # 3. Train Residual Model
        self.likelihood_res = GaussianLikelihood().to(self.device)
        self.model_res = BaseGP(train_X_high, residuals, self.likelihood_res).to(self.device)
        self._train_model(self.model_res, self.likelihood_res, train_X_high, residuals)

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
        self.model_low.eval()
        self.model_res.eval()

        test_X = test_X.to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            low_post = self.likelihood_low(self.model_low(test_X))
            res_post = self.likelihood_res(self.model_res(test_X))

            mean = low_post.mean + res_post.mean
            var = low_post.variance + res_post.variance

        return mean + self.y0, var
