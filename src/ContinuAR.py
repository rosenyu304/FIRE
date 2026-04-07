"""
ContinuAR: Continuous Autoregressive Gaussian Process

Implements the ContinuAR model based on Linear Fidelity Differential Equations:
    dy(x,t)/dt + beta * y(x,t) = u(x,t)

This allows continuous fidelity modeling with ODE-based kernel structure.
"""

import torch
import numpy as np
import gpytorch
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import Positive


class ContinuARKernel(Kernel):
    """
    Kernel for ContinuAR based on Linear Fidelity Differential Equations.

    The process is defined by:
        dy(x,t)/dt + beta * y(x,t) = u(x,t)

    Covariance Structure:
        k((x,t), (x',t')) = Term1 + Term2

        Term1 (Decay of initial fidelity):
            exp(-beta * (t + t' - 2*t0)) * k_0(x, x')

        Term2 (Accumulated uncertainty):
            k_u(x, x') * (1 / (2*beta)) * (exp(-beta * |t - t'|) - exp(-beta * (t + t' - 2*t0)))

    Args:
        kernel_0: Kernel for the lowest fidelity y(x, t0)
        kernel_u: Kernel for the driving source term u(x, t)
        t_min: Minimum fidelity value (anchor point)
    """

    def __init__(self, kernel_0, kernel_u, t_min=0.0):
        super().__init__()
        self.kernel_0 = kernel_0
        self.kernel_u = kernel_u
        self.t_min = t_min

        # Beta parameter (must be positive for stability)
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.tensor(0.5)))
        self.register_constraint("raw_beta", Positive())

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    def forward(self, x1, x2, diag=False, **params):
        # Extract features and fidelity (last column is fidelity 't')
        x1_vals, x1_t = x1[..., :-1], x1[..., -1]
        x2_vals, x2_t = x2[..., :-1], x2[..., -1]

        # Compute spatial kernels
        k0_x = self.kernel_0(x1_vals, x2_vals, diag=diag, **params)
        ku_x = self.kernel_u(x1_vals, x2_vals, diag=diag, **params)

        # Relative to t_min
        t1 = x1_t - self.t_min
        t2 = x2_t - self.t_min

        beta = self.beta

        if diag:
            # Diagonal case: t1 == t2
            decay_term = torch.exp(-2 * beta * t1)
            integral_term = (1.0 / (2 * beta)) * (1.0 - torch.exp(-2 * beta * t1))
            covar = decay_term * k0_x + integral_term * ku_x
        else:
            t1 = t1.unsqueeze(-1)  # (N, 1)
            t2 = t2.unsqueeze(-2)  # (1, M)

            # Term 1: Decay
            decay_factor = torch.exp(-beta * (t1 + t2))
            term1 = decay_factor * k0_x

            # Term 2: Integral (OU-like structure)
            diff_t = torch.abs(t1 - t2)
            sum_t = t1 + t2
            time_cov = (1.0 / (2 * beta)) * (torch.exp(-beta * diff_t) - torch.exp(-beta * sum_t))
            term2 = time_cov * ku_x

            covar = term1 + term2

        return covar


class SimpleGP(gpytorch.models.ExactGP):
    """Simple GP wrapper for custom kernel."""
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ContinuAR:
    """
    Continuous Autoregressive GP for multi-fidelity regression.

    Uses continuous fidelity values treated as time in an ODE framework.
    Supports arbitrary number of fidelity levels through the fidelity column.
    """

    def __init__(self, train_X, train_Y, fidelity_col_idx=-1, device='cpu', seed=42, train_iter=200):
        """
        Args:
            train_X: Training inputs with fidelity column (N, D+1)
            train_Y: Training targets (N,)
            fidelity_col_idx: Index of fidelity column (default: -1, last column)
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

        # Prepare data
        if not torch.is_tensor(train_X):
            train_X = torch.tensor(train_X)
        if not torch.is_tensor(train_Y):
            train_Y = torch.tensor(train_Y)

        self.train_X = train_X.to(torch.double).to(self.device)
        self.train_Y = train_Y.to(torch.double).to(self.device)
        self.fid_col = fidelity_col_idx

        # Identify fidelity stats
        fidelities = self.train_X[..., self.fid_col]
        self.t_min = fidelities.min().item()
        self.t_max = fidelities.max().item()

        # Model initialization
        self.likelihood = GaussianLikelihood().to(self.device)

        k_0 = ScaleKernel(RBFKernel())
        k_u = ScaleKernel(RBFKernel())

        self.kernel = ContinuARKernel(kernel_0=k_0, kernel_u=k_u, t_min=self.t_min).to(self.device)
        self.model = SimpleGP(self.train_X, self.train_Y, self.likelihood, self.kernel).to(self.device).to(torch.double)

        # Training
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(self.train_iter):
            optimizer.zero_grad()
            output = self.model(self.train_X)
            loss = -mll(output, self.train_Y)
            loss.backward()
            optimizer.step()

    def predict(self, test_X):
        """
        Predict at the highest fidelity.

        Args:
            test_X: Test inputs WITHOUT fidelity column (N_test, D)

        Returns:
            mean: Predicted mean (N_test,)
            var: Predicted variance (N_test,)
        """
        self.model.eval()
        self.likelihood.eval()

        if not torch.is_tensor(test_X):
            test_X = torch.tensor(test_X)
        test_X = test_X.to(self.device).to(torch.double)

        # Append max fidelity value
        fid_column = torch.full((test_X.shape[0], 1), self.t_max, device=self.device)
        test_X_aug = torch.cat([test_X, fid_column], dim=1)

        with torch.no_grad():
            posterior = self.model(test_X_aug)
            pred_dist = self.likelihood(posterior)
            mean = pred_dist.mean
            var = pred_dist.variance

        return mean, var
