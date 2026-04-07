"""
AR(1): Autoregressive Multi-fidelity Gaussian Process

Implements the classic AR(1) model for multi-fidelity regression where:
    y_high(x) = rho * y_low(x) + delta(x) + epsilon
"""

import torch
import numpy as np
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


class LinearTransferMean(gpytorch.means.Mean):
    """
    Mean function for AR(1): mu_high(x) = rho * mu_low(x)
    'rho' is a learnable scalar parameter.
    """
    def __init__(self, input_size=1):
        super().__init__()
        self.rho = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.double))

    def forward(self, input):
        # Last column is the mean prediction from the previous fidelity
        previous_mean = input[..., -1]
        return self.rho * previous_mean


class BaseGP(gpytorch.models.ExactGP):
    """Standard GP for the lowest fidelity."""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ResidualGP(gpytorch.models.ExactGP):
    """
    GP for higher fidelities with AR(1) mean structure.
    Expects input with previous fidelity's mean appended as last column.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = LinearTransferMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        features = x[..., :-1]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class AR1:
    """
    AR(1) Multi-fidelity Gaussian Process.

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
            train_iter: Number of training iterations per GP
        """
        self.device = device
        self.train_iter = train_iter

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        # Prepare data
        train_X = torch.as_tensor(train_X, device=device).double()
        train_Y = torch.as_tensor(train_Y, device=device).double()

        # Separate features and fidelity
        self.fid_col = fidelity_col_idx
        if self.fid_col == -1:
            features = train_X[..., :-1]
        else:
            features = torch.cat([train_X[..., :self.fid_col],
                                  train_X[..., self.fid_col+1:]], dim=1)
        fidelities = train_X[..., self.fid_col]

        unique_fids = torch.sort(fidelities.unique())[0]
        self.num_fidelities = len(unique_fids)
        self.models = []
        self.likelihoods = []

        # Sequential training loop
        for i, fid_val in enumerate(unique_fids):
            mask = (fidelities == fid_val)
            X_curr = features[mask]
            y_curr = train_Y[mask]

            lik = GaussianLikelihood().to(device).double()

            if i == 0:
                # Lowest fidelity: Standard GP
                model = BaseGP(X_curr, y_curr, lik).to(device).double()
                self._train_gp(model, lik, X_curr, y_curr)
            else:
                # Higher fidelities: Residual GP with AR(1) mean
                prev_model = self.models[i-1]
                prev_model.eval()
                with torch.no_grad():
                    prev_mean = self._recursive_predict_mean(i-1, X_curr)
                    prev_mean = prev_mean.unsqueeze(-1)

                X_curr_aug = torch.cat([X_curr, prev_mean], dim=1)
                model = ResidualGP(X_curr_aug, y_curr, lik).to(device).double()
                self._train_gp(model, lik, X_curr_aug, y_curr)

            self.models.append(model)
            self.likelihoods.append(lik)

    def _train_gp(self, model, likelihood, X, y):
        """Train a single GP model."""
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self.train_iter):
            optimizer.zero_grad()
            output = model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

    def _recursive_predict_mean(self, level_idx, X):
        """Recursively predict mean up to level_idx."""
        if level_idx == 0:
            return self.models[0](X).mean

        prev_mean = self._recursive_predict_mean(level_idx - 1, X)
        X_aug = torch.cat([X, prev_mean.unsqueeze(-1)], dim=1)
        return self.models[level_idx](X_aug).mean

    def predict(self, test_X):
        """
        Predict at the highest fidelity.

        Args:
            test_X: Test inputs WITHOUT fidelity column (N_test, D)

        Returns:
            mean: Predicted mean (N_test,)
            var: Predicted variance (N_test,)
        """
        test_X = torch.as_tensor(test_X, device=self.device).double()

        final_level = self.num_fidelities - 1

        if final_level == 0:
            self.models[0].eval()
            self.likelihoods[0].eval()
            with torch.no_grad():
                posterior = self.models[0](test_X)
                pred_dist = self.likelihoods[0](posterior)
                return pred_dist.mean, pred_dist.variance

        # Multi-fidelity case
        prev_mean = self._recursive_predict_mean(final_level - 1, test_X)
        test_X_aug = torch.cat([test_X, prev_mean.unsqueeze(-1)], dim=1)

        model = self.models[final_level]
        lik = self.likelihoods[final_level]

        model.eval()
        lik.eval()
        with torch.no_grad():
            posterior = model(test_X_aug)
            pred_dist = lik(posterior)

        return pred_dist.mean, pred_dist.variance
