"""
MFKG: Multi-fidelity Kriging using BoTorch

Uses BoTorch's SingleTaskMultiFidelityGP with non-linear truncated kernel.
"""

import torch
import numpy as np
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


class MFKG:
    """
    Multi-fidelity Kriging using BoTorch.

    Uses SingleTaskMultiFidelityGP with non-linear (truncated=False) fidelity kernel.
    """

    def __init__(self, train_X_low, train_y_low, train_X_high, train_y_high,
                 device='cuda:0', seed=42):
        """
        Args:
            train_X_low: Low-fidelity training inputs (N_low, D)
            train_y_low: Low-fidelity training targets (N_low,)
            train_X_high: High-fidelity training inputs (N_high, D)
            train_y_high: High-fidelity training targets (N_high,)
            device: Device for computation
            seed: Random seed
        """
        self.device = torch.device(device) if isinstance(device, str) else device

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        # Add fidelity column (0 for low, 1 for high)
        train_X_low = torch.concatenate([train_X_low, torch.zeros(train_X_low.shape[0], 1)], dim=1)
        train_X_high = torch.concatenate([train_X_high, torch.ones(train_X_high.shape[0], 1)], dim=1)

        train_X = torch.cat([train_X_low, train_X_high], dim=0).to(torch.double).to(self.device)
        train_Y = torch.cat([train_y_low, train_y_high], dim=0).to(torch.double).to(self.device)

        if len(train_Y.shape) < 2:
            train_Y = train_Y.unsqueeze(-1)

        # Model initialization with non-linear fidelity kernel
        self.model = SingleTaskMultiFidelityGP(
            train_X=train_X,
            train_Y=train_Y,
            data_fidelities=[train_X_high.shape[1]-1],  # Fidelity column index
            linear_truncated=False,  # Non-linear fidelity kernel
            outcome_transform=Standardize(m=1)
        )

        # Training
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def predict(self, test_X_high):
        """
        Predict at high fidelity for test inputs.

        Args:
            test_X_high: Test inputs (N_test, D)

        Returns:
            mean: Predicted mean (N_test,)
            var: Predicted variance (N_test,)
        """
        # Add fidelity column (1 for high fidelity)
        test_X_high = torch.concatenate([test_X_high, torch.ones(test_X_high.shape[0], 1)], dim=1)
        test_X_high = test_X_high.to(torch.double).to(self.device)

        with torch.no_grad():
            posterior = self.model.posterior(test_X_high)
            mean = posterior.mean
            std = posterior.stddev
            var = std ** 2

        return mean.squeeze(-1), var.squeeze(-1)
