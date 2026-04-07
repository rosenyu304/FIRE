"""
FIRE: Fidelity-aware In-context REgression

This module implements the FIRE algorithm for multi-fidelity regression using:
1. FIRE_TFM: Uses Tabular Foundation Models (TabPFN) for zero-shot inference
2. FIRE_GP: Uses Gaussian Processes with BoTorch/GPyTorch
"""

import torch
import numpy as np
from tabpfn import TabPFNRegressor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


class FIRE:
    """
    FIRE with Tabular Foundation Models (TabPFN).

    Two-stage approach:
    1. Train LF model on low-fidelity data with fidelity encoding
    2. Train residual model on high-fidelity data, conditioned on LF predictions
       (mean, variance, quantiles)
    """

    def __init__(self, train_X_low, train_y_low, train_X_high, train_y_high,
                 device='cuda:0', seed=42):
        """
        Args:
            train_X_low: Low-fidelity training inputs (N_low, D)
            train_y_low: Low-fidelity training targets (N_low,)
            train_X_high: High-fidelity training inputs (N_high, D)
            train_y_high: High-fidelity training targets (N_high,)
            device: Device for computation ('cuda:0' or 'cpu')
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.device = device

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        # Add fidelity column (0 for low, 1 for high)
        train_X_low = torch.concatenate([train_X_low, torch.zeros(train_X_low.shape[0], 1)], dim=1)
        train_X_high = torch.concatenate([train_X_high, torch.ones(train_X_high.shape[0], 1)], dim=1)

        # Stage 1: Train low-fidelity model
        self.lf_model = TabPFNRegressor(device=self.device, random_state=self.seed)
        self.lf_model.fit(train_X_low, train_y_low)

        # Get LF predictions at HF training points
        lf_pred = self.lf_model.predict(train_X_high)
        lf_pred_variance = self.lf_model.predict(train_X_high, output_type="variance")
        lf_pred_quantile = self.lf_model.predict(train_X_high, output_type="quantiles")
        lf_pred_quantile = [q.reshape(-1, 1) for q in lf_pred_quantile]
        lf_pred_quantile = np.hstack(lf_pred_quantile)

        # Convert to tensors
        if not isinstance(lf_pred, torch.Tensor):
            lf_pred = torch.tensor(lf_pred)
        if not isinstance(lf_pred_variance, torch.Tensor):
            lf_pred_variance = torch.tensor(lf_pred_variance)
        if not isinstance(lf_pred_quantile, torch.Tensor):
            lf_pred_quantile = torch.tensor(lf_pred_quantile)
        if len(lf_pred.shape) > 1:
            lf_pred = lf_pred.squeeze(-1)
        if len(train_y_high.shape) > 1:
            train_y_high = train_y_high.squeeze(-1)
        if len(lf_pred_variance.shape) > 1:
            lf_pred_variance = lf_pred_variance.squeeze(-1)

        # Compute residuals
        residuals = train_y_high - lf_pred

        # Stage 2: Train residual model with distribution-conditioned inputs
        self.residual_model = TabPFNRegressor(device=self.device, random_state=self.seed)
        self.residual_model.fit(
            torch.cat([train_X_high,
                      lf_pred.reshape(-1, 1),
                      lf_pred_variance.reshape(-1, 1),
                      lf_pred_quantile], dim=1),
            residuals
        )

    def predict(self, test_X_high):
        """
        Predict at high fidelity for test inputs.

        Args:
            test_X_high: Test inputs (N_test, D) - without fidelity column

        Returns:
            mean: Predicted mean (N_test,)
            var: Predicted variance (N_test,)
        """
        # Add fidelity column (1 for high fidelity)
        test_X_high = torch.concatenate([test_X_high, torch.ones(test_X_high.shape[0], 1)], dim=1)

        # Get LF predictions
        lf_pred = self.lf_model.predict(test_X_high)
        lf_pred_variance = self.lf_model.predict(test_X_high, output_type="variance")
        lf_pred_quantile = self.lf_model.predict(test_X_high, output_type="quantiles")
        lf_pred_quantile = [q.reshape(-1, 1) for q in lf_pred_quantile]
        lf_pred_quantile = np.hstack(lf_pred_quantile)

        # Convert to tensors
        if not isinstance(lf_pred, torch.Tensor):
            lf_pred = torch.tensor(lf_pred)
        if not isinstance(lf_pred_variance, torch.Tensor):
            lf_pred_variance = torch.tensor(lf_pred_variance)
        if not isinstance(lf_pred_quantile, torch.Tensor):
            lf_pred_quantile = torch.tensor(lf_pred_quantile)
        if len(lf_pred.shape) > 1:
            lf_pred = lf_pred.squeeze(-1)
        if len(lf_pred_variance.shape) > 1:
            lf_pred_variance = lf_pred_variance.squeeze(-1)

        # Augment test inputs with LF distribution features
        test_X_aug = torch.cat([test_X_high,
                                lf_pred.reshape(-1, 1),
                                lf_pred_variance.reshape(-1, 1),
                                lf_pred_quantile], dim=1)

        # Predict residuals
        test_residual_pred = self.residual_model.predict(test_X_aug)
        test_residual_var = self.residual_model.predict(test_X_aug, output_type="variance")

        # Combine: HF prediction = LF prediction + residual
        return lf_pred + test_residual_pred, lf_pred_variance + test_residual_var


class FIRE_GP:
    """
    FIRE with Gaussian Processes (BoTorch/GPyTorch).

    Two-stage approach using GP models instead of TabPFN.
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
            train_iter: Number of training iterations (not used, kept for API compatibility)
        """
        self.seed = seed
        self.device = torch.device(device) if isinstance(device, str) else device
        self.train_iter = train_iter

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        # Prepare data
        train_X_low = self._prep_tensor(train_X_low)
        train_y_low = self._prep_tensor(train_y_low).reshape(-1, 1)
        train_X_high = self._prep_tensor(train_X_high)
        train_y_high = self._prep_tensor(train_y_high).reshape(-1, 1)

        # Stage 1: Train Low-Fidelity Model
        self.lf_model = SingleTaskGP(train_X_low, train_y_low)
        self.lf_model.to(self.device)

        mll_lf = ExactMarginalLogLikelihood(self.lf_model.likelihood, self.lf_model)
        fit_gpytorch_mll(mll_lf)

        # Generate LF features for HF training
        self.lf_model.eval()
        with torch.no_grad():
            posterior = self.lf_model.posterior(train_X_high)
            lf_pred = posterior.mean
            lf_pred_var = posterior.variance

            # Generate quantiles (0.1, 0.9)
            q_list = torch.tensor([0.1, 0.9], device=self.device, dtype=torch.float64)
            lf_quantiles = posterior.quantile(q_list)
            lf_quantiles = lf_quantiles.squeeze(-1).t()

        # Compute residuals
        residuals = train_y_high - lf_pred

        # Stage 2: Train Residual Model
        train_X_aug = torch.cat([train_X_high, lf_pred, lf_pred_var, lf_quantiles], dim=1)

        self.residual_model = SingleTaskGP(train_X_aug, residuals)
        self.residual_model.to(self.device)

        mll_res = ExactMarginalLogLikelihood(self.residual_model.likelihood, self.residual_model)
        fit_gpytorch_mll(mll_res)

    def predict(self, test_X_high):
        """
        Predict at high fidelity for test inputs.

        Args:
            test_X_high: Test inputs (N_test, D)

        Returns:
            mean: Predicted mean (N_test,)
            var: Predicted variance (N_test,)
        """
        test_X_high = self._prep_tensor(test_X_high)

        # LF inference
        self.lf_model.eval()
        with torch.no_grad():
            posterior = self.lf_model.posterior(test_X_high)
            lf_pred = posterior.mean
            lf_pred_var = posterior.variance

            q_list = torch.tensor([0.1, 0.9], device=self.device, dtype=torch.float64)
            lf_quantiles = posterior.quantile(q_list).squeeze(-1).t()

        # Augment test inputs
        test_X_aug = torch.cat([test_X_high, lf_pred, lf_pred_var, lf_quantiles], dim=1)

        # Residual inference
        self.residual_model.eval()
        with torch.no_grad():
            res_posterior = self.residual_model.posterior(test_X_aug)
            test_res_pred = res_posterior.mean
            test_res_var = res_posterior.variance

        # Combine predictions
        final_mean = lf_pred + test_res_pred
        final_var = lf_pred_var + test_res_var

        return final_mean.squeeze(-1), final_var.squeeze(-1)

    def _prep_tensor(self, x):
        """Helper to ensure float64 and correct device."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.to(device=self.device, dtype=torch.float64)
