"""
MFBNN: Multi-fidelity Bayesian Neural Network

Implements recursive multi-fidelity BNN using the mfbml library.

Note: Requires mfbml package (check out https://github.com/bessagroup/mfbml)
"""

import torch
import numpy as np


class SingleLevelBNN_Wrapper:
    """
    Wrapper for DNNLinearRegressionBNN from mfbml library.
    Makes it behave like a standard fit/predict model.
    """

    def __init__(self, input_dim, design_space, device='cpu'):
        from mfbml import DNNLinearRegressionBNN

        self.device = device

        self.lf_configure = {
            "in_features": input_dim,
            "hidden_features": [50, 50],
            "out_features": 1,
            "activation": "Tanh",
            "optimizer": "Adam",
            "lr": 0.001,
            "weight_decay": 0.000001,
            "loss": "mse",
        }
        self.hf_configure = {
            "in_features": input_dim,
            "hidden_features": [500, 500],
            "out_features": 1,
            "activation": "Tanh",
            "lr": 0.001,
            "sigma": 0.05,
        }
        self.lf_train_config = {"batch_size": None, "num_epochs": 2000, "print_iter": 500, "data_split": True}
        self.hf_train_config = {"num_epochs": 1500, "sample_freq": 100, "print_info": False, "burn_in_epochs": 500}

        self.model = DNNLinearRegressionBNN(
            design_space=design_space,
            lf_configure=self.lf_configure,
            hf_configure=self.hf_configure,
            beta_optimize=True,
            lf_order=1,
            beta_bounds=[-1, 1],
            optimizer_restart=10,
            discrepancy_normalization="diff",
        )

    def fit(self, X_low, y_low, X_high, y_high):
        """Train the BNN model."""
        # mfbml expects list format: [High, Low]
        samples = [X_high, X_low]
        responses = [y_high, y_low]
        self.model.train(X=samples, Y=responses,
                        lf_train_config=self.lf_train_config,
                        hf_train_config=self.hf_train_config)

    def predict(self, X):
        """Predict with the BNN model."""
        pred_hy, pred_epistemic, pred_total_unc, pred_aleatoric = self.model.predict(X=X)
        return pred_hy, pred_epistemic


class MFBNN:
    """
    Multi-fidelity Bayesian Neural Network.

    Recursive architecture where each level learns to correct the previous level's predictions.
    Supports arbitrary number of fidelity levels through the fidelity column.
    """

    def __init__(self, train_X, train_y, fidelity_col_idx=-1, device='cpu', seed=42):
        """
        Args:
            train_X: Training inputs with fidelity column (N, D+1)
            train_y: Training targets (N,)
            fidelity_col_idx: Index of fidelity column (default: -1, last column)
            device: Device for computation
            seed: Random seed
        """
        from mfbml import DNNLinearRegressionBNN

        self.device = device
        self.seed = seed
        self.fidelity_col_idx = fidelity_col_idx

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        # Prepare data
        if not torch.is_tensor(train_X):
            train_X = torch.tensor(train_X, dtype=torch.float32)
        if not torch.is_tensor(train_y):
            train_y = torch.tensor(train_y, dtype=torch.float32)

        if train_y.ndim > 1:
            train_y = train_y.squeeze()

        if train_X.ndim == 1:
            train_X = train_X.reshape(-1, 1)

        self.train_X = train_X.to(self.device).to(torch.float32)
        self.train_y = train_y.to(self.device).to(torch.float32)

        # Identify fidelity levels
        self.fidelities = torch.unique(self.train_X[:, fidelity_col_idx]).sort()[0].tolist()

        # Define design space (excluding fidelity column)
        if fidelity_col_idx == -1:
            features = self.train_X[:, :-1]
        else:
            cols = [c for c in range(self.train_X.shape[1])
                    if c != (self.train_X.shape[1] + fidelity_col_idx if fidelity_col_idx < 0 else fidelity_col_idx)]
            features = self.train_X[:, cols]

        min_vals = features.min(dim=0)[0]
        max_vals = features.max(dim=0)[0]
        self.design_space = torch.stack([min_vals, max_vals], dim=1).to(self.device)
        self.input_dim = features.shape[1]

        self.models = []
        self.data_cache = {}

        # Iterative training loop
        for i, fid in enumerate(self.fidelities):
            print(f"--- Training Level {i} (Fidelity {fid}) ---")

            # Extract data for this level
            mask = (self.train_X[:, fidelity_col_idx] == fid)
            X_curr_full = self.train_X[mask]
            y_curr = self.train_y[mask].reshape(-1, 1)

            # Remove fidelity column
            if fidelity_col_idx == -1:
                X_curr = X_curr_full[:, :-1]
            else:
                cols = [c for c in range(X_curr_full.shape[1])
                        if c != (X_curr_full.shape[1] + fidelity_col_idx if fidelity_col_idx < 0 else fidelity_col_idx)]
                X_curr = X_curr_full[:, cols]

            self.data_cache[fid] = (X_curr, y_curr)

            # Prepare inputs for BNN
            if i == 0:
                # Level 0: Base model
                X_low_input = X_curr
                y_low_input = y_curr
                X_high_input = X_curr
                y_high_input = y_curr
            else:
                # Level > 0: Use predictions from previous level
                prev_fid = self.fidelities[i-1]
                X_prev, y_prev = self.data_cache[prev_fid]

                # Check for exact matches (nested design)
                dist = (X_curr.unsqueeze(1) - X_prev.unsqueeze(0)).abs().sum(dim=2)
                is_match = (dist < 1e-6)
                has_match = is_match.any(dim=1)
                match_indices = is_match.float().argmax(dim=1)

                y_prev_at_curr = torch.zeros_like(y_curr, dtype=torch.float32)

                # Exact match: use observed data
                if has_match.any():
                    y_prev_at_curr[has_match] = y_prev[match_indices[has_match]]

                # No match: use prediction from previous chain
                if (~has_match).any():
                    X_missing = X_curr[~has_match]
                    with torch.no_grad():
                        pred_mean, _ = self.models[-1].predict(X_missing)
                        y_prev_at_curr[~has_match] = torch.tensor(pred_mean, dtype=torch.float32)

                X_low_input = X_curr
                y_low_input = y_prev_at_curr
                X_high_input = X_curr
                y_high_input = y_curr

            # Train BNN for this level
            bnn_wrapper = SingleLevelBNN_Wrapper(self.input_dim, self.design_space, self.device)
            bnn_wrapper.fit(X_low_input, y_low_input, X_high_input, y_high_input)
            self.models.append(bnn_wrapper)

    def predict(self, test_X):
        """
        Predict at the highest fidelity.

        Args:
            test_X: Test inputs (N_test, D) or (N_test, D+1)
                    If D+1, fidelity column will be removed.

        Returns:
            mean: Predicted mean (N_test,)
            var: Predicted variance (N_test,)
        """
        if not torch.is_tensor(test_X):
            test_X = torch.tensor(test_X, dtype=torch.float32)
        test_X = test_X.to(self.device).to(torch.float32)

        # Handle fidelity column removal if present
        expected_dim = self.input_dim
        if test_X.shape[1] == expected_dim + 1:
            if self.fidelity_col_idx == -1:
                test_X = test_X[:, :-1]
            else:
                cols = [c for c in range(test_X.shape[1])
                        if c != (test_X.shape[1] + self.fidelity_col_idx if self.fidelity_col_idx < 0 else self.fidelity_col_idx)]
                test_X = test_X[:, cols]

        # Predict using the highest level model
        mean, var = self.models[-1].predict(test_X)

        return mean.reshape(-1), var.reshape(-1)
