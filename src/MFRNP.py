"""
MFRNP: Multi-fidelity Residual Neural Process

Implements a neural process architecture for multi-fidelity regression
with residual learning at the highest fidelity level.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class MLP_Encoder(nn.Module):
    """
    Encodes (x, y) pairs into latent representation parameters (mu, cov).
    """

    def __init__(self, in_dim, out_dim, hidden_layers=2, hidden_dim=32):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Sigmoid()

    def forward(self, x):
        output = self.model(x)
        mean = self.mean_out(output)
        cov = 0.1 + 0.9 * self.cov_m(self.cov_out(output))
        return mean, cov


class MLP_Decoder(nn.Module):
    """
    Decodes (x, z) into output parameters (mu, cov).
    """

    def __init__(self, in_dim, out_dim, hidden_layers=2, hidden_dim=32):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ELU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.model = nn.Sequential(*layers)
        self.mean_out = nn.Linear(hidden_dim, out_dim)
        self.cov_out = nn.Linear(hidden_dim, out_dim)
        self.cov_m = nn.Softplus()

    def forward(self, x):
        output = self.model(x)
        mean = self.mean_out(output)
        cov = self.cov_m(self.cov_out(output))
        return mean, cov


class MultiFidelityNPModel(nn.Module):
    """
    Core Multi-fidelity Neural Process Model.
    Handles multi-level encoding/decoding and residual aggregation.
    """

    def __init__(self, levels, input_dim, output_dims, device='cpu', **kwargs):
        super().__init__()
        self.device = device
        self.levels = levels
        self.input_dim = input_dim

        # Hyperparameters
        self.hidden_dim = int(kwargs.get('hidden_dim', 64))
        self.z_dim = int(kwargs.get('z_dim', 32))
        self.hidden_layers = int(kwargs.get('hidden_layers', 2))
        self.context_percentage_low = float(kwargs.get('context_percentage_low', 0.2))
        self.context_percentage_high = float(kwargs.get('context_percentage_high', 0.8))

        # Initialize encoders and decoders for each level
        for level in range(1, levels + 1):
            output_dim = output_dims[level-1]
            setattr(self, f"l{level}_output_dim", output_dim)

            # Encoder: (x, y) -> z
            setattr(self, f"l{level}_encoder_model", MLP_Encoder(
                self.input_dim + output_dim, self.z_dim, self.hidden_layers, self.hidden_dim).to(self.device))

            # Decoder: (x, z) -> y
            setattr(self, f"l{level}_decoder_model", MLP_Decoder(
                self.z_dim + self.input_dim, output_dim, self.hidden_layers, self.hidden_dim).to(self.device))

    def split_context_target(self, x, y):
        """Randomly splits data into context and target sets."""
        context_percentage = np.random.uniform(self.context_percentage_low, self.context_percentage_high)
        n_context = int(x.shape[0] * context_percentage)
        n_context = max(1, n_context) if x.shape[0] > 0 else 0

        ind = np.arange(x.shape[0])
        np.random.shuffle(ind)
        mask_c = ind[:n_context]
        mask_t = ind[n_context:]

        return x[mask_c], y[mask_c], x[mask_t], y[mask_t], mask_c, mask_t

    def sample_z(self, mean, var, n):
        """Reparameterization trick."""
        eps = torch.randn(n, var.size(0)).to(self.device)
        std = torch.sqrt(var)
        return mean.unsqueeze(0) + std.unsqueeze(0) * eps

    def z_to_y(self, x, zs, level):
        """Decoder forward pass."""
        zs_expanded = zs.expand(x.shape[0], -1)
        output = getattr(self, f"l{level}_decoder_model")(torch.cat([x, zs_expanded], dim=-1))
        return output

    def xy_to_r(self, x, y, level):
        """Encoder forward pass."""
        return getattr(self, f"l{level}_encoder_model")(torch.cat([x, y], dim=-1))

    def ba_z_agg(self, r_mu, r_cov):
        """Bayesian aggregation of latent variables (Product of Gaussians)."""
        z_mu_prior = torch.zeros(r_mu.shape[1]).to(self.device)
        z_cov_prior = torch.ones(r_cov.shape[1]).to(self.device)

        if r_mu.shape[0] == 0:
            return z_mu_prior, z_cov_prior

        w_cov_inv = 1 / r_cov
        z_cov_new = 1 / (1 / z_cov_prior + torch.sum(w_cov_inv, dim=0))
        v = r_mu * w_cov_inv
        z_mu_new = z_cov_new * (torch.sum(v, dim=0))

        return z_mu_new, z_cov_new

    def forward(self, xs, ys):
        """Training forward pass."""
        results = {
            "targets": [], "output_mus": [], "output_covs": [],
            "z_mu_all": [], "z_cov_all": [], "z_mu_cs": [], "z_cov_cs": []
        }

        levels = len(xs)
        for level in range(1, levels + 1):
            x, y = xs[level-1], ys[level-1]
            x_c, y_c, x_t, y_t, mask_c, mask_t = self.split_context_target(x, y)

            # Residual logic for highest fidelity
            if level == levels and levels > 1:
                residual_predictions = []
                for lvl in range(levels - 1):
                    zs = self.sample_z(results["z_mu_all"][lvl], results["z_cov_all"][lvl], x.size(0))
                    res_pred_mu, _ = self.z_to_y(x, zs[0], lvl + 1)
                    residual_predictions.append(res_pred_mu)

                ensemble_agg = torch.mean(torch.stack(residual_predictions), dim=0)
                y = y - ensemble_agg
                y_c = y[mask_c]
                y_t = y[mask_t]

            # Standard NP encoding/decoding
            r_mu_all, r_cov_all = self.xy_to_r(x, y, level)
            z_mu_all, z_cov_all = self.ba_z_agg(r_mu_all, r_cov_all)

            r_mu_c, r_cov_c = self.xy_to_r(x_c, y_c, level)
            z_mu_c, z_cov_c = self.ba_z_agg(r_mu_c, r_cov_c)

            zs = self.sample_z(z_mu_c, z_cov_c, x_t.size(0))
            output_mu, output_cov = self.z_to_y(x_t, zs[0], level)

            results["targets"].append(y_t)
            results["output_mus"].append(output_mu)
            results["output_covs"].append(output_cov)
            results["z_mu_all"].append(z_mu_all)
            results["z_cov_all"].append(z_cov_all)
            results["z_mu_cs"].append(z_mu_c)
            results["z_cov_cs"].append(z_cov_c)

        return results

    def get_context_encoding(self, xs, ys):
        """Calculate Z distribution from full training set."""
        temp_z_mu, temp_z_cov = [], []
        levels = len(xs)

        for level in range(1, levels + 1):
            x, y = xs[level-1], ys[level-1]

            if level == levels and levels > 1:
                preds = []
                for lvl in range(levels - 1):
                    zs = self.sample_z(temp_z_mu[lvl], temp_z_cov[lvl], x.size(0))
                    mu, _ = self.z_to_y(x, zs[0], lvl + 1)
                    preds.append(mu)
                ensemble = torch.mean(torch.stack(preds), dim=0)
                y = y - ensemble

            r_mu, r_cov = self.xy_to_r(x, y, level)
            z_mu, z_cov = self.ba_z_agg(r_mu, r_cov)

            temp_z_mu.append(z_mu)
            temp_z_cov.append(z_cov)

        return temp_z_mu, temp_z_cov


def nll_loss(pred_mu, pred_cov, y):
    """Negative log-likelihood loss."""
    pred_std = torch.sqrt(pred_cov)
    gaussian = Normal(pred_mu, pred_std)
    nll = torch.mean(-gaussian.log_prob(y))
    return nll


def kld_gaussian_loss(z_mean_all, z_var_all, z_mean_context, z_var_context):
    """KL divergence between two Gaussians."""
    std_all = torch.sqrt(z_var_all)
    std_context = torch.sqrt(z_var_context)
    dist_all = Normal(z_mean_all, std_all)
    dist_context = Normal(z_mean_context, std_context)
    kld = torch.distributions.kl_divergence(dist_all, dist_context)
    return torch.mean(kld)


class MFRNP:
    """
    Multi-fidelity Residual Neural Process.

    Supports arbitrary number of fidelity levels through the fidelity column.
    """

    def __init__(self, train_X, train_Y, fidelity_col_idx=-1, device='cpu', seed=42, train_iter=1000):
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
        self.device = torch.device(device)
        self.train_iter = train_iter

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Data formatting
        if not torch.is_tensor(train_X):
            train_X = torch.tensor(train_X)
        if not torch.is_tensor(train_Y):
            train_Y = torch.tensor(train_Y)

        train_X = train_X.float().to(self.device)
        train_Y = train_Y.float().to(self.device)
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)

        # Split by fidelity
        fidelities = train_X[:, fidelity_col_idx].long()
        unique_fids = torch.unique(fidelities).sort()[0]
        self.levels = len(unique_fids)

        self.xs = []
        self.ys = []

        # Remove fidelity column
        feature_mask = torch.ones(train_X.shape[1], dtype=torch.bool).to(self.device)
        feature_mask[fidelity_col_idx] = False

        for fid in unique_fids:
            mask = (fidelities == fid)
            self.xs.append(train_X[mask][:, feature_mask])
            self.ys.append(train_Y[mask])

        input_dim = self.xs[0].shape[1]
        output_dims = [y.shape[1] for y in self.ys]

        # Model initialization
        self.model = MultiFidelityNPModel(
            levels=self.levels,
            input_dim=input_dim,
            output_dims=output_dims,
            device=self.device,
            hidden_dim=64,
            z_dim=32
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-3)
        self._train()

    def _train(self):
        """Train the model."""
        self.model.train()
        fidelity_weight = 5.0
        lower_fidelity_weight = 0.2

        for i in range(self.train_iter):
            self.optimizer.zero_grad()
            output = self.model(self.xs, self.ys)

            total_loss = 0
            for level in range(1, self.levels + 1):
                idx = level - 1
                nll = nll_loss(output["output_mus"][idx], output["output_covs"][idx], output["targets"][idx])
                kld = kld_gaussian_loss(
                    output["z_mu_all"][idx], output["z_cov_all"][idx],
                    output["z_mu_cs"][idx], output["z_cov_cs"][idx]
                )

                if level == self.levels:
                    total_loss += nll * fidelity_weight + kld
                else:
                    total_loss += nll * lower_fidelity_weight + kld

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

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
        if not torch.is_tensor(test_X):
            test_X = torch.tensor(test_X)
        test_X = test_X.float().to(self.device)

        with torch.no_grad():
            z_mus, z_covs = self.model.get_context_encoding(self.xs, self.ys)

            # Aggregate lower fidelity predictions
            lower_preds = []
            for lvl in range(self.levels - 1):
                z = self.model.sample_z(z_mus[lvl], z_covs[lvl], test_X.size(0))
                mu, _ = self.model.z_to_y(test_X, z[0], level=lvl + 1)
                lower_preds.append(mu)

            if lower_preds:
                ensemble_agg = torch.mean(torch.stack(lower_preds), dim=0)
            else:
                ensemble_agg = 0

            # Predict residual at highest fidelity
            z_last = self.model.sample_z(z_mus[-1], z_covs[-1], test_X.size(0))
            residual_mu, residual_cov = self.model.z_to_y(test_X, z_last[0], level=self.levels)

            # Combine
            final_mean = ensemble_agg + residual_mu
            final_var = residual_cov

        return final_mean.squeeze(-1), final_var.squeeze(-1)
