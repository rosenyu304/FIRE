"""
Utility functions for FIRE: Multi-fidelity Regression with Distribution-conditioned In-context Learning
"""

import os
import math
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


def get_metrics(y_true, y_pred, y_var):
    """
    Compute evaluation metrics for regression predictions.

    Args:
        y_true: Ground truth values
        y_pred: Predicted mean values
        y_var: Predicted variance values

    Returns:
        r2: R-squared score
        nrmse: Normalized Root Mean Squared Error
        nll: Negative Log Likelihood (Gaussian)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_var, torch.Tensor):
        y_var = y_var.cpu().numpy()
    if len(y_true.shape) > 1:
        y_true = y_true.squeeze(-1)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.squeeze(-1)
    if len(y_var.shape) > 1:
        y_var = y_var.squeeze(-1)

    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / (y_range if y_range != 0 else 1e-12)

    var = np.maximum(y_var, 1e-12)
    NLL = float(0.5 * np.mean(np.log(2.0 * math.pi * var) + ((y_true - y_pred)**2) / var))

    return r2, nrmse, NLL


def encode_2fidelity_data(X_lf, y_lf, X_hf, y_hf, X_test, y_test_hf,
                          preprocess_X=False, preprocess_Y=False):
    """
    Encode 2-fidelity data with fidelity column for general multi-fidelity methods.

    Args:
        X_lf: Low-fidelity input features (N_lf, D)
        y_lf: Low-fidelity target values (N_lf,)
        X_hf: High-fidelity input features (N_hf, D)
        y_hf: High-fidelity target values (N_hf,)
        X_test: Test input features (N_test, D)
        y_test_hf: Test target values at high fidelity (N_test,)
        preprocess_X: Whether to min-max normalize input features
        preprocess_Y: Preprocessing method for targets ("minmax", "standardize", or False)

    Returns:
        train_X_encoded: Encoded training inputs with fidelity column (N_lf + N_hf, D + 1)
        train_y_encoded: Concatenated training targets (N_lf + N_hf,)
        X_test_encoded: Test inputs (no fidelity column) (N_test, D)
        y_test_encoded: Test targets (N_test,)
    """
    if preprocess_X:
        X_lf_origin = X_lf.copy()
        X_hf_origin = X_hf.copy()
        X_test_origin = X_test.copy()

        # Min-max normalize X
        X_all = np.concatenate([X_lf, X_hf, X_test], axis=0)
        minn_ = np.min(X_all, axis=0)
        maxx_ = np.max(X_all, axis=0)
        range_ = maxx_ - minn_ + 1e-12

        X_lf = (X_lf - minn_) / range_
        X_hf = (X_hf - minn_) / range_
        X_test = (X_test - minn_) / range_

        if (np.isinf(X_lf).any()) or (np.isinf(X_hf).any()) or (np.isinf(X_test).any()):
            X_lf = X_lf_origin
            X_hf = X_hf_origin
            X_test = X_test_origin

    if preprocess_Y == "minmax":
        minn_ = np.min([np.min(y_lf), np.min(y_hf), np.min(y_test_hf)])
        maxx_ = np.max([np.max(y_lf), np.max(y_hf), np.max(y_test_hf)])
        range_ = maxx_ - minn_ + 1e-12
        y_lf = (y_lf - minn_) / range_
        y_hf = (y_hf - minn_) / range_
        y_test_hf = (y_test_hf - minn_) / range_

    elif preprocess_Y == "standardize":
        minn_ = np.mean(np.concatenate([y_lf, y_hf, y_test_hf], axis=0))
        maxx_ = np.std(np.concatenate([y_lf, y_hf, y_test_hf], axis=0))
        y_lf = (y_lf - minn_) / maxx_
        y_hf = (y_hf - minn_) / maxx_
        y_test_hf = (y_test_hf - minn_) / maxx_

    # Encode fidelity as additional column (0 = low, 1 = high)
    X_lf_encoded = np.concatenate([X_lf, np.zeros((X_lf.shape[0], 1))], axis=1)
    X_hf_encoded = np.concatenate([X_hf, np.ones((X_hf.shape[0], 1))], axis=1)

    train_X_encoded = np.concatenate([X_lf_encoded, X_hf_encoded], axis=0)
    train_y_encoded = np.concatenate([y_lf, y_hf], axis=0)

    X_test_encoded = X_test
    y_test_encoded = y_test_hf

    return (torch.tensor(train_X_encoded),
            torch.tensor(train_y_encoded),
            torch.tensor(X_test_encoded),
            torch.tensor(y_test_encoded))


def encode_3fidelity_data(X_lf_0, y_lf_0, X_lf_1, y_lf_1, X_hf, y_hf, X_test, y_test_hf,
                          preprocess_X=False, preprocess_Y=False):
    """
    Encode 3-fidelity data with fidelity column.

    Fidelity encoding: 0 = lowest, 1 = middle, 2 = highest
    """
    if preprocess_X:
        X_all = np.concatenate([X_lf_0, X_lf_1, X_hf, X_test], axis=0)
        minn_ = np.min(X_all, axis=0)
        maxx_ = np.max(X_all, axis=0)
        range_ = maxx_ - minn_ + 1e-12
        X_lf_0 = (X_lf_0 - minn_) / range_
        X_lf_1 = (X_lf_1 - minn_) / range_
        X_hf = (X_hf - minn_) / range_
        X_test = (X_test - minn_) / range_

    if preprocess_Y:
        minn_ = np.min([np.min(y_lf_0), np.min(y_lf_1), np.min(y_hf), np.min(y_test_hf)])
        maxx_ = np.max([np.max(y_lf_0), np.max(y_lf_1), np.max(y_hf), np.max(y_test_hf)])
        range_ = maxx_ - minn_ + 1e-12
        y_lf_0 = (y_lf_0 - minn_) / range_
        y_lf_1 = (y_lf_1 - minn_) / range_
        y_hf = (y_hf - minn_) / range_
        y_test_hf = (y_test_hf - minn_) / range_

    X_lf_0_encoded = np.concatenate([X_lf_0, np.zeros((X_lf_0.shape[0], 1))], axis=1)
    X_lf_1_encoded = np.concatenate([X_lf_1, np.ones((X_lf_1.shape[0], 1))], axis=1)
    X_hf_encoded = np.concatenate([X_hf, np.ones((X_hf.shape[0], 1)) * 2], axis=1)

    train_X_encoded = np.concatenate([X_lf_0_encoded, X_lf_1_encoded, X_hf_encoded], axis=0)
    train_y_encoded = np.concatenate([y_lf_0, y_lf_1, y_hf], axis=0)

    X_test_encoded = X_test
    y_test_encoded = y_test_hf

    return (torch.tensor(train_X_encoded),
            torch.tensor(train_y_encoded),
            torch.tensor(X_test_encoded),
            torch.tensor(y_test_encoded))


def encode_5fidelity_data(X_lf_0, y_lf_0, X_lf_1, y_lf_1,
                          X_lf_2, y_lf_2, X_lf_3, y_lf_3,
                          X_hf, y_hf, X_test, y_test_hf,
                          preprocess_X=False, preprocess_Y=False):
    """
    Encode 5-fidelity data with fidelity column.

    Fidelity encoding: 0, 1, 2, 3 = Low Fidelities; 4 = High Fidelity
    """
    if preprocess_X:
        X_all = np.concatenate([X_lf_0, X_lf_1, X_lf_2, X_lf_3, X_hf, X_test], axis=0)
        minn_ = np.min(X_all, axis=0)
        maxx_ = np.max(X_all, axis=0)
        range_ = maxx_ - minn_ + 1e-12

        X_lf_0 = (X_lf_0 - minn_) / range_
        X_lf_1 = (X_lf_1 - minn_) / range_
        X_lf_2 = (X_lf_2 - minn_) / range_
        X_lf_3 = (X_lf_3 - minn_) / range_
        X_hf = (X_hf - minn_) / range_
        X_test = (X_test - minn_) / range_

    if preprocess_Y == "minmax":
        minn_ = np.min([np.min(y_lf_0), np.min(y_lf_1), np.min(y_lf_2),
                        np.min(y_lf_3), np.min(y_hf), np.min(y_test_hf)])
        maxx_ = np.max([np.max(y_lf_0), np.max(y_lf_1), np.max(y_lf_2),
                        np.max(y_lf_3), np.max(y_hf), np.max(y_test_hf)])
        range_ = maxx_ - minn_ + 1e-12

        y_lf_0 = (y_lf_0 - minn_) / range_
        y_lf_1 = (y_lf_1 - minn_) / range_
        y_lf_2 = (y_lf_2 - minn_) / range_
        y_lf_3 = (y_lf_3 - minn_) / range_
        y_hf = (y_hf - minn_) / range_
        y_test_hf = (y_test_hf - minn_) / range_

    elif preprocess_Y == "standardize":
        y_all = np.concatenate([y_lf_0, y_lf_1, y_lf_2, y_lf_3, y_hf, y_test_hf], axis=0)
        mean_ = np.mean(y_all)
        std_ = np.std(y_all) + 1e-12

        y_lf_0 = (y_lf_0 - mean_) / std_
        y_lf_1 = (y_lf_1 - mean_) / std_
        y_lf_2 = (y_lf_2 - mean_) / std_
        y_lf_3 = (y_lf_3 - mean_) / std_
        y_hf = (y_hf - mean_) / std_
        y_test_hf = (y_test_hf - mean_) / std_

    # Encode fidelity levels (0 to 4)
    X_lf_0_encoded = np.concatenate([X_lf_0, np.zeros((X_lf_0.shape[0], 1))], axis=1)
    X_lf_1_encoded = np.concatenate([X_lf_1, np.ones((X_lf_1.shape[0], 1))], axis=1)
    X_lf_2_encoded = np.concatenate([X_lf_2, np.ones((X_lf_2.shape[0], 1)) * 2], axis=1)
    X_lf_3_encoded = np.concatenate([X_lf_3, np.ones((X_lf_3.shape[0], 1)) * 3], axis=1)
    X_hf_encoded = np.concatenate([X_hf, np.ones((X_hf.shape[0], 1)) * 4], axis=1)

    train_X_encoded = np.concatenate([
        X_lf_0_encoded, X_lf_1_encoded, X_lf_2_encoded, X_lf_3_encoded, X_hf_encoded
    ], axis=0)

    train_y_encoded = np.concatenate([
        y_lf_0, y_lf_1, y_lf_2, y_lf_3, y_hf
    ], axis=0)

    X_test_encoded = X_test
    y_test_encoded = y_test_hf

    return (torch.tensor(train_X_encoded),
            torch.tensor(train_y_encoded),
            torch.tensor(X_test_encoded),
            torch.tensor(y_test_encoded))


def load_2fidelity_data(filename):
    """
    Load 2-fidelity data from npz file.

    Expected keys: X_lf, y_lf, X_hf, y_hf, X_test, y_test_hf
    """
    data = np.load(filename)
    return (data['X_lf'], data['y_lf'],
            data['X_hf'], data['y_hf'],
            data['X_test'], data['y_test_hf'])


def load_3fidelity_data(filename):
    """
    Load 3-fidelity data from npz file.

    Expected keys: X_lf_0, y_lf_0, X_lf_1, y_lf_1, X_hf, y_hf, X_test, y_test_hf
    """
    data = np.load(filename)
    return (data['X_lf_0'], data['y_lf_0'],
            data['X_lf_1'], data['y_lf_1'],
            data['X_hf'], data['y_hf'],
            data['X_test'], data['y_test_hf'])


def load_5fidelity_data(filename):
    """
    Load 5-fidelity data from npz file.

    Expected keys: X_lf_0...3, y_lf_0...3, X_hf, y_hf, X_test, y_test_hf
    """
    data = np.load(filename)
    return (data['X_lf_0'], data['y_lf_0'],
            data['X_lf_1'], data['y_lf_1'],
            data['X_lf_2'], data['y_lf_2'],
            data['X_lf_3'], data['y_lf_3'],
            data['X_hf'], data['y_hf'],
            data['X_test'], data['y_test_hf'])