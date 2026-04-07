"""
Generate synthetic 2-fidelity high-dimensional (BNN) datasets.

Five problems with dim = 10, 20, 30, 40, 50.
Analytical functions:
  HF: f(x) = (x_0 - 1)^2 + sum_{i=1}^{d-1} (2*x_i^2 - x_{i-1})^2
  LF: g(x) = 0.8 * f(x) - sum_{i=0}^{d-2} 0.4*x_i*x_{i+1} - 50
Bounds: [-3, 3]^d for all dimensions.

Each function returns a dict containing:
  - X_lf, y_lf:         2000 low-fidelity training samples (LHS)
  - X_hf_{n}, y_hf_{n}: HF training samples for n in {40, 80, 100, 200, 400, 500}
  - X_test, y_test:     1000 HF test samples (LHS, disjoint from training)

Usage:
    from generate_bnn import HD10
    data = HD10(seed=42)
    X_lf, y_lf = data['X_lf'], data['y_lf']
    X_hf, y_hf = data['X_hf_100'], data['y_hf_100']
    X_test, y_test = data['X_test'], data['y_test']
"""

import numpy as np
from pyDOE import lhs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_LF = 2000
N_HF_LIST = [40, 80, 100, 200, 400, 500]
N_TEST = 1000
HD_DIMS = [10, 20, 30, 40, 50]


# ---------------------------------------------------------------------------
# Analytical functions
# ---------------------------------------------------------------------------
def _hf_func(X):
    """High-fidelity: (x_0 - 1)^2 + sum_i (2*x_i^2 - x_{i-1})^2"""
    return (X[:, 0] - 1.0) ** 2 + np.sum((2.0 * X[:, 1:] ** 2 - X[:, :-1]) ** 2, axis=1)


def _lf_func(X):
    """Low-fidelity: 0.8 * HF - sum_i 0.4*x_i*x_{i+1} - 50"""
    return 0.8 * _hf_func(X) - np.sum(0.4 * X[:, :-1] * X[:, 1:], axis=1) - 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lhs_in_bounds(n, d, lb, ub):
    """LHS scaled to [lb, ub]. Caller must set np.random.seed first."""
    raw = lhs(d, samples=n)
    return raw * (ub - lb) + lb


def _subseed(name, role, base_seed):
    """Deterministic sub-seed per (function, role)."""
    return abs(hash((name, role, base_seed))) % (2**31)


def _generate_hd(d, seed=42):
    """Core generator for a high-dimensional BNN dataset of dimension d."""
    name = f"high_dim_{d}"
    lb = -3.0 * np.ones(d)
    ub = 3.0 * np.ones(d)

    result = {}

    # --- LF training data ---
    np.random.seed(_subseed(name, "lf_train", seed))
    X_lf = _lhs_in_bounds(N_LF, d, lb, ub)
    result['X_lf'] = X_lf.astype(np.float64)
    result['y_lf'] = _lf_func(X_lf).astype(np.float64)

    # --- HF training data (nested subsets) ---
    max_hf = max(N_HF_LIST)
    np.random.seed(_subseed(name, "hf_train", seed))
    X_hf_all = _lhs_in_bounds(max_hf, d, lb, ub)
    y_hf_all = _hf_func(X_hf_all)

    for n_hf in N_HF_LIST:
        result[f'X_hf_{n_hf}'] = X_hf_all[:n_hf].astype(np.float64)
        result[f'y_hf_{n_hf}'] = y_hf_all[:n_hf].astype(np.float64)

    # --- HF test data ---
    np.random.seed(_subseed(name, "test", seed))
    X_test = _lhs_in_bounds(N_TEST, d, lb, ub)
    result['X_test'] = X_test.astype(np.float64)
    result['y_test'] = _hf_func(X_test).astype(np.float64)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def HD10(seed=42):
    return _generate_hd(10, seed)

def HD20(seed=42):
    return _generate_hd(20, seed)

def HD30(seed=42):
    return _generate_hd(30, seed)

def HD40(seed=42):
    return _generate_hd(40, seed)

def HD50(seed=42):
    return _generate_hd(50, seed)


BNN_FUNCTIONS = {
    'HD10': HD10,
    'HD20': HD20,
    'HD30': HD30,
    'HD40': HD40,
    'HD50': HD50,
}


if __name__ == "__main__":
    for name, fn in BNN_FUNCTIONS.items():
        data = fn(seed=42)
        dim = data['X_lf'].shape[1]
        print(f"{name:>6s}  dim={dim:2d}  "
              f"LF={data['X_lf'].shape[0]}  "
              f"HF_max={data['X_hf_500'].shape[0]}  "
              f"test={data['X_test'].shape[0]}")
