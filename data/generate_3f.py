"""
Generate 3-fidelity Branin and Hartmann datasets.

Branin3f (2D):
  Fidelity 1 (HF):  Standard Branin function
  Fidelity 2 (MF):  10*sqrt(y1(x-2)) + 2(x1-0.5) - 3(3x2-1) - 1
  Fidelity 3 (LF):  y2(1.2*(x+2)) - 3*x2 + 1

Hartmann3f (3D):
  3-fidelity Hartmann with alpha perturbation:
    alpha_f = alpha_base + (3-f)*delta,  f=1(LF), f=2(MF), f=3(HF)

Each function returns a dict containing:
  - X_lf_0, y_lf_0:     200 lowest-fidelity (LF) training samples
  - X_lf_1, y_lf_1:     50 mid-fidelity (MF) training samples
  - X_hf_{n}, y_hf_{n}: HF training samples for n in {40, 80, 100, 200, 400, 500}
  - X_test, y_test:      100 HF test samples

Usage:
    from generate_3f import branin3f, hartmann3f
    data = branin3f(seed=42)
"""

import numpy as np
from pyDOE import lhs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_LF_0 = 200   # lowest fidelity
N_LF_1 = 50    # mid fidelity
N_HF_LIST = [40, 80, 100, 200, 400, 500]
N_TEST = 100


# ---------------------------------------------------------------------------
# Branin 3-fidelity analytical functions
# ---------------------------------------------------------------------------
def _branin_hf(X):
    """Fidelity 1 (highest): standard Branin."""
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    pi = np.pi
    term1 = (-1.275 * x1**2 / pi**2 + 5 * x1 / pi + x2 - 6)**2
    term2 = (10 - 5 / (4 * pi)) * np.cos(x1) + 10
    return term1 + term2


def _branin_mf(X):
    """Fidelity 2 (mid): depends on HF."""
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    y_high_shifted = _branin_hf(X - 2.0)
    return 10 * np.sqrt(y_high_shifted) + 2 * (x1 - 0.5) - 3 * (3 * x2 - 1) - 1


def _branin_lf(X):
    """Fidelity 3 (lowest): depends on MF."""
    X = np.atleast_2d(X)
    x2 = X[:, 1]
    X_transformed = 1.2 * (X + 2.0)
    y_med = _branin_mf(X_transformed)
    return y_med - 3 * x2 + 1


# ---------------------------------------------------------------------------
# Hartmann 3-fidelity analytical functions
# ---------------------------------------------------------------------------
_HARTMANN_A = np.array([
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
    [3.0, 10.0, 30.0],
    [0.1, 10.0, 35.0],
])

_HARTMANN_P = np.array([
    [0.3689, 0.1170, 0.2673],
    [0.4699, 0.4387, 0.7470],
    [0.1091, 0.8732, 0.5547],
    [0.0381, 0.5743, 0.8828],
])

_HARTMANN_ALPHA_BASE = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN_DELTA = np.array([0.01, -0.01, -0.1, 0.1])


def _hartmann_fidelity(X, fidelity_level):
    """Hartmann-3D at fidelity_level (1=lowest, 2=mid, 3=highest).
    alpha_f = alpha_base + (3 - f) * delta
    """
    X = np.atleast_2d(X)
    alpha_f = _HARTMANN_ALPHA_BASE + (3 - fidelity_level) * _HARTMANN_DELTA
    y = np.zeros(X.shape[0])
    for i in range(4):
        inner = np.sum(_HARTMANN_A[i] * (X - _HARTMANN_P[i])**2, axis=1)
        y += alpha_f[i] * np.exp(-inner)
    return y


def _hartmann_hf(X):
    return _hartmann_fidelity(X, 3)

def _hartmann_mf(X):
    return _hartmann_fidelity(X, 2)

def _hartmann_lf(X):
    return _hartmann_fidelity(X, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lhs_in_bounds(n, d, lb, ub):
    """LHS scaled to [lb, ub]. Caller must set np.random.seed first."""
    raw = lhs(d, samples=n)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    return raw * (ub - lb) + lb


def _subseed(name, role, base_seed):
    return abs(hash((name, role, base_seed))) % (2**31)


def _generate_3f(name, d, lb, ub, lf_fn, mf_fn, hf_fn, seed=42):
    """Core generator for a 3-fidelity dataset."""
    result = {}

    # --- LF_0 (lowest fidelity) training data ---
    np.random.seed(_subseed(name, "lf0_train", seed))
    X_lf_0 = _lhs_in_bounds(N_LF_0, d, lb, ub)
    result['X_lf_0'] = X_lf_0.astype(np.float64)
    result['y_lf_0'] = lf_fn(X_lf_0).astype(np.float64)

    # --- LF_1 (mid fidelity) training data ---
    np.random.seed(_subseed(name, "lf1_train", seed))
    X_lf_1 = _lhs_in_bounds(N_LF_1, d, lb, ub)
    result['X_lf_1'] = X_lf_1.astype(np.float64)
    result['y_lf_1'] = mf_fn(X_lf_1).astype(np.float64)

    # --- HF training data (nested subsets) ---
    max_hf = max(N_HF_LIST)
    np.random.seed(_subseed(name, "hf_train", seed))
    X_hf_all = _lhs_in_bounds(max_hf, d, lb, ub)
    y_hf_all = hf_fn(X_hf_all)
    for n_hf in N_HF_LIST:
        result[f'X_hf_{n_hf}'] = X_hf_all[:n_hf].astype(np.float64)
        result[f'y_hf_{n_hf}'] = y_hf_all[:n_hf].astype(np.float64)

    # --- HF test data ---
    np.random.seed(_subseed(name, "test", seed))
    X_test = _lhs_in_bounds(N_TEST, d, lb, ub)
    result['X_test'] = X_test.astype(np.float64)
    result['y_test'] = hf_fn(X_test).astype(np.float64)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def branin3f(seed=42):
    lb = np.array([-5.0, 0.0])
    ub = np.array([10.0, 15.0])
    return _generate_3f("branin3f", 2, lb, ub, _branin_lf, _branin_mf, _branin_hf, seed)


def hartmann3f(seed=42):
    lb = np.zeros(3)
    ub = np.ones(3)
    return _generate_3f("hartmann3f", 3, lb, ub, _hartmann_lf, _hartmann_mf, _hartmann_hf, seed)


THREE_FIDELITY_FUNCTIONS = {
    'branin3f': branin3f,
    'hartmann3f': hartmann3f,
}


if __name__ == "__main__":
    for name, fn in THREE_FIDELITY_FUNCTIONS.items():
        data = fn(seed=42)
        dim = data['X_lf_0'].shape[1]
        print(f"{name:>12s}  dim={dim}  "
              f"LF0={data['X_lf_0'].shape[0]}  "
              f"LF1={data['X_lf_1'].shape[0]}  "
              f"HF_max={data['X_hf_500'].shape[0]}  "
              f"test={data['X_test'].shape[0]}")
