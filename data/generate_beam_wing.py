"""
Generate 2-fidelity Beam and Wing datasets from analytical functions.

Beam (5D):
  Bounds: [0.1, 0.2, 4.0, 5e9, 1e4] to [0.2, 0.4, 6.0, 35e9, 1.4e4]
  LF (fidelity 0): fixed parameters (0.15, 0.3, 5, 30e9, 1.2e4)
  HF (fidelity 1): full parameterized deflection formula
  Train: 200 LF, {4, 8, 10, 20, 40, 50} HF
  Test: 100 HF

Wing (10D):
  Bounds: [150,220,6,-10,16,0.7,0.08,2,1700,0.025] to [200,300,10,10,45,0.9,0.20,3.5,2500,0.08]
  LF (fidelity 0): fixed q=40, lambda=0.85, tc=0.17, Nz=3
  HF (fidelity 1): full parameterized wing weight formula
  Train: 2000 LF, {40, 80, 100, 200, 400, 500} HF
  Test: 1000 HF

Usage:
    from generate_beam_wing import beam, wing
    data = beam(seed=42)
"""

import numpy as np
from pyDOE import lhs


# ---------------------------------------------------------------------------
# Beam analytical functions (5D)
# ---------------------------------------------------------------------------
BEAM_LB = np.array([0.1, 0.2, 4.0, 5e9, 1e4])
BEAM_UB = np.array([0.2, 0.4, 6.0, 35e9, 1.4e4])
BEAM_DIM = 5


def _beam_formula(X):
    """Core beam deflection: (5/32) * (load * length^4) / (E * width * height^3)"""
    return (5.0 / 32.0) * (X[..., 4] * X[..., 2]**4) / (X[..., 3] * X[..., 0] * X[..., 1]**3)


def _beam_low(X):
    """LF: fixed parameters."""
    X = X.copy()
    X[..., 0] = 0.15
    X[..., 1] = 0.3
    X[..., 2] = 5.0
    X[..., 3] = 30e9
    X[..., 4] = 1.2e4
    return _beam_formula(X)


def _beam_high(X):
    """HF: full parameterized."""
    return _beam_formula(X)


# ---------------------------------------------------------------------------
# Wing analytical functions (10D)
# ---------------------------------------------------------------------------
WING_LB = np.array([150, 220, 6, -10, 16, 0.7, 0.08, 2, 1700, 0.025])
WING_UB = np.array([200, 300, 10, 10, 45, 0.9, 0.20, 3.5, 2500, 0.08])
WING_DIM = 10


def _wing_high(X):
    """HF (fidelity 1): full wing weight formula."""
    Sw = X[..., 0]
    Wfw = X[..., 1]
    A = X[..., 2]
    Gama = X[..., 3] * (np.pi / 180.0)
    q = X[..., 4]
    lamb = X[..., 5]
    tc = X[..., 6]
    Nz = X[..., 7]
    Wdg = X[..., 8]
    Wp = X[..., 9]
    return (0.036 * Sw**0.758 * Wfw**0.0035
            * (A / np.cos(Gama)**2)**0.6
            * q**0.006 * lamb**0.04
            * (100 * tc / np.cos(Gama))**(-0.3)
            * (Nz * Wdg)**0.49
            + Wp)


def _wing_low(X):
    """LF (fidelity 0): fixed q, lambda, tc, Nz."""
    X = X.copy()
    X[..., 4] = 40.0     # q
    X[..., 5] = 0.85      # lambda
    X[..., 6] = 0.17      # tc
    X[..., 7] = 3.0       # Nz
    return _wing_high(X)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lhs_in_bounds(n, d, lb, ub):
    raw = lhs(d, samples=n)
    return raw * (ub - lb) + lb


def _subseed(name, role, base_seed):
    return abs(hash((name, role, base_seed))) % (2**31)


def _generate_2f(name, d, lb, ub, low_fn, high_fn, n_lf, n_hf_list, n_test, seed=42):
    result = {}

    # LF training
    np.random.seed(_subseed(name, "lf_train", seed))
    X_lf = _lhs_in_bounds(n_lf, d, lb, ub)
    result['X_lf'] = X_lf.astype(np.float64)
    result['y_lf'] = low_fn(X_lf).astype(np.float64)

    # HF training (nested)
    max_hf = max(n_hf_list)
    np.random.seed(_subseed(name, "hf_train", seed))
    X_hf_all = _lhs_in_bounds(max_hf, d, lb, ub)
    y_hf_all = high_fn(X_hf_all)
    for n_hf in n_hf_list:
        result[f'X_hf_{n_hf}'] = X_hf_all[:n_hf].astype(np.float64)
        result[f'y_hf_{n_hf}'] = y_hf_all[:n_hf].astype(np.float64)

    # HF test
    np.random.seed(_subseed(name, "test", seed))
    X_test = _lhs_in_bounds(n_test, d, lb, ub)
    result['X_test'] = X_test.astype(np.float64)
    result['y_test'] = high_fn(X_test).astype(np.float64)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def beam(seed=42):
    """Beam: LF=200, HF={4,8,10,20,40,50}, test=100."""
    return _generate_2f("beam", BEAM_DIM, BEAM_LB, BEAM_UB,
                        _beam_low, _beam_high,
                        n_lf=200, n_hf_list=[4, 8, 10, 20, 40, 50],
                        n_test=100, seed=seed)


def wing(seed=42):
    """Wing: LF=2000, HF={40,80,100,200,400,500}, test=1000."""
    return _generate_2f("wing", WING_DIM, WING_LB, WING_UB,
                        _wing_low, _wing_high,
                        n_lf=2000, n_hf_list=[40, 80, 100, 200, 400, 500],
                        n_test=1000, seed=seed)


BEAM_WING_FUNCTIONS = {
    'beam': beam,
    'wing': wing,
}


if __name__ == "__main__":
    for name, fn in BEAM_WING_FUNCTIONS.items():
        data = fn(seed=42)
        dim = data['X_lf'].shape[1]
        max_key = 'X_hf_50' if name == 'beam' else 'X_hf_500'
        print(f"{name:>6s}  dim={dim:2d}  "
              f"LF={data['X_lf'].shape[0]}  "
              f"HF_max={data[max_key].shape[0]}  "
              f"test={data['X_test'].shape[0]}")
