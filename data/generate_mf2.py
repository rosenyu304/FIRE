"""
Generate synthetic 2-fidelity datasets from the mf2 benchmark library.

Each function returns a dict containing:
  - X_lf, y_lf:       200 low-fidelity training samples (LHS)
  - X_hf_{n}, y_hf_{n}: HF training samples for n in {4, 8, 10, 20, 40, 50}
  - X_test, y_test:    100 HF test samples (LHS, disjoint from training)

Usage:
    from generate_mf2 import branin
    data = branin(seed=42)
    X_lf, y_lf = data['X_lf'], data['y_lf']
    X_hf, y_hf = data['X_hf_10'], data['y_hf_10']  # 10 HF samples
    X_test, y_test = data['X_test'], data['y_test']
"""

import numpy as np
from pyDOE import lhs
import mf2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_LF = 200
N_HF_LIST = [4, 8, 10, 20, 40, 50]
N_TEST = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lhs_in_bounds(n, d, l_bound, u_bound):
    """Latin Hypercube Sampling scaled to [l_bound, u_bound].
    Caller must set np.random.seed before calling."""
    raw = lhs(d, samples=n)
    lb = np.asarray(l_bound, dtype=float)
    ub = np.asarray(u_bound, dtype=float)
    if lb.ndim == 0:
        lb = np.full(d, float(lb))
    if ub.ndim == 0:
        ub = np.full(d, float(ub))
    return raw * (ub - lb) + lb


def _subseed(name, role, base_seed):
    """Deterministic sub-seed so each (function, role) pair is reproducible."""
    return abs(hash((name, role, base_seed))) % (2**31)


def _generate_2f(func_name, mf2_func, seed=42):
    """
    Core generator for any mf2 two-fidelity function.

    Sampling strategy (no data leak):
      1. Draw N_LF LF training points with one sub-seed.
      2. Draw max(N_HF_LIST) HF training points with another sub-seed.
         Smaller HF sets are strict subsets of the largest one.
      3. Draw N_TEST HF test points with a third sub-seed.
    All three pools use independent LHS designs.
    """
    d = mf2_func.ndim
    lb = np.asarray(mf2_func.l_bound, dtype=float)
    ub = np.asarray(mf2_func.u_bound, dtype=float)
    low_fn = mf2_func.low
    high_fn = mf2_func.high

    result = {}

    # --- LF training data ---
    np.random.seed(_subseed(func_name, "lf_train", seed))
    X_lf = _lhs_in_bounds(N_LF, d, lb, ub)
    y_lf = low_fn(X_lf)
    result['X_lf'] = X_lf.astype(np.float64)
    result['y_lf'] = y_lf.astype(np.float64)

    # --- HF training data (nested subsets) ---
    max_hf = max(N_HF_LIST)
    np.random.seed(_subseed(func_name, "hf_train", seed))
    X_hf_all = _lhs_in_bounds(max_hf, d, lb, ub)
    y_hf_all = high_fn(X_hf_all)

    for n_hf in N_HF_LIST:
        result[f'X_hf_{n_hf}'] = X_hf_all[:n_hf].astype(np.float64)
        result[f'y_hf_{n_hf}'] = y_hf_all[:n_hf].astype(np.float64)

    # --- HF test data ---
    np.random.seed(_subseed(func_name, "test", seed))
    X_test = _lhs_in_bounds(N_TEST, d, lb, ub)
    y_test = high_fn(X_test)
    result['X_test'] = X_test.astype(np.float64)
    result['y_test'] = y_test.astype(np.float64)

    return result


# ---------------------------------------------------------------------------
# Public API — one function per mf2 benchmark
# ---------------------------------------------------------------------------
def forrester(seed=42):
    return _generate_2f("forrester", mf2.forrester, seed)

def booth(seed=42):
    return _generate_2f("booth", mf2.booth, seed)

def branin(seed=42):
    return _generate_2f("branin", mf2.branin, seed)

def bohachevsky(seed=42):
    return _generate_2f("bohachevsky", mf2.bohachevsky, seed)

def borehole(seed=42):
    return _generate_2f("borehole", mf2.borehole, seed)

def currin(seed=42):
    return _generate_2f("currin", mf2.currin, seed)

def hartmann6(seed=42):
    return _generate_2f("hartmann6", mf2.hartmann6, seed)

def himmelblau(seed=42):
    return _generate_2f("himmelblau", mf2.himmelblau, seed)

def park91a(seed=42):
    return _generate_2f("park91a", mf2.park91a, seed)

def park91b(seed=42):
    return _generate_2f("park91b", mf2.park91b, seed)

def six_hump_camelback(seed=42):
    return _generate_2f("six_hump_camelback", mf2.six_hump_camelback, seed)


# ---------------------------------------------------------------------------
# Registry for convenience (iterate over all problems)
# ---------------------------------------------------------------------------
MF2_FUNCTIONS = {
    'forrester': forrester,
    'booth': booth,
    'branin': branin,
    'bohachevsky': bohachevsky,
    'borehole': borehole,
    'currin': currin,
    'hartmann6': hartmann6,
    'himmelblau': himmelblau,
    'park91a': park91a,
    'park91b': park91b,
    'six_hump_camelback': six_hump_camelback,
}


if __name__ == "__main__":
    # Quick sanity check
    for name, fn in MF2_FUNCTIONS.items():
        data = fn(seed=42)
        dim = data['X_lf'].shape[1]
        print(f"{name:>20s}  dim={dim:2d}  "
              f"LF={data['X_lf'].shape[0]}  "
              f"HF_max={data['X_hf_50'].shape[0]}  "
              f"test={data['X_test'].shape[0]}")
