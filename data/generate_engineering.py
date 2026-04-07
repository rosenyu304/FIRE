"""
Generate 2-fidelity engineering datasets from CSV files.

Datasets and their configurations (from Table 4):
  car:      dim=23, LF=500,  HF={10,20,25,50,100,125}, test=250
  concrete: dim=8,  LF=200,  HF={4,8,10,20,40,50},     test=100
  bwb_cd:   dim=14, LF=1000, HF={20,40,50,100,200,250}, test=500
  bwb_cl:   dim=14, LF=1000, HF={20,40,50,100,200,250}, test=500

CSV column structure:
  car:      cols 3: → X (23D), col 1: y_lf, col 2: y_hf
  concrete: cols 0:8 → X (8D), col 8: y_hf, col 9: y_lf
  bwb:      14 feature cols → X (14D), CL_LF/CL_HF, CD_LF1/CD_HF

Usage:
    from generate_engineering import car, concrete, bwb_cd, bwb_cl
    data = car(seed=42)
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_DIR = os.path.join(os.path.dirname(__file__), "eng_csv")

CONFIGS = {
    'car':      {'n_lf': 500,  'n_hf_list': [10, 20, 25, 50, 100, 125], 'n_test': 250},
    'concrete': {'n_lf': 200,  'n_hf_list': [4, 8, 10, 20, 40, 50],     'n_test': 100},
    'bwb_cd':   {'n_lf': 1000, 'n_hf_list': [20, 40, 50, 100, 200, 250], 'n_test': 500},
    'bwb_cl':   {'n_lf': 1000, 'n_hf_list': [20, 40, 50, 100, 200, 250], 'n_test': 500},
}

BWB_FEATURE_COLS = [
    "alt_kft", "Re_L", "M_inf", "alpha_deg",
    "B1", "B2", "B3", "C1", "C2", "C3", "C4", "S1", "S2", "S3",
]


# ---------------------------------------------------------------------------
# CSV Loaders
# ---------------------------------------------------------------------------
def _load_car():
    df = pd.read_csv(os.path.join(CSV_DIR, "data_car.csv"), header=0)
    X = df.iloc[:, 3:].values.astype(np.float64)   # 23 features
    y_lf = df.iloc[:, 1].values.astype(np.float64)  # Low Fidelity
    y_hf = df.iloc[:, 2].values.astype(np.float64)  # High Fidelity
    return X, y_lf, y_hf


def _load_concrete():
    df = pd.read_csv(os.path.join(CSV_DIR, "data_concrete.csv"), header=0)
    X = df.iloc[:, 0:8].values.astype(np.float64)   # 8 features
    y_hf = df.iloc[:, 8].values.astype(np.float64)   # fc (HF)
    y_lf = df.iloc[:, 9].values.astype(np.float64)   # fc_ext_abrams (LF)
    return X, y_lf, y_hf


def _load_bwb():
    df = pd.read_csv(os.path.join(CSV_DIR, "data_bwb.csv"), header=0)
    X = df[BWB_FEATURE_COLS].values.astype(np.float64)  # 14 features
    return {
        'bwb_cd': (X, df["CD_LF1"].values.astype(np.float64), df["CD_HF"].values.astype(np.float64)),
        'bwb_cl': (X, df["CL_LF"].values.astype(np.float64), df["CL_HF"].values.astype(np.float64)),
    }


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
def _subseed(name, role, base_seed):
    return abs(hash((name, role, base_seed))) % (2**31)


def _generate_eng(name, X, y_lf_all, y_hf_all, seed=42):
    """Generate 2-fidelity dataset from tabular data.

    Sampling:
      - Shuffle all rows with a deterministic seed.
      - First n_test rows → test set (HF only).
      - Remaining rows → training pool.
      - From training pool: first n_lf for LF, nested subsets for HF.
    """
    cfg = CONFIGS[name]
    n_lf = cfg['n_lf']
    n_hf_list = cfg['n_hf_list']
    n_test = cfg['n_test']
    max_hf = max(n_hf_list)

    result = {}

    # Global shuffle to split train/test
    rng_split = np.random.RandomState(_subseed(name, "split", seed))
    perm = rng_split.permutation(len(X))

    # Test: first n_test indices
    test_idx = perm[:n_test]
    result['X_test'] = X[test_idx].astype(np.float64)
    result['y_test'] = y_hf_all[test_idx].astype(np.float64)

    # Training pool: remaining indices
    train_pool = perm[n_test:]

    # LF training: first n_lf from training pool
    lf_idx = train_pool[:n_lf]
    result['X_lf'] = X[lf_idx].astype(np.float64)
    result['y_lf'] = y_lf_all[lf_idx].astype(np.float64)

    # HF training: shuffle training pool independently, nested subsets
    rng_hf = np.random.RandomState(_subseed(name, "hf_train", seed))
    hf_perm = rng_hf.permutation(len(train_pool))[:max_hf]
    hf_idx = train_pool[hf_perm]

    for n_hf in n_hf_list:
        result[f'X_hf_{n_hf}'] = X[hf_idx[:n_hf]].astype(np.float64)
        result[f'y_hf_{n_hf}'] = y_hf_all[hf_idx[:n_hf]].astype(np.float64)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def car(seed=42):
    X, y_lf, y_hf = _load_car()
    return _generate_eng("car", X, y_lf, y_hf, seed)


def concrete(seed=42):
    X, y_lf, y_hf = _load_concrete()
    return _generate_eng("concrete", X, y_lf, y_hf, seed)


def bwb_cd(seed=42):
    data = _load_bwb()
    X, y_lf, y_hf = data['bwb_cd']
    return _generate_eng("bwb_cd", X, y_lf, y_hf, seed)


def bwb_cl(seed=42):
    data = _load_bwb()
    X, y_lf, y_hf = data['bwb_cl']
    return _generate_eng("bwb_cl", X, y_lf, y_hf, seed)


ENGINEERING_FUNCTIONS = {
    'car': car,
    'concrete': concrete,
    'bwb_cd': bwb_cd,
    'bwb_cl': bwb_cl,
}


if __name__ == "__main__":
    for name, fn in ENGINEERING_FUNCTIONS.items():
        data = fn(seed=42)
        dim = data['X_lf'].shape[1]
        max_key = max(CONFIGS[name]['n_hf_list'])
        print(f"{name:>10s}  dim={dim:2d}  "
              f"LF={data['X_lf'].shape[0]}  "
              f"HF_max={data[f'X_hf_{max_key}'].shape[0]}  "
              f"test={data['X_test'].shape[0]}")
