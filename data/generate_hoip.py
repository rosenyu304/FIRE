"""
Generate 3-fidelity HOIP dataset from HOIP_noisy.csv.

The CSV contains 480 rows with 5 columns:
  col 0: input feature 1
  col 1: fidelity indicator (0, 1, 2)  — used for grouping, not as input
  col 2: input feature 2
  col 3: input feature 3 (constant 0, but kept for consistency)
  col 4: target value

Input features used: columns [0, 2, 3] → 3D input.
160 rows per fidelity level (0=lowest, 1=mid, 2=highest).

Dataset structure:
  X_lf_0, y_lf_0:     100 lowest-fidelity training samples
  X_lf_1, y_lf_1:     25 mid-fidelity training samples
  X_hf_{n}, y_hf_{n}: HF training samples for n in {2, 4, 5, 10, 20, 25}
  X_test, y_test:      40 HF test samples (disjoint from HF training)

Usage:
    from generate_hoip import hoip
    data = hoip(seed=42)
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "eng_csv", "HOIP_noisy.csv")

N_LF_0 = 100   # fidelity 0 (lowest)
N_LF_1 = 25    # fidelity 1 (mid)
N_HF_LIST = [2, 4, 5, 10, 20, 25]
N_TEST = 40

INPUT_COLS = [0, 2, 3]  # 3D input (skip fidelity column 1)
TARGET_COL = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _subseed(name, role, base_seed):
    return abs(hash((name, role, base_seed))) % (2**31)


def hoip(seed=42):
    """Generate 3-fidelity HOIP dataset.

    Sampling:
      - Fidelity 0 rows are shuffled, first N_LF_0 selected for training.
      - Fidelity 1 rows are shuffled, first N_LF_1 selected for training.
      - Fidelity 2 rows are shuffled; first max(N_HF_LIST) for HF training
        (nested), next N_TEST for test.  No overlap.
    """
    raw = np.genfromtxt(CSV_PATH, delimiter=",")

    # Split by fidelity indicator (column 1)
    f0_mask = raw[:, 1] == 0
    f1_mask = raw[:, 1] == 1
    f2_mask = raw[:, 1] == 2

    f0_data = raw[f0_mask]  # 160 rows
    f1_data = raw[f1_mask]  # 160 rows
    f2_data = raw[f2_mask]  # 160 rows

    result = {}

    # --- LF_0 training (fidelity 0) ---
    rng0 = np.random.RandomState(_subseed("hoip", "lf0_train", seed))
    idx0 = rng0.permutation(len(f0_data))[:N_LF_0]
    result['X_lf_0'] = f0_data[idx0][:, INPUT_COLS].astype(np.float64)
    result['y_lf_0'] = f0_data[idx0][:, TARGET_COL].astype(np.float64)

    # --- LF_1 training (fidelity 1) ---
    rng1 = np.random.RandomState(_subseed("hoip", "lf1_train", seed))
    idx1 = rng1.permutation(len(f1_data))[:N_LF_1]
    result['X_lf_1'] = f1_data[idx1][:, INPUT_COLS].astype(np.float64)
    result['y_lf_1'] = f1_data[idx1][:, TARGET_COL].astype(np.float64)

    # --- HF training + test (fidelity 2) ---
    rng2 = np.random.RandomState(_subseed("hoip", "hf_train_test", seed))
    max_hf = max(N_HF_LIST)
    total_needed = max_hf + N_TEST  # 25 + 40 = 65 (out of 160 available)
    idx2 = rng2.permutation(len(f2_data))[:total_needed]

    # First max_hf for training (nested)
    hf_idx = idx2[:max_hf]
    for n_hf in N_HF_LIST:
        result[f'X_hf_{n_hf}'] = f2_data[hf_idx[:n_hf]][:, INPUT_COLS].astype(np.float64)
        result[f'y_hf_{n_hf}'] = f2_data[hf_idx[:n_hf]][:, TARGET_COL].astype(np.float64)

    # Remaining for test
    test_idx = idx2[max_hf:max_hf + N_TEST]
    result['X_test'] = f2_data[test_idx][:, INPUT_COLS].astype(np.float64)
    result['y_test'] = f2_data[test_idx][:, TARGET_COL].astype(np.float64)

    return result


HOIP_FUNCTIONS = {
    'hoip': hoip,
}


if __name__ == "__main__":
    data = hoip(seed=42)
    dim = data['X_lf_0'].shape[1]
    print(f"HOIP  dim={dim}  "
          f"LF0={data['X_lf_0'].shape[0]}  "
          f"LF1={data['X_lf_1'].shape[0]}  "
          f"HF_max={data['X_hf_25'].shape[0]}  "
          f"test={data['X_test'].shape[0]}")
