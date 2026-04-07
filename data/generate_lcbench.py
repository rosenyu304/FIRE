"""
Generate 5-fidelity LCBench HPO datasets from raw CSV files.

Fidelity levels correspond to training epochs: 42, 44, 46, 48, 50.
  F0 (lowest):  epoch 42  — 1000 samples
  F1:           epoch 44  — 750 samples
  F2:           epoch 46  — 500 samples
  F3:           epoch 48  — 250 samples
  HF (highest): epoch 50  — {20, 40, 50, 100, 200, 250} samples
  Test:         epoch 50  — 100 samples

Features (7): batch_size, learning_rate, max_dropout, max_units,
              momentum, num_layers, weight_decay
Target: val_accuracy

Requires raw CSVs in RAW_DATA_DIR (default: same directory as this script, under lcbench_csv/).

Usage:
    from generate_lcbench import adult, higgs
    data = adult(seed=42)
    X_lf_0, y_lf_0 = data['X_lf_0'], data['y_lf_0']  # 1000 samples, epoch 42
    X_hf, y_hf = data['X_hf_50'], data['y_hf_50']      # 50 samples, epoch 50
    X_test, y_test = data['X_test'], data['y_test']      # 100 samples, epoch 50
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "lcbench_csv")

EPOCHS = [42, 44, 46, 48, 50]  # F0, F1, F2, F3, HF
N_LF = [1000, 750, 500, 250]   # samples for F0, F1, F2, F3
N_HF_LIST = [20, 40, 50, 100, 200, 250]
N_TEST = 100

FEATURE_COLS = [
    "batch_size", "learning_rate", "max_dropout", "max_units",
    "momentum", "num_layers", "weight_decay",
]
TARGET_COL = "val_accuracy"

DATASETS = ["adult", "Fashion-MNIST", "higgs", "jasmine", "vehicle", "volkert"]

# Training pool: first 1600 rows; Test pool: rows 1600 onwards
TRAIN_POOL_SIZE = 1600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_csv(dataset_name, epoch):
    """Load one epoch's worth of rows from a merged LCBench CSV.

    Each CSV is a single file containing all epochs, with an 'epoch' column
    as the first column.  This function filters to the requested epoch and
    returns the cleaned features and target.
    """
    path = os.path.join(RAW_DATA_DIR, f"{dataset_name}.csv")
    df = pd.read_csv(path)
    df = df[df["epoch"] == epoch].reset_index(drop=True)
    df = df[FEATURE_COLS + [TARGET_COL]]
    # num_layers can be boolean string in some CSVs
    df["num_layers"] = df["num_layers"].apply(
        lambda x: 1 if x is True or str(x).lower() == "true" else
                  (0 if x is False or str(x).lower() == "false" else x)
    )
    df = df.apply(lambda x: pd.to_numeric(x, errors="coerce"))
    df = df.dropna()
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)
    return X, y


def _subseed(name, role, base_seed):
    return abs(hash((name, role, base_seed))) % (2**31)


def _generate_lcbench(dataset_name, seed=42):
    """Generate 5-fidelity data for one LCBench dataset.

    Sampling strategy:
      - Training pool = first TRAIN_POOL_SIZE rows of each epoch CSV.
      - Test pool = remaining rows of epoch-50 CSV.
      - Within each pool, rows are shuffled with a deterministic seed,
        then the first N are selected.  This avoids data leak.
    """
    result = {}

    # --- Load all epoch CSVs ---
    all_X = {}
    all_y = {}
    for epoch in EPOCHS:
        X, y = _load_csv(dataset_name, epoch)
        all_X[epoch] = X[:TRAIN_POOL_SIZE]
        all_y[epoch] = y[:TRAIN_POOL_SIZE]

    # --- LF training data (F0-F3) ---
    for i, (epoch, n_lf) in enumerate(zip(EPOCHS[:4], N_LF)):
        rng = np.random.RandomState(_subseed(dataset_name, f"lf{i}_train", seed))
        idx = rng.permutation(len(all_X[epoch]))[:n_lf]
        result[f'X_lf_{i}'] = all_X[epoch][idx]
        result[f'y_lf_{i}'] = all_y[epoch][idx]

    # --- HF training data (epoch 50, nested subsets) ---
    rng_hf = np.random.RandomState(_subseed(dataset_name, "hf_train", seed))
    max_hf = max(N_HF_LIST)
    hf_X = all_X[50]
    hf_y = all_y[50]
    hf_idx = rng_hf.permutation(len(hf_X))[:max_hf]

    for n_hf in N_HF_LIST:
        result[f'X_hf_{n_hf}'] = hf_X[hf_idx[:n_hf]]
        result[f'y_hf_{n_hf}'] = hf_y[hf_idx[:n_hf]]

    # --- HF test data (epoch 50, from test pool) ---
    X_test_full, y_test_full = _load_csv(dataset_name, 50)
    X_test_pool = X_test_full[TRAIN_POOL_SIZE:]
    y_test_pool = y_test_full[TRAIN_POOL_SIZE:]

    rng_test = np.random.RandomState(_subseed(dataset_name, "test", seed))
    test_idx = rng_test.permutation(len(X_test_pool))[:N_TEST]
    result['X_test'] = X_test_pool[test_idx]
    result['y_test'] = y_test_pool[test_idx]

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def adult(seed=42):
    return _generate_lcbench("adult", seed)

def fashion_mnist(seed=42):
    return _generate_lcbench("Fashion-MNIST", seed)

def higgs(seed=42):
    return _generate_lcbench("higgs", seed)

def jasmine(seed=42):
    return _generate_lcbench("jasmine", seed)

def vehicle(seed=42):
    return _generate_lcbench("vehicle", seed)

def volkert(seed=42):
    return _generate_lcbench("volkert", seed)


LCBENCH_FUNCTIONS = {
    'adult': adult,
    'Fashion-MNIST': fashion_mnist,
    'higgs': higgs,
    'jasmine': jasmine,
    'vehicle': vehicle,
    'volkert': volkert,
}


if __name__ == "__main__":
    for name, fn in LCBENCH_FUNCTIONS.items():
        data = fn(seed=42)
        dim = data['X_lf_0'].shape[1]
        print(f"{name:>15s}  dim={dim}  "
              f"F0={data['X_lf_0'].shape[0]}  "
              f"F1={data['X_lf_1'].shape[0]}  "
              f"F2={data['X_lf_2'].shape[0]}  "
              f"F3={data['X_lf_3'].shape[0]}  "
              f"HF_max={data['X_hf_250'].shape[0]}  "
              f"test={data['X_test'].shape[0]}")
