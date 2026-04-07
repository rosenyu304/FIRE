"""
Example usage of FIRE and baseline methods for multi-fidelity regression.

This example demonstrates:
1. Generating synthetic multi-fidelity data
2. Training FIRE and baseline methods
3. Evaluating predictions

Note: Some methods require specific hardware (GPU) or packages.
"""

import numpy as np
import torch
from src.util import *
from data.generate_mf2 import borehole


def load_borehole_data(n_high=10, seed=42):
    """
    Load borehole 2-fidelity data via the FIRE data generation pipeline.

    Returns LF (200 samples), HF (n_high samples chosen from {4, 8, 10, 20, 40, 50}),
    and HF test (100 samples).
    """
    if n_high not in {4, 8, 10, 20, 40, 50}:
        raise ValueError(f"n_high must be one of {{4, 8, 10, 20, 40, 50}}, got {n_high}")
    data = borehole(seed=seed)
    return (data['X_lf'], data['y_lf'],
            data[f'X_hf_{n_high}'], data[f'y_hf_{n_high}'],
            data['X_test'], data['y_test'])


def main():
    print("=" * 60)
    print("FIRE: Multi-fidelity Regression Example")
    print("=" * 60)

    # Load borehole multi-fidelity data via the FIRE data pipeline
    print("\n[0] Loading borehole multi-fidelity data...")
    X_lf, y_lf, X_hf, y_hf, X_test, y_test = load_borehole_data(n_high=10, seed=42)

    print(f"    Low-fidelity:  {X_lf.shape[0]} samples")
    print(f"    High-fidelity: {X_hf.shape[0]} samples")
    print(f"    Test:          {X_test.shape[0]} samples")

    # Min-max normalize X to [0,1] and y to [0,1] across all splits
    X_all = np.concatenate([X_lf, X_hf, X_test], axis=0)
    x_min = np.min(X_all, axis=0)
    x_range = np.max(X_all, axis=0) - x_min + 1e-12
    X_lf = (X_lf - x_min) / x_range
    X_hf = (X_hf - x_min) / x_range
    X_test = (X_test - x_min) / x_range

    y_min = min(np.min(y_lf), np.min(y_hf), np.min(y_test))
    y_max = max(np.max(y_lf), np.max(y_hf), np.max(y_test))
    y_range = y_max - y_min + 1e-12
    y_lf = (y_lf - y_min) / y_range
    y_hf = (y_hf - y_min) / y_range
    y_test = (y_test - y_min) / y_range

    # Convert to tensors
    X_lf = torch.tensor(X_lf, dtype=torch.float32)
    y_lf = torch.tensor(y_lf, dtype=torch.float32)
    X_hf = torch.tensor(X_hf, dtype=torch.float32)
    y_hf = torch.tensor(y_hf, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_np = y_test.copy()

    # Encode data for general methods (with fidelity column); already normalized
    train_X, train_y, _, _ = encode_2fidelity_data(
        X_lf.numpy(), y_lf.numpy(), X_hf.numpy(), y_hf.numpy(),
        X_test.numpy(), y_test_np, preprocess_X=False, preprocess_Y=False
    )

    results = {}
    
    # =====================================================
    # Method 1: FIRE (works on GPU)
    # =====================================================
    print("\n[2] Training FIRE...")
    try:
        from src.FIRE import FIRE
        model = FIRE(X_lf, y_lf, X_hf, y_hf, device='cpu', seed=42)
        y_pred, y_var = model.predict(X_test)
        r2, nrmse, nll = get_metrics(y_test_np, y_pred, y_var)
        results['FIRE'] = {'R2': r2, 'NRMSE': nrmse, 'NLL': nll}
        print(f"    R²: {r2:.4f}, NRMSE: {nrmse:.4f}, NLL: {nll:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

    # =====================================================
    # Method 1: FIRE_GP (works on CPU)
    # =====================================================
    print("\n[2] Training FIRE_GP...")
    try:
        from src.FIRE import FIRE_GP
        model = FIRE_GP(X_lf, y_lf, X_hf, y_hf, device='cpu', seed=42)
        y_pred, y_var = model.predict(X_test)
        r2, nrmse, nll = get_metrics(y_test_np, y_pred, y_var)
        results['FIRE_GP'] = {'R2': r2, 'NRMSE': nrmse, 'NLL': nll}
        print(f"    R²: {r2:.4f}, NRMSE: {nrmse:.4f}, NLL: {nll:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

    # =====================================================
    # Method 2: ResGP
    # =====================================================
    print("\n[3] Training ResGP...")
    try:
        from src.ResGP import ResGP
        model = ResGP(X_lf, y_lf, X_hf, y_hf, device='cpu', seed=42, train_iter=100)
        y_pred, y_var = model.predict(X_test)
        r2, nrmse, nll = get_metrics(y_test_np, y_pred, y_var)
        results['ResGP'] = {'R2': r2, 'NRMSE': nrmse, 'NLL': nll}
        print(f"    R²: {r2:.4f}, NRMSE: {nrmse:.4f}, NLL: {nll:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

    # =====================================================
    # Method 3: NARGP
    # =====================================================
    print("\n[4] Training NARGP...")
    try:
        from src.NARGP import NARGP
        model = NARGP(X_lf, y_lf, X_hf, y_hf, device='cpu', seed=42, train_iter=100)
        y_pred, y_var = model.predict(X_test)
        r2, nrmse, nll = get_metrics(y_test_np, y_pred, y_var)
        results['NARGP'] = {'R2': r2, 'NRMSE': nrmse, 'NLL': nll}
        print(f"    R²: {r2:.4f}, NRMSE: {nrmse:.4f}, NLL: {nll:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

    # =====================================================
    # Method 4: AR1
    # =====================================================
    print("\n[5] Training AR1...")
    try:
        from src.AR1 import AR1
        model = AR1(train_X, train_y, fidelity_col_idx=-1, device='cpu', seed=42, train_iter=100)
        y_pred, y_var = model.predict(X_test)
        r2, nrmse, nll = get_metrics(y_test_np, y_pred, y_var)
        results['AR1'] = {'R2': r2, 'NRMSE': nrmse, 'NLL': nll}
        print(f"    R²: {r2:.4f}, NRMSE: {nrmse:.4f}, NLL: {nll:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

    # =====================================================
    # Method 5: ContinuAR
    # =====================================================
    print("\n[6] Training ContinuAR...")
    try:
        from src.ContinuAR import ContinuAR
        model = ContinuAR(train_X, train_y, fidelity_col_idx=-1, device='cpu', seed=42, train_iter=100)
        y_pred, y_var = model.predict(X_test)
        r2, nrmse, nll = get_metrics(y_test_np, y_pred, y_var)
        results['ContinuAR'] = {'R2': r2, 'NRMSE': nrmse, 'NLL': nll}
        print(f"    R²: {r2:.4f}, NRMSE: {nrmse:.4f}, NLL: {nll:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

    # =====================================================
    # Method 6: MFRNP
    # =====================================================
    print("\n[7] Training MFRNP...")
    try:
        from src.MFRNP import MFRNP
        model = MFRNP(train_X, train_y, fidelity_col_idx=-1, device='cpu', seed=42, train_iter=100)
        y_pred, y_var = model.predict(X_test)
        r2, nrmse, nll = get_metrics(y_test_np, y_pred, y_var)
        results['MFRNP'] = {'R2': r2, 'NRMSE': nrmse, 'NLL': nll}
        print(f"    R²: {r2:.4f}, NRMSE: {nrmse:.4f}, NLL: {nll:.4f}")
    except Exception as e:
        print(f"    Error: {e}")

    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "=" * 60)
    print("Summary of Results")
    print("=" * 60)
    print(f"{'Method':<15} {'R²':>10} {'NRMSE':>10} {'NLL':>10}")
    print("-" * 45)
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['R2']:>10.4f} {metrics['NRMSE']:>10.4f} {metrics['NLL']:>10.4f}")

    print("\n" + "=" * 60)
    print("Notes:")
    print("- FIRE_TFM requires GPU and TabPFN package")
    print("- MFKG requires GPU and BoTorch package")
    print("- MFBNN requires mfbml package")
    print("=" * 60)


if __name__ == "__main__":
    main()
