# 🔥 FIRE: Multi-fidelity Regression with Distribution-conditioned In-context Learning


Official implementation of **"FIRE: Multi-fidelity Regression with Distribution-conditioned In-context Learning using Tabular Foundation Models"**.

## 📋 Overview

FIRE (**F**idelity-aware **I**n-context **RE**gression) is a novel multi-fidelity regression framework that leverages Tabular Foundation Models (TFMs) to achieve zero-shot Bayesian inference under extreme data imbalance. Our key contributions include:

- 🎯 **Distribution-conditioned Residual Learning**: Augment residual model inputs with full predictive distributions (mean, variance, quantiles) from lower fidelity
- 🚀 **Zero-shot In-context Learning**: No gradient-based training required for regression tasks
- 📊 **State-of-the-art Performance**: Consistently outperforms GP-based and deep learning baselines across diverse benchmarks

## 🛠️ Installation

### Environment Setup

```bash
conda create -n FIRE python=3.12
conda activate FIRE
pip install -r requirements.txt
```

### TabPFN Setup (Required for FIRE_TFM)

Install TabPFN v2.5:
```bash
pip install tabpfn
```

### BNN baseline setup

Install MFBNN: Follow the instruction on https://github.com/bessagroup/mfbml to create a new environment and pip install the package.



### Alternative TFMs
Recommendation: Make a conda environment for each TFM. 
#### TabPFN v2 (Optional)
```bash
pip install tabpfn==2.0.6
```

#### Mitra (Optional)
```bash
pip install autogluon.tabular[mitra] 
```

### ⚙️ TabPFN Modifications for Variance/Quantile Support

To enable variance and quantile predictions with TabPFN, you need to modify the TabPFN source code:

1. **Locate the TabPFN installation**:
```bash
python -c "import tabpfn; print(tabpfn.__file__)"
```
Or install it using the GitHub developer mode
```bash
git clone https://github.com/PriorLabs/TabPFN.git --depth 1
```

2. **Edit the regressor file** (typically `tabpfn/regressor.py`):

Find the `predict` method and add support for `output_type` parameter:

```python
# Around line 150-200 in regressor.py
def _logits_to_output(
    *,
    output_type: str,
    logits: torch.Tensor,
    criterion: FullSupportBarDistribution,
    quantiles: list[float],
) -> np.ndarray | list[np.ndarray]:
    # ... existing code ...

    if output_type == "mean":
        return predictions.mean(axis=0)
    elif output_type == "variance":
        return predictions.var(axis=0)
    
    # ... existing code ...
```

Then change the output type settings at line ~100
```python
# ... existing code ...
_OUTPUT_TYPES_BASIC = ("mean", "median", "mode", "variance")
# ... existing code ...
OutputType = Literal["mean", "median", "mode", "quantiles", "full", "main", "variance"]
```


## 📦 Repository Structure

```
├── README.md
├── requirements.txt
├── example.py              # Usage examples with synthetic data
└── src/
    ├── util.py             # Data loading and encoding utilities
    ├── FIRE.py             # 🔥 FIRE (TFM) and FIRE_GP implementations
    ├── AR1.py              # AR(1) baseline
    ├── ResGP.py            # Residual GP baseline
    ├── NARGP.py            # Non-linear Autoregressive GP
    ├── ContinuAR.py        # ContinuAR GP
    ├── MFKG.py             # Multi-fidelity Knowledge Gradient GP (BoTorch)
    ├── MFBNN.py            # Multi-fidelity Bayesian Neural Network
    └── MFRNP.py            # Multi-fidelity Residual Neural Process
```

## 🚀 Quick Start

See `example.py`


## 📊 Baselines Comparison

| Method | Type | Training | Key Features |
|--------|------|----------|--------------|
| **FIRE_TFM** | TFM | Zero-shot | Distribution-conditioned residuals |
| FIRE_GP | GP | Kernel Learning | Distribution-conditioned residuals |
| AR1 | GP | Kernel Learning | Linear correlation |
| ResGP | GP | Kernel Learning | Residual learning |
| NARGP | GP | Kernel Learning | Non-linear autoregression |
| ContinuAR | GP | Kernel Learning | Continuous fidelity ODE |
| MFKG | GP | Kernel Learning | Multi-fidelity Knowledge Gradient |
| MFBNN | DL | Deep Learning | Bayesian Neural Network |
| MFRNP | DL | Deep Learning | Residual Neural Process |


## 📂 Data Availability

All 31 benchmark problems used in the paper are reproducible from the scripts under `data/`.
Synthetic problems are generated on-the-fly from analytical functions; real-world problems
are produced from CSV files shipped under `data/eng_csv/` and `data/lcbench_csv/`.

### Datasets

The benchmark suite mirrors **Table 4** of the paper:

| Category | Source | Datasets | Dim | Fidelity Levels | Generator |
|----------|--------|----------|-----|-----------------|-----------|
| Synthetic (MF2) | [`mf2`](https://mf2.readthedocs.io) library | Bohachevsky, Booth, Borehole, Branin, Currin, Forrester, Hartmann, Himmelblau, Park91a, Park91b, Six-Hump Camelback | 1–8 | 2 | `data/generate_mf2.py` |
| Synthetic (BNN) | Analytical (HD suite) | HD10, HD20, HD30, HD40, HD50 | 10–50 | 2 | `data/generate_HD.py` |
| Synthetic (GP+) | Analytical 3-fidelity | branin3f, hartmann3f | 2, 3 | 3 | `data/generate_3f.py` |
| HPO (LCBench) | LCBench CSVs | adult, Fashion-MNIST, higgs, jasmine, vehicle, volkert | 7 | 5 | `data/generate_lcbench.py` |
| Engineering | Analytical / CSV | Beam, Wing | 5, 10 | 2 | `data/generate_beam_wing.py` |
| Engineering | CSV (HOIP) | HOIP | 3 | 3 | `data/generate_hoip.py` |
| Engineering (Ours) | CSV | Concrete, BWB-CD, BWB-CL, Car | 8, 14, 14, 23 | 2 | `data/generate_engineering.py` |

Sample-size configurations follow Table 4 of the paper. For 2-fidelity problems each
generator returns one LF set and six nested HF sets at varying budgets; for 3- and
5-fidelity problems the lower fidelities have fixed sizes and the highest fidelity is
swept across six budgets. A 5- or 10-fold seeded test set is included in every dataset.

### How to Load the Data

Each generator exposes one Python function per dataset. Calling it returns a `dict`
of NumPy arrays:

```python
from data.generate_mf2 import branin

data = branin(seed=42)
X_lf,    y_lf    = data['X_lf'],    data['y_lf']      # 200 LF samples
X_hf,    y_hf    = data['X_hf_10'], data['y_hf_10']    # 10 HF samples (∈ {4,8,10,20,40,50})
X_test,  y_test  = data['X_test'],  data['y_test']     # 100 HF test samples
```

Returned keys (varies by category):

- **2-fidelity**: `X_lf`, `y_lf`, `X_hf_{n}`, `y_hf_{n}`, `X_test`, `y_test`
- **3-fidelity**: `X_lf_0`, `y_lf_0`, `X_lf_1`, `y_lf_1`, `X_hf_{n}`, `y_hf_{n}`, `X_test`, `y_test`
- **5-fidelity**: `X_lf_0..3`, `y_lf_0..3`, `X_hf_{n}`, `y_hf_{n}`, `X_test`, `y_test`

All arrays are `np.float64`; `X_*` has shape `(N, D)` and `y_*` has shape `(N,)`.
HF subsets are nested (the smallest HF budget is a prefix of the largest), and the
training pool is disjoint from the test set. Sampling is fully reproducible via the
`seed` argument.

A walkthrough that prints the shapes returned by every category lives in
[`example_generate_data.py`](example_generate_data.py); a complete training run on
Borehole data using FIRE and the baselines is in [`example.py`](example.py).

### Original Dataset Sources

Several real-world datasets are redistributed in `data/eng_csv/` and `data/lcbench_csv/`
under their original licenses. Please refer to the upstream repositories for their
license terms:

- **HOIP, Wing, Beam** — [Bostanabad-Research-Group/GP-Plus](https://github.com/Bostanabad-Research-Group/GP-Plus)
  (MIT License)
- **Car** — [Mohamedelrefaie/DrivAerNet](https://github.com/Mohamedelrefaie/DrivAerNet?tab=License-1-ov-file#readme)
  (Attribution-NonCommercial 4.0 International)
- **LCBench** (adult, Fashion-MNIST, higgs, jasmine, vehicle, volkert) — [automl/LCBench](https://github.com/automl/LCBench)
  (Apache-2.0)
- **BWB-CD, BWB-CL** — [BlendedNet Multi-Fidelity Extension Dataset](https://doi.org/10.7910/DVN/M2LDF2)
  (see linked DOI page for license terms)

Please consult each original page for the full license text and any usage restrictions
(in particular, the Car dataset is non-commercial only).

