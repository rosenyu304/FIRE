"""
Example: showcase one dataset from each category in FIRE/data/.
For each dataset, prints the shape of every LF/MF/HF/test array.
"""

from data.generate_mf2 import branin
from data.generate_HD import HD10
from data.generate_3f import branin3f
from data.generate_lcbench import adult
from data.generate_beam_wing import wing
from data.generate_hoip import hoip
from data.generate_engineering import concrete


def show(title, data):
    print("=" * 70)
    print(title)
    print("=" * 70)
    for k in sorted(data.keys()):
        if not k.startswith("X"):
            continue
        v = data[k]
        shape = v.shape if hasattr(v, "shape") else type(v).__name__
        print(f"  {k:>12s}  shape={shape}")
    print()


def main():
    # 1. MF2 synthetic (2-fidelity)
    show("[MF2 synthetic]  branin (2D, 2-fidelity)", branin(seed=42))

    # 2. High-dimensional analytical (2-fidelity)
    show("[High-dim BNN]  HD10 (10D, 2-fidelity)", HD10(seed=42))

    # 3. 3-fidelity analytical
    show("[3-fidelity analytical]  branin3f (2D, 3-fidelity)", branin3f(seed=42))

    # 4. LCBench HPO (5-fidelity)
    show("[LCBench HPO]  adult (7D, 5-fidelity)", adult(seed=42))

    # 5. Beam & Wing analytical (2-fidelity)
    show("[Beam/Wing analytical]  wing (10D, 2-fidelity)", wing(seed=42))

    # 6. HOIP from CSV (3-fidelity)
    show("[HOIP CSV]  hoip (3D, 3-fidelity)", hoip(seed=42))

    # 7. Engineering CSV (2-fidelity)
    show("[Engineering CSV]  concrete (8D, 2-fidelity)", concrete(seed=42))


if __name__ == "__main__":
    main()
