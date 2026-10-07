# Smoke test for conditions.py: ConditionRegistry and CLAP text encoding.
# Downloads the CLAP weights on first run.
# Run with `pytest test_functions` or `python test_functions/test_conditions.py`.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from conditions import ConditionRegistry


def test_condition_registry():
    print("=" * 60)
    print("Test ConditionRegistry (CLAP-based, no label)")
    print("=" * 60)

    reg = ConditionRegistry()
    print(reg)
    print(f"\nFrame cond dims:    {reg.frame_cond_dims}")
    print(f"Global cond configs: {reg.global_cond_configs}")

    print("\n--- Test CLAP text encoding (single prompt) ---")
    if "text" in reg.global_extractors:
        t = reg.global_extractors["text"]
        emb = t.encode_text("baroque sacred music")
        print(f"  Embedding shape: {emb.shape}, "
              f"norm: {np.linalg.norm(emb):.4f} (expected ~1.0)")
        t.unload()
        print("  CLAP offloaded from GPU.")


if __name__ == "__main__":
    test_condition_registry()