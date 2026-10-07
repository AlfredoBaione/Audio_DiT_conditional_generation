# Smoke test for audio_dataset_npy.py on a real latent dataset.
# pytest:     AUDIO_LATENT_ROOT=<root> [AUDIO_DURATION_S=5.0] [AUDIO_NORMALIZER=<path>] pytest test_functions
#             (skipped when the dataset root does not exist)
# standalone: python test_functions/test_audio_dataset_npy.py [root] [duration_s] [normalizer_path]
#             (also saves the fitted normalizer to checkpoints_v2/ when no path is given)

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_dataset_npy import build_datasets

DEFAULT_ROOT = "./dataset_ready_cond/latents"


def check_dataset(root, duration_s=5.0, norm_path=None):
    print(f"Test AudioLatentDataset on: {root}")
    print(f"duration_s={duration_s}s | normalizer_path={norm_path}\n")

    train_dataset, val_dataset, test_dataset, normalizer, label_map, split_info = \
        build_datasets(latent_root=root, duration_s=duration_s,
                       normalizer_path=norm_path, preload=False)

    print(f"\nSplit: {split_info['file_counts']} files | "
          f"{split_info['chunk_counts']} chunks | {split_info['n_classes']} classes")
    sample, label = train_dataset[0]
    print(f"\nSingle sample:")
    print(f"  shape   : {sample.shape}  (n_frames, token_dim)")
    print(f"  label   : {label} ({train_dataset.idx_to_label[label]})")
    print(f"  Mean    : {sample.mean():.4f}")
    print(f"  Std     : {sample.std():.4f}")
    return normalizer


def test_audio_latent_dataset():
    import pytest
    root = os.environ.get("AUDIO_LATENT_ROOT", DEFAULT_ROOT)
    if not Path(root).exists():
        pytest.skip(f"dataset not found: {root} (set AUDIO_LATENT_ROOT)")
    check_dataset(root,
                  duration_s=float(os.environ.get("AUDIO_DURATION_S", 5.0)),
                  norm_path=os.environ.get("AUDIO_NORMALIZER"))


if __name__ == "__main__":
    root       = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    norm_path  = sys.argv[3] if len(sys.argv) > 3 else None

    normalizer = check_dataset(root, duration_s, norm_path)

    if norm_path is None:
        os.makedirs("checkpoints_v2", exist_ok=True)
        normalizer.save("checkpoints_v2/normalizer.pt")