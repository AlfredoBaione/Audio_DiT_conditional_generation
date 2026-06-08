"""
extract_conditions.py

Extracts frame-level conditions (melody, chroma, rhythm, ...) from the WAV files
of a dataset already produced by preprocess_dataset_cond.py.

Design:
    - Reads from CONDITION_CONFIG (conditions.py) which conditions are enabled;
      a per-run subset can be selected with --conditions.
    - Processes ALL the WAVs in <dataset>/wav (train/val/test). With
      preprocess_dataset_cond.py the train WAVs are kept on disk, so the
      conditions are extracted from the ORIGINAL audio of every split.
    - Saves conditions to <dataset>/conditions/split/class/file.npz
    - If an .npz already exists and contains ALL the requested conditions, skip.
    - If it exists but some conditions are missing, they are added (merge), so a
      new condition can be added later without recomputing the others.
    - DAC decode of the latent is only a FALLBACK, used when a WAV is missing
      (should not normally happen for any split with the conditioned dataset).

Expected input:
    dataset_ready_cond/
        latents/train|val|test/class/*.npy
        wav/train|val|test/class/*.wav     <- all splits (train included)

Output:
    dataset_ready_cond/
        conditions/train|val|test/class/*.npz
            -> contains: melody, chroma, rhythm, ... (per CONDITION_CONFIG and
               the --conditions selection)

Usage:
    # all conditions enabled in CONDITION_CONFIG
    python extract_conditions.py dataset_ready_cond --device cuda

    # only a subset (modular / incremental):
    python extract_conditions.py dataset_ready_cond --conditions melody --device cuda
    # later, add rhythm WITHOUT recomputing melody (merged into the same .npz):
    python extract_conditions.py dataset_ready_cond --conditions rhythm --device cuda

    python extract_conditions.py dataset_ready_cond --force    # recompute all
"""

# IRCAM: force transformers to use the PyTorch backend only. Without this,
# `transformers` (pulled in via conditions.py -> laion_clap / FAD) tries to
# import TensorFlow from the tf2.18 base and crashes on protobuf. Must be set
# BEFORE any transformers import.
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# IRCAM: redirect ALL model caches (DAC, basic-pitch, beat_this, HuggingFace
# CLAP/CLIP) onto the machine-LOCAL disk instead of the NFS HOME. Downloading
# model weights into ~ (NFS) is slow and can hit the HOME quota. Mirrors the
# redirection already done in training_cond.py. Guarded by the local-dir check
# so the script stays portable on Windows / other machines (where the dir does
# not exist and the platform default cache is used).
_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ.setdefault("HF_HOME", os.path.join(_cache, "huggingface"))


import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from conditions import (
    ConditionRegistry,
    DAC_HOP_LENGTH, DAC_SAMPLE_RATE,
)


# ============================================================
# LATENT → WAV (fallback per train set dove i WAV non esistono)
# ============================================================

_dac_model = None

def get_dac_model(device: str = "cpu"):
    """Lazy-load the DAC model used to decode latents -> WAV (fallback only)."""
    global _dac_model
    if _dac_model is None:
        import dac
        _dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        _dac_model.to(device)
        _dac_model.eval()
        print(f"[DAC] Model loaded on {device}")
    return _dac_model


@torch.no_grad()
def decode_latent_to_wav(npy_path: Path, device: str = "cpu") -> np.ndarray:
    """Decode a .npy file of DAC latents -> numpy waveform (fallback)."""
    z = np.load(str(npy_path)).astype(np.float32)
    z_t = torch.from_numpy(z).unsqueeze(0).to(device)  # (1, 1024, T)
    dac_model = get_dac_model(device)
    wav = dac_model.decode(z_t).squeeze().cpu().numpy()
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    return wav.astype(np.float32)


# ============================================================
# PROCESS SINGLE FILE
# ============================================================

def process_file(
    latent_path: Path,
    wav_path: Path,
    cond_path: Path,
    registry: ConditionRegistry,
    force: bool = False,
    dac_device: str = "cpu",
) -> bool:
    """
    Extract the conditions for a single file.
    Saves the missing conditions to cond_path (merging with existing ones).

    Returns:
        True se ha fatto qualcosa (nuovo o aggiornato), False se skip
    """
    required_conds = set(registry.frame_names)

    # Check what already exists
    existing_conds = {}
    if cond_path.exists() and not force:
        try:
            data = np.load(str(cond_path))
            existing_conds = {k: data[k] for k in data.keys()}
            # If it already has all the requested conditions, skip
            if required_conds.issubset(set(existing_conds.keys())):
                return False
        except Exception:
            existing_conds = {}

    # Determine which conditions are missing
    missing_conds = required_conds - set(existing_conds.keys()) if not force else required_conds

    if not missing_conds:
        return False

    # Determine n_frames from the latent file (the alignment reference)
    try:
        z_shape = np.load(str(latent_path), mmap_mode='r').shape
        n_frames = z_shape[1]  # (1024, T)
    except Exception as e:
        tqdm.write(f"  [ERR] Leggo latenti {latent_path.name}: {e}")
        return False

    # Load the audio: from the WAV if present, otherwise decode from latents
    if wav_path.exists():
        try:
            audio, sr = sf.read(str(wav_path), dtype='float32')
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        except Exception as e:
            tqdm.write(f"  [ERR] Leggo WAV {wav_path.name}: {e}")
            return False
    else:
        # Fallback: decode from latents (only if the WAV is missing)
        try:
            audio = decode_latent_to_wav(latent_path, device=dac_device)
            sr = DAC_SAMPLE_RATE
        except Exception as e:
            tqdm.write(f"  [ERR] Decode DAC {latent_path.name}: {e}")
            return False

    # Extract ONLY the missing conditions
    new_conds = {}
    for name in missing_conds:
        extractor = registry.frame_extractors[name]
        try:
            new_conds[name] = extractor.extract(audio, sr, n_frames)
        except Exception as e:
            tqdm.write(f"  [ERR] Extract {name} per {latent_path.name}: {e}")
            return False

    # Merge with existing
    final_conds = {**existing_conds, **new_conds}

    # Save
    cond_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cond_path), **final_conds)
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Frame-level condition extraction (melody, chroma, rhythm, ...) "
                     "from the preprocessed dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_conditions.py dataset_ready_cond --device cuda
    python extract_conditions.py dataset_ready_cond --conditions melody --device cuda
    python extract_conditions.py dataset_ready_cond --conditions rhythm --device cuda
    python extract_conditions.py dataset_ready_cond --force
        """,
    )
    parser.add_argument("dataset_root", type=str,
                        help="Output directory of preprocess_dataset_cond.py "
                             "(contains latents/ and wav/)")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated subset of frame conditions to "
                             "extract, e.g. 'melody' or 'melody,rhythm'. Each "
                             "name must be enabled=True in CONDITION_CONFIG "
                             "(conditions.py). Default (None): extract ALL the "
                             "conditions enabled in CONDITION_CONFIG. Existing "
                             ".npz are merged, so you can add a new condition "
                             "later without recomputing the others.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for the DAC decoder (only used as a "
                             "fallback when a WAV is missing on disk)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute everything even if it already exists")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test"],
                        help="Splits to process")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    latent_root = dataset_root / "latents"
    wav_root = dataset_root / "wav"
    cond_root = dataset_root / "conditions"

    if not latent_root.exists():
        print(f"[ERROR] {latent_root} not found")
        print(f"Run first: python preprocess_dataset_cond.py <src> {dataset_root}")
        return

    # n_classes from the train split (kept for back-compat with the registry
    # signature; LabelCondition has been removed).
    train_dir = latent_root / "train"
    n_classes = 0
    if train_dir.exists():
        n_classes = sum(1 for d in train_dir.iterdir() if d.is_dir())

    # Per-run selection of which frame conditions to extract. None -> all the
    # ones enabled in CONDITION_CONFIG; a list -> only that subset.
    enabled_frame = None
    if args.conditions:
        enabled_frame = [c.strip() for c in args.conditions.split(",") if c.strip()]

    registry = ConditionRegistry(n_classes=n_classes, enabled_frame=enabled_frame)

    if not registry.frame_names:
        print("[ERROR] No frame-level condition enabled "
              "(check CONDITION_CONFIG and --conditions)")
        return

    print(f"{'='*60}")
    print(f"CONDITION EXTRACTION")
    print(f"{'='*60}")
    print(f"  Dataset:         {dataset_root}")
    print(f"  Conditions:      {registry.frame_names}")
    print(f"  Dims:            {registry.frame_cond_dims}")
    print(f"  DAC device:      {args.device}")
    print(f"  Force rebuild:   {args.force}")
    print(f"  Splits:          {args.splits}")
    print(f"{'='*60}\n")

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for split in args.splits:
        split_latent_dir = latent_root / split
        if not split_latent_dir.exists():
            print(f"[SKIP] Split {split} not found")
            continue

        # Collect all the .npy files of the split
        npy_files = sorted(split_latent_dir.rglob("*.npy"))
        if not npy_files:
            continue

        print(f"\n[{split}] {len(npy_files)} file da processare")

        for npy_path in tqdm(npy_files, desc=f"Extract {split}"):
            # Corresponding paths
            rel = npy_path.relative_to(latent_root)  # split/class/file.npy
            wav_path = wav_root / rel.with_suffix(".wav")
            cond_path = cond_root / rel.with_suffix(".npz")

            try:
                processed = process_file(
                    latent_path=npy_path,
                    wav_path=wav_path,
                    cond_path=cond_path,
                    registry=registry,
                    force=args.force,
                    dac_device=args.device,
                )
                if processed:
                    total_processed += 1
                else:
                    total_skipped += 1
            except Exception as e:
                tqdm.write(f"  [ERR] {npy_path.name}: {e}")
                total_errors += 1

    print(f"\n{'='*60}")
    print(f"COMPLETATO")
    print(f"{'='*60}")
    print(f"  Processed:       {total_processed}")
    print(f"  Skipped:         {total_skipped}")
    print(f"  Errors:          {total_errors}")
    print(f"\n  Output:          {cond_root}/train|val|test/<class>/*.npz")


if __name__ == "__main__":
    main()
