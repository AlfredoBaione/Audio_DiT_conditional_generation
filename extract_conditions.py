"""
extract_conditions.py

Standalone tool to add/extract frame-level conditions (f0, chroma, rhythm,
energy, f0, ...) onto an EXISTING latents dataset produced by preprocess_stream.py.

When to use this vs preprocess_stream.py:
    * preprocess_stream.py is the PRIMARY path: it extracts conditions while it
      creates the latents, and re-running it with a new --conditions re-chunks
      from the SOURCE audio (cheap, no DAC re-encode) and merges the new
      condition. Prefer it whenever the source audio is still available.
    * extract_conditions.py is the FALLBACK for when you only have the latents
      (and optionally the per-chunk WAVs) on disk and cannot/don't want to go
      back to the source. It reads the audio from wav/<...>.wav if present, else
      DECODES the latent back to audio via DAC (lossy reconstruction, so the
      conditions are approximate -- prefer --save_wav at preprocessing time, or
      re-run preprocess_stream.py from source, for exact conditions).

Design:
    - Reads from CONDITION_CONFIG (conditions.py) which conditions are enabled;
      a per-run subset can be selected with --conditions (same registry as the
      training/preprocessing).
    - SPLIT-LESS: processes ALL the .npy latents under <dataset>/latents/
      (recursively, mirroring the source class tree). There is no train/val/test
      directory anymore; the split is decided at training time.
    - Each condition is aligned to the latent's own T (read per file), exactly
      like preprocess_stream.py, so latents and conditions stay frame-aligned.
    - Saves conditions to <dataset>/conditions/<class...>/file.npz
    - If an .npz already exists and contains ALL the requested conditions, skip.
    - If it exists but some conditions are missing, they are added (merge), so a
      new condition can be added later without recomputing the others.

Expected input:
    dataset_ready_cond/
        latents/<class...>/*.npy          <- (72, T) pre-quant DAC latents
        wav/<class...>/*.wav              <- optional (else latents are decoded)

Output:
    dataset_ready_cond/
        conditions/<class...>/*.npz
            -> contains: f0, chroma, rhythm, energy, ... (per
               CONDITION_CONFIG and the --conditions selection)

Usage:
    # all conditions enabled in CONDITION_CONFIG
    python extract_conditions.py dataset_ready_cond --device cuda

    # only a subset (modular / incremental):
    python extract_conditions.py dataset_ready_cond --conditions f0 --device cuda
    # later, add energy WITHOUT recomputing f0 (merged into the same .npz):
    python extract_conditions.py dataset_ready_cond --conditions f0 --device cuda

    python extract_conditions.py dataset_ready_cond --force    # recompute all
"""

# IRCAM: force transformers to use the PyTorch backend only. Without this,
# `transformers` (pulled in via conditions.py -> laion_clap / FAD) tries to
# import TensorFlow from the tf2.18 base and crashes on protobuf. Must be set
# BEFORE any transformers import.
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# IRCAM: redirect ALL model caches (DAC, CREPE, beat_this, HuggingFace
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
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from conditions import (
    ConditionRegistry,
    DAC_SAMPLE_RATE,
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
    z_t = torch.from_numpy(z).unsqueeze(0).to(device)  # (1, 72, T) pre-quant latents
    dac_model = get_dac_model(device)
    z_q, _, _ = dac_model.quantizer.from_latents(z_t)  # (1,72,T) -> (1,1024,T)
    wav = dac_model.decode(z_q).squeeze().cpu().numpy()
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

    # Always load what is already on disk (even with force=True) so re-computing
    # the REQUESTED conditions never drops the OTHERS (report #1).
    existing_conds = {}
    if cond_path.exists():
        try:
            data = np.load(str(cond_path))
            existing_conds = {k: data[k] for k in data.keys()}
        except Exception:
            existing_conds = {}
        if not force and required_conds.issubset(set(existing_conds.keys())):
            return False

    # Determine which conditions to (re)compute
    missing_conds = required_conds if force else (required_conds - set(existing_conds.keys()))

    if not missing_conds:
        return False

    # Determine n_frames from the latent file (the alignment reference)
    try:
        z_shape = np.load(str(latent_path), mmap_mode='r').shape
        n_frames = z_shape[1]  # (72, T)
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
        description="Frame-level condition extraction (f0, chroma, rhythm, ...) "
                     "from the preprocessed dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_conditions.py dataset_ready_cond --device cuda
    python extract_conditions.py dataset_ready_cond --conditions f0 --device cuda
    python extract_conditions.py dataset_ready_cond --conditions rhythm --device cuda
    python extract_conditions.py dataset_ready_cond --force
        """,
    )
    parser.add_argument("dataset_root", type=str,
                        help="Output directory of preprocess_stream.py "
                             "(contains latents/ and optionally wav/)")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated subset of frame conditions to "
                             "extract, e.g. 'f0' or 'f0,rhythm'. Each "
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

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    latent_root = dataset_root / "latents"
    wav_root = dataset_root / "wav"
    cond_root = dataset_root / "conditions"

    if not latent_root.exists():
        print(f"[ERROR] {latent_root} not found")
        print(f"Run first: python preprocess_stream.py <src> {dataset_root} --device cuda")
        return

    # Per-run selection of which frame conditions to extract. None -> all the
    # ones enabled in CONDITION_CONFIG; a list -> only that subset. The registry
    # is the SAME one used by preprocess_stream.py / training_cond.py.
    enabled_frame = None
    if args.conditions:
        enabled_frame = [c.strip() for c in args.conditions.split(",") if c.strip()]

    registry = ConditionRegistry(enabled_frame=enabled_frame)

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
    print(f"{'='*60}\n")

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    # SPLIT-LESS: single recursive scan over all latents (mirrors the class tree)
    npy_files = sorted(latent_root.rglob("*.npy"))
    if not npy_files:
        print(f"[ERROR] No .npy latents under {latent_root}")
        return

    print(f"{len(npy_files)} latent files to process")

    for npy_path in tqdm(npy_files, desc="Extract"):
        # Corresponding paths mirror the latent tree (no split level).
        rel = npy_path.relative_to(latent_root)          # <class...>/file.npy
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
    print(f"\n  Output:          {cond_root}/<class...>/*.npz")


if __name__ == "__main__":
    main()
