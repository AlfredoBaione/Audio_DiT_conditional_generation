# Extracts the frame conditions of an audio file into an .npz.

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ.setdefault("HF_HOME", os.path.join(_cache, "huggingface"))
    os.environ["TORCH_HOME"] = os.path.join(_cache, "torch")


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
import latent_codec as lc


_dac_model = None

def get_dac_model(device: str = "cpu"):
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
    z = np.load(str(npy_path)).astype(np.float32)
    z_t = torch.from_numpy(z).unsqueeze(0).to(device)
    if lc.active().name != "dac_44khz":
        wav = lc.decode(lc.load_model(lc.active().name, device), z_t)
        wav = wav.squeeze().cpu().numpy()
        return (wav.mean(axis=0) if wav.ndim > 1 else wav).astype(np.float32)
    dac_model = get_dac_model(device)
    z_q, _, _ = dac_model.quantizer.from_latents(z_t)
    wav = dac_model.decode(z_q).squeeze().cpu().numpy()
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    return wav.astype(np.float32)


def process_file(
    latent_path: Path,
    wav_path: Path,
    cond_path: Path,
    registry: ConditionRegistry,
    force: bool = False,
    dac_device: str = "cpu",
) -> bool:
    required_conds = set(registry.frame_names)

    existing_conds = {}
    if cond_path.exists():
        try:
            data = np.load(str(cond_path))
            existing_conds = {k: data[k] for k in data.keys()}
        except Exception:
            existing_conds = {}
        if not force and required_conds.issubset(set(existing_conds.keys())):
            return False

    missing_conds = required_conds if force else (required_conds - set(existing_conds.keys()))

    if not missing_conds:
        return False

    try:
        z_shape = np.load(str(latent_path), mmap_mode='r').shape
        n_frames = z_shape[1]
    except Exception as e:
        tqdm.write(f"  [ERR] Leggo latenti {latent_path.name}: {e}")
        return False

    if wav_path.exists():
        try:
            audio, sr = sf.read(str(wav_path), dtype='float32')
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        except Exception as e:
            tqdm.write(f"  [ERR] Leggo WAV {wav_path.name}: {e}")
            return False
    else:
        try:
            audio = decode_latent_to_wav(latent_path, device=dac_device)
            sr = lc.active().sample_rate
        except Exception as e:
            tqdm.write(f"  [ERR] Decode DAC {latent_path.name}: {e}")
            return False

    new_conds = {}
    for name in missing_conds:
        extractor = registry.frame_extractors[name]
        try:
            new_conds[name] = extractor.extract(audio, sr, n_frames)
        except Exception as e:
            tqdm.write(f"  [ERR] Extract {name} per {latent_path.name}: {e}")
            return False

    final_conds = {**existing_conds, **new_conds}

    cond_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cond_path), **final_conds)
    return True


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
    _codec = lc.activate(lc.dataset_codec(latent_root))
    print(f"[codec] {_codec.name} (from the dataset)")
    wav_root = dataset_root / "wav"
    cond_root = dataset_root / "conditions"

    if not latent_root.exists():
        print(f"[ERROR] {latent_root} not found")
        print(f"Run first: python preprocess_stream.py <src> {dataset_root} --device cuda")
        return

    enabled_frame = None
    if args.conditions:
        enabled_frame = [c.strip() for c in args.conditions.split(",") if c.strip()]

    registry = ConditionRegistry(enabled_frame=enabled_frame)

    companion = [n for n, e in registry.frame_extractors.items()
                 if getattr(e, "companion_suffixes", None)]
    if companion:
        if enabled_frame is not None:
            print(f"[ERROR] {companion} is read from the file with the same "
                  f"name next to each SOURCE audio file, which this tool never "
                  f"sees. Use: python preprocess_stream.py SRC "
                  f"{dataset_root} --conditions {','.join(companion)} "
                  f"(same SRC and parameters as the dataset; --skip_dac).")
            return
        print(f"  [note] {companion} left out: read from the source's MIDI, "
              f"only preprocess_stream.py can add it.")
        registry = ConditionRegistry(enabled_frame=[
            n for n in registry.frame_names if n not in companion])

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

    npy_files = sorted(latent_root.rglob("*.npy"))
    if not npy_files:
        print(f"[ERROR] No .npy latents under {latent_root}")
        return

    print(f"{len(npy_files)} latent files to process")

    for npy_path in tqdm(npy_files, desc="Extract"):
        rel = npy_path.relative_to(latent_root)
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
