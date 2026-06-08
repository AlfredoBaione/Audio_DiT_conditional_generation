"""
extract_conditions.py

Estrae le condizioni frame-level (melody, chroma, rhythm, ...) dai WAV di un dataset
già preprocessato da preprocess_dataset.py.

Design:
    - Legge da CONDITION_CONFIG in conditions.py quali condizioni sono abilitate
    - Processa TUTTI i WAV in dataset_ready/wav (train/val/test)
    - Anche i WAV di train NON sono su disco → li rigeneriamo dai latenti DAC
    - Salva condizioni in dataset_ready/conditions/split/class/file.npz
    - Se un .npz esiste già e contiene TUTTE le condizioni richieste, skip
    - Se esiste ma mancano alcune condizioni, le aggiunge (merge)

Input atteso:
    dataset_ready/
        latents/train|val|test/class/*.npy
        wav/val|test/class/*.wav     ← solo val e test (train non salvato)

Output:
    dataset_ready/
        conditions/train|val|test/class/*.npz
            → contiene: melody, chroma, rhythm, ... (secondo CONDITION_CONFIG)

Uso:
    python extract_conditions.py dataset_ready
    python extract_conditions.py dataset_ready --device cuda
    python extract_conditions.py dataset_ready --force    # ricalcola tutto
"""

# IRCAM: force transformers to use the PyTorch backend only. Without this,
# `transformers` (pulled in via conditions.py -> laion_clap / FAD) tries to
# import TensorFlow from the tf2.18 base and crashes on protobuf. Must be set
# BEFORE any transformers import.
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


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
    """Lazy-load del modello DAC per decodificare latenti → WAV."""
    global _dac_model
    if _dac_model is None:
        import dac
        _dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        _dac_model.to(device)
        _dac_model.eval()
        print(f"[DAC] Modello caricato su {device}")
    return _dac_model


@torch.no_grad()
def decode_latent_to_wav(npy_path: Path, device: str = "cpu") -> np.ndarray:
    """Decodifica un file .npy di latenti DAC → waveform numpy."""
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
    Estrae le condizioni per un singolo file.
    Salva in cond_path le condizioni mancanti (merge con esistenti).

    Returns:
        True se ha fatto qualcosa (nuovo o aggiornato), False se skip
    """
    required_conds = set(registry.frame_names)

    # Controlla cosa esiste già
    existing_conds = {}
    if cond_path.exists() and not force:
        try:
            data = np.load(str(cond_path))
            existing_conds = {k: data[k] for k in data.keys()}
            # Se ha già tutte le condizioni richieste, skip
            if required_conds.issubset(set(existing_conds.keys())):
                return False
        except Exception:
            existing_conds = {}

    # Determina quali condizioni servono
    missing_conds = required_conds - set(existing_conds.keys()) if not force else required_conds

    if not missing_conds:
        return False

    # Determina n_frames dal file di latenti (è il riferimento)
    try:
        z_shape = np.load(str(latent_path), mmap_mode='r').shape
        n_frames = z_shape[1]  # (1024, T)
    except Exception as e:
        tqdm.write(f"  [ERR] Leggo latenti {latent_path.name}: {e}")
        return False

    # Carica l'audio: prima dal WAV se esiste, altrimenti decodifica dai latenti
    if wav_path.exists():
        try:
            audio, sr = sf.read(str(wav_path), dtype='float32')
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        except Exception as e:
            tqdm.write(f"  [ERR] Leggo WAV {wav_path.name}: {e}")
            return False
    else:
        # Fallback: decodifica dai latenti (caso train dove WAV non salvati)
        try:
            audio = decode_latent_to_wav(latent_path, device=dac_device)
            sr = DAC_SAMPLE_RATE
        except Exception as e:
            tqdm.write(f"  [ERR] Decode DAC {latent_path.name}: {e}")
            return False

    # Estrai SOLO le condizioni mancanti
    new_conds = {}
    for name in missing_conds:
        extractor = registry.frame_extractors[name]
        try:
            new_conds[name] = extractor.extract(audio, sr, n_frames)
        except Exception as e:
            tqdm.write(f"  [ERR] Extract {name} per {latent_path.name}: {e}")
            return False

    # Merge con esistenti
    final_conds = {**existing_conds, **new_conds}

    # Salva
    cond_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cond_path), **final_conds)
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Estrazione condizioni frame-level (melody, chroma, rhythm, ...) "
                     "dal dataset preprocessato",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
    python extract_conditions.py dataset_ready
    python extract_conditions.py dataset_ready --device cuda
    python extract_conditions.py dataset_ready --force
        """,
    )
    parser.add_argument("dataset_root", type=str,
                        help="Directory output di preprocess_dataset.py "
                             "(contiene latents/ e wav/)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device per DAC decoder (fallback train)")
    parser.add_argument("--force", action="store_true",
                        help="Ricalcola tutto anche se già esistente")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "val", "test"],
                        help="Split da processare")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    latent_root = dataset_root / "latents"
    wav_root = dataset_root / "wav"
    cond_root = dataset_root / "conditions"

    if not latent_root.exists():
        print(f"[ERROR] {latent_root} non trovata")
        print(f"Lancia prima: python preprocess_dataset.py <src> {dataset_root}")
        return

    # Determina n_classes dal train set (serve per inizializzare LabelCondition,
    # anche se qui non la usiamo direttamente)
    train_dir = latent_root / "train"
    n_classes = 0
    if train_dir.exists():
        n_classes = sum(1 for d in train_dir.iterdir() if d.is_dir())

    # Crea registry solo per condizioni frame-level
    registry = ConditionRegistry(n_classes=n_classes)

    if not registry.frame_names:
        print("[ERROR] Nessuna condizione frame-level abilitata in CONDITION_CONFIG")
        return

    print(f"{'='*60}")
    print(f"ESTRAZIONE CONDIZIONI")
    print(f"{'='*60}")
    print(f"  Dataset:         {dataset_root}")
    print(f"  Condizioni:      {registry.frame_names}")
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
            print(f"[SKIP] Split {split} non trovato")
            continue

        # Raccogli tutti i file .npy dello split
        npy_files = sorted(split_latent_dir.rglob("*.npy"))
        if not npy_files:
            continue

        print(f"\n[{split}] {len(npy_files)} file da processare")

        for npy_path in tqdm(npy_files, desc=f"Extract {split}"):
            # Path corrispondenti
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
    print(f"  Processati:      {total_processed}")
    print(f"  Skippati:        {total_skipped}")
    print(f"  Errori:          {total_errors}")
    print(f"\n  Output:          {cond_root}/train|val|test/<class>/*.npz")


if __name__ == "__main__":
    main()
