"""
preprocess_dataset.py (v6 - Universal, lazy GPU acquisition)

Unified preprocessing for the Audio DiT (conditional + unconditional).

Pipeline:
    Raw audio -> qualitative preprocessing -> chunk -> DAC encode -> latents .npy
Phases 1-4 are CPU only (ffmpeg + multiprocessing). Phase 5 (DAC) is the ONLY
phase that touches the GPU, and the GPU is acquired ONLY when phase 5 starts.

Why lazy GPU acquisition matters on IRCAM:
    Locking a GPU at launch and then running the long CPU phases 1-4 leaves the
    card LOCKED-but-IDLE for a long time. Such jobs are killed by the lab's
    GPU-idle policy (which produces orphaned Pool workers and a cascade of
    BrokenPipeError). Acquiring the GPU only inside encode_chunks_dac avoids the
    locked-and-idle window entirely. Do NOT wrap this script in python_gpu_lock
    when using --gpu_mode lock/auto: the script locks the GPU itself, at phase 5.

GPU strategy (--gpu_mode, only relevant when --device cuda):
    auto   : if manage_gpus is importable (IRCAM) -> internal lock at phase 5;
             otherwise -> use cuda directly without any lock (non-IRCAM box).
    lock   : force IRCAM internal lock at phase 5 (error if manage_gpus absent).
    direct : use cuda directly, never lock (any machine with a free GPU).
    (with --device cpu, gpu_mode is ignored.)

Train WAVs:
    --keep_train_wav keeps the train-split WAVs on disk (needed by the
    conditioned pipeline / extract_conditions.py). Without it, train WAVs are
    deleted after DAC encoding (unconditional behaviour, saves disk).

Usage:
    # Unconditional, IRCAM, GPU auto-locked only at phase 5:
    python preprocess_dataset.py SRC OUT --device cuda --max_workers 8

    # Conditioned (keep train wavs), IRCAM:
    python preprocess_dataset.py SRC OUT --device cuda --max_workers 8 --keep_train_wav

    # Non-IRCAM machine with a free GPU (no locking):
    python preprocess_dataset.py SRC OUT --device cuda --gpu_mode direct --keep_train_wav

    # CPU only:
    python preprocess_dataset.py SRC OUT --device cpu --max_workers 8 --keep_train_wav

    # CPU phases only, stop before DAC:
    python preprocess_dataset.py SRC OUT --skip_dac --max_workers 8 --keep_train_wav
"""

import os
import re
import csv
import json
import math
import random
import shutil
import argparse
import subprocess
import tempfile
import multiprocessing
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter

import numpy as np
# NOTE: torch is imported lazily inside encode_chunks_dac (after the Pools and
# after the GPU lock), never at module import time.


# ============================================================
# GLOBAL CONSTANTS (safe for multiprocessing)
# ============================================================
SUPPORTED_AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".wma", ".mpc", ".oma", ".ape", ".aac",
}


# ============================================================
# SANITIZE
# ============================================================
def sanitize_class_name(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name, flags=re.ASCII)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name if name else "unknown"


def sanitize_filename(name: str) -> str:
    name = Path(name).stem
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name, flags=re.ASCII)
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name if name else "unknown"


# ============================================================
# FFMPEG / FFPROBE
# ============================================================
def get_audio_duration(file_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(file_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_chunk_rms_db(file_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(file_path),
             "-af", "volumedetect", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in (result.stderr or "").splitlines():
            if "mean_volume" in line:
                return float(line.split("mean_volume:")[1].strip().replace(" dB", ""))
        return -100.0
    except Exception:
        return -100.0


# ============================================================
# DETECT SILENCE
# ============================================================
def detect_trim_points(file_path: str, threshold_db: float) -> tuple:
    duration = get_audio_duration(file_path)
    if duration == 0:
        return 0.0, 0.0

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(file_path),
             "-af", f"silencedetect=noise={threshold_db}dB:d=0.1",
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except Exception:
        return 0.0, duration

    stderr = result.stderr or ""
    silence_regions = []
    current_start = None

    for line in stderr.splitlines():
        if "silence_start:" in line:
            try:
                current_start = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                current_start = None
        elif "silence_end:" in line and current_start is not None:
            try:
                end_val = float(line.split("silence_end:")[1].strip().split()[0])
                silence_regions.append((current_start, end_val))
            except (ValueError, IndexError):
                pass
            current_start = None

    if current_start is not None:
        silence_regions.append((current_start, duration))

    if not silence_regions:
        return 0.0, duration

    trim_start = 0.0
    if silence_regions[0][0] < 0.05:
        trim_start = silence_regions[0][1]

    trim_end = duration
    if silence_regions[-1][1] >= duration - 0.05:
        trim_end = silence_regions[-1][0]

    return trim_start, trim_end


# ============================================================
# LOUDNESS ANALYSIS (pass 1)
# ============================================================
def analyze_loudness(file_path: str, config: dict) -> dict:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(file_path),
             "-af", (f"loudnorm=I={config['TARGET_LUFS']}:TP={config['TARGET_TP']}:"
                     f"LRA={config['TARGET_LRA']}:print_format=json"),
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        stderr = result.stderr or ""
        json_start = stderr.rfind("{")
        json_end = stderr.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            return None
        data = json.loads(stderr[json_start:json_end])
        return {
            "measured_I": data.get("input_i", "-24.0"),
            "measured_TP": data.get("input_tp", "-1.0"),
            "measured_LRA": data.get("input_lra", "7.0"),
            "measured_thresh": data.get("input_thresh", "-34.0"),
        }
    except Exception as e:
        print(f"[WARN] Loudness analysis failed: {e}")
        return None


# ============================================================
# PREPROCESSING: TRIM + NORMALIZE  (worker: phase 2)
# ============================================================
def preprocess_file(args: tuple) -> dict:
    file_path, class_name, safe_class_name, file_index, config = args
    file_path = Path(file_path)

    trim_start, trim_end = detect_trim_points(str(file_path), config["SILENCE_TRIM_DB"])
    trimmed_duration = trim_end - trim_start

    if trimmed_duration < config["MIN_CHUNK_SEC"]:
        print(f"[SKIP] Too short after trim ({trimmed_duration:.1f}s): {file_path.name}")
        return None

    loudness = analyze_loudness(str(file_path), config)

    filters = []
    if loudness:
        # linear=true => one constant gain is applied to reach the target
        # integrated loudness. With a high TARGET_LRA the dynamic range is NOT
        # compressed: internal forte/piano relationships are preserved.
        filters.append(
            f"loudnorm=I={config['TARGET_LUFS']}:TP={config['TARGET_TP']}:LRA={config['TARGET_LRA']}:"
            f"measured_I={loudness['measured_I']}:"
            f"measured_TP={loudness['measured_TP']}:"
            f"measured_LRA={loudness['measured_LRA']}:"
            f"measured_thresh={loudness['measured_thresh']}:"
            f"linear=true"
        )
    else:
        # Measurement failed: without measured_* ffmpeg can only run loudnorm in
        # dynamic (compressing) mode, which would crush the dynamics. To protect
        # them, skip loudness normalization for this file (keep original level).
        filters = []

    temp_out = Path(config["TEMP_DIR"]) / f"{safe_class_name}_{file_index:05d}.wav"

    command = [
        "ffmpeg", "-y", "-hide_banner",
        "-ss", str(trim_start),
        "-i", str(file_path),
        "-t", str(trimmed_duration),
    ]
    # Only add the -af flag if we actually have filters (the dynamics-preserving
    # fallback above can leave filters empty: in that case no loudnorm is applied).
    if filters:
        command += ["-af", ",".join(filters)]
    command += [
        "-ar", str(config["SR"]), "-ac", "1",
        "-loglevel", "error",
        str(temp_out),
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        stderr_msg = (e.stderr.decode("utf-8", errors="replace").strip()) if e.stderr else ""
        print(f"[ERROR] Preprocessing failed: {file_path.name}: {stderr_msg}")
        return None

    actual_duration = get_audio_duration(str(temp_out))
    if actual_duration < config["MIN_CHUNK_SEC"]:
        temp_out.unlink(missing_ok=True)
        return None

    return {
        "temp_path":       str(temp_out),
        "source_name":     file_path.name,
        "class_name":      class_name,
        "safe_class_name": safe_class_name,
        "duration":        actual_duration,
    }


# ============================================================
# GATHERING SOURCE FILES
# ============================================================
def get_audio_files(source_dir: str) -> list:
    tasks = []
    file_index = 0
    for class_name in sorted(os.listdir(source_dir)):
        class_path = Path(source_dir) / class_name
        if not class_path.is_dir():
            continue
        safe_class = sanitize_class_name(class_name)
        for file_path in sorted(class_path.iterdir()):
            if file_path.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                tasks.append((file_path, class_name, safe_class, file_index))
                file_index += 1
    return tasks


# ============================================================
# CHUNK PREPARATION + SPLIT + NAMING
# ============================================================
def plan_and_assign_chunks(preprocessed_files: list, split_ratios: dict,
                           seed: int, config: dict) -> list:
    rng = random.Random(seed)
    thresholds = (split_ratios["train"], split_ratios["train"] + split_ratios["val"])

    class_counters = Counter()
    all_chunks = []

    for pf in preprocessed_files:
        n_chunks = math.ceil(pf["duration"] / config["CHUNK_LENGTH_SEC"])
        safe_class = pf["safe_class_name"]

        for i in range(n_chunks):
            start_sec = i * config["CHUNK_LENGTH_SEC"]
            actual_duration = min(config["CHUNK_LENGTH_SEC"], pf["duration"] - start_sec)

            if actual_duration < config["MIN_CHUNK_SEC"]:
                continue

            r = rng.random()
            if r < thresholds[0]:
                split = "train"
            elif r < thresholds[1]:
                split = "val"
            else:
                split = "test"

            class_counters[safe_class] += 1
            seg_num = class_counters[safe_class]
            short_name = f"{safe_class}_seg{seg_num:05d}"

            # Keep WAV on disk for val/test always; for train only if requested.
            if split in ("val", "test") or config["KEEP_TRAIN_WAV"]:
                wav_out = str(Path(config["OUTPUT_DIR"]) / "wav" / split / safe_class / f"{short_name}.wav")
            else:
                wav_out = str(Path(config["TEMP_DIR"]) / f"{short_name}.wav")

            all_chunks.append({
                "temp_path":       pf["temp_path"],
                "source_name":     pf["source_name"],
                "class_name":      pf["class_name"],
                "safe_class_name": safe_class,
                "seg_index":       i,
                "start_sec":       round(start_sec, 3),
                "duration_sec":    round(actual_duration, 3),
                "split":           split,
                "short_name":      short_name,
                "wav_path":        wav_out,
            })

    return all_chunks


# ============================================================
# EXTRACT CHUNK TO WAV  (worker: phase 4)
# ============================================================
# config is passed once per worker via initializer (NOT pickled per-chunk).
_WORKER_CONFIG = None

def _chunk_worker_init(config: dict):
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def extract_chunk_to_file(chunk: dict, out_path: str, config: dict) -> bool:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", chunk["temp_path"],
        "-ss", str(chunk["start_sec"]),
        "-t", str(config["CHUNK_LENGTH_SEC"]),
        "-c:a", "pcm_s16le",
        "-ar", str(config["SR"]), "-ac", "1",
        "-loglevel", "error",
        str(out_path),
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        return False

    rms_db = get_chunk_rms_db(str(out_path))
    if rms_db < config["SILENCE_THRESH_DB"]:
        Path(out_path).unlink(missing_ok=True)
        return False

    return True


def process_chunk(chunk: dict) -> dict:
    config = _WORKER_CONFIG
    wav_path = chunk["wav_path"]

    ok = extract_chunk_to_file(chunk, wav_path, config)
    if not ok:
        return None

    chunk["rms_db"] = round(get_chunk_rms_db(wav_path), 2)
    return chunk


# ============================================================
# GPU ACQUISITION (lazy, only for phase 5)
# ============================================================
def acquire_gpu_for_dac(device: str, gpu_mode: str):
    """
    Decide and perform GPU acquisition right before phase 5, BEFORE importing
    torch. Returns the (possibly adjusted) device string to use.

    gpu_mode (only relevant if device starts with 'cuda'):
        auto   : try IRCAM manage_gpus lock; if unavailable, use cuda directly.
        lock   : force IRCAM lock (raise if manage_gpus is not installed).
        direct : never lock; use cuda directly.
    """
    if not device.startswith("cuda"):
        return device  # CPU: nothing to acquire

    def _try_ircam_lock():
        import manage_gpus as gpl  # only available on IRCAM servers
        gpu_id = gpl.get_gpu_lock(soft=False)
        print(f"[GPU] IRCAM lock acquired (device id {gpu_id}). "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
        return True

    if gpu_mode == "direct":
        print("[GPU] gpu_mode=direct -> using CUDA without locking.")
        return device

    if gpu_mode == "lock":
        try:
            _try_ircam_lock()
        except ImportError:
            raise RuntimeError(
                "gpu_mode=lock requested but manage_gpus is not available "
                "(not an IRCAM server?). Use --gpu_mode direct instead."
            )
        return device

    # auto
    try:
        _try_ircam_lock()
        print("[GPU] gpu_mode=auto -> IRCAM detected, GPU locked at phase 5.")
    except ImportError:
        print("[GPU] gpu_mode=auto -> manage_gpus absent, using CUDA directly "
              "(non-IRCAM machine).")
    return device


# ============================================================
# ENCODING DAC (phase 5; torch imported here, AFTER GPU acquisition)
# ============================================================
def encode_chunks_dac(saved_chunks: list, device: str, gpu_mode: str,
                      output_dir: str, keep_train_wav: bool):
    # Acquire the GPU (IRCAM lock or direct) BEFORE importing torch, as required
    # by the IRCAM locking mechanism. For CPU this is a no-op.
    device = acquire_gpu_for_dac(device, gpu_mode)

    # Import torch ONLY here, after the multiprocessing pools are done AND after
    # the GPU has been acquired.
    import torch
    import soundfile as sf
    try:
        import dac
    except ImportError:
        print("[ERROR] DAC not installed. Run: pip install descript-audio-codec")
        return

    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested {device} but CUDA is not available. Falling back to CPU.")
        device = "cpu"

    print(f"\n[DAC] Loading model on {device}...")
    model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(model_path)
    dac_model.to(device)
    dac_model.eval()
    print(f"[DAC] Model loaded.\n")

    n_ok = 0
    n_err = 0

    for chunk in tqdm(saved_chunks, desc="Encoding DAC"):
        wav_path = chunk["wav_path"]
        safe_class = chunk["safe_class_name"]
        short_name = chunk["short_name"]
        split = chunk["split"]

        latent_path = Path(output_dir) / "latents" / split / safe_class / f"{short_name}.npy"
        latent_path.parent.mkdir(parents=True, exist_ok=True)

        if latent_path.exists():
            if split == "train" and not keep_train_wav and Path(wav_path).exists():
                Path(wav_path).unlink(missing_ok=True)
            n_ok += 1
            continue

        try:
            audio, sr = sf.read(str(wav_path), always_2d=True)
            waveform = torch.from_numpy(audio.T).float()

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.unsqueeze(0).to(device)

            with torch.no_grad():
                x = dac_model.preprocess(waveform, sr)
                z, codes, latents, _, _ = dac_model.encode(x)

            latents_np = z.squeeze(0).cpu().numpy().astype(np.float16)
            np.save(str(latent_path), latents_np)
            n_ok += 1

        except Exception as e:
            print(f"[DAC ERROR] {short_name}: {e}")
            n_err += 1

        if split == "train" and not keep_train_wav and Path(wav_path).exists():
            Path(wav_path).unlink(missing_ok=True)

    del dac_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[DAC] Completed: {n_ok} OK, {n_err} errors")


# ============================================================
# METADATA CSV
# ============================================================
def write_metadata(chunks: list, output_path: str):
    fieldnames = [
        "short_name", "split", "class_name", "safe_class_name",
        "source_name", "seg_index", "start_sec", "duration_sec", "rms_db",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in sorted(chunks, key=lambda x: (x["safe_class_name"], x["short_name"])):
            writer.writerow({k: c.get(k, "") for k in fieldnames})


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing: raw audio -> DAC latents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--chunk_length", type=float, default=5)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--gpu_mode", type=str, default="auto",
                        choices=["auto", "lock", "direct"],
                        help="How to obtain the GPU at phase 5 (only if "
                             "--device cuda). auto: IRCAM lock if available, "
                             "else direct. lock: force IRCAM lock. direct: no "
                             "lock. Do NOT wrap in python_gpu_lock with auto/lock.")
    parser.add_argument("--skip_dac", action="store_true")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--split_train", type=float, default=0.8)
    parser.add_argument("--split_val", type=float, default=0.1)
    parser.add_argument("--split_test", type=float, default=0.1)
    parser.add_argument("--keep_train_wav", action="store_true",
                        help="Keep train .wav files in the output (conditioned).")

    args = parser.parse_args()

    split_ratios = {"train": args.split_train, "val": args.split_val, "test": args.split_test}
    assert abs(sum(split_ratios.values()) - 1.0) < 1e-6

    # Temp dir on local disk; created fresh for this run.
    base_tmp_dir = "/data/anasynth_nonbp/baione"
    try:
        os.makedirs(base_tmp_dir, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="audio_preprocess_", dir=base_tmp_dir)
    except (OSError, PermissionError):
        print(f"[WARN] Cannot access '{base_tmp_dir}'. Using system default temp dir.")
        temp_dir = tempfile.mkdtemp(prefix="audio_preprocess_")

    config = {
        "CHUNK_LENGTH_SEC": args.chunk_length,
        "MIN_CHUNK_SEC": args.chunk_length,
        "SR": args.sr,
        "OUTPUT_DIR": args.output_dir,
        "TEMP_DIR": temp_dir,
        "SILENCE_THRESH_DB": -40.0,
        "SILENCE_TRIM_DB": -35.0,
        "TARGET_LUFS": -14.0,
        "TARGET_TP": -1.0,
        "TARGET_LRA": 20.0,   # high target LRA => loudnorm (linear) does NOT
                              # compress the dynamic range; preserves the wide
                              # forte/piano range typical of classical music.
        "KEEP_TRAIN_WAV": args.keep_train_wav,
    }

    n_cores = min(args.max_workers, cpu_count())

    print(f"{'='*60}")
    print(f"PREPROCESSING AUDIO DATASET (v6 - lazy GPU)")
    print(f"{'='*60}")
    print(f"  Source:         {args.source_dir}")
    print(f"  Output:         {args.output_dir}")
    print(f"  Temp Dir:       {temp_dir}")
    print(f"  Workers:        {args.max_workers}")
    print(f"  Device:         {args.device}  (gpu_mode={args.gpu_mode})")
    print(f"  Keep Train WAV: {'YES (conditioned)' if args.keep_train_wav else 'NO (saves disk)'}")
    print(f"{'='*60}\n")

    try:
        # 1. SCANNING
        print("Phase 1/5 - Scanning the source files...")
        tasks = get_audio_files(args.source_dir)
        if not tasks:
            print(f"[ERROR] No audio file found in {args.source_dir}")
            return

        class_mapping = {}
        for _, orig, safe, _ in tasks:
            if orig not in class_mapping:
                class_mapping[orig] = safe

        # 2. PREPROCESSING (CPU)
        print(f"Phase 2/5 - Preprocessing with {args.max_workers} workers...")
        preprocess_args = [(str(fp), cn, sc, fi, config) for fp, cn, sc, fi in tasks]

        preprocessed = []
        with Pool(args.max_workers) as pool:
            for result in tqdm(pool.imap_unordered(preprocess_file, preprocess_args),
                               total=len(preprocess_args), desc="Preprocessing"):
                if result is not None:
                    preprocessed.append(result)

        if not preprocessed:
            print("[ERROR] No files preprocessed!")
            return

        # Deterministic order before split assignment (reproducible split/naming).
        preprocessed.sort(key=lambda x: (x["safe_class_name"], x["source_name"]))

        # 3. CHUNK + SPLIT
        print("Phase 3/5 - Chunk preparation + split...")
        all_chunks = plan_and_assign_chunks(preprocessed, split_ratios, args.seed, config)

        # 4. CHUNK WAV EXTRACTION (CPU; config via initializer, not per-chunk pickle)
        print(f"Phase 4/5 - Chunk extraction with {n_cores} workers...")
        saved_chunks = []
        skipped = 0

        with Pool(n_cores, initializer=_chunk_worker_init, initargs=(config,)) as pool:
            for result in tqdm(pool.imap_unordered(process_chunk, all_chunks),
                               total=len(all_chunks), desc="Extracting chunks"):
                if result is not None:
                    saved_chunks.append(result)
                else:
                    skipped += 1

        for pf in preprocessed:
            Path(pf["temp_path"]).unlink(missing_ok=True)

        os.makedirs(args.output_dir, exist_ok=True)
        write_metadata(saved_chunks, str(Path(args.output_dir) / "metadata.csv"))
        with open(str(Path(args.output_dir) / "class_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=2, ensure_ascii=False)

        print(f"  Saved chunks: {len(saved_chunks)}, discarded: {skipped}\n")

        # 5. ENCODING DAC (GPU acquired here, only now)
        if not args.skip_dac:
            print("Phase 5/5 - Encoding DAC...")
            encode_chunks_dac(saved_chunks, args.device, args.gpu_mode,
                              args.output_dir, args.keep_train_wav)
        else:
            print("Phase 5/5 - Encoding DAC skipped\n")

        print("\nCOMPLETED SUCCESSFULLY!")

    finally:
        # Always clean the temp dir, even on error/interruption.
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # spawn avoids fork-related CUDA/threading deadlocks in the worker pools.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()