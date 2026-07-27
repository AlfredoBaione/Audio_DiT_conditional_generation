"""
preprocess_stream.py

Streaming preprocessing for the conditioned Audio DiT, built on the CHUNKING
PHILOSOPHY of the supervisor's `datasets.py` (ChunkedAudioFileDataset):

    load audio -> mono -> resample -> trim -> silence/peak filter -> clip
                -> stream fixed-length chunks
                -> encode EACH chunk with DAC on the GPU immediately
                -> save only the latent (.npy); WAV and conditions are OPTIONAL

The whole point is that a full-file WAV is NEVER written to disk: a chunk is
produced in memory and encoded on the spot, so the on-disk footprint is just the
latents (and, only if asked, per-chunk WAV / conditions). This mirrors the
supervisor's stream-then-consume design, but for OFFLINE latent creation instead
of training-time streaming.

WHY NOT reuse ChunkedAudioFileDataset directly?
    That class is an *IterableDataset* meant for training: it is infinite,
    shuffled (reservoir buffer), sharded across DDP ranks + workers, and it
    yields bare chunk tensors WITHOUT provenance (source filename + chunk index).
    Offline preprocessing needs that provenance for (a) stable, deterministic
    output naming and (b) incremental condition addition. So its two PURE steps
    -- `_load_audio` and `_stream_chunks` -- are ported here verbatim in
    behaviour and driven file by file in deterministic order:
      * same offset math: num_offsets = (length - chunk_length)//hop_length + 1,
        offset_i = i * hop_length, hop_length = chunk_length - chunk_overlap;
      * same keep_num_chunks_per_file pruning (linspace bin centers);
      * same file gates (min_duration, peak/silence ratios, clip) and per-chunk
        RMS/peak gates.
    This IS the supervisor's chunking procedure, with no `musicbox` dependency,
    so it runs identically on IRCAM and on the Windows VM. Deliberate deviations
    are documented on VendoredChunker (min_duration on the time axis; the
    deterministic sub-selection path only).

NO TRAIN/VAL/TEST SPLIT is produced here (by design): the split is a training-time
concern and will be handled on the training side. The output MIRRORS THE SOURCE
DIRECTORY TREE verbatim (the class folders of the raw dataset are reproduced), so
training can use those same directories as the classes to split on:

    OUT/
        latents/<same/subdir/as/source>/<name>.npy   <- (72, T) float32, DAC
        wav/<same/subdir/as/source>/<name>.wav        <- only with --save_wav
        conditions/<same/subdir/as/source>/<name>.npz <- only with --conditions
        global_conditions/<text|image>/<class>.npy    <- only with --global
        dataset_meta.json                             <- chunk params, re-run safety

    e.g. SRC/rock/song.mp3  ->  OUT/latents/rock/song__c0000.npy, ...__c0001.npy
    Source directory names are preserved verbatim; only per-chunk file names are
    sanitized. The chunk index cNNNN is a positional counter over the kept chunks
    of that source file.

INCREMENTAL / IDEMPOTENT
    Every stage is idempotent per chunk, so you can:
      1) run once with latents + some conditions:
           python preprocess_stream.py SRC OUT --conditions melody
      2) later add another condition WITHOUT recomputing latents / other conds:
           python preprocess_stream.py SRC OUT --conditions energy
         Because chunking is deterministic, re-running regenerates the exact same
         chunk audio in memory; latents already on disk are NOT re-encoded (DAC is
         skipped), and only the MISSING condition is extracted and merged into the
         existing .npz. The source audio must still be reachable on the re-run.

Comparison with preprocess_dataset.py (ffmpeg pipeline):
    By default this script does supervisor-style loading only (mono/resample/
    trim-to-duration/clip + optional silence & peak FILTERING), with NO loudness
    normalization. Your previous acoustic treatment is available VERBATIM behind
    --acoustic_rules: silence edge-trim + constant-gain loudness normalization
    (true-peak-capped, never compressing) + per-channel stereo split. It is
    applied per source file in-stream (transient temp WAV, never the whole
    dataset). If you switch loudness on/off, latent statistics change, so the
    latent normalizer and the FD-DAC reference cache MUST be refit in a fresh
    cache dir; dataset_meta.json records the acoustic params and refuses to mix
    differently-normalized latents in one OUT dir.

Usage:
    # latents only, GPU:
    python preprocess_stream.py SRC OUT --device cuda

    # latents + per-chunk melody + energy conditions:
    python preprocess_stream.py SRC OUT --device cuda --conditions melody,energy

    # with YOUR acoustic treatment (stereo split + constant-gain loudnorm + trim):
    python preprocess_stream.py SRC OUT --device cuda --acoustic_rules

    # also keep the per-chunk WAV (for FAD / listening):
    python preprocess_stream.py SRC OUT --device cuda --save_wav

    # add f0 (CREPE) later, incrementally, without recomputing latents:
    python preprocess_stream.py SRC OUT --device cuda --conditions f0

    # large dataset: parallel CPU workers + batched DAC encode
    python preprocess_stream.py SRC OUT --device cuda --conditions melody,energy \
        --num_workers 8 --batch_size 16

Throughput:
    The CPU work (audio load, acoustic ffmpeg, chunking, condition extraction,
    WAV writing) runs on DataLoader workers (--num_workers), in parallel with the
    GPU, which batches the DAC encode (--batch_size chunks per forward). Workers
    write conditions/WAV straight to disk; the main process only encodes+saves
    the latents. Conditions are frame-aligned to the constant DAC latent length T
    (discovered once), so workers never wait on the GPU. num_workers=0 (default)
    runs the identical path in a single process.

MULTIPROCESSING SAFETY
    Workers use the "spawn" start method by default, prefetch only one batch and
    transfer an owned clone of each chunk (never a view backed by the complete
    source file). Torch/BLAS/FFmpeg threads are capped to avoid multiplying native
    thread pools by --num_workers. Latents and sidecars are published with an
    atomic same-directory replace, so a crash cannot turn a partial file into a
    valid-looking cache entry. For a large first run start conservatively with:
        --num_workers 4 --loader_batch_size 8 --batch_size 16 \
        --prefetch_factor 1
"""

import os

# IRCAM: transformers must use the torch backend only (conditions.py may pull in
# CLAP/CLIP via transformers); set before any transformers import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# A spawned worker imports this module from scratch, so native libraries see
# these limits before NumPy/Torch are imported. Users can override them in the
# shell, but one thread is the safe default: otherwise N workers each create a
# full OpenMP/BLAS pool, while every worker also launches FFmpeg subprocesses.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# IRCAM: redirect model caches (DAC, basic-pitch, HuggingFace) to the local disk
# instead of the NFS HOME. Guarded so the script stays portable off-IRCAM.
_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ.setdefault("HF_HOME", os.path.join(_cache, "huggingface"))

import re
import json
import math
import shutil
import hashlib
import argparse
import subprocess
import tempfile
import faulthandler
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

# torch is imported at module level so the IterableDataset subclass below is a
# real, picklable, module-level class (needed for DataLoader workers on 'spawn').
# Guarded so `--help` still works on a machine without torch (the script can't
# actually run without it, but reading the help shouldn't require it).
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import IterableDataset, DataLoader
    _TORCH_OK = True
except Exception:                       # pragma: no cover
    torch = None
    F = None
    IterableDataset = object            # lets the class definition parse
    DataLoader = None
    _TORCH_OK = False


def _identity_collate(batch):
    """Return worker items as-is; main accumulates IPC batches into DAC batches."""
    return batch


def _worker_init_fn(worker_id: int):
    """Keep every spawned CPU worker small and make native crashes diagnosable."""
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before inter-op work starts. A fresh
        # spawn normally succeeds; keeping the intra-op cap is still sufficient.
        pass
    print(f"[worker {worker_id}] pid={os.getpid()} torch_threads=1", flush=True)


def _atomic_save_npy(path: Path, array):
    """Close a complete sibling temp file, then atomically publish it as .npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_save_npz(path: Path, arrays: dict):
    """Atomic equivalent of np.savez_compressed for condition sidecars."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict):
    """Atomic UTF-8 JSON publication for metadata and the source manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_save_wav(path: Path, audio, sr: int):
    """Write a complete WAV beside its destination before os.replace()."""
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".part.wav", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        sf.write(str(tmp), audio, sr)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _latent_file_is_valid(path: Path, n_frames: int) -> bool:
    """A resume skips only a readable float32 latent with geometry (72, T)."""
    path = Path(path)
    if not path.exists():
        return False
    z = None
    try:
        z = np.load(str(path), mmap_mode="r", allow_pickle=False)
        valid = z.shape == (72, int(n_frames)) and z.dtype == np.dtype(np.float32)
    except Exception as e:
        print(f"[resume] invalid latent will be regenerated: {path} ({e})")
        return False
    finally:
        if z is not None:
            mm = getattr(z, "_mmap", None)
            if mm is not None:
                mm.close()
    if not valid:
        print(f"[resume] invalid latent will be regenerated: {path} "
              f"(expected shape=(72,{n_frames}), dtype=float32)")
    return valid


SUPPORTED_AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".wma", ".mpc", ".oma", ".ape", ".aac",
}

PREPROCESS_BUILD = "2026-07-18-multiprocessing-safe-v1"


# ============================================================
# NAMING (pure, filesystem-safe) -- ported from preprocess_dataset.py
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
# FILE SCAN -> (path, rel_parent, leaf_class)
# ============================================================
def scan_audio_files(
    source_dir: str,
    single_class: bool = False,
    class_name: Optional[str] = None,
) -> List[Tuple[Path, Path, str, str, str]]:
    """
    Walk source_dir and return [(path, rel_parent, leaf_class, src_hash, rel_posix),
    ...] in a deterministic order.

    `rel_parent` is the source subdirectory of the file RELATIVE to source_dir.
    Output mirrors it verbatim, so the encoded dataset reproduces THE SAME
    directory tree as the raw dataset (e.g. SRC/rock/song.mp3 -> latents/rock/).
    Directory names are preserved exactly (not sanitized) so they match the
    source; only the per-chunk FILE names are sanitized.

    `leaf_class` is the last component of rel_parent -- the class label used for
    the (per-class) global-condition sidecars.

    `src_hash` is a short, deterministic hash of the file's path relative to
    source_dir (posix-normalized, so it is identical on Windows and Linux). It is
    embedded in the chunk file name as `<stem>_<src_hash>__c<idx>` so that two
    different sources whose sanitized stems collide (e.g. "A-B.wav" and "A B.wav"
    both -> "a_b") get DISTINCT output names AND distinct source-group keys
    (the split groups by the pre-"__" token, i.e. `<stem>_<src_hash>`), avoiding
    both silent overwrites and cross-source leakage (report #7).

    With --single_class, every file is placed under one directory
    (class_name or the source basename).
    """
    src = Path(source_dir)
    files = sorted(
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
    )
    out: List[Tuple[Path, Path, str, str, str]] = []
    for p in files:
        if single_class:
            rel_parent = Path(class_name or src.name)
        else:
            rp = p.parent.relative_to(src)
            # files sitting directly in source_dir have no class subfolder
            rel_parent = rp if str(rp) != "." else Path(class_name or src.name)
        leaf_class = rel_parent.name
        rel_posix = p.relative_to(src).as_posix()
        src_hash = hashlib.sha1(rel_posix.encode("utf-8")).hexdigest()[:8]
        out.append((p, rel_parent, leaf_class, src_hash, rel_posix))
    return out


# ============================================================
# SOURCE MANIFEST -- what each source looked like, and what it produced
# ============================================================
#
# dataset_meta.json records WITH WHICH PARAMETERS the dataset was built. The
# manifest records WHAT IT WAS BUILT FROM: for every source, its size/mtime at
# the time and the chunk names it produced. Without it two failure modes are
# invisible:
#   * a source EDITED in place keeps the same output names (src_hash is a hash of
#     the PATH), so the existing latents are silently kept and the dataset holds
#     the OLD audio;
#   * a source DELETED or renamed leaves its latents behind, and they still feed
#     the split / normalizer / training while corresponding to nothing.
# Identity is (size, mtime_ns), which comes free from the stat() the scan already
# does; a full content digest would cost a re-read of the whole corpus (~80 min
# for SHS over the network) and is not needed to catch a real edit.
# NOTE: this is deliberately NOT folded into src_hash -- that hash is part of the
# chunk FILE NAMES, so making it content-dependent would rename every output and
# force a full rebuild of existing datasets.

MANIFEST_NAME = "source_manifest.json"


def _source_identity(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def load_source_manifest(out_root: Path) -> dict:
    p = out_root / MANIFEST_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("sources", {})
    except Exception:
        print(f"[manifest] WARNING: {p} is unreadable, ignoring it.")
        return {}


def audit_sources(prev: dict, files) -> Tuple[list, list]:
    """Compare the manifest against what is on disk NOW.

    Returns (changed, removed):
      changed -- sources whose bytes differ from when they were encoded: their
                 existing outputs are STALE (and would be silently skipped);
      removed -- sources that no longer exist: their outputs are ORPHANS.
    """
    changed, current = [], {}
    for p, _rp, _leaf, _h, rel in files:
        try:
            current[rel] = _source_identity(p)
        except OSError:
            continue
    for rel, ident in current.items():
        old = prev.get(rel)
        if old and (old.get("size"), old.get("mtime_ns")) != (ident["size"],
                                                             ident["mtime_ns"]):
            changed.append(rel)
    removed = [rel for rel in prev if rel not in current]
    return sorted(changed), sorted(removed)


def write_source_manifest(out_root: Path, prev: dict, produced: dict,
                          removed: list, prune: bool):
    """Merge this run's observations into the manifest and persist it.

    `produced` is {rel_source: {"size","mtime_ns","chunks":[...]}} collected from
    the workers. Sources not seen in this run keep their previous entry, unless
    they were pruned.
    """
    merged = dict(prev)
    if prune:
        for rel in removed:
            merged.pop(rel, None)
    merged.update(produced)
    p = out_root / MANIFEST_NAME
    _atomic_write_json(p, {"sources": merged})
    return len(merged)


def prune_orphans(out_root: Path, prev: dict, removed: list) -> int:
    """Delete the outputs of sources that no longer exist. Only touches files the
    manifest attributes to those sources -- never a blind directory sweep."""
    latents = out_root / "latents"
    conds = out_root / "conditions"
    wavs = out_root / "wav"
    n = 0
    for rel in removed:
        for chunk in prev.get(rel, {}).get("chunks", []):
            for root, ext in ((latents, ".npy"), (conds, ".npz"), (wavs, ".wav")):
                f = root / f"{chunk}{ext}"
                if f.exists():
                    f.unlink()
                    n += 1
    return n


# ============================================================
# CHUNKING BACKEND (self-contained port of the supervisor's two pure steps)
# ============================================================
class VendoredChunker:
    """
    Behaviourally-aligned port of ChunkedAudioFileDataset._load_audio and
    ._stream_chunks (supervisor datasets.py). Kept dependency-free (no musicbox,
    no DDP) and provenance-aware: yields (chunk_index, chunk[1, L]) per file.

    Differences from the supervisor, all deliberate and flagged:
      * min_duration uses wav.shape[-1] (samples), NOT wav.shape[0]. In the
        supervisor, _load_audio applies mono FIRST (wav -> [1, t]) and then
        checks `wav.shape[0] / sr` -- but shape[0] is the CHANNEL count (== 1
        after mono), so that gate is effectively `1/sr < min_duration`, i.e. it
        discards ALL files whenever min_duration is set. Here we use the time
        axis so the gate is correct. (Worth reporting upstream.)
      * chunk sub-selection (keep_num_chunks_per_file) uses ONLY the deterministic
        linspace-centers path -- never the RNG path -- so re-runs are reproducible.
    """

    def __init__(
        self,
        chunk_length: int,
        chunk_overlap: int = 0,
        duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        audio_sr: int = 44100,
        mono: bool = True,
        silence_threshold: Optional[float] = None,
        max_silence_ratio: Optional[float] = None,
        peak_threshold: Optional[float] = 1.0,
        max_peak_ratio: Optional[float] = None,
        clip: bool = True,
        chunk_min_rms_threshold: Optional[float] = None,
        chunk_min_peak_threshold: Optional[float] = None,
        pad_and_keep_last_chunk: bool = False,
        pad_value: float = 0.0,
        keep_num_chunks_per_file: Optional[int] = None,
    ):
        assert chunk_length > 0
        assert 0 <= chunk_overlap < chunk_length
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        self.hop_length = chunk_length - chunk_overlap
        self.duration = duration
        self.min_duration = min_duration
        self.audio_sr = audio_sr
        self.mono = mono
        self.silence_threshold = silence_threshold
        self.max_silence_ratio = max_silence_ratio
        self.peak_threshold = peak_threshold if peak_threshold is not None else 1.0
        self.max_peak_ratio = max_peak_ratio
        self.clip = clip
        self.chunk_min_rms_threshold = chunk_min_rms_threshold
        self.chunk_min_peak_threshold = chunk_min_peak_threshold
        self.pad_and_keep_last_chunk = pad_and_keep_last_chunk
        self.pad_value = pad_value
        self.keep_num_chunks_per_file = keep_num_chunks_per_file

    def load_audio(self, filepath):

        import numpy as np
        # Decoding ladder (no torchaudio/torchcodec: needs FFmpeg DLLs and breaks
        # on bare Windows):
        #   1. soundfile/libsndfile -- wav/flac/ogg, no subprocess. This is the HOT
        #      path: with --acoustic_rules the input is always an ffmpeg-made temp
        #      WAV, so we never leave this branch.
        #   2. ffmpeg -- for what libsndfile cannot open (mp3/m4a/...). NOT librosa:
        #      librosa silently falls back to `audioread`, which is slow, prints a
        #      UserWarning per file, and is removed in librosa 0.11.
        # A file neither can decode is REPORTED and skipped, instead of being
        # hidden by a broad except (a corrupt file must not look like an
        # unsupported format).
        try:
            import soundfile as sf
            data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)  # (t, c)
            wav = torch.from_numpy(data.T.copy())                                # (c, t)
        except Exception as e:
            wav, sr = _ff_decode(filepath)
            if wav is None:
                print(f"[skip] cannot decode {Path(filepath).name} "
                      f"(soundfile: {type(e).__name__}; ffmpeg failed too)")
                return None

        if self.mono:
            wav = wav.mean(dim=0, keepdim=True)           # [1, t]

        # min_duration on the TIME axis (see class docstring re: supervisor bug).
        if self.min_duration is not None and (wav.shape[-1] / sr < self.min_duration):
            return None

        # discard files with too many peaks (likely compression artefacts)
        if self.max_peak_ratio is not None and self.peak_threshold is not None:
            peak_ratio = (wav.abs() >= self.peak_threshold).float().mean().item()
            if peak_ratio > self.max_peak_ratio:
                return None

        # discard files that are mostly silence (likely empty)
        if self.max_silence_ratio is not None and self.silence_threshold is not None:
            silence_ratio = (wav.abs() < self.silence_threshold).float().mean().item()
            if silence_ratio >= self.max_silence_ratio:
                return None

        if self.audio_sr is not None and sr != self.audio_sr:
            import librosa
            y = librosa.resample(wav.cpu().numpy(), orig_sr=sr,
                                 target_sr=self.audio_sr, axis=-1)
            wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            sr = self.audio_sr

        if self.duration is not None:
            wav = wav[..., :int(self.duration * sr)]

        # residual peaks are likely clicks -> clip
        if self.clip:
            wav = wav.clamp(-self.peak_threshold, self.peak_threshold)

        return wav

    def _chunk_offsets(self, length: int) -> List[int]:
        if length < self.chunk_length:
            return []
        num_offsets = (length - self.chunk_length) // self.hop_length + 1
        if (self.keep_num_chunks_per_file is not None
                and self.keep_num_chunks_per_file < num_offsets):
            import torch
            bins = torch.linspace(0, num_offsets, steps=self.keep_num_chunks_per_file + 1)
            centers = ((bins[:-1] + bins[1:]) / 2).long().tolist()
            indices = sorted(set(int(c) for c in centers))
        else:
            indices = list(range(num_offsets))
        return [i * self.hop_length for i in indices]

    def _keep_chunk(self, chunk) -> bool:
        # Keep-if-loud logic (report #8): a chunk is kept when its RMS is high
        # ENOUGH *or* its peak is high enough -- so an isolated transient/attack
        # (low RMS, high peak) is not discarded, matching the CLI help. If both
        # thresholds are None -> keep everything.
        rms_thr = self.chunk_min_rms_threshold
        peak_thr = self.chunk_min_peak_threshold
        if rms_thr is None and peak_thr is None:
            return True
        rms_ok = (rms_thr is not None) and bool((chunk ** 2).mean().sqrt() > rms_thr)
        peak_ok = (peak_thr is not None) and bool(chunk.abs().max() > peak_thr)
        return rms_ok or peak_ok

    def iter_file_chunks(self, filepath) -> Iterator[Tuple[int, "object"]]:

        import torch.nn.functional as F
        wav = self.load_audio(filepath)
        if wav is None:
            return
        length = wav.shape[-1]
        offsets = self._chunk_offsets(length)

        kept = 0
        last_end = 0
        for off in offsets:
            chunk = wav[..., off:off + self.chunk_length]
            last_end = off + self.hop_length
            if self._keep_chunk(chunk):
                yield kept, chunk
                kept += 1

        # optional padded last chunk
        if self.pad_and_keep_last_chunk:
            if not offsets and length > 0:
                pad = self.chunk_length - length
                chunk = F.pad(wav, (0, pad), value=self.pad_value)
                if self._keep_chunk(chunk):
                    yield kept, chunk
            elif offsets and last_end < length:
                pad = self.chunk_length - (length - last_end)
                chunk = F.pad(wav[..., last_end:], (0, pad), value=self.pad_value)
                if self._keep_chunk(chunk):
                    yield kept, chunk


# ============================================================
# DAC ENCODER (loaded once, encodes a chunk tensor -> (72, T) numpy)
# ============================================================
class DACEncoder:
    def __init__(self, device: str = "cuda"):

        import dac
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[DAC] CUDA not available -> CPU")
            device = "cpu"
        self.device = device
        self.torch = torch
        print(f"[DAC] loading 44khz model on {device} ...")
        self.model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        self.model.to(device)
        self.model.eval()
        print("[DAC] model loaded.")

    def encode(self, chunk, sr: int):
        """chunk: torch [1, L] mono -> latents numpy (72, T) float32."""
        return self.encode_batch([chunk], sr)[0]

    def encode_batch(self, chunks, sr: int):
        """
        chunks: list of torch tensors [L] or [1, L], ALL the same length L.
        Returns a list of (72, T) float32 numpy arrays, one per chunk. A single
        batched DAC forward keeps the GPU busy instead of one call per chunk.
        """
        import numpy as np
        torch = self.torch
        mats = []
        for c in chunks:
            w = c
            if w.dim() == 1:
                w = w.unsqueeze(0)       # [1, L]
            if w.dim() == 2:
                w = w.unsqueeze(0)       # [1, 1, L]
            mats.append(w)
        batch = torch.cat(mats, dim=0).to(self.device)   # [B, 1, L]
        with torch.no_grad():
            x = self.model.preprocess(batch, sr)
            _z, _codes, latents, _, _ = self.model.encode(x)
        latents = latents.cpu().numpy().astype(np.float32)   # (B, 72, T)
        return [latents[i] for i in range(latents.shape[0])]

    def n_frames_for(self, chunk_length: int, sr: int) -> int:
        """Discover the DAC latent length T for a chunk of `chunk_length` samples.
        Content-independent, so a single silent encode gives the exact T shared by
        EVERY equal-length chunk -- lets the workers frame-align conditions without
        waiting for each chunk's DAC pass."""
        z = self.encode(self.torch.zeros(1, chunk_length), sr)
        return int(z.shape[1])


# ============================================================
# CONDITIONS (frame-level, incremental merge into per-chunk .npz)
# ============================================================
def extract_and_merge_frame_conditions(
    registry, chunk_audio_np, sr: int, n_frames: int,
    cond_path: Path, force: bool = False,
) -> bool:
    """
    Extract ONLY the frame conditions that are missing from cond_path and merge
    them in (mirrors extract_conditions.py). Returns True if the .npz changed.
    """
    import numpy as np
    required = set(registry.frame_names)
    if not required:
        return False

    # Always load what is already on disk (even with force=True) so that
    # re-computing the REQUESTED conditions never drops the OTHERS (report #1).
    existing = {}
    if cond_path.exists():
        try:
            data = np.load(str(cond_path))
            existing = {k: data[k] for k in data.keys()}
        except Exception:
            existing = {}
        if not force and required.issubset(existing.keys()):
            return False

    # force -> recompute the required set; otherwise only the missing ones.
    missing = required if force else (required - set(existing.keys()))
    if not missing:
        return False

    new = {}
    for name in missing:
        new[name] = registry.frame_extractors[name].extract(chunk_audio_np, sr, n_frames)

    final = {**existing, **new}   # keep others; force overwrites only `required`
    _atomic_save_npz(cond_path, final)
    return True


# ============================================================
# GLOBAL CONDITIONS (optional, class-level sidecars)
# ============================================================
def extract_global_conditions(
    registry, out_root: Path, classes: List[str], image_root: Optional[str],
    force: bool = False,
):
    """
    OPTIONAL / forward-looking. The current training dataset derives global
    conditions (text/image) at load time from the class name / image folder and
    does NOT read these sidecars yet; this only pre-caches them.

    text  -> global_conditions/text/<class>.npy   (CLAP embedding of class name)
    image -> global_conditions/image/<class>.npy  (stacked CLIP embeddings)
    """
    import numpy as np
    gnames = registry.global_names
    if not gnames:
        return

    if "text" in gnames:
        ext = registry.global_extractors["text"]
        d = out_root / "global_conditions" / "text"
        d.mkdir(parents=True, exist_ok=True)
        prompts = [c.replace("_", " ") for c in classes]
        embs = ext.encode_batch(prompts)
        for c, e in zip(classes, embs):
            p = d / f"{sanitize_class_name(c)}.npy"
            if p.exists() and not force:
                continue
            _atomic_save_npy(p, np.asarray(e, dtype=np.float32))
        if hasattr(ext, "unload"):
            ext.unload()
        print(f"[global/text] cached {len(classes)} class embeddings")

    if "image" in gnames:
        if not image_root or not Path(image_root).exists():
            print("[global/image] --image_root missing/not found -> skipped")
        else:
            ext = registry.global_extractors["image"]
            d = out_root / "global_conditions" / "image"
            d.mkdir(parents=True, exist_ok=True)
            for c in classes:
                p = d / f"{sanitize_class_name(c)}.npy"
                if p.exists() and not force:
                    continue
                cls_dir = Path(image_root) / c            # raw name matches source folder
                if not cls_dir.exists():
                    continue
                imgs = sorted(
                    q for q in cls_dir.rglob("*")
                    if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                )
                if not imgs:
                    continue
                stack = [np.asarray(ext.encode_image(str(q)), dtype=np.float32) for q in imgs]
                _atomic_save_npy(p, np.stack(stack, axis=0))
            if hasattr(ext, "unload"):
                ext.unload()
            print("[global/image] cached per-class image embeddings")


# ============================================================
# ACOUSTIC RULES (optional) -- ported VERBATIM from preprocess_dataset.py
# ------------------------------------------------------------
# Your previous ffmpeg treatment, preserved exactly, but applied per source file
# IN-STREAM: it produces 1-2 transient temp WAV(s) that feed the chunker, so the
# full-dataset WAVs are NEVER materialised on disk (only one source at a time).
#   * silence EDGE-TRIM  (silencedetect, SILENCE_TRIM_DB)
#   * constant-gain LOUDNESS  gain_dB = min(LUFS - measured_I, TP - measured_TP)
#     (pure `volume=` gain: never compresses, never clips; capped by true-peak)
#   * STEREO per-channel: each channel -> its own mono example (pan=mono|c0=cN),
#     NOT an L+R average (which phase-cancels dense stereo mixes)
# Requires ffmpeg/ffprobe on PATH (as your old pipeline did).
# ============================================================
def _ff_duration(path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _ff_channels(path) -> int:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        return int(r.stdout.strip())
    except Exception:
        return 1


def _ff_sample_rate(path) -> Optional[int]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        return int(r.stdout.strip())
    except Exception:
        return None


def _ff_decode(path):
    """
    Decode ANY ffmpeg-readable file to (torch float32 (c, t), sr) at its NATIVE
    sample rate and channel count -- the same contract as
    librosa.load(sr=None, mono=False), but decoded by ffmpeg.

    Used for the formats libsndfile cannot open (mp3/m4a/...). librosa would work
    too, but it silently falls back to `audioread`: slow, one UserWarning per
    file, and removed in librosa 0.11 -- i.e. a path that will simply stop working.
    ffmpeg is already a dependency of this script, is much faster, and handles
    every format uniformly.

    Returns (None, None) if the file cannot be decoded (the caller then skips it
    with a message rather than silently degrading).
    """
    import numpy as np
    sr = _ff_sample_rate(path)
    ch = _ff_channels(path)
    if not sr or not ch:
        return None, None
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
         "-filter_threads", "1", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],   # native sr/channels
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0 or not r.stdout:
        return None, None
    a = np.frombuffer(r.stdout, dtype=np.float32)       # interleaved
    n = (a.size // ch) * ch                             # drop a partial frame
    if n == 0:
        return None, None
    a = a[:n].reshape(-1, ch).T                         # (c, t)
    return torch.from_numpy(np.ascontiguousarray(a)), sr


def _detect_trim_points(path, threshold_db: float) -> Tuple[float, float]:
    duration = _ff_duration(path)
    if duration == 0:
        return 0.0, 0.0
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-threads", "1",
             "-filter_threads", "1", "-i", str(path),
             "-af", f"silencedetect=noise={threshold_db}dB:d=0.1",
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
    except Exception:
        return 0.0, duration
    regions = []
    cur = None
    for line in (r.stderr or "").splitlines():
        if "silence_start:" in line:
            try:
                cur = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                cur = None
        elif "silence_end:" in line and cur is not None:
            try:
                regions.append((cur, float(line.split("silence_end:")[1].strip().split()[0])))
            except (ValueError, IndexError):
                pass
            cur = None
    if cur is not None:
        regions.append((cur, duration))
    if not regions:
        return 0.0, duration
    trim_start = regions[0][1] if regions[0][0] < 0.05 else 0.0
    trim_end = regions[-1][0] if regions[-1][1] >= duration - 0.05 else duration
    return trim_start, trim_end


def _analyze_loudness(path, target_lufs, target_tp, target_lra):
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-threads", "1",
             "-filter_threads", "1", "-i", str(path),
             "-af", (f"loudnorm=I={target_lufs}:TP={target_tp}:"
                     f"LRA={target_lra}:print_format=json"),
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        stderr = r.stderr or ""
        js, je = stderr.rfind("{"), stderr.rfind("}") + 1
        if js == -1 or je == 0:
            return None
        data = json.loads(stderr[js:je])
        return float(data.get("input_i", "nan")), float(data.get("input_tp", "nan"))
    except Exception:
        return None


def acoustic_preprocess_file(
    path, tmp_dir, sr, target_lufs, target_tp, target_lra,
    silence_trim_db, min_sec, stereo_split,
) -> List[Tuple[str, int]]:
    """
    Port of preprocess_dataset.preprocess_file (trim + constant-gain + stereo).
    Returns [(temp_wav_path, channel), ...] (1 for mono / averaged, 2 for a
    stereo file with --stereo_split). Empty if the file is too short after trim.
    """
    trim_start, trim_end = _detect_trim_points(path, silence_trim_db)
    dur = trim_end - trim_start
    if dur < min_sec:
        return []

    meas = _analyze_loudness(path, target_lufs, target_tp, target_lra)
    gain_filter = None
    if meas is not None:
        mi, mtp = meas
        if math.isfinite(mi) and math.isfinite(mtp):
            gain_db = min(target_lufs - mi, target_tp - mtp)
            gain_filter = f"volume={gain_db:.2f}dB"
    # measurement failed -> no gain (keep original level; dynamics-safe fallback)

    n_channels = _ff_channels(path)
    channels = [0, 1] if (stereo_split and n_channels >= 2) else [None]

    out: List[Tuple[str, int]] = []
    for ch in channels:
        filt = []
        if ch is not None:
            filt.append(f"pan=mono|c0=c{ch}")
        if gain_filter:
            filt.append(gain_filter)
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
        os.close(fd)
        cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-threads", "1",
               "-filter_threads", "1",
               "-ss", str(trim_start), "-i", str(path), "-t", str(dur)]
        if filt:
            cmd += ["-af", ",".join(filt)]
        cmd += ["-ar", str(sr), "-ac", "1", "-loglevel", "error", tmp]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            Path(tmp).unlink(missing_ok=True)
            continue
        if _ff_duration(tmp) < min_sec:
            Path(tmp).unlink(missing_ok=True)
            continue
        out.append((tmp, ch if ch is not None else 0))
    return out


def _chunk_mean_dbfs(chunk) -> float:
    """Mean-amplitude dBFS of a chunk (mirrors ffmpeg volumedetect mean_volume,
    used by the SILENCE_THRESH_DB per-chunk gate in preprocess_dataset.py)."""
    x = float(chunk.abs().mean().item())
    return 20.0 * math.log10(x + 1e-7)


# ============================================================
# PARALLEL STREAMING (DataLoader workers do the CPU work; the main process
# batches the DAC on the GPU)
# ------------------------------------------------------------
# Division of labour by RESOURCE, to overlap CPU and GPU and to keep CUDA out of
# forked workers:
#   * WORKERS (CPU, parallel): load + acoustic ffmpeg + chunking + condition
#     extraction + WAV writing. Each of these is a per-chunk side effect written
#     straight to disk (distinct files, no contention). Conditions are aligned to
#     n_frames_fixed (the constant DAC T for a fixed chunk length, discovered once
#     on the main GPU), so a worker never needs to wait for the DAC.
#   * MAIN (GPU): batched DAC encode of the chunks that still need a latent.
# Workers yield ONLY chunks whose latent is missing (or --force); everything else
# is fully handled worker-side. With num_workers=0 this same path runs in-process.
# ============================================================
def _shard_files(files, worker_id: int, num_workers: int):
    return files[worker_id::num_workers] if num_workers and num_workers > 1 else files


def _force_cpu_extractors(registry):
    """Force any model-backed extractor onto CPU. CUDA inside forked DataLoader
    workers is fragile and would contend with the main DAC; f0 on GPU is only
    advisable with --num_workers 0."""
    if registry is None:
        return
    for ext in list(getattr(registry, "frame_extractors", {}).values()):
        if hasattr(ext, "device"):
            ext.device = "cpu"


def _encode_and_save_batch(dac_enc, batch, sr, n_frames: int) -> int:
    """Batched DAC encode of a list of worker items -> save each latent."""
    if not batch:
        return 0
    audios = [it["audio"] for it in batch]
    lats = dac_enc.encode_batch(audios, sr)      # list of (72, T)
    if len(lats) != len(batch):
        raise RuntimeError(
            f"[DAC] Encoder returned {len(lats)} latents for "
            f"{len(batch)} input chunks. Refusing a partial batch."
        )
    n = 0
    for it, lat in zip(batch, lats):
        p = Path(it["latent_path"])
        lat = np.asarray(lat)
        if lat.shape != (72, int(n_frames)):
            raise RuntimeError(
                f"[DAC] Refusing latent with shape {lat.shape}; expected "
                f"(72, {n_frames}) for {p}"
            )
        if lat.dtype != np.dtype(np.float32):
            lat = lat.astype(np.float32)
        if not np.isfinite(lat).all():
            raise RuntimeError(f"[DAC] Refusing NaN/Inf latent for {p}")
        _atomic_save_npy(p, lat)
        n += 1
    return n


class StreamingChunkDataset(IterableDataset):
    """
    Yields, per chunk that still needs a latent, {"audio": [L], "latent_path": str}.
    Conditions and WAV are written to disk as a side effect inside the worker.
    """

    def __init__(self, files, chunker, registry, latent_root, wav_root, cond_root,
                 sr, save_wav, force, n_frames_fixed,
                 acoustic, acoustic_kw, silence_thresh_db, tmp_dir):
        self.files = files
        self.chunker = chunker
        self.registry = registry
        self.latent_root = Path(latent_root)
        self.wav_root = Path(wav_root)
        self.cond_root = Path(cond_root)
        self.sr = sr
        self.save_wav = save_wav
        self.force = force
        self.n_frames_fixed = n_frames_fixed
        self.acoustic = acoustic
        self.acoustic_kw = acoustic_kw          # dict for acoustic_preprocess_file
        self.silence_thresh_db = silence_thresh_db
        self.tmp_dir = tmp_dir

    def __iter__(self):
        import soundfile as sf
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            shard = self.files
        else:
            shard = _shard_files(self.files, wi.id, wi.num_workers)

        has_conditions = self.registry is not None and bool(self.registry.frame_names)

        for path, rel_parent, _leaf, src_hash, rel_src in shard:
            if self.acoustic:
                items = acoustic_preprocess_file(path, self.tmp_dir, self.sr, **self.acoustic_kw)
                cleanup = [w for w, _ in items]
            else:
                items = [(str(path), 0)]
                cleanup = []
            produced_chunks = []          # what THIS source yields, for the manifest
            try:
                for wav_src, channel in items:
                    ch_suffix = f"__ch{channel}" if self.acoustic else ""
                    for idx, chunk in self.chunker.iter_file_chunks(wav_src):
                        if self.acoustic and \
                                _chunk_mean_dbfs(chunk) < self.silence_thresh_db:
                            continue

                        name = (f"{sanitize_filename(Path(path).name)}_{src_hash}"
                                f"{ch_suffix}__c{idx:04d}")
                        rel_chunk = f"{rel_parent.as_posix()}/{name}"
                        produced_chunks.append(rel_chunk)
                        latent_path = self.latent_root / rel_parent / f"{name}.npy"
                        wav_path = self.wav_root / rel_parent / f"{name}.wav"
                        cond_path = self.cond_root / rel_parent / f"{name}.npz"

                        # conditions (CPU, in the worker), aligned to the fixed T
                        if has_conditions:
                            chunk_np = chunk.squeeze(0).cpu().numpy()
                            extract_and_merge_frame_conditions(
                                self.registry, chunk_np, self.sr,
                                self.n_frames_fixed, cond_path, force=self.force)

                        # optional per-chunk WAV (also worker-side)
                        if self.save_wav and (self.force or not wav_path.exists()):
                            _atomic_save_wav(
                                wav_path, chunk.squeeze(0).cpu().numpy(), self.sr
                            )

                        # Hand the main process an OWNED chunk. A narrow slice of
                        # the source tensor is already "contiguous", so calling
                        # .contiguous() would return the same view and PyTorch IPC
                        # could share the storage of the complete source file.
                        # clone() limits each queued tensor to exactly one chunk.
                        needs_latent = (
                            self.force
                            or not _latent_file_is_valid(
                                latent_path, self.n_frames_fixed
                            )
                        )
                        if needs_latent:
                            audio = chunk.squeeze(0).detach().clone()
                            yield {"audio": audio,
                                   "latent_path": str(latent_path)}
            finally:
                for w in cleanup:
                    Path(w).unlink(missing_ok=True)

            # Tell the main process this SOURCE FILE is fully handled, so it can
            # drive one "Files" bar with a known total (len(files)) and a real ETA
            # -- which the per-chunk stream cannot provide (an IterableDataset has
            # no length, and chunks-per-file varies). Emitted even for files that
            # produced no chunk (skipped/filtered), so the count stays exact.
            # It also carries this source's manifest entry: its identity now, and
            # the chunks it owns (INCLUDING those skipped because their latent
            # already existed -- ownership does not depend on who encoded them).
            # Workers are separate processes, so this stream is how they report.
            # Carries no audio; the main loop filters it out of the DAC batch.
            try:
                ident = _source_identity(Path(path))
            except OSError:
                ident = {"size": None, "mtime_ns": None}
            yield {"file_done": 1, "src": rel_src,
                   "ident": {**ident, "chunks": produced_chunks}}


# ============================================================
# DATASET META (chunk params) -- re-run safety
# ============================================================
def _meta_dict(args, chunk_length, chunk_overlap, latent_frames_per_chunk=None):
    """
    The identity of the dataset: EVERY parameter that changes the audio bytes,
    which chunks survive, or the latent geometry. check_or_write_meta() refuses
    to mix runs that disagree on any of these, which is what stops a re-run with
    different settings from silently producing a half-old/half-new dataset.
    """
    return {
        "sr": args.sr,
        "chunk_length_samples": chunk_length,
        "chunk_overlap_samples": chunk_overlap,
        "chunk_duration_s": args.chunk_duration,
        "chunk_overlap_s": args.chunk_overlap,
        "duration_s": args.duration,
        "min_duration_s": args.min_duration,
        "mono": True,
        "pad_last_chunk": args.pad_last_chunk,
        "keep_num_chunks_per_file": args.keep_num_chunks_per_file,
        # latent geometry (the CONTRACT the training/sampling must honor):
        # the REAL DAC frame count per chunk, discovered by encoding once -- NOT
        # int(duration*fps), which truncates (e.g. 5s -> 430 vs the real 431).
        "latent_frames_per_chunk": latent_frames_per_chunk,
        "latent_dim": 72,
        "dac_model": "44khz",
        # per-FILE gates: they decide which sources are dropped entirely
        "min_chunk_sec": args.min_chunk_sec,
        "silence_threshold": args.silence_threshold,
        "max_silence_ratio": args.max_silence_ratio,
        "peak_threshold": args.peak_threshold,
        "max_peak_ratio": args.max_peak_ratio,
        "clip": not args.no_clip,
        # per-CHUNK gates: they decide which chunks survive -> and the surviving
        # chunks are RENUMBERED, so the same name can hold different audio if
        # these change. They must be part of the dataset identity.
        "chunk_min_rms": args.chunk_min_rms,
        "chunk_min_peak": args.chunk_min_peak,
        # acoustic treatment (changes the audio content -> the latents)
        "acoustic_rules": args.acoustic_rules,
        "target_lufs": args.target_lufs if args.acoustic_rules else None,
        "target_tp": args.target_tp if args.acoustic_rules else None,
        "target_lra": args.target_lra if args.acoustic_rules else None,
        "silence_trim_db": args.silence_trim_db if args.acoustic_rules else None,
        "silence_thresh_db": args.silence_thresh_db if args.acoustic_rules else None,
        "stereo_split": (not args.no_stereo_split) if args.acoustic_rules else None,
    }


def check_or_write_meta(out_root: Path, meta: dict):
    """
    Persist the dataset identity; on re-run, HARD-FAIL if ANY parameter differs
    from the one that produced the existing latents. Changing chunk geometry
    would break latent<->cond alignment / incremental naming; changing a gate or
    the acoustic treatment would mix differently-selected or differently-
    normalized latents in the same dataset (surviving chunks are RENUMBERED, so
    the same file name can end up holding different audio).

    Every key of _meta_dict() is compared -- deliberately not a hand-kept subset,
    which is how parameters silently escaped the check before.
    """
    p = out_root / "dataset_meta.json"
    if p.exists():
        old = json.loads(p.read_text())
        # A parameter that is RECORDED and DIFFERENT is a real conflict -> stop.
        # A parameter simply ABSENT from an older meta is NOT a conflict: it just
        # was not recorded back then, so it cannot be verified. Failing on those
        # would block every pre-existing dataset (e.g. adding a condition to one
        # built before these keys existed) even when nothing actually changed.
        conflicts = {k: (old[k], meta.get(k)) for k in meta
                     if k in old and old[k] != meta.get(k)}
        unverifiable = [k for k in meta if k not in old]
        if conflicts:
            raise SystemExit(
                f"[meta] the parameters differ from the existing dataset in "
                f"{out_root} (old vs new): {conflicts}\n"
                f"Re-running with different parameters would mix incompatible "
                f"chunks/latents in one dataset. Use a FRESH output dir.\n"
                f"(--force does NOT rebuild: it overwrites the chunks it "
                f"re-encounters and LEAVES the others, which is exactly how a "
                f"mixed dataset is produced. It is meant for re-extracting "
                f"conditions with the SAME parameters.)"
            )
        if unverifiable:
            # Do NOT write these into the meta. They were not recorded when the
            # dataset was built, so their real values are unknown: persisting the
            # CURRENT ones would turn an unverified guess into apparent historical
            # fact, and the next run would then "verify" the dataset against
            # numbers nobody ever confirmed. Staying silent-but-honest is better:
            # the keys remain absent, and this warning appears every time.
            print(f"[meta] WARNING: {len(unverifiable)} parameter(s) are NOT "
                  f"recorded in {p} (dataset built by an older version): "
                  f"{sorted(unverifiable)}.\n"
                  f"      They CANNOT be checked against this run, and they are "
                  f"deliberately not written into the meta (recording today's "
                  f"values would certify them as the original ones without "
                  f"evidence). Make sure you are passing the same values used "
                  f"originally. For a dataset whose provenance is fully "
                  f"verifiable, rebuild into a FRESH output dir with this "
                  f"version.")
    else:
        _atomic_write_json(p, meta)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Streaming preprocessing (chunk -> DAC encode -> latents), "
                    "supervisor-style. Optional per-chunk WAV/conditions, "
                    "incremental, no train/val/test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source_dir", type=str)
    parser.add_argument("output_dir", type=str)

    # chunking
    parser.add_argument("--sr", type=int, default=44100,
                        help="Target sample rate. MUST be 44100 for the 44khz DAC.")
    parser.add_argument("--chunk_duration", type=float, default=5.0,
                        help="Chunk length in seconds (default: 5.0).")
    parser.add_argument("--chunk_overlap", type=float, default=0.0,
                        help="Chunk overlap in seconds (default: 0).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Trim each source file to N s BEFORE chunking "
                             "(default: None = whole file).")
    parser.add_argument("--min_duration", type=float, default=None,
                        help="Discard source files shorter than N s.")
    parser.add_argument("--pad_last_chunk", action="store_true",
                        help="Pad and keep the last incomplete chunk of each file.")
    parser.add_argument("--keep_num_chunks_per_file", type=int, default=None,
                        help="Keep at most N chunks per file (deterministic "
                             "linspace selection).")

    # file/chunk filtering (supervisor-style)
    parser.add_argument("--silence_threshold", type=float, default=None)
    parser.add_argument("--max_silence_ratio", type=float, default=None)
    parser.add_argument("--peak_threshold", type=float, default=1.0)
    parser.add_argument("--max_peak_ratio", type=float, default=None)
    parser.add_argument("--no_clip", action="store_true",
                        help="Disable clamp to [-peak_threshold, peak_threshold].")
    parser.add_argument("--chunk_min_rms", type=float, default=None,
                        help="Drop chunks whose RMS is below this value.")
    parser.add_argument("--chunk_min_peak", type=float, default=None,
                        help="Keep a low-RMS chunk if its peak exceeds this value.")

    # acoustic rules (your previous preprocess_dataset.py treatment; opt-in)
    parser.add_argument("--acoustic_rules", action="store_true",
                        help="Apply the previous ffmpeg treatment before chunking: "
                             "silence edge-trim + constant-gain loudness "
                             "normalization (true-peak-capped, no compression) + "
                             "per-channel stereo split. Needs ffmpeg/ffprobe.")
    parser.add_argument("--target_lufs", type=float, default=-18.0)
    parser.add_argument("--target_tp", type=float, default=-1.0)
    parser.add_argument("--target_lra", type=float, default=20.0)
    parser.add_argument("--silence_trim_db", type=float, default=-55.0,
                        help="Edge-trim threshold (silencedetect noise).")
    parser.add_argument("--silence_thresh_db", type=float, default=-60.0,
                        help="Per-chunk mean-volume gate: drop chunks below this.")
    parser.add_argument("--no_stereo_split", action="store_true",
                        help="With --acoustic_rules, average stereo to mono "
                             "instead of emitting one example per channel.")
    parser.add_argument("--min_chunk_sec", type=float, default=None,
                        help="Min seconds after trim to keep a file "
                             "(default: --chunk_duration).")

    # outputs
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_wav", action="store_true",
                        help="Also save the per-chunk WAV under wav/<class>/.")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated frame conditions to extract, e.g. "
                             "'melody' or 'melody,energy'. None = skip conditions.")
    parser.add_argument("--global", dest="global_conds", type=str, default=None,
                        help="Comma-separated global conditions to pre-cache "
                             "('text' and/or 'image'). Class-level sidecars.")
    parser.add_argument("--image_root", type=str, default=None,
                        help="image_root/<class>/*.jpg for --global image.")

    # class layout
    parser.add_argument("--single_class", action="store_true")
    parser.add_argument("--class_name", type=str, default=None)

    # misc
    parser.add_argument("--skip_dac", action="store_true",
                        help="Do everything except DAC encoding (debug).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute the latents/conditions of the chunks this "
                             "run encounters, instead of skipping the ones already "
                             "on disk. It is NOT a rebuild: outputs that this run "
                             "no longer produces are LEFT in place, and it does "
                             "NOT let you change the chunking/gate/acoustic "
                             "parameters (those are still checked against "
                             "dataset_meta.json -- use a fresh OUT dir for that). "
                             "Intended use: re-extract conditions with the SAME "
                             "parameters, e.g. after fixing an extractor.")

    parser.add_argument("--prune_orphans", action="store_true",
                        help="Delete the outputs of sources listed in "
                             "source_manifest.json that no longer exist on disk "
                             "(deleted/renamed). Without this they stay and keep "
                             "feeding the split/normalizer/training while "
                             "corresponding to nothing. Only files the manifest "
                             "attributes to those sources are removed.")

    # throughput
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers doing the CPU work (load, "
                             "acoustic, chunking, condition extraction) in "
                             "parallel with the GPU DAC. 0 = single process "
                             "(default). Start with 4 on a large dataset; each "
                             "worker loads one condition model when conditions "
                             "are enabled.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Chunks encoded per DAC forward pass (default: 8).")
    parser.add_argument("--loader_batch_size", type=int, default=8,
                        help="Chunks transported in each worker IPC batch "
                             "(default: 8). Kept separate from --batch_size so "
                             "GPU batching can grow without multiplying shared "
                             "memory in every worker.")
    parser.add_argument("--prefetch_factor", type=int, default=1,
                        help="Batches prefetched by EACH worker (default: 1). "
                             "Higher values multiply shared-memory use by "
                             "num_workers * loader_batch_size.")
    parser.add_argument("--worker_start_method", type=str, default="spawn",
                        choices=("spawn", "forkserver"),
                        help="Safe multiprocessing start method (default: "
                             "spawn). Avoids forking workers from a process "
                             "that already initialized CUDA.")

    args = parser.parse_args()

    if not _TORCH_OK:
        raise SystemExit("[preprocess_stream] PyTorch is required to run "
                         "(only --help works without it).")
    print(f"[preprocess] build={PREPROCESS_BUILD}")
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    if args.num_workers < 0:
        raise SystemExit("[preprocess_stream] --num_workers must be >= 0.")
    if args.batch_size < 1:
        raise SystemExit("[preprocess_stream] --batch_size must be >= 1.")
    if args.loader_batch_size < 1:
        raise SystemExit("[preprocess_stream] --loader_batch_size must be >= 1.")
    if args.prefetch_factor < 1:
        raise SystemExit("[preprocess_stream] --prefetch_factor must be >= 1.")

    chunk_length = int(round(args.chunk_duration * args.sr))
    chunk_overlap = int(round(args.chunk_overlap * args.sr))

    # The DAC codec is the 44khz model: any other sample rate produces incoherent
    # latents / frame rate. Enforce it instead of only documenting it (report #14).
    if args.sr != 44100:
        raise SystemExit(
            f"[preprocess_stream] --sr must be 44100 for the 44khz DAC "
            f"(got {args.sr}). The latent frame rate and all conditions assume it.")

    out_root = Path(args.output_dir)
    latent_root = out_root / "latents"
    wav_root = out_root / "wav"
    cond_root = out_root / "conditions"

    # NOTE: dataset_meta.json is written AFTER the real latent T is discovered
    # (see below), so it records latent_frames_per_chunk as part of the contract.

    # ---- condition registry (frame + optional global selection) ----
    enabled_frame = None
    if args.conditions is not None:
        enabled_frame = [c.strip() for c in args.conditions.split(",") if c.strip()]
    enabled_global = None
    if args.global_conds is not None:
        enabled_global = [c.strip() for c in args.global_conds.split(",") if c.strip()]

    registry = None
    if enabled_frame or enabled_global:
        from conditions import ConditionRegistry
        registry = ConditionRegistry(
            enabled_frame=enabled_frame if enabled_frame else [],
            enabled_global=enabled_global if enabled_global else [],
        )
        if args.num_workers > 0:
            # CUDA in forked workers is fragile; keep model-backed extractors on CPU.
            _force_cpu_extractors(registry)

    # ---- scan ----
    print("Scanning source files ...")
    files = scan_audio_files(args.source_dir, args.single_class, args.class_name)
    if not files:
        print(f"[ERROR] No audio files under {args.source_dir}")
        return
    classes = sorted({leaf for _, _, leaf, _, _ in files})
    print(f"  {len(files)} files, {len(classes)} classes")

    # ---- chunking backend ----
    backend_kw = dict(
        chunk_length=chunk_length, chunk_overlap=chunk_overlap,
        duration=args.duration, min_duration=args.min_duration,
        audio_sr=args.sr, mono=True,
        silence_threshold=args.silence_threshold,
        max_silence_ratio=args.max_silence_ratio,
        peak_threshold=args.peak_threshold,
        max_peak_ratio=args.max_peak_ratio,
        clip=not args.no_clip,
        chunk_min_rms_threshold=args.chunk_min_rms,
        chunk_min_peak_threshold=args.chunk_min_peak,
        pad_and_keep_last_chunk=args.pad_last_chunk,
        pad_value=0.0,
        keep_num_chunks_per_file=args.keep_num_chunks_per_file,
    )
    # The chunker is a faithful, dependency-free port of the supervisor's
    # ChunkedAudioFileDataset (_load_audio / _stream_chunks): same offset math
    # (num_offsets = (length - chunk_length) // hop_length + 1), same
    # keep_num_chunks_per_file pruning, same RMS/peak gates. It is THE procedure,
    # not one of two interchangeable backends.
    chunker = VendoredChunker(**backend_kw)

    # ---- DAC ----
    dac_enc = None
    if not args.skip_dac:
        dac_enc = DACEncoder(device=args.device)

    # ---- fixed latent length T = the REAL DAC frame count for a full chunk ----
    # (content-independent, so one silent encode gives the exact T shared by every
    # equal-length chunk). Discovered ALWAYS (not only when conditions are asked)
    # so it can be recorded in dataset_meta.json as the latent-geometry contract.
    if dac_enc is not None:
        n_frames_fixed = dac_enc.n_frames_for(chunk_length, args.sr)
    else:
        # --skip_dac: recover T from an existing dataset_meta.json, else a latent.
        n_frames_fixed = None
        meta_path = out_root / "dataset_meta.json"
        if meta_path.exists():
            try:
                n_frames_fixed = json.loads(meta_path.read_text()).get(
                    "latent_frames_per_chunk")
            except Exception:
                n_frames_fixed = None
        if n_frames_fixed is None:
            for existing in latent_root.rglob("*.npy"):
                z = None
                try:
                    z = np.load(
                        str(existing), mmap_mode="r", allow_pickle=False
                    )
                    if (z.ndim == 2 and z.shape[0] == 72
                            and z.dtype == np.dtype(np.float32)):
                        n_frames_fixed = int(z.shape[1])
                        break
                except Exception:
                    continue
                finally:
                    if z is not None:
                        mm = getattr(z, "_mmap", None)
                        if mm is not None:
                            mm.close()
        if n_frames_fixed is None:
            raise SystemExit(
                "[preprocess_stream] --skip_dac but no latent/meta on disk to read "
                "the latent length from: run once without --skip_dac first.")
    print(f"[latents] T (real DAC frames per chunk) = {n_frames_fixed}")
    if registry is not None and registry.frame_names:
        print(f"[conditions] frame-aligned to T={n_frames_fixed}")

    # ---- dataset_meta.json (written now that the real T is known) ----
    meta = _meta_dict(args, chunk_length, chunk_overlap, n_frames_fixed)
    # The meta check runs ALWAYS, --force included. --force means "recompute the
    # files you encounter", NOT "ignore that the parameters changed": bypassing
    # the check let a re-run with different chunking/gates/acoustics overwrite
    # PART of an existing dataset and leave the rest, which is precisely how a
    # silently mixed dataset is produced. Different parameters => fresh OUT dir.
    check_or_write_meta(out_root, meta)

    # ---- source manifest: audit BEFORE doing any work ----
    prev_manifest = load_source_manifest(out_root)
    changed_srcs, removed_srcs = ([], [])
    if prev_manifest:
        changed_srcs, removed_srcs = audit_sources(prev_manifest, files)
        if changed_srcs:
            print(f"[manifest] WARNING: {len(changed_srcs)} source(s) CHANGED since "
                  f"they were encoded (size/mtime differ). Their existing latents "
                  f"are STALE and, without --force, would be silently kept:")
            for rel in changed_srcs[:5]:
                print(f"             {rel}")
            if len(changed_srcs) > 5:
                print(f"             ... and {len(changed_srcs) - 5} more")
            print(f"           -> re-run with --force to re-encode them, or use a "
                  f"fresh OUT dir.")
        if removed_srcs:
            n_orph = sum(len(prev_manifest[r].get("chunks", [])) for r in removed_srcs)
            print(f"[manifest] {len(removed_srcs)} source(s) no longer exist, "
                  f"leaving ~{n_orph} ORPHAN output(s) that would still feed the "
                  f"split/normalizer/training.")
            if args.prune_orphans:
                n_del = prune_orphans(out_root, prev_manifest, removed_srcs)
                print(f"           -> --prune_orphans: deleted {n_del} file(s).")
            else:
                print(f"           -> pass --prune_orphans to delete them.")

    # ---- acoustic-rules setup (optional) ----
    tmp_dir = None
    min_chunk_sec = args.min_chunk_sec if args.min_chunk_sec is not None else args.chunk_duration
    acoustic_kw = None
    if args.acoustic_rules:
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise SystemExit("[acoustic_rules] ffmpeg/ffprobe not found on PATH.")
        base = _IRCAM_LOCAL if os.path.isdir(_IRCAM_LOCAL) else None
        tmp_dir = tempfile.mkdtemp(prefix="preprocess_stream_", dir=base)
        acoustic_kw = dict(
            target_lufs=args.target_lufs, target_tp=args.target_tp,
            target_lra=args.target_lra, silence_trim_db=args.silence_trim_db,
            min_sec=min_chunk_sec, stereo_split=not args.no_stereo_split,
        )
        print(f"[acoustic_rules] ON (LUFS={args.target_lufs}, TP={args.target_tp}, "
              f"trim={args.silence_trim_db}dB, gate={args.silence_thresh_db}dB, "
              f"stereo_split={not args.no_stereo_split}) tmp={tmp_dir}")

    # ---- streaming dataset + parallel workers; main process batches the DAC ----
    dataset = StreamingChunkDataset(
        files=files, chunker=chunker, registry=registry,
        latent_root=latent_root, wav_root=wav_root, cond_root=cond_root,
        sr=args.sr, save_wav=args.save_wav, force=args.force,
        n_frames_fixed=n_frames_fixed,
        acoustic=args.acoustic_rules, acoustic_kw=acoustic_kw,
        silence_thresh_db=args.silence_thresh_db, tmp_dir=tmp_dir,
    )
    loader_kw = dict(
        dataset=dataset,
        batch_size=args.loader_batch_size,
        num_workers=args.num_workers,
        collate_fn=_identity_collate,
        persistent_workers=False,
    )
    if args.num_workers > 0:
        # The defaults that caused the crash were 32 workers * 2 prefetched
        # batches * 64 chunks. Spawn avoids CUDA's unsafe post-init fork, while
        # the explicit prefetch cap bounds queued/shared audio tensors.
        loader_kw.update(
            prefetch_factor=args.prefetch_factor,
            multiprocessing_context=args.worker_start_method,
            worker_init_fn=_worker_init_fn,
        )
    loader = DataLoader(**loader_kw)
    print(f"[run] num_workers={args.num_workers}, "
          f"loader_batch_size={args.loader_batch_size}, "
          f"dac_batch_size={args.batch_size}, "
          f"prefetch_factor={args.prefetch_factor if args.num_workers else 0}, "
          f"start_method={args.worker_start_method if args.num_workers else 'none'}, "
          f"skip_dac={args.skip_dac}")

    # ONE progress bar, owned by the main process, driven by the per-file markers
    # the workers emit. Works identically for num_workers=0 and >0 (no clashing
    # per-worker bars) and, unlike a bar over the chunk stream, it has a known
    # total -> real percentage + ETA.
    try:
        from tqdm import tqdm
        file_bar = tqdm(total=len(files), desc="Files", unit="file")
    except Exception:
        file_bar = None

    n_lat = 0
    pending = []          # real chunks awaiting a full DAC batch
    produced = {}         # rel_source -> {size, mtime_ns, chunks} for the manifest
    try:
        for batch in loader:                        # each batch = list of items
            for it in batch:
                if "file_done" in it:
                    if file_bar is not None:
                        file_bar.update(1)
                    src = it.get("src")
                    if src is not None:
                        produced[src] = it["ident"]
                    continue
                pending.append(it)
            if args.skip_dac or dac_enc is None:
                pending.clear()                     # workers already wrote conds/wav
                continue
            # keep the DAC batches at exactly batch_size (the markers must not
            # shrink them), flushing the remainder after the stream ends.
            while len(pending) >= args.batch_size:
                n_lat += _encode_and_save_batch(
                    dac_enc, pending[:args.batch_size], args.sr,
                    n_frames_fixed)
                del pending[:args.batch_size]
        if pending and not (args.skip_dac or dac_enc is None):
            n_lat += _encode_and_save_batch(
                dac_enc, pending, args.sr, n_frames_fixed
            )
            pending.clear()
    finally:
        if file_bar is not None:
            file_bar.close()
        # Persist whatever was observed, even on Ctrl-C / crash: a PARTIAL
        # manifest is strictly better than none (it still records the sources
        # actually handled). Written atomically via a temp file + replace.
        if produced or prev_manifest:
            n_src = write_source_manifest(out_root, prev_manifest, produced,
                                          removed_srcs, args.prune_orphans)
            print(f"[manifest] {out_root / MANIFEST_NAME}: {n_src} source(s) "
                  f"({len(produced)} seen this run)")
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- global conditions (class-level, after the stream) ----
    if registry is not None and registry.global_names:
        extract_global_conditions(
            registry, out_root, classes, args.image_root, force=args.force
        )

    print("\nDONE")
    print(f"  latents encoded this run: {n_lat}")
    print("  conditions / wav were written in-stream by the workers "
          "(counts not aggregated across processes).")
    print(f"  output: {out_root}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
