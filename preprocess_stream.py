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

THE TRAIN/VAL/TEST SPLIT IS DECIDED HERE and recorded in splits.json. It is
assigned over the SOURCE FILES (the leakage-safe unit) before anything is
decoded, which is what makes "save only the validation wavs" expressible at all.
The output still MIRRORS THE SOURCE DIRECTORY TREE verbatim -- there are no
train/val/test directories on disk, the split is a lookup table:

    OUT/
        latents/<same/subdir/as/source>/<name>.npy   <- (72, T) float32, DAC
        wav/<same/subdir/as/source>/<name>.wav        <- only with --save_wav
        conditions/<same/subdir/as/source>/<name>.npz <- only with --conditions
        global_conditions/<text|image>/<class>.npy    <- only with --global
        dataset_meta.json                             <- chunk params, re-run safety
        splits.json                                   <- source -> train/val/test
        source_manifest.json                          <- what each source produced

    The split NEVER moves a source it has already assigned: a re-run only places
    the NEW ones. Re-deciding it needs --resplit, because promoting yesterday's
    training material to today's test set silently invalidates every evaluation
    of an existing checkpoint. Datasets built before the split lived here get
    theirs with --split_only (fresh) or --import_legacy_split (reproduces the
    split the training used to compute in-code, for runs already in flight).

    e.g. SRC/rock/song.mp3  ->  OUT/latents/rock/song__c0000.npy, ...__c0001.npy
    Source directory names are preserved verbatim; only per-chunk file names are
    sanitized. The chunk index cNNNN is a positional counter over the kept chunks
    of that source file.

INCREMENTAL / IDEMPOTENT
    Every stage is idempotent per chunk, so you can:
      1) run once with latents + some conditions:
           python preprocess_stream.py SRC OUT --conditions f0
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

    # latents + per-chunk f0 + energy conditions:
    python preprocess_stream.py SRC OUT --device cuda --conditions f0,energy

    # with YOUR acoustic treatment (stereo split + constant-gain loudnorm + trim):
    python preprocess_stream.py SRC OUT --device cuda --acoustic_rules

    # keep the per-chunk WAV of the VALIDATION split only (FAD reference):
    python preprocess_stream.py SRC OUT --device cuda --save_wav val

    # everything driven by a config file instead of flags:
    python preprocess_stream.py --config configs/preprocess_default.yaml

    # give an ALREADY-PREPROCESSED dataset a split, without redoing any work:
    python preprocess_stream.py SRC OUT --split_only

    # add f0 (CREPE) later, incrementally, without recomputing latents:
    python preprocess_stream.py SRC OUT --device cuda --conditions f0

    # large dataset: parallel CPU workers + batched DAC encode
    python preprocess_stream.py SRC OUT --device cuda --conditions f0,energy \
        --num_workers 8 --batch_size 16

    # keep f0 on CPU even though the DAC is on the GPU:
    python preprocess_stream.py SRC OUT --device cuda --cond_device cpu \
        --conditions f0

Throughput:
    Work is divided by RESOURCE, not by stage.

    WORKERS (--num_workers, CPU, parallel): audio load (the soundfile/ffmpeg
    decoding ladder), acoustic ffmpeg pass, chunking, the CPU-only conditions
    (chroma, energy) and WAV writing -- each a per-chunk side effect
    written straight to disk.

    MAIN PROCESS (the only owner of the GPU): the batched DAC encode
    (--batch_size chunks per forward) AND the GPU-capable conditions -- f0 via
    torchcrepe, rhythm via beat_this -- over the SAME chunks, in the same pass.
    They cannot share one tensor (the DAC takes 44.1 kHz, CREPE 16 kHz), but they
    share the pass, which is what keeps the card busy instead of alternating
    between a loaded CPU and an idle GPU.

    This is why --cond_device exists and why no worker ever touches CUDA: one
    process owns the card, so N workers cannot contend with the DAC for VRAM or
    time-slice the GPU between themselves. --cond_device cpu puts the GPU-capable
    conditions back in the workers, which is the pre-split behaviour.

    Conditions are frame-aligned to the constant DAC latent length T (discovered
    once), so workers never wait on the GPU. num_workers=0 (default) runs the
    identical path in a single process.

    A chunk is handed to the main process if it still needs a latent OR still
    needs a GPU-side condition, tracked independently -- so adding f0 to a
    dataset whose latents already exist re-reads the audio without re-encoding a
    single latent.

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

# IRCAM: redirect model caches (DAC, CREPE, beat_this, HuggingFace) to the local disk
# instead of the NFS HOME. Guarded so the script stays portable off-IRCAM.
_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ.setdefault("HF_HOME", os.path.join(_cache, "huggingface"))
    # TORCH_HOME is an ASSIGNMENT, not a setdefault: the IRCAM nodes already
    # export it, pointing inside the SHARED conda env
    # (.../envs/tf2.18/share/TORCH), which is read-only for us. torch.hub
    # prefers TORCH_HOME over XDG_CACHE_HOME, so the first download on a
    # machine with a cold cache (beat_this fetching its checkpoint) dies with
    # PermissionError. Only overwriting the variable fixes it.
    os.environ["TORCH_HOME"] = os.path.join(_cache, "torch")

import re
import json
import math
import random
import shutil
import hashlib
import argparse
import subprocess
import tempfile
import faulthandler
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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


def _latent_file_is_valid(path: Path, n_frames: int, verbose: bool = True) -> bool:
    """A resume skips only a readable float32 latent with geometry (72, T).

    mmap_mode reads the .npy HEADER only, so this costs one open per file and
    never the array -- which is what makes it usable as a pre-flight check over
    a whole dataset, not just per chunk. `verbose=False` silences the per-file
    report for that bulk pass (the chunk that gets re-encoded reports itself)."""
    path = Path(path)
    if not path.exists():
        return False
    z = None
    try:
        z = np.load(str(path), mmap_mode="r", allow_pickle=False)
        valid = z.shape == (72, int(n_frames)) and z.dtype == np.dtype(np.float32)
    except Exception as e:
        if verbose:
            print(f"[resume] invalid latent will be regenerated: {path} ({e})")
        return False
    finally:
        if z is not None:
            mm = getattr(z, "_mmap", None)
            if mm is not None:
                mm.close()
    if not valid and verbose:
        print(f"[resume] invalid latent will be regenerated: {path} "
              f"(expected shape=(72,{n_frames}), dtype=float32)")
    return valid


SUPPORTED_AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".wma", ".mpc", ".oma", ".ape", ".aac",
}

PREPROCESS_BUILD = "2026-09-04-split-and-config-v1"


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
# CLASS LABELS -- read from the directory tree, or from a CSV
# ============================================================
#
# In this pipeline the class of a source is ONE thing: the leaf of the output
# directory it is written into. The stratified split, `_class_of_file` in
# audio_dataset_cond.py, the per-class image bank and the panel captions all
# read it back off that folder name, and none of them knows how it was decided.
#
# So a CSV-labelled corpus needs no second mechanism: the CSV only has to DECIDE
# THE OUTPUT FOLDER. With `label_source: csv` the label looked up for a file
# becomes its `rel_parent`, and everything downstream keeps working untouched --
# including corpora whose audio sits in ONE FLAT DIRECTORY, which is precisely
# the layout that has no folder to read a class from. The raw audio is never
# moved or copied: only the encoded output is grouped by class.
#
# WHAT DELIBERATELY DOES NOT CHANGE: `rel_posix`, and therefore `src_hash` and
# every chunk file name, are still computed from the SOURCE path. Turning CSV
# labelling on does not rename a single chunk, so the manifest, an interrupted
# run's resume and the split's source groups are all unaffected by it.
#
# THE CSV CONTRACT IS FIXED, AND THERE ARE NO OPTIONS AROUND IT:
#
#     file,label
#     violin_01.wav,violin
#     ...
#
#   * a file column, named `file` or `filename` (case does not matter);
#   * a label column, named `label` or `class`;
#   * the file column holds the name, or the path relative to source_dir;
#   * the label is the class, used VERBATIM as the output folder name;
#   * every source file must have a row.
#
# Anything else is fixed IN THE CSV, not by a flag: a different column name, a
# typo in a label, a file with no row. A CSV is a text file under your control,
# and every option that "handles" one of those cases is a rule that will be
# wrong on some other corpus, silently. The whole surface is therefore two
# flags -- --label_source and --label_csv -- and a contract.

LABEL_SOURCES = ("dir", "csv")

# The column names this tool accepts, matched IGNORING CASE. Two spellings each,
# because these are the two the world actually uses and they are mutually
# exclusive in practice -- a CSV that carries both `label` and `class` is a CSV
# whose author meant two different things, and guessing which is the class would
# be exactly the kind of silent choice this file refuses to make. Finding both
# is therefore an error, resolved by renaming or removing one column in the CSV.
# This is a short CLOSED list, not an option: no flag chooses between them.
CSV_FILE_COLS = ("file", "filename")
CSV_LABEL_COLS = ("label", "class")

# Characters that cannot appear in a directory name on Windows, plus the path
# separators. A class label becomes a real folder VERBATIM (dir-mode labels are
# folders already, so they are legal by construction); a CSV cell is arbitrary
# text and has to be checked, because silently sanitizing it would break the
# one invariant everything downstream relies on -- class name == folder name --
# and would make `image_root/<class>` stop matching for exactly those classes.
_LABEL_FORBIDDEN = set('/\\:*?"<>|') | {chr(c) for c in range(32)}

_AMBIGUOUS = object()      # a lookup key that two rows claim with DIFFERENT labels


def _validate_class_label(label: str, where: str) -> str:
    """A CSV cell -> a label usable as a directory name, or a hard stop.

    Not sanitized, CHECKED: see _LABEL_FORBIDDEN for why.

    A label is also handed to a text encoder when the `text` condition is
    active, so it is worth naming classes as plain words -- `violin`, not
    `Sound_Violin`. Measured 9 Sept 2026 with laion/clap-htsat-unfused over four
    instrument classes: mean cosine between DIFFERENT classes 0.883 when the
    labels shared a `Sound_` prefix, 0.103 with bare nouns. That is the
    difference between four conditioning vectors on top of each other and four
    that can be told apart. Nothing here rewrites a name to fix that -- a name
    rewritten at encode time would stop being the name the folder, the split and
    the panels talk about, and any rule clever enough to do it is clever enough
    to be wrong in silence.
    """
    s = str(label).strip()
    if not s:
        raise SystemExit(f"[labels] {where}: empty class label.")
    bad = sorted(set(s) & _LABEL_FORBIDDEN)
    if bad or s in (".", ".."):
        raise SystemExit(
            f"[labels] {where}: the label {s!r} cannot be a directory name "
            f"(offending character(s): {bad or [s]}). The class name IS the "
            f"output folder name here, so it is refused rather than silently "
            f"rewritten -- a rewritten name would no longer match "
            f"image_root/<class>. Fix it in the CSV.")
    return s


def _norm_rel_key(s) -> str:
    """A CSV path cell -> the same shape as a scanned file's rel_posix."""
    s = str(s).strip().strip('"').replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


def _index_put(d: dict, key: str, row: int, label_of) -> None:
    """Insert key -> row, marking the key AMBIGUOUS if a previous row claimed it
    with a DIFFERENT label. Two rows agreeing on the label are a duplicate, not
    a conflict, and the first one is kept."""
    if not key:
        return
    prev = d.get(key)
    if prev is None:
        d[key] = row
    elif prev is _AMBIGUOUS:
        return
    elif label_of(prev) != label_of(row):
        d[key] = _AMBIGUOUS


class CsvLabelResolver:
    """Answers "which class is this source file?" from a metadata CSV.

    MATCHING, strongest first: the file's path relative to source_dir, then its
    bare name, then those two again case-insensitively (a CSV and a filesystem
    routinely disagree on case, and on Windows the disagreement is invisible).
    The first index that HAS the key decides -- including deciding that it is
    ambiguous: a weaker index must never rescue a key a stronger one has already
    found to be claimed twice.

    Ambiguity is a hard stop: two rows naming the same file with different
    labels mean the CSV cannot answer the question, and picking one would put a
    random half of those sources in the wrong class.

    `used` records which ROWS actually matched something, so the caller can
    report the rows that matched no file -- the usual sign that the CSV and the
    audio folder are not of the same vintage.
    """

    def __init__(self, csv_path: Path, rows: List[Tuple[str, str]]):
        self.csv_path = Path(csv_path)
        self.rows = list(rows)              # [(rel_key, label), ...] in file order
        self.used = set()

        def label_of(i):
            return self.rows[i][1]

        self._by_rel = {}
        self._by_name = {}
        self._by_rel_lc = {}
        self._by_name_lc = {}
        for i, (rel, _lab) in enumerate(self.rows):
            name = rel.rsplit("/", 1)[-1]
            _index_put(self._by_rel, rel, i, label_of)
            _index_put(self._by_name, name, i, label_of)
            _index_put(self._by_rel_lc, rel.lower(), i, label_of)
            _index_put(self._by_name_lc, name.lower(), i, label_of)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def labels(self) -> List[str]:
        return sorted({lab for _, lab in self.rows})

    def unmatched_rows(self) -> List[str]:
        """The rel keys of the rows that never matched a file, in file order."""
        return [rel for i, (rel, _) in enumerate(self.rows) if i not in self.used]

    def lookup(self, rel_posix: str) -> Tuple[Optional[str], str]:
        """-> (label, status), status in {"ok", "missing", "ambiguous"}."""
        name = rel_posix.rsplit("/", 1)[-1]
        for d, key in ((self._by_rel, rel_posix),
                       (self._by_name, name),
                       (self._by_rel_lc, rel_posix.lower()),
                       (self._by_name_lc, name.lower())):
            hit = d.get(key)
            if hit is None:
                continue
            if hit is _AMBIGUOUS:
                return None, "ambiguous"
            self.used.add(hit)
            return self.rows[hit][1], "ok"
        return None, "missing"


def _pick_csv_column(path, fields, accepted, what: str) -> str:
    """The one column of `fields` whose name is in `accepted`, or a hard stop.

    Case-insensitive, because a header is written by a human. Zero matches and
    two matches are both errors, and both messages say what to change in the
    CSV -- there is no flag to answer them with, on purpose.
    """
    hits = [f for f in fields if f.lower() in accepted]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"[labels] {Path(path).name} has no {what} column. Accepted names "
            f"(any case): {list(accepted)}. The file has "
            f"{fields or 'no header'}. Rename the column in the CSV -- there is "
            f"deliberately no flag for this.")
    raise SystemExit(
        f"[labels] {Path(path).name} has {len(hits)} columns that could be the "
        f"{what} column: {hits}. Which one is the {what} cannot be guessed, and "
        f"guessing would put a whole corpus in the wrong folders. Rename or "
        f"remove one of them in the CSV.")


def load_label_csv(csv_path: str) -> "CsvLabelResolver":
    """Read the metadata CSV into a resolver, or stop with a message that says
    what is wrong with the file. Columns: see CSV_FILE_COLS / CSV_LABEL_COLS."""
    import csv as _csv
    p = Path(csv_path)
    if not p.exists():
        raise SystemExit(f"[labels] --label_csv {p} does not exist.")

    # utf-8-sig: a CSV exported from Excel starts with a BOM, which would
    # otherwise become part of the FIRST column's name and make it unfindable.
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = _csv.DictReader(fh)
        fields = [(f or "").strip() for f in (reader.fieldnames or [])]
        fcol = _pick_csv_column(p, fields, CSV_FILE_COLS, "file")
        lcol = _pick_csv_column(p, fields, CSV_LABEL_COLS, "label")
        rows, n_blank, n_seen = [], 0, 0
        for raw in reader:
            n_seen += 1
            rel = _norm_rel_key(raw.get(fcol) or "")
            lab = str(raw.get(lcol) or "").strip()
            if not rel or not lab:
                n_blank += 1
                continue
            rows.append((rel, _validate_class_label(lab, f"{p.name} row {n_seen}")))

    if not rows:
        raise SystemExit(
            f"[labels] {p}: no usable row ({n_seen} read, {n_blank} with an "
            f"empty cell).")
    res = CsvLabelResolver(p, rows)
    print(f"[labels] {p.name}: {len(rows)} row(s), {len(res.labels)} label(s)"
          + (f", {n_blank} row(s) skipped for an empty cell" if n_blank else ""))
    return res


def build_label_resolver(args):
    """The resolver this run labels with, or None for plain directory labelling.

    Also the place where the mutually exclusive options are refused, BEFORE any
    audio is touched: --single_class flattens every source into one class and a
    CSV assigns one per file, so asking for both is not a preference to settle
    silently.
    """
    src = str(getattr(args, "label_source", "dir") or "dir").strip().lower()
    if src not in LABEL_SOURCES:
        raise SystemExit(f"[labels] label_source must be one of "
                         f"{list(LABEL_SOURCES)}, got {src!r}.")
    if src == "dir":
        if getattr(args, "label_csv", None):
            print("[labels] label_csv is set but label_source is 'dir': the CSV "
                  "is NOT read, classes come from the directory tree.")
        return None
    if not getattr(args, "label_csv", None):
        raise SystemExit("[labels] --label_source csv needs --label_csv "
                         "<metadata.csv>.")
    if getattr(args, "single_class", False):
        raise SystemExit(
            "[labels] --single_class and --label_source csv contradict each "
            "other: the first puts every source in ONE class, the second gives "
            "each file the class its CSV row names. Choose one.")
    return load_label_csv(args.label_csv)


def print_class_histogram(files, max_shown: int = 24) -> None:
    """How many SOURCE FILES each class holds, in one capped block.

    Printed for both labelling modes because it is the cheapest possible check
    that the labels are the ones expected -- a class that should not exist, or
    one holding two files, shows up here and nowhere else until training starts.
    """
    if not files:
        return
    counts = {}
    for _p, _rel_parent, leaf, _h, _rel in files:
        counts[leaf] = counts.get(leaf, 0) + 1
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    body = ", ".join(f"{c} {n}" for c, n in order[:max_shown])
    tail = ""
    if len(order) > max_shown:
        rest = sum(n for _c, n in order[max_shown:])
        tail = (f", ... {len(order) - max_shown} more class(es), "
                f"{rest} file(s)")
    print(f"  classes: {body}{tail}")


# ============================================================
# LABEL PROVENANCE -- recorded next to the sources it labelled
# ============================================================
#
# WHY IT IS RECORDED. The labelling mode decides which FOLDER every latent is
# written into. Re-running the same output dir with the other mode does not
# overwrite anything: it writes a second complete copy of the dataset under
# different class folders, and the split, the manifest and the image banks then
# describe a mixture of the two. Nothing downstream can detect that afterwards,
# because by then a folder name is all there is. So the mode is stored WITH the
# dataset and a change of it is refused, exactly as dataset_meta.json refuses a
# change of chunk geometry.
#
# The CSV's own fields (path, columns, aliases) are reported as a WARNING and
# not refused: adding rows to a CSV as a corpus grows is normal, and only the
# mode change is guaranteed to relocate what already exists.

def labels_provenance(args, resolver) -> dict:
    prov = {"source": "csv" if resolver is not None else "dir"}
    if resolver is not None:
        prov.update({"csv": str(args.label_csv),
                     "csv_name": resolver.csv_path.name})
    elif getattr(args, "single_class", False):
        prov["single_class"] = str(getattr(args, "class_name", None) or "")
    return prov


def load_labels_provenance(out_root: Path) -> dict:
    p = out_root / MANIFEST_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("labels", {}) or {}
    except Exception:
        return {}


def check_labels_provenance(out_root: Path, prov: dict) -> None:
    old = load_labels_provenance(out_root)
    if not old:
        # Nothing recorded. A manifest that HAS sources but no labels block was
        # written before this option existed, and back then the directory tree
        # was the only way a class could be decided -- so "dir" here is a fact
        # about the code's history, not a guess, and it lets an existing dataset
        # be protected from a CSV re-run just like a new one. No manifest at all
        # means nothing has been encoded yet: nothing to protect.
        if not load_source_manifest(out_root):
            return
        old = {"source": "dir"}
    if old.get("source") != prov.get("source"):
        raise SystemExit(
            f"[labels] {out_root} was built with label_source="
            f"{old.get('source')!r} and this run uses {prov.get('source')!r}.\n"
            f"The labelling mode decides the output FOLDER of every latent, so "
            f"re-running with the other one does not overwrite the dataset -- "
            f"it writes a SECOND copy under different class folders, and the "
            f"split and the manifest then describe both at once. Use a FRESH "
            f"output dir.")
    differing = {k: (old.get(k), prov.get(k)) for k in ("csv_name",)
                 if k in old and old.get(k) != prov.get(k)}
    if differing:
        print(f"[labels] WARNING: a different CSV from the one recorded for "
              f"{out_root} (old vs new): {differing}. A source whose label "
              f"changed keeps its OLD latents under the OLD class folder -- "
              f"those become orphans (see --prune_orphans).")


# ============================================================
# FILE SCAN -> (path, rel_parent, leaf_class)
# ============================================================
def scan_audio_files(
    source_dir: str,
    single_class: bool = False,
    class_name: Optional[str] = None,
    label_resolver=None,
) -> List[Tuple[Path, Path, str, str, str]]:
    """
    Walk source_dir and return [(path, rel_parent, leaf_class, src_hash, rel_posix),
    ...] in a deterministic order.

    `rel_parent` is the output subdirectory of the file. With directory
    labelling it is the file's source subdirectory RELATIVE to source_dir, so
    the encoded dataset reproduces THE SAME directory tree as the raw dataset
    (e.g. SRC/rock/song.mp3 -> latents/rock/). Directory names are preserved
    exactly (not sanitized) so they match the source; only the per-chunk FILE
    names are sanitized.

    `leaf_class` is the last component of rel_parent -- the class label used for
    the stratified split and for the (per-class) global-condition sidecars.

    `src_hash` is a short, deterministic hash of the file's path relative to
    source_dir (posix-normalized, so it is identical on Windows and Linux). It is
    embedded in the chunk file name as `<stem>_<src_hash>__c<idx>` so that two
    different sources whose sanitized stems collide (e.g. "A-B.wav" and "A B.wav"
    both -> "a_b") get DISTINCT output names AND distinct source-group keys
    (the split groups by the pre-"__" token, i.e. `<stem>_<src_hash>`), avoiding
    both silent overwrites and cross-source leakage (report #7).

    With --single_class, every file is placed under one directory
    (class_name or the source basename).

    With a `label_resolver` (--label_source csv) the CSV decides rel_parent
    instead: the label of a file becomes its output folder, whatever the source
    tree looks like. Nothing else changes -- src_hash and rel_posix are still
    computed from the SOURCE path, so no chunk is renamed by switching mode.
    A source file the CSV does not name is an ERROR: the CSV is under your
    control, so a gap in it is a mistake to fix there and not a policy to
    choose per run.
    """
    src = Path(source_dir)
    files = sorted(
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
    )

    out: List[Tuple[Path, Path, str, str, str]] = []
    unlabelled: List[str] = []
    ambiguous: List[str] = []
    for p in files:
        rel_posix = p.relative_to(src).as_posix()
        if label_resolver is not None:
            label, status = label_resolver.lookup(rel_posix)
            if status == "ambiguous":
                ambiguous.append(rel_posix)
                continue
            if status == "missing":
                unlabelled.append(rel_posix)
                continue
            rel_parent = Path(label)
        elif single_class:
            rel_parent = Path(class_name or src.name)
        else:
            rp = p.parent.relative_to(src)
            # files sitting directly in source_dir have no class subfolder
            rel_parent = rp if str(rp) != "." else Path(class_name or src.name)
        leaf_class = rel_parent.name
        src_hash = hashlib.sha1(rel_posix.encode("utf-8")).hexdigest()[:8]
        out.append((p, rel_parent, leaf_class, src_hash, rel_posix))

    if label_resolver is not None:
        _report_csv_labelling(label_resolver, len(files), len(out),
                              unlabelled, ambiguous)
    return out


def _report_csv_labelling(resolver, n_files: int, n_kept: int,
                          unlabelled: List[str], ambiguous: List[str]) -> None:
    """Say what the CSV actually covered, and stop on what it cannot answer.

    This is the only moment where the CSV and the audio folder are both in hand.
    A silent partial match here becomes, hours later, a dataset that is quietly
    missing a class or has a class nobody expected -- so the counts are printed
    every run, not only when something is wrong.
    """
    if ambiguous:
        raise SystemExit(
            f"[labels] {len(ambiguous)} source file(s) match SEVERAL rows of "
            f"{resolver.csv_path.name} carrying DIFFERENT labels, so the CSV "
            f"cannot say what class they are: "
            f"{ambiguous[:8]}{' ...' if len(ambiguous) > 8 else ''}\n"
            f"This happens when the file column holds bare names and the same "
            f"name appears in two folders under different labels. Fix by "
            f"putting the path relative to source_dir in the file column, or "
            f"by removing the duplicate rows.")

    if unlabelled:
        raise SystemExit(
            f"[labels] {len(unlabelled)} source file(s) have no row in "
            f"{resolver.csv_path.name}: "
            f"{unlabelled[:8]}{' ...' if len(unlabelled) > 8 else ''}\n"
            f"Every source must be named by the CSV. Add the rows, or point "
            f"--label_csv at a CSV that covers this source_dir.")

    print(f"[labels] {n_kept}/{n_files} source file(s) labelled from "
          f"{resolver.csv_path.name} ({resolver.n_rows} row(s))")
    unmatched = resolver.unmatched_rows()
    if unmatched:
        # Not an error: one CSV can legitimately cover several folders (a train
        # and a test directory preprocessed separately, say). It IS reported,
        # because the same symptom appears when the CSV and the audio are of
        # different vintages, and then it means files are missing.
        print(f"[labels] {len(unmatched)}/{resolver.n_rows} CSV row(s) matched "
              f"no file under this source_dir: "
              f"{unmatched[:8]}{' ...' if len(unmatched) > 8 else ''}")


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
                          removed: list, prune: bool,
                          labels: Optional[dict] = None):
    """Merge this run's observations into the manifest and persist it.

    `produced` is {rel_source: {"size","mtime_ns","chunks":[...]}} collected from
    the workers. Sources not seen in this run keep their previous entry, unless
    they were pruned.

    `labels` is how this run decided the class of a source (see
    labels_provenance). It is written alongside the sources rather than in
    dataset_meta.json because it does not change a single audio byte -- it
    changes which FOLDER the bytes land in, which is a property of what the
    dataset was built FROM. A run that does not pass it keeps whatever was
    already recorded instead of dropping it.
    """
    merged = dict(prev)
    if prune:
        for rel in removed:
            merged.pop(rel, None)
    merged.update(produced)
    p = out_root / MANIFEST_NAME
    payload = {"sources": merged}
    labels = labels or load_labels_provenance(out_root)
    if labels:
        payload["labels"] = labels
    _atomic_write_json(p, payload)
    return len(merged)


def source_outputs_complete(chunks, latent_root: Path, cond_root: Path,
                            wav_root: Path, n_frames: int, cond_names,
                            check_latents: bool, want_wav: bool) -> bool:
    """True if EVERY output this run would produce for one source already exists.

    The manifest records which chunks a source owns, so this answers "is there
    anything left to do for this file?" from METADATA ALONE -- a handful of
    stat()s and .npy/.npz headers -- instead of decoding the source, re-chunking
    it and discovering chunk by chunk that everything was already there.

    That is the whole cost of a no-op re-run: on a dataset that is already
    complete the pipeline currently re-decodes every mp3 to conclude it has
    nothing to write. Skipping the file outright is what makes "add the wavs of
    the validation split" touch ~10% of the corpus instead of all of it.

    Deliberately conservative: a source with NO recorded chunks is never
    reported complete (it may simply never have been processed), and the checks
    mirror exactly what this run would write -- latents only when it encodes
    them, conditions only for the names it was asked for, wavs only where they
    were requested.
    """
    if not chunks:
        return False
    cond_names = list(cond_names or ())
    for rel_chunk in chunks:
        if check_latents and not _latent_file_is_valid(
                latent_root / f"{rel_chunk}.npy", n_frames, verbose=False):
            return False
        if cond_names and _npz_missing(cond_root / f"{rel_chunk}.npz", cond_names):
            return False
        if want_wav and not (wav_root / f"{rel_chunk}.wav").exists():
            return False
    return True


def filter_complete_sources(files, prev_manifest: dict, changed_srcs,
                            latent_root: Path, cond_root: Path, wav_root: Path,
                            n_frames: int, cond_names, check_latents: bool,
                            wav_sources) -> Tuple[list, int]:
    """Drop the sources that have nothing left to produce. Returns (files, n_skipped).

    A source is only considered when the manifest both KNOWS it and agrees with
    its bytes on disk: an unknown source (dataset built before the manifest, or
    never processed) and a CHANGED one are always re-processed, so the fast path
    can never be the reason something stale survives.
    """
    changed = set(changed_srcs or ())
    kept, skipped = [], 0
    for item in files:
        rel_src = item[4]
        entry = prev_manifest.get(rel_src)
        if entry is None or rel_src in changed:
            kept.append(item)
            continue
        want_wav = (wav_sources == "all"
                    or (isinstance(wav_sources, (set, frozenset))
                        and rel_src in wav_sources))
        if source_outputs_complete(
                entry.get("chunks", []), latent_root, cond_root, wav_root,
                n_frames, cond_names, check_latents, want_wav):
            skipped += 1
        else:
            kept.append(item)
    return kept, skipped


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
# TRAIN / VAL / TEST SPLIT -- decided HERE, at preprocessing time
# ============================================================
# The split used to be recomputed by the training at every startup from whatever
# latents happened to be on disk. Deciding it here makes it a PROPERTY OF THE
# DATASET instead of a property of one training run:
#   * it is decided over the SOURCE FILES, before a single byte is decoded, so
#     "save only the validation wavs" becomes answerable at preprocessing time;
#   * it is WRITTEN DOWN (splits.json) instead of being re-derived, so two runs
#     over the same dataset cannot silently disagree on what the test set was;
#   * it is leakage-safe by construction (the source file is the unit) rather
#     than by reconstructing the source grouping from chunk names afterwards.
#
# The group key is byte-for-byte the one audio_dataset_npy._source_group_of()
# derives from a latent path -- "<rel_parent>/<sanitized_stem>_<src_hash>" -- so
# the training side only has to LOOK EACH LATENT UP. There is exactly one
# implementation of the grouping rule, and it is the one used to name the files.
#
# GROWTH RULE (the one that matters): a re-run NEVER reassigns a source that
# already has one. New sources are placed into the existing split and the
# recorded assignments are read back verbatim. Re-deciding the split of an
# existing dataset silently promotes yesterday's training material to today's
# test set -- an already-trained checkpoint would then be evaluated on data it
# has seen -- so it requires an explicit --resplit.

SPLITS_NAME = "splits.json"
SPLIT_NAMES = ("train", "val", "test")


def source_group_key(rel_parent: Path, src_name: str, src_hash: str) -> str:
    """The leakage-safe id of one source file, as the SPLIT and the CHUNK NAMES
    both see it.

    Chunks are named "<sanitized_stem>_<src_hash>[__ch<n>]__c<idx>" under
    rel_parent, and audio_dataset_npy._source_group_of() recovers the group of a
    latent as "<rel_parent>/<stem before the first '__'>". Building the key from
    the SAME two pieces here is what lets the training resolve a latent to its
    split with a dict lookup instead of a second, drift-prone reimplementation.
    """
    return (Path(rel_parent) / f"{sanitize_filename(src_name)}_{src_hash}").as_posix()


def _split_params(ratios, seed: int, stratify_by_class: bool) -> dict:
    """The knobs that DEFINE the split. Recorded in splits.json and compared on
    every re-run: changing one of them changes who is in the test set."""
    return {
        "ratios": [float(r) for r in ratios],
        "seed": int(seed),
        "stratify_by_class": bool(stratify_by_class),
        # Recorded for the reader's benefit: the split unit is always the source
        # file here (that IS the leakage-safe unit), so it is not a knob.
        "group_by_source": True,
        "unit": "source",
    }


def load_splits(out_root: Path) -> Optional[dict]:
    """Read splits.json, or None if this dataset has no split yet."""
    p = Path(out_root) / SPLITS_NAME
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception as e:
        raise SystemExit(
            f"[split] {p} exists but is unreadable ({e}). Refusing to continue: "
            f"silently recomputing the split would change the test set of a "
            f"dataset that already has one. Fix or delete the file."
        )
    if "groups" not in payload:
        raise SystemExit(f"[split] {p} has no 'groups' key -- refusing to guess.")
    return payload


def assign_source_splits(files, ratios, seed: int, stratify_by_class: bool,
                         existing: Optional[dict] = None) -> Tuple[dict, int]:
    """Assign every scanned source to train/val/test, honouring `existing`.

    `files` are the scan tuples; `existing` maps group_key -> split for the
    sources already assigned. Returns (group_key -> split, n_new).

    The allocation math is IMPORTED from audio_dataset_npy rather than copied:
    the two sides must agree on how a bucket of N groups is cut into
    (train, val, test), and one shared implementation is the only way to keep
    that true as the code moves.
    """
    from audio_dataset_npy import _allocate_three, _seed_for

    existing = dict(existing or {})
    # bucket -> keys, in the same stratification the training used: per class,
    # or one global bucket when stratification is off.
    buckets: dict = {}
    key_class = {}
    for path, rel_parent, leaf_class, src_hash, _rel in files:
        gk = source_group_key(rel_parent, Path(path).name, src_hash)
        bucket = leaf_class if stratify_by_class else "__all__"
        key_class[gk] = bucket
        buckets.setdefault(bucket, set()).add(gk)

    out = {}
    n_new = 0
    for bucket in sorted(buckets):
        keys = sorted(buckets[bucket])
        have = {k: existing[k] for k in keys if k in existing}
        new_keys = [k for k in keys if k not in existing]
        # Deterministic order, independent of PYTHONHASHSEED and of the order
        # the filesystem happened to return the files in.
        random.Random(_seed_for(seed, bucket)).shuffle(new_keys)
        out.update(have)
        if not new_keys:
            continue
        n_new += len(new_keys)

        if not have:
            # Fresh bucket: exactly the training's allocation, same order
            # (train first, then val, then test).
            n_tr, n_val, _n_te = _allocate_three(len(new_keys), ratios)
            for i, k in enumerate(new_keys):
                if i < n_tr:
                    out[k] = "train"
                elif i < n_tr + n_val:
                    out[k] = "val"
                else:
                    out[k] = "test"
            continue

        # GROWING bucket: aim the TOTAL at the ratios instead of cutting the new
        # arrivals on their own -- otherwise adding 3 files to a 300-file class
        # would hand one of them to test regardless of how the class already
        # stands. Existing assignments are read, never rewritten.
        total = len(have) + len(new_keys)
        target = dict(zip(SPLIT_NAMES, _allocate_three(total, ratios)))
        current = {s: sum(1 for v in have.values() if v == s) for s in SPLIT_NAMES}
        # Fill the small splits first: a deficit in test/val is what actually
        # breaks an evaluation, while train absorbs any remainder harmlessly.
        it = iter(new_keys)
        for s in ("test", "val", "train"):
            deficit = max(0, target[s] - current[s])
            for _ in range(deficit):
                k = next(it, None)
                if k is None:
                    break
                out[k] = s
        for k in it:                      # rounding remainder -> train
            out[k] = "train"

    # A key that is in splits.json but no longer on disk stays recorded: its
    # latents may still exist (a source can be deleted after encoding), and
    # forgetting its assignment is how a former test file quietly returns as
    # training data. --prune_orphans is the deliberate way to drop them.
    for k, v in existing.items():
        out.setdefault(k, v)
    return out, n_new


def summarize_splits(groups: dict) -> dict:
    counts = {s: 0 for s in SPLIT_NAMES}
    for v in groups.values():
        if v in counts:
            counts[v] += 1
    return counts


def write_splits(out_root: Path, groups: dict, params: dict, source: str):
    payload = {
        "version": 1,
        "unit": "source",
        "source": source,          # "assigned" | "imported-from-training-split"
        "params": params,
        "counts": summarize_splits(groups),
        "groups": dict(sorted(groups.items())),
    }
    _atomic_write_json(Path(out_root) / SPLITS_NAME, payload)
    return payload


def count_chunks_by_split(out_root: Path, groups: dict) -> Optional[dict]:
    """How many CHUNKS -- the unit the TRAINING actually sees -- each split holds.

    splits.json counts SOURCES, because the source file is the unit the split is
    assigned over (that is what makes it leakage-safe). The number of training
    samples is a different number: a source yields as many chunks as its duration
    allows, so 80/10/10 over sources is only approximately 80/10/10 over samples.

    The manifest records which chunks every source owns, and the group key is
    recovered from a chunk name exactly as audio_dataset_npy._source_group_of()
    does it -- the stem before the first "__", which strips __ch<n>/__c<idx> --
    so this count cannot drift from the one the training reports on startup.

    Returns None when there is no manifest to count (nothing encoded yet): an
    absent number is honest, a zero would be a lie.
    """
    sources = load_source_manifest(Path(out_root))
    if not sources:
        return None
    counts = {s: 0 for s in SPLIT_NAMES}
    unassigned = 0
    for info in sources.values():
        for rel_chunk in info.get("chunks", []):
            parent, _, name = rel_chunk.rpartition("/")
            stem = name.split("__")[0]
            gk = f"{parent}/{stem}" if parent else stem
            split = groups.get(gk)
            if split in counts:
                counts[split] += 1
            else:
                unassigned += 1
    return {"counts": counts, "unassigned": unassigned}


def record_chunk_counts(out_root: Path, groups: dict) -> Optional[dict]:
    """Write the per-split CHUNK counts into splits.json, next to the source ones.

    Written at the END of a run, when the manifest is final: the split is decided
    before anything is encoded, so how many samples it holds is simply not known
    when write_splits() first runs. write_splits() rebuilds the payload from
    scratch, so a later run that CHANGES the assignment drops these numbers
    instead of leaving stale ones behind -- and this puts them back.

    Only 'chunk_counts' is added; 'groups' and 'params' are untouched, so the
    training's cache fingerprint (a digest of the assignment) does not move.
    """
    res = count_chunks_by_split(out_root, groups)
    payload = load_splits(out_root)
    if res is None or payload is None:
        return None
    payload["chunk_counts"] = res["counts"]
    payload["chunk_counts_unassigned"] = res["unassigned"]
    _atomic_write_json(Path(out_root) / SPLITS_NAME, payload)
    return res


def print_chunk_counts(out_root: Path, groups: dict):
    """Record the chunk counts and say them out loud at the end of the run."""
    res = record_chunk_counts(out_root, groups)
    if res is None:
        print("  samples per split: not counted "
              f"({MANIFEST_NAME} records no source yet -- nothing encoded)")
        return
    c = res["counts"]
    print(f"  samples (latent chunks) per split: train {c['train']} | "
          f"val {c['val']} | test {c['test']}  (tot {sum(c.values())})")
    print(f"  recorded as 'chunk_counts' in {Path(out_root) / SPLITS_NAME}")
    if res["unassigned"]:
        print(f"  WARNING: {res['unassigned']} chunk(s) belong to a source with "
              f"no split assignment -- re-run with --split_only to assign it.")


def resolve_splits(out_root: Path, files, ratios, seed: int,
                   stratify_by_class: bool, resplit: bool) -> dict:
    """Load-or-create the dataset's split and return group_key -> split.

    This is the only place a split is decided. It refuses two things loudly:
    a parameter change (the recorded split answers to different numbers than the
    ones asked for) and an implicit re-decision of an existing split.
    """
    want = _split_params(ratios, seed, stratify_by_class)
    prev = load_splits(out_root)

    if prev is not None and resplit:
        print(f"[split] --resplit: discarding the recorded split "
              f"({summarize_splits(prev['groups'])}) and deciding it again. "
              f"Any checkpoint trained on the previous split is now evaluated "
              f"on data it may have seen.")
        prev = None

    existing = None
    if prev is not None:
        old = prev.get("params", {})
        diff = {k: (old.get(k), want[k]) for k in want if old.get(k) != want[k]}
        if diff:
            raise SystemExit(
                f"[split] {Path(out_root) / SPLITS_NAME} was written with "
                f"different split parameters (recorded vs requested): {diff}\n"
                f"  The recorded split is what the existing latents were (or "
                f"will be) trained against. Honour it by passing the same "
                f"values, or pass --resplit to decide the split again from "
                f"scratch -- which reshuffles train/val/test and invalidates "
                f"every evaluation made against the old one."
            )
        existing = prev["groups"]

    groups, n_new = assign_source_splits(
        files, ratios, seed, stratify_by_class, existing=existing)

    if existing is None:
        write_splits(out_root, groups, want, "assigned")
        print(f"[split] created {Path(out_root) / SPLITS_NAME}: "
              f"{summarize_splits(groups)} source(s)")
    elif n_new:
        write_splits(out_root, groups, want, prev.get("source", "assigned"))
        print(f"[split] {n_new} new source(s) assigned into the existing split "
              f"-> {summarize_splits(groups)}")
    else:
        print(f"[split] reusing {Path(out_root) / SPLITS_NAME}: "
              f"{summarize_splits(groups)} source(s), nothing new to assign")

    # A ratio > 0 that produced nothing is a silent trap: the training would
    # later build an empty val/test dataset and fail at the first index.
    counts = summarize_splits(groups)
    r = dict(zip(SPLIT_NAMES, want["ratios"]))
    for name in ("val", "test"):
        if r[name] > 0 and counts[name] == 0:
            raise SystemExit(
                f"[split] the '{name}' split is EMPTY despite ratio={r[name]} > 0. "
                f"Every class has too few source files (a class with a single "
                f"source goes entirely to train, which is what keeps the split "
                f"leakage-safe). Add more sources, disable stratification, or "
                f"set the '{name}' ratio to 0 on purpose.")
    return groups


def import_legacy_split(out_root: Path, latent_root: Path, ratios, seed: int,
                        stratify_by_class: bool) -> dict:
    """Freeze the split the TRAINING would have computed in-code into splits.json.

    Datasets built before the split moved here have no splits.json, and a run
    already in flight was trained against the in-code split. Recomputing it from
    the sources would be a different assignment, so a resumed run would evaluate
    on latents it had already trained on. This reproduces the OLD split exactly
    -- it calls the training's own compute_split() over the latents on disk --
    and records it in the new format, keeping those runs reproducible.
    """
    from audio_dataset_npy import compute_split, _source_group_of

    latent_root = Path(latent_root)
    if not latent_root.exists():
        raise SystemExit(f"[split] --import_legacy_split: no latents under "
                         f"{latent_root}. Nothing to import.")
    res = compute_split(
        latent_root, ratios=tuple(ratios), seed=seed,
        group_by_source=True, stratify_by_class=stratify_by_class,
        save_test_manifest=False,
    )
    groups = {}
    for name in SPLIT_NAMES:
        for f in res["splits"][name]:
            groups[_source_group_of(Path(f), latent_root)] = name
    params = _split_params(ratios, seed, stratify_by_class)
    write_splits(out_root, groups, params, "imported-from-training-split")
    print(f"[split] imported the in-code training split into "
          f"{out_root / SPLITS_NAME}: {summarize_splits(groups)} source(s)")
    return groups


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
# CONDITIONS (incremental merge into per-chunk .npz)
# ------------------------------------------------------------
# THE AXIS THAT MATTERS HERE IS NOT frame-vs-global.
#
# conditions.py divides conditions by what the MODEL does with them: a frame
# condition is concatenated per timestep, a global one is added to the AdaLN
# vector. That is the right axis there and the wrong one here, because the
# preprocessing does not care how a value is consumed -- it cares WHERE the
# value comes from, which is what decides where in this file the work happens:
#
#   PER-CHUNK   computed from one chunk's own audio, so it is produced inside
#               the stream, once per chunk, and merged into that chunk's .npz.
#               Every frame condition is per-chunk, and so is the "text"
#               global (the CLAP embedding of the chunk's audio).
#   PER-CLASS   computed once for a whole class from something that is not the
#               audio at all -- the "image" global reads a folder of pictures
#               -- so it is produced after the stream, into a sidecar.
#
# A global condition is per-chunk exactly when its extractor exposes
# encode_audio(); nothing here holds a list of names, so a condition added to
# conditions.py later lands on the correct side by itself. It is also what lets
# the two kinds compose freely: --conditions and --global_conds only SELECT
# and each name then flows down whichever path its extractor implies. Asking
# for frame conditions alone, globals alone, or any subset of both is therefore
# not a special case anywhere below -- it is the same code with a shorter list.
# ============================================================
def _global_is_chunk_level(ext) -> bool:
    """True when this global condition is computed from a chunk's own audio.

    Read off the OBJECT -- an extractor that can encode_audio() is one -- and
    not from a hardcoded name list, for the same reason _extractor_device_attr
    reads the device off the object: a list here goes stale the moment
    conditions.py gains a condition, and it goes stale SILENTLY, by routing the
    new condition down the wrong path."""
    return callable(getattr(ext, "encode_audio", None))


def _split_globals_by_stage(registry):
    """-> (per-chunk global names, per-class global names), both sorted.

    Both empty when no global was selected, which is what lets a frame-only run
    behave exactly as it did before any of this existed."""
    if registry is None:
        return [], []
    chunk, klass = [], []
    for name, ext in getattr(registry, "global_extractors", {}).items():
        (chunk if _global_is_chunk_level(ext) else klass).append(name)
    return sorted(chunk), sorted(klass)


def _chunk_extractor(registry, name):
    """The object that produces `name` for one chunk -> (extractor, kind).

    `kind` is "frame" or "global", and it exists only because the two are
    CALLED differently: a frame extractor takes (audio, sr, n_frames) and
    returns (n_frames, dim); a per-chunk global takes (audio, sr) and returns
    (dim,). That signature is the last place the distinction survives here."""
    ext = getattr(registry, "frame_extractors", {}).get(name)
    if ext is not None:
        return ext, "frame"
    ext = getattr(registry, "global_extractors", {}).get(name)
    if ext is not None and _global_is_chunk_level(ext):
        return ext, "global"
    raise KeyError(
        f"'{name}' is not a per-chunk condition of this registry "
        f"(frame: {sorted(getattr(registry, 'frame_extractors', {}))}, "
        f"per-chunk global: {_split_globals_by_stage(registry)[0]})")


def _npz_missing(cond_path: Path, names, force: bool = False) -> bool:
    """True if `names` are not all already stored in cond_path (or force).

    Reads only the archive's key list, never the arrays: np.load on an .npz is
    lazy, so `.files` costs one read of the zip central directory. Used to
    decide whether a chunk still has GPU-side work to do, WITHOUT paying for the
    data of a chunk that turns out to be complete."""
    names = set(names or ())
    if not names:
        return False
    if force or not cond_path.exists():
        return True
    try:
        with np.load(str(cond_path)) as data:
            return not names.issubset(set(data.files))
    except Exception:
        return True          # unreadable -> treat as missing, it gets rewritten


def extract_and_merge_chunk_conditions(
    registry, chunk_audio_np, sr: int, n_frames: int,
    cond_path: Path, force: bool = False, names=None,
) -> bool:
    """
    Extract ONLY the per-chunk conditions that are missing from cond_path and
    merge them in (mirrors extract_conditions.py). Returns True if the .npz
    changed.

    Handles both kinds of per-chunk condition, and stores them side by side in
    the same archive under their own names:
        frame  -> (n_frames, dim), one row per latent frame
        global -> (dim,), one vector for the whole chunk
    They are told apart on the way OUT by their shape, so no extra bookkeeping
    file is needed and a reader can stay agnostic.

    `names` restricts the work to a SUBSET of the registry's conditions, which
    is what lets the CPU-side conditions be extracted in the DataLoader workers
    while the GPU-side ones are extracted later, in the main process, from the
    same chunk. None = every per-chunk condition in the registry.

    Splitting the .npz across two producers is safe because the merge below
    always re-reads what is on disk and preserves the keys it was not asked to
    compute -- and because the worker writes BEFORE handing the chunk over, so
    the two writes are ordered by the queue, never concurrent.
    """
    import numpy as np
    if names is None:
        # Everything this registry can produce per chunk: the frame conditions
        # plus the globals that read audio. The per-class globals are NOT here
        # -- they have no per-chunk value to compute.
        names = list(registry.frame_names) + _split_globals_by_stage(registry)[0]
    required = set(names)
    if not required:
        return False

    # Always load what is already on disk (even with force=True) so that
    # re-computing the REQUESTED conditions never drops the OTHERS (report #1).
    existing = {}
    if cond_path.exists():
        try:
            # Read INSIDE a context manager: np.load on an .npz returns a lazy
            # NpzFile that keeps the file OPEN. Leaving it open makes the
            # os.replace() at the end of this function fail on Windows with
            # PermissionError/WinError 5 (POSIX allows renaming over an open
            # file, Windows does not) -- so the whole run dies on the first chunk
            # that already has conditions. The dict comprehension materialises
            # every array before the handle closes, so nothing is lost.
            with np.load(str(cond_path)) as data:
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
        ext, kind = _chunk_extractor(registry, name)
        if kind == "frame":
            arr = ext.extract(chunk_audio_np, sr, n_frames)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != int(n_frames):
                raise RuntimeError(
                    f"frame condition '{name}' produced {arr.shape} for "
                    f"{cond_path.name}; expected ({n_frames}, dim).")
        else:
            arr = np.asarray(ext.encode_audio(chunk_audio_np, sr),
                             dtype=np.float32).reshape(-1)
            # A global embedding is one vector and nothing downstream can tell
            # a degenerate one from a good one by looking at it, so the only
            # cheap check worth making is that it is a number at all. A silent
            # NaN here would reach the AdaLN vector and poison the whole batch.
            if arr.size == 0:
                raise RuntimeError(
                    f"global condition '{name}' produced an empty embedding "
                    f"for {cond_path.name}.")
        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"condition '{name}' contains NaN/Inf for {cond_path.name}.")
        new[name] = arr

    final = {**existing, **new}   # keep others; force overwrites only `required`
    _atomic_save_npz(cond_path, final)
    return True


# ============================================================
# PER-CLASS GLOBAL CONDITIONS (sidecars, written after the stream)
# ============================================================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def write_text_label_vocab(registry, out_root: Path, vocab_path: Optional[str],
                           force: bool = False) -> int:
    """Encode the label vocabulary once and store it WITH the dataset.

        global_conditions/text_vocab.npy    (n_phrases, dim) float32, L2-normed
        global_conditions/text_vocab.json   the phrases, in row order

    WHAT IT IS FOR. The stored 'text' condition is the CLAP embedding of a
    chunk's own audio, and CLAP cannot be run backwards, so a panel has no words
    to put next to a validation sample. With this table it can name the nearest
    phrases instead -- a retrieval, not a translation, which is why the reader
    is always shown the cosine too.

    WHY IT LIVES IN THE DATASET rather than being rebuilt per run: encoding the
    phrases needs CLAP, and doing it at every training start would put that
    model back in the training process for a table that never changes. Written
    once, next to the vectors it will be compared against, and in the same
    space -- built by the run's own text encoder, so a dataset can never end up
    with a vocabulary from a different checkpoint than its chunks.

    Changing the vocabulary later costs ONE re-run of this function (a few
    hundred short strings) and no audio is touched: the labels are recomputed
    from the vectors already on disk.

    `vocab_path` is an optional text file, one phrase per line, replacing the
    built-in list. Returns how many phrases were written (0 = nothing to do).
    """
    import numpy as np
    ext = getattr(registry, "global_extractors", {}).get("text")
    if ext is None:
        return 0

    if vocab_path:
        p = Path(vocab_path)
        if not p.exists():
            print(f"[global/text] --text_vocab {p} not found -> using the "
                  f"built-in vocabulary")
            phrases = None
        else:
            phrases = [ln.strip() for ln in
                       p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            print(f"[global/text] vocabulary from {p}: {len(phrases)} phrases")
    else:
        phrases = None
    if not phrases:
        from conditions import TEXT_LABEL_VOCAB
        phrases = list(TEXT_LABEL_VOCAB)

    d = out_root / "global_conditions"
    d.mkdir(parents=True, exist_ok=True)
    npy, js = d / "text_vocab.npy", d / "text_vocab.json"
    if npy.exists() and js.exists() and not force:
        try:
            old = json.loads(js.read_text(encoding="utf-8"))
            if list(old.get("phrases", [])) == phrases:
                print(f"[global/text] label vocabulary already current "
                      f"({len(phrases)} phrases)")
                return len(phrases)
        except Exception:
            pass          # unreadable -> rebuild

    embs = np.asarray(ext.encode_batch(phrases), dtype=np.float32)
    _atomic_save_npy(npy, embs)
    _atomic_write_json(js, {"phrases": phrases,
                            "model_name": getattr(ext, "model_name", None)})
    print(f"[global/text] label vocabulary: {len(phrases)} phrases encoded "
          f"-> {npy.name}")
    return len(phrases)


TEXT_LABELS_JSONL = "text_labels.jsonl"
TEXT_LABELS_COS = "text_labels_cos.npy"
TEXT_LABELS_EMB = "text_labels_emb.npy"
TEXT_LABELS_TOK = "text_labels_tok.npy"
TEXT_LABELS_TOKLEN = "text_labels_tok_len.npy"
TEXT_LABELS_META = "text_labels.json"


def _text_labels_fingerprint(phrases, model_name, n_terms) -> str:
    """What the sidecar was built from. Any change here invalidates it."""
    h = hashlib.sha1()
    h.update(repr(list(phrases)).encode("utf-8"))
    h.update(str(model_name).encode("utf-8"))
    h.update(str(int(n_terms)).encode("utf-8"))
    return h.hexdigest()[:16]


def format_text_caption(class_name: str, phrases_cos, quote: bool = True) -> str:
    """The one place the caption STRING is built, in its two forms.

    quote=True (for a HUMAN):
        `class · "phrase" (+0.31), "phrase" (+0.24)` -- the class first because
        it is the only part that is true by construction, the phrases after it
        with their cosine because they are a closed-vocabulary retrieval and
        must never be read as a description.

    quote=False (for the CLAP TEXT ENCODER):
        `class, phrase, phrase` -- plain prose. The cosines and the quotes are
        bookkeeping for a reader; handing them to a tokenizer would spend tokens
        on punctuation and numbers that mean nothing in CLAP's space.
    """
    if not phrases_cos:
        return str(class_name)
    if not quote:
        return ", ".join([str(class_name)] + [str(p) for p, _c in phrases_cos])
    body = ", ".join(f"\"{p}\" ({c:+.2f})" for p, c in phrases_cos)
    return f"{class_name} · {body}"


def write_text_labels(out_root: Path, cond_root: Path, n_terms: int = 2,
                      force: bool = False, text_extractor=None) -> int:
    """Write, for EVERY chunk, the text that describes it: its class plus the
    nearest phrases of the vocabulary to its stored CLAP vector.

        global_conditions/text_labels.jsonl    one row per chunk, in a fixed
                                               order: chunk, class, phrases
                                               (with cosines), and the ready
                                               made caption string
        global_conditions/text_labels_cos.npy  (n_chunks, n_phrases) float16,
                                               the FULL cosine table, rows in
                                               the same order as the .jsonl
        global_conditions/text_labels_emb.npy  (n_distinct_captions, dim)
                                               float32, the CLAP TEXT embedding
                                               of each DISTINCT caption
        global_conditions/text_labels.json     what it was built from, plus the
                                               distinct caption list

    WHY THE CAPTION EMBEDDINGS. The validation can be asked to generate FROM the
    description instead of from the chunk's own audio vector (see
    sampling.validation_text_from_caption in the training config): the slot then
    receives the CLAP TEXT embedding of the caption, which is what a prompt
    would put there at inference, and the audio-text similarity becomes a real
    audio-vs-text score instead of an audio-vs-audio one. Encoding those strings
    needs the CLAP text tower, which the preprocessing already holds and the
    training deliberately does not -- so they are computed here, once.

    Only the DISTINCT captions are encoded, with an index per chunk: with
    --text_labels_n 1 a four-class corpus has four of them, not one per chunk.

    WHY IT EXISTS AT ALL. This description used to be computed inside the
    training process, at panel-drawing time, for the handful of validation
    samples that get a panel -- and it existed nowhere else. It is a property of
    the DATASET, not of a run: computing it for every chunk costs one matrix
    product against vectors that are already on disk, and it makes the text
    condition inspectable, countable and reusable for anything else later. The
    number of panels a machine can afford must not decide how much of the
    dataset gets described.

    WHY THE FULL COSINE TABLE TOO. `n_terms` is a presentation choice, and
    storing only the top-N would freeze it: wanting three phrases instead of two
    next month would mean recomputing from the .npz files. The table is
    (n_chunks x n_phrases) float16 -- 20 MB for 50k chunks and 200 phrases --
    and any N, any threshold, any other question can be re-derived from it.

    NOT A DESCRIPTION, A RETRIEVAL. CLAP cannot be run backwards: these phrases
    are the nearest entries of a closed list, and the cosine is stored beside
    every one of them so the two are never confused. A phrase at +0.08 is the
    least bad match in the list, not a statement about the audio.

    `n_terms` counts the caption's terms INCLUDING the class: 1 = the class
    alone, 2 = class + the nearest phrase, 3 = class + the two nearest.

    Returns the number of chunks described (0 = nothing to do).
    """
    import numpy as np

    d = out_root / "global_conditions"
    vocab_npy, vocab_js = d / "text_vocab.npy", d / "text_vocab.json"
    if not (vocab_npy.exists() and vocab_js.exists()):
        print("[global/text] no text_vocab on disk -> no per-chunk labels "
              "written (nothing to compare the vectors against).")
        return 0
    try:
        meta = json.loads(vocab_js.read_text(encoding="utf-8"))
        phrases = list(meta.get("phrases", []))
        model_name = meta.get("model_name")
        vocab = np.load(str(vocab_npy)).astype(np.float32)
    except Exception as e:
        print(f"[global/text] text_vocab unreadable ({type(e).__name__}: {e}) "
              f"-> no per-chunk labels written.")
        return 0
    if not phrases or vocab.ndim != 2 or vocab.shape[0] != len(phrases):
        print(f"[global/text] text_vocab is inconsistent "
              f"({len(phrases)} phrases vs {getattr(vocab, 'shape', None)}) "
              f"-> no per-chunk labels written.")
        return 0

    n_terms = max(1, int(n_terms))
    k = min(max(0, n_terms - 1), len(phrases))     # phrases beside the class

    # Every chunk that HAS a stored text vector, in a deterministic order.
    files = sorted(p for p in cond_root.rglob("*.npz")) if cond_root.exists() else []
    if not files:
        print("[global/text] no per-chunk conditions on disk -> no labels.")
        return 0

    fp = _text_labels_fingerprint(phrases, model_name, n_terms)
    jsonl_p, cos_p, meta_p = (d / TEXT_LABELS_JSONL, d / TEXT_LABELS_COS,
                              d / TEXT_LABELS_META)
    if not force and jsonl_p.exists() and cos_p.exists() and meta_p.exists():
        try:
            old = json.loads(meta_p.read_text(encoding="utf-8"))
            # Compared against the number of .npz SCANNED, not the number of
            # rows written: a chunk without a text vector produces no row, so
            # comparing rows would make an incomplete dataset rebuild the
            # sidecar on every single run.
            # The fingerprint says the INPUTS are unchanged. It says nothing
            # about the OUTPUTS, and this function has grown new ones: a sidecar
            # written before the caption embeddings, or before the token
            # sequences, has a matching fingerprint and is still missing the
            # files a run needs. Without this the only way to get them would be
            # a global --force (which re-encodes every latent) or an unrelated
            # change to n_terms, and the user would be told "already current"
            # by a function that had just decided not to produce what they came
            # for.
            wants = []
            if text_extractor is not None:
                wants.append(d / TEXT_LABELS_EMB)
                if hasattr(text_extractor, "encode_tokens"):
                    wants += [d / TEXT_LABELS_TOK, d / TEXT_LABELS_TOKLEN]
            absent = [p.name for p in wants if not p.exists()]
            if (old.get("fingerprint") == fp
                    and int(old.get("n_npz", -1)) == len(files)
                    and not absent):
                print(f"[global/text] per-chunk labels already current "
                      f"({old.get('n_chunks')} chunks, {n_terms} term(s))")
                return int(old.get("n_chunks", 0))
            if absent and old.get("fingerprint") == fp:
                print(f"[global/text] labels are current but {', '.join(absent)} "
                      f"{'is' if len(absent) == 1 else 'are'} missing -> "
                      f"rewriting the sidecar (no audio is read, no latent is "
                      f"re-encoded)")
        except Exception:
            pass          # unreadable -> rebuild

    rows, cos_rows, n_missing = [], [], 0
    for p in files:
        rel = p.relative_to(cond_root).with_suffix("").as_posix()
        try:
            with np.load(str(p)) as z:
                if "text" not in z.files:
                    n_missing += 1
                    continue
                vec = np.asarray(z["text"], dtype=np.float32).reshape(-1)
        except Exception as e:
            print(f"  [global/text] {rel}: {type(e).__name__}: {e} -> skipped")
            n_missing += 1
            continue
        if vec.shape[0] != vocab.shape[1] or not np.isfinite(vec).all():
            n_missing += 1
            continue
        # Both sides are L2-normalized, so the dot product IS the cosine.
        sims = vocab @ vec
        cos_rows.append(sims.astype(np.float16))
        top = np.argsort(-sims)[:k] if k else []
        best = [[phrases[i], round(float(sims[i]), 4)] for i in top]
        # The class is the parent directory of the chunk, exactly as
        # audio_dataset_cond._class_of_file reads it back.
        cls = Path(rel).parent.name or ""
        rows.append({"chunk": rel, "class": cls, "phrases": best,
                     "caption": format_text_caption(cls, best)})

    if not rows:
        print(f"[global/text] no chunk carries a 'text' vector "
              f"({len(files)} .npz inspected) -> no labels written.")
        return 0

    # ---- the distinct captions, and their CLAP TEXT embedding --------------
    # Distinct, because a caption is a function of the class and of a handful of
    # phrases: with n_terms=1 a four-class corpus has exactly four of them. The
    # per-chunk row keeps an index into this table rather than a copy of a
    # 512-float vector.
    # TWO STRINGS PER CAPTION, and the difference is ONLY punctuation. `caption`
    # is what a human reads: `class · "phrase" (+0.31)`. `caption_text` is what
    # the CLAP text encoder is given: `class, phrase` -- the quotes and the
    # cosines are bookkeeping for a reader and would spend tokens on nothing.
    # THE CLASS NAME ITSELF IS NEVER TOUCHED, in either. If it does not read as
    # plain words, fix it where it is decided -- the CSV, or --label_aliases --
    # rather than here: a class rewritten at encode time would no longer be the
    # class the folder, the split and the panels talk about, and a rule clever
    # enough to do it would also be clever enough to be wrong in silence.
    # How well the resulting vectors separate is measured and printed below.
    # DISTINCT ON THE ENCODED STRING, not on the human one. The two differ by
    # the cosines: `drum · "electronic dance music" (+0.54)` and the same phrase
    # at (+0.53) are two different HUMAN captions and one single input to CLAP,
    # so keying the table on the human string stored the identical vector twice
    # -- and, since the token sequences arrived, the identical (L, 768) matrix
    # twice as well. Measured 14 Sept 2026 on 20 chunks: 15 rows keyed the old
    # way, 6 keyed this way, and the embeddings of the collapsed rows were
    # bit-identical. The per-chunk `caption` in the .jsonl is untouched, so a
    # panel still prints that chunk's own cosines.
    captions, captions_text, caption_id = [], [], {}
    for r in rows:
        enc = format_text_caption(r["class"], r["phrases"], quote=False)
        if enc not in caption_id:
            caption_id[enc] = len(captions_text)
            captions_text.append(enc)
            # The human string of the FIRST chunk that produced this encoding,
            # kept parallel to captions_text so every reader that checks one
            # length against the other keeps working.
            captions.append(r["caption"])
        r["caption_id"] = caption_id[enc]
    cap_emb = None
    if text_extractor is not None:
        try:
            cap_emb = np.asarray(text_extractor.encode_batch(captions_text),
                                 dtype=np.float32)
            if cap_emb.ndim != 2 or cap_emb.shape[0] != len(captions):
                raise ValueError(f"encode_batch returned {cap_emb.shape} for "
                                 f"{len(captions)} caption(s)")
        except Exception as e:
            # The captions themselves are still worth writing: losing their
            # embeddings costs the caption-conditioned validation, not the file.
            print(f"[global/text] caption embeddings NOT written "
                  f"({type(e).__name__}: {e}). The descriptions are still "
                  f"stored; sampling.validation_text_from_caption will have "
                  f"nothing to read.")
            cap_emb = None

    # ---- the same captions as TOKEN SEQUENCES, for the cross-attention -----
    # A second encoding of the SAME strings, and the two are not redundant: the
    # pooled vector above is one point of the shared audio-text space and is
    # what the AdaLN slot takes, while these are the text tower's per-token
    # states and are what a cross-attention can actually attend over. One
    # vector cannot be attended to -- an attention with a single key is a
    # learned bias -- so the model needs both or neither.
    #
    # Distinct captions again, padded to the longest, with the true lengths
    # beside them: a token of padding that reaches K and V is a token the model
    # reads as a word. float16 like the cosine table, and for the same reason --
    # these are inputs to a Linear, not accumulators, and the file is small
    # enough to keep in full rather than storing a per-chunk copy.
    cap_tok = cap_len = None
    if text_extractor is not None and hasattr(text_extractor, "encode_tokens"):
        try:
            cap_tok, cap_len = text_extractor.encode_tokens(captions_text)
            if cap_tok.ndim != 3 or cap_tok.shape[0] != len(captions):
                raise ValueError(f"encode_tokens returned {cap_tok.shape} for "
                                 f"{len(captions)} caption(s)")
        except Exception as e:
            print(f"[global/text] caption TOKEN sequences NOT written "
                  f"({type(e).__name__}: {e}). A run with "
                  f"model.text_cross_every > 0 will refuse to start on this "
                  f"dataset; everything else is unaffected.")
            cap_tok = cap_len = None

    d.mkdir(parents=True, exist_ok=True)
    tmp = jsonl_p.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(jsonl_p))
    _atomic_save_npy(cos_p, np.stack(cos_rows, axis=0))
    if cap_emb is not None:
        _atomic_save_npy(d / TEXT_LABELS_EMB, cap_emb)
    else:
        # A stale table from an earlier build would silently describe other
        # captions, so it goes rather than being left behind.
        try:
            (d / TEXT_LABELS_EMB).unlink()
        except FileNotFoundError:
            pass
    if cap_tok is not None:
        _atomic_save_npy(d / TEXT_LABELS_TOK, cap_tok.astype(np.float16))
        _atomic_save_npy(d / TEXT_LABELS_TOKLEN, cap_len.astype(np.int32))
    else:
        # Same rule, and it matters more here: a stale token table would feed
        # the cross-attention the words of a vocabulary that no longer exists,
        # and nothing downstream could tell.
        for stale in (TEXT_LABELS_TOK, TEXT_LABELS_TOKLEN):
            try:
                (d / stale).unlink()
            except FileNotFoundError:
                pass
    _atomic_write_json(meta_p, {
        "fingerprint": fp, "n_chunks": len(rows), "n_npz": len(files),
        "n_terms": n_terms,
        "n_phrases": len(phrases), "model_name": model_name,
        "phrases": phrases,
        "captions": captions,
        "captions_text": captions_text,
        "caption_emb": bool(cap_emb is not None),
        "caption_tokens": bool(cap_tok is not None),
        "caption_tok_dim": int(cap_tok.shape[2]) if cap_tok is not None else 0,
        "caption_tok_max_len": int(cap_tok.shape[1]) if cap_tok is not None else 0,
        "cos_dtype": "float16",
        "note": ("Row i of text_labels_cos.npy is line i of text_labels.jsonl. "
                 "The phrases are a closed-vocabulary retrieval over the stored "
                 "CLAP vector, not a description: read every one with its "
                 "cosine. The class is exact, read off the chunk's folder."),
    })

    distinct = len({tuple(p for p, _c in r["phrases"]) for r in rows})
    print(f"[global/text] per-chunk labels: {len(rows)} chunk(s), "
          f"{n_terms} term(s) each, {distinct} distinct phrase combination(s) "
          f"over {len(phrases)} phrases -> {jsonl_p.name}")
    print(f"[global/text] {len(captions)} distinct caption(s)"
          + (f", CLAP-text encoded -> {TEXT_LABELS_EMB}" if cap_emb is not None
             else ", NOT encoded (no text extractor)"))
    if cap_tok is not None:
        print(f"[global/text] caption token sequences: "
              f"{cap_tok.shape[0]} x {cap_tok.shape[1]} token(s) x "
              f"{cap_tok.shape[2]} -> {TEXT_LABELS_TOK} "
              f"(lengths {cap_len.min()}..{cap_len.max()}, float16). "
              f"This is what model.text_cross_every reads.")
    if cap_emb is not None:
        # The spread of the conditioning vectors IS the signal the text slot can
        # carry at validation. Two captions at 0.97 cannot be told apart by the
        # model or by the similarity metric, so the number is printed rather
        # than left to be discovered in a flat curve three hours later.
        if len(captions) > 1:
            M = cap_emb @ cap_emb.T
            off = M[~np.eye(len(captions), dtype=bool)]
            print(f"[global/text] caption embeddings: mean cosine between "
                  f"DIFFERENT captions {float(off.mean()):+.3f} "
                  f"(max {float(off.max()):+.3f}) -- the lower, the more the "
                  f"text slot can distinguish them")
        for _c, _t in list(zip(captions, captions_text))[:4]:
            print(f"    {_c!r}  ->  encoded as {_t!r}")
    if n_missing:
        print(f"[global/text] {n_missing} chunk(s) had no usable 'text' vector "
              f"and are absent from the sidecar.")
    return len(rows)


def extract_class_global_conditions(
    registry, out_root: Path, classes: List[str], image_root: Optional[str],
    force: bool = False,
) -> dict:
    """
    Build the sidecar of every PER-CLASS global condition. Currently that is
    "image": every picture of a class, encoded with CLIP, stacked into one bank.

        global_conditions/image/<class>.npy    (n_images, dim) float32
        global_conditions/image/<class>.json   the file names, same order

    THE WHOLE BANK IS STORED, not a sample of it. The training draws a random
    image of the sound's class at every epoch -- that is the augmentation the
    condition exists for -- so capping the bank here would silently cap the
    augmentation for every run that ever reads this dataset. (The training used
    to encode at most 10 images per class on the fly, at startup, with CLIP
    loaded in the training process; this replaces that entirely.)

    The .json is not bookkeeping for its own sake: the validation panels have
    to SHOW the image a generation was conditioned on, and an embedding cannot
    be turned back into a picture. Same order as the rows, so index i in the
    bank is file i in the list.

    Returns {class: n_images} for what it wrote or found, so the caller can say
    something true about coverage instead of guessing.

    NOT here: "text". Its value is per CHUNK (the CLAP embedding of that chunk's
    audio) and lives in the chunk's own .npz, written during the stream. It used
    to be a per-class sidecar holding the embedding of the class NAME; that file
    is no longer produced, and a leftover one from an older run is stale.
    """
    import numpy as np
    _, class_globals = _split_globals_by_stage(registry)
    if not class_globals:
        return {}

    written = {}
    if "image" in class_globals:
        if not image_root or not Path(image_root).exists():
            print(f"[global/image] --image_root {image_root or '(unset)'} "
                  f"missing/not found -> NOTHING WRITTEN. The image condition "
                  f"will have no bank to read.")
            return {}
        ext = registry.global_extractors["image"]
        d = out_root / "global_conditions" / "image"
        d.mkdir(parents=True, exist_ok=True)
        missing_dir, empty_dir, refreshed = [], [], []
        for c in classes:
            p = d / f"{sanitize_class_name(c)}.npy"
            jp = p.with_suffix(".json")
            cls_dir = Path(image_root) / c      # raw name: matches the source folder
            if not cls_dir.exists():
                # An existing bank for a class whose folder is gone is kept: it
                # is still the bank the dataset was built with, and refusing to
                # report it would look like the condition had vanished.
                if p.exists() and jp.exists():
                    try:
                        written[c] = int(np.load(str(p), mmap_mode="r").shape[0])
                        continue
                    except Exception:
                        pass
                missing_dir.append(c)
                continue
            imgs = sorted(q for q in cls_dir.rglob("*")
                          if q.suffix.lower() in IMAGE_EXTS)
            # A bank on disk is reused only while it still describes the FOLDER.
            # Comparing the recorded file list with what is there now is one
            # directory listing, and it is what makes "drop ten new pictures in
            # and re-run" work: the old test (the file merely exists) meant a
            # grown class was silently stuck with yesterday's bank until someone
            # ran --force, which also recomputes every chunk condition in the
            # dataset. A class whose folder is unchanged is still not re-encoded.
            if p.exists() and jp.exists() and not force:
                try:
                    recorded = list(json.loads(
                        jp.read_text(encoding="utf-8")).get("files", []))
                    n_rows = int(np.load(str(p), mmap_mode="r").shape[0])
                    if (len(recorded) == n_rows
                            and recorded == [q.name for q in imgs]):
                        written[c] = n_rows
                        continue
                    if imgs:
                        refreshed.append(f"{c}({len(recorded)}->{len(imgs)})")
                except Exception:
                    pass          # unreadable -> fall through and rebuild
            if not imgs:
                empty_dir.append(c)
                continue
            stack, kept = [], []
            for q in imgs:
                try:
                    stack.append(np.asarray(ext.encode_image(str(q)),
                                            dtype=np.float32))
                    kept.append(q.name)
                except Exception as e:
                    # One unreadable picture must not cost the whole class its
                    # bank; say which, and carry on with the rest.
                    print(f"  [global/image] {c}/{q.name}: "
                          f"{type(e).__name__}: {e} -> skipped")
            if not stack:
                empty_dir.append(c)
                continue
            _atomic_save_npy(p, np.stack(stack, axis=0))
            _atomic_write_json(jp, {"class": c, "files": kept})
            written[c] = len(kept)

        if hasattr(ext, "unload"):
            ext.unload()
        total = sum(written.values())
        print(f"[global/image] {total} images over "
              f"{len(written)}/{len(classes)} classes -> {d}")
        if refreshed:
            # Say it out loud: the bank changed, so the image condition of those
            # classes is not the one an earlier checkpoint was trained with.
            print(f"[global/image] re-encoded {len(refreshed)} class(es) whose "
                  f"folder changed since the bank was written: "
                  f"{refreshed[:8]}{' ...' if len(refreshed) > 8 else ''}")
        # A class with no images is not fatal -- the training falls back to a
        # null image for it -- but it IS the difference between conditioning
        # and not conditioning those samples, so it is reported, not swallowed.
        if missing_dir:
            print(f"[global/image] WARNING: no folder under {image_root} for "
                  f"{len(missing_dir)} class(es): "
                  f"{missing_dir[:8]}{' ...' if len(missing_dir) > 8 else ''}")
        if empty_dir:
            print(f"[global/image] WARNING: folder present but no usable image "
                  f"for {len(empty_dir)} class(es): "
                  f"{empty_dir[:8]}{' ...' if len(empty_dir) > 8 else ''}")
    return written


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
# the workers entirely:
#   * WORKERS (CPU, parallel): load + acoustic ffmpeg + chunking + the CPU-only
#     conditions (chroma, energy) + WAV writing. Each is a per-chunk side
#     effect written straight to disk (distinct files, no contention). Conditions
#     are aligned to n_frames_fixed (the constant DAC T for a fixed chunk length,
#     discovered once on the main GPU), so a worker never waits for the DAC.
#   * MAIN (GPU): batched DAC encode AND the GPU-capable conditions (f0, rhythm)
#     over the same chunks, in one pass. The card has exactly one owner, which is
#     what makes worker parallelism and GPU conditions compatible instead of an
#     either/or -- N workers each holding a model would contend for VRAM and
#     time-slice the GPU between processes.
# Workers yield a chunk when it still needs a latent OR a GPU-side condition, the
# two tracked independently so each resumes on its own; everything else is fully
# handled worker-side. With num_workers=0 this same path runs in-process.
# ============================================================
def _shard_files(files, worker_id: int, num_workers: int):
    return files[worker_id::num_workers] if num_workers and num_workers > 1 else files


def _extractor_device_attr(ext) -> Optional[str]:
    """Name of the attribute an extractor keeps its device in, or None if it has
    no device at all (a pure-DSP extractor: chroma, energy).

    Two names because the classes disagree: CrepeF0Extractor exposes `device`,
    RhythmExtractor `_device`. Checking only the public one -- which is what this
    file used to do -- silently missed rhythm, so a call meant to pin everything
    to CPU left it wherever it was. Its default happens to be "cpu", so nothing
    broke; the guard was simply not doing what its name said."""
    for attr in ("device", "_device"):
        if hasattr(ext, attr):
            return attr
    return None


def _all_chunk_extractors(registry) -> dict:
    """{name: extractor} for everything computed per chunk: the frame
    conditions and the globals that read audio. The per-class globals are left
    out -- they never see a chunk, so nothing about devices or workers applies
    to them."""
    if registry is None:
        return {}
    out = dict(getattr(registry, "frame_extractors", {}))
    chunk_globals, _ = _split_globals_by_stage(registry)
    gexts = getattr(registry, "global_extractors", {})
    out.update({n: gexts[n] for n in chunk_globals})
    return out


def _split_extractors_by_device(registry):
    """Partition the PER-CHUNK conditions into (gpu_capable, cpu_only) names.

    GPU-capable = has a device knob (f0 via torchcrepe, rhythm via beat_this,
    text via the CLAP audio tower). CPU-only = pure DSP (chroma, energy) or a
    backend with no device knob exposed here. Driven by the extractor objects
    rather than a hardcoded name list, so a condition added later lands on the
    right side by itself -- which is how the CLAP audio encoder ended up on the
    GPU side without a line here mentioning it."""
    gpu, cpu = [], []
    for name, ext in _all_chunk_extractors(registry).items():
        (gpu if _extractor_device_attr(ext) else cpu).append(name)
    return sorted(gpu), sorted(cpu)


def _set_extractor_device(registry, names, device: str):
    """Point the named extractors at `device`. The device never changes the
    values an extractor produces, only the speed -- so this is safe to flip per
    run and is deliberately NOT part of any output fingerprint."""
    if registry is None:
        return
    exts = _all_chunk_extractors(registry)
    for name in names or ():
        ext = exts.get(name)
        if ext is None:
            continue
        attr = _extractor_device_attr(ext)
        if attr:
            setattr(ext, attr, device)


def _force_cpu_extractors(registry, names=None):
    """Pin the named extractors to CPU (all of them if `names` is None).

    Why it still exists now that the GPU-side conditions run in the main
    process: the extractors that STAY in the DataLoader workers must never touch
    CUDA. N workers each holding a model on the one card would contend with the
    DAC for VRAM and time-slice the GPU between processes. Since the GPU-side
    conditions have left the workers entirely, this now applies to the CPU-side
    group only -- applying it to all of them, as before, would put f0 back on
    CPU and undo the point of the split."""
    if registry is None:
        return
    if names is None:
        names = list(_all_chunk_extractors(registry).keys())
    _set_extractor_device(registry, names, "cpu")


def _process_gpu_batch(dac_enc, batch, sr, n_frames: int,
                       registry=None, gpu_names=None, force=False):
    """Everything the GPU owes this batch of chunks, in the MAIN process.

    Two GPU steps over the SAME chunks, back to back:
      1. the batched DAC encode -> latents;
      2. the GPU-side conditions (f0 / rhythm) -> merged into each chunk's .npz.

    They cannot share one tensor -- the DAC wants 44.1 kHz audio and CREPE wants
    16 kHz, they are different models with different inputs -- but they run over
    the same chunks in the same pass, which is what keeps the card busy instead
    of alternating between a busy CPU and an idle GPU.

    The conditions are NOT batched across chunks, on purpose: torchcrepe already
    batches internally at the FRAME level (its `batch_size` is CREPE frames per
    forward), and one 5 s chunk is ~500 frames -- a full forward on its own once
    `batch_size` is raised to match. Stacking chunks on top of that would add
    framing complexity at the boundaries for no throughput.

    Each item carries its own two flags: `latent_path` is None when the latent is
    already valid, `cond_path` is None when the .npz already holds every GPU-side
    condition. A chunk is handed over if EITHER is outstanding, so the two kinds
    of work resume independently -- adding f0 to a dataset whose latents are all
    encoded re-reads the audio without re-encoding a single latent.

    Returns (n_latents, n_conditions) actually written.
    """
    if not batch:
        return 0, 0

    n_lat = _encode_and_save_batch(
        dac_enc, [it for it in batch if it.get("latent_path")], sr, n_frames)

    n_cond = 0
    if registry is not None and gpu_names:
        for it in batch:
            cp = it.get("cond_path")
            if not cp:
                continue
            audio = it["audio"]
            chunk_np = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
            extract_and_merge_chunk_conditions(
                registry, chunk_np, sr, n_frames, Path(cp),
                force=force, names=gpu_names)
            n_cond += 1
    return n_lat, n_cond


def _encode_and_save_batch(dac_enc, batch, sr, n_frames: int) -> int:
    """Batched DAC encode of a list of worker items -> save each latent."""
    if not batch or dac_enc is None:
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
                 sr, wav_sources, force, n_frames_fixed,
                 acoustic, acoustic_kw, silence_thresh_db, tmp_dir,
                 cpu_cond_names=None, gpu_cond_names=None):
        self.files = files
        self.chunker = chunker
        self.registry = registry
        # Which conditions this worker extracts itself, and which it only has to
        # REPORT as outstanding so the main process can do them on the GPU.
        self.cpu_cond_names = list(cpu_cond_names or [])
        self.gpu_cond_names = list(gpu_cond_names or [])
        self.latent_root = Path(latent_root)
        self.wav_root = Path(wav_root)
        self.cond_root = Path(cond_root)
        self.sr = sr
        # "all" | None | set of rel_source. The SET form is what makes
        # "keep only the validation wavs" cost nothing to express here: main
        # resolves the split once and ships the (small) selected subset, so a
        # worker never needs the split table to decide.
        self.wav_sources = wav_sources
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

        has_cpu_conditions = self.registry is not None and bool(self.cpu_cond_names)
        has_gpu_conditions = self.registry is not None and bool(self.gpu_cond_names)

        for path, rel_parent, _leaf, src_hash, rel_src in shard:
            want_wav = (self.wav_sources == "all"
                        or (isinstance(self.wav_sources, (set, frozenset))
                            and rel_src in self.wav_sources))
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

                        # CPU-side per-chunk conditions, in the worker, aligned
                        # to the fixed T. The GPU-side ones (f0 / rhythm, and
                        # the CLAP embedding behind the 'text' global) are NOT
                        # done here: they belong to the main process, which is
                        # the only one allowed to touch CUDA.
                        if has_cpu_conditions:
                            chunk_np = chunk.squeeze(0).cpu().numpy()
                            extract_and_merge_chunk_conditions(
                                self.registry, chunk_np, self.sr,
                                self.n_frames_fixed, cond_path,
                                force=self.force, names=self.cpu_cond_names)

                        # optional per-chunk WAV (also worker-side)
                        if want_wav and (self.force or not wav_path.exists()):
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
                        # The .npz was just written above (CPU conditions), so
                        # this reads the CURRENT state: only the GPU-side names
                        # still absent make the chunk worth handing over.
                        needs_gpu_cond = has_gpu_conditions and _npz_missing(
                            cond_path, self.gpu_cond_names, force=self.force)
                        if needs_latent or needs_gpu_cond:
                            audio = chunk.squeeze(0).detach().clone()
                            # Each flag travels as "the path to write, or None":
                            # the main process then does exactly the outstanding
                            # half and never redoes the finished one.
                            yield {"audio": audio,
                                   "latent_path": (str(latent_path)
                                                   if needs_latent else None),
                                   "cond_path": (str(cond_path)
                                                 if needs_gpu_cond else None)}
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
def build_parser():
    parser = argparse.ArgumentParser(
        description="Streaming preprocessing (chunk -> DAC encode -> latents), "
                    "supervisor-style. Optional per-chunk WAV/conditions, "
                    "incremental, with the train/val/test split decided here "
                    "and recorded in splits.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Optional so the whole run can be described by --config alone; a positional
    # given on the command line still wins over the file (see _resolve_args).
    parser.add_argument("source_dir", type=str, nargs="?", default=None)
    parser.add_argument("output_dir", type=str, nargs="?", default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file with any of these options (keys are the "
                             "long flag names without '--'). Precedence: "
                             "built-in defaults < config file < command line, so "
                             "a flag you type always wins over the file.")

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
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for the DAC encoder (and, unless "
                             "--cond_device says otherwise, for the GPU-capable "
                             "condition extractors).")
    parser.add_argument("--cond_device", type=str, default=None,
                        help="Device for the GPU-capable condition extractors "
                             "(f0 via torchcrepe, rhythm via beat_this). "
                             "Default: follow --device. These run in the MAIN "
                             "process, next to the DAC batch, so no worker ever "
                             "touches CUDA. Pure-DSP conditions (chroma, energy) "
                             "always run on CPU in the workers, "
                             "whatever this says. The device changes speed only, "
                             "never the extracted values.")
    parser.add_argument("--save_wav", type=str, nargs="?", default="none",
                        const="all", metavar="SPLITS",
                        help="Also save the per-chunk WAV under wav/<class>/, for "
                             "the listed splits: 'none' (default), 'all', or a "
                             "comma-separated subset of train,val,test (e.g. "
                             "--save_wav val). A bare --save_wav means 'all', as "
                             "before. Anything but 'none'/'all' needs the split, "
                             "so it is resolved from splits.json. The wavs are "
                             "the REAL source audio (never DAC round-trips): "
                             "'val' is what a standard FAD reference is built "
                             "from, and it costs ~10x less disk than 'all'.")

    # ---- train / val / test split (decided here, recorded in splits.json) ----
    parser.add_argument("--split_ratios", type=str, default="0.8,0.1,0.1",
                        help="train,val,test ratios over SOURCE FILES (not "
                             "chunks). Applied per class when stratifying.")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed of the split. Part of its identity: changing "
                             "it on an existing dataset is refused without "
                             "--resplit.")
    parser.add_argument("--no_stratify", action="store_true",
                        help="Do NOT balance the split per class (default is to "
                             "stratify, matching the training's old behaviour).")
    parser.add_argument("--resplit", action="store_true",
                        help="Decide the split again from scratch, discarding "
                             "the recorded one. This REASSIGNS sources across "
                             "train/val/test, so any checkpoint trained on the "
                             "old split is then evaluated on data it has seen.")
    parser.add_argument("--split_only", action="store_true",
                        help="Only create/extend splits.json and exit: no "
                             "decoding, no encoding, no conditions. This is how "
                             "an already-preprocessed dataset gets a split.")
    parser.add_argument("--import_legacy_split", action="store_true",
                        help="Write splits.json by reproducing the split the "
                             "TRAINING used to compute in-code, from the latents "
                             "on disk. Use it once on datasets built before the "
                             "split moved here, so runs already in flight keep "
                             "the exact same val/test sets. Implies --split_only.")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated frame conditions to extract, e.g. "
                             "'f0' or 'f0,energy'. None = skip conditions.")
    # TWO SPELLINGS, one dest. `--global_conds` is the primary one because it
    # matches the YAML key, which is the rule every other option in this parser
    # follows: a config file key IS the long flag without its dashes, and one
    # option that broke the rule was one option you had to remember. `--global`
    # stays as an alias -- it is in commands that are already written down, and
    # silently removing it would turn them into "unrecognized arguments".
    parser.add_argument("--global_conds", "--global", dest="global_conds",
                        type=str, default=None,
                        help="Comma-separated global conditions to extract "
                             "('text' and/or 'image'). Independent of "
                             "--conditions: either flag can be used alone, "
                             "both together, and each takes any subset. "
                             "'text' is computed PER CHUNK (the CLAP embedding "
                             "of that chunk's audio) and is stored in the same "
                             ".npz as the frame conditions, so it is extracted "
                             "and resumed exactly like f0. 'image' is computed "
                             "PER CLASS from --image_root into "
                             "global_conditions/image/<class>.npy.")
    parser.add_argument("--text_vocab", type=str, default=None,
                        help="Text file, one phrase per line, replacing the "
                             "built-in label vocabulary (conditions.py "
                             "TEXT_LABEL_VOCAB). It is encoded once into "
                             "global_conditions/text_vocab.npy and is what the "
                             "validation panels use to put WORDS next to a "
                             "stored CLAP vector -- a nearest-phrase retrieval, "
                             "shown with its cosine, not a translation. "
                             "Changing it never re-touches the audio.")
    parser.add_argument("--image_root", type=str, default=None,
                        help="image_root/<class>/*.jpg for --global_conds image. The "
                             "class folder names must match the source audio's "
                             "class folders. EVERY image of a class is encoded: "
                             "the training draws one at random per epoch.")

    # class layout
    parser.add_argument("--single_class", action="store_true")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--label_source", type=str, default="dir",
                        help="Where the CLASS of each source comes from: 'dir' "
                             "(default) the source subdirectory, as before; "
                             "'csv' a metadata file. The class is the output "
                             "folder either way, so a CSV-labelled corpus can "
                             "live in one flat directory and still be split, "
                             "conditioned and reported per class. Recorded in "
                             "source_manifest.json: changing it on an existing "
                             "output dir is refused (it would relocate every "
                             "latent instead of overwriting it).")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="The metadata CSV for --label_source csv. One "
                             "file column ('file' or 'filename') and one label "
                             "column ('label' or 'class'), any case; every "
                             "source must have a row. Rows are matched to files "
                             "by path relative to source_dir, then by bare file "
                             "name, then by either case-insensitively.")
    parser.add_argument("--text_labels_n", type=int, default=2,
                        help="How many TERMS the per-chunk text description "
                             "holds, the class included: 1 = the class alone, "
                             "2 = class + the nearest vocabulary phrase, 3 = "
                             "class + the two nearest. Written for EVERY chunk "
                             "into global_conditions/text_labels.jsonl (with "
                             "--global_conds text), together with the full cosine "
                             "table, so another N can be re-derived later "
                             "without re-reading a single .npz.")

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
    return parser


_UNSET = object()      # a default argparse will never try to type-convert


def _explicitly_given(argv=None) -> dict:
    """Which options the user actually TYPED, as opposed to inherited defaults.

    A second parser whose every default is a unique sentinel returns only the
    options present on the command line, which is what lets the config file fill
    in the rest without ever overriding something asked for explicitly.

    The sentinel is an object() and not argparse.SUPPRESS on purpose: SUPPRESS is
    the STRING '==SUPPRESS==', and argparse runs `type=` over string defaults --
    so '--sr' would fail with int('==SUPPRESS==') before parsing even started.
    """
    p = build_parser()
    for a in p._actions:
        if a.dest != "help":
            a.default = _UNSET          # the action, not parser.set_defaults()
    ns = vars(p.parse_args(argv))
    return {k: v for k, v in ns.items() if v is not _UNSET}


def _load_preprocess_config(path: str, parser) -> dict:
    """Read the YAML and validate its keys against the parser.

    An unknown key is an ERROR, not a shrug: a typo in a config that is meant to
    be the memory of how a dataset was built would otherwise be indistinguishable
    from a setting that silently did nothing."""
    import yaml
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[config] no such file: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise SystemExit(f"[config] {p} is not readable YAML: {e}")
    if not isinstance(raw, dict):
        raise SystemExit(f"[config] {p} must contain a mapping of option -> value.")
    known = {a.dest for a in parser._actions if a.dest != "help"} - {"config"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise SystemExit(
            f"[config] {p} has {len(unknown)} unknown option(s): {unknown}\n"
            f"  Keys are the long flags without '--'. Known: {sorted(known)}")
    return raw


def _resolve_args(argv=None):
    """Built-in defaults < --config file < command line."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        cfg = _load_preprocess_config(args.config, parser)
        typed = _explicitly_given(argv)
        applied = []
        for k, v in cfg.items():
            if k in typed:                # the command line wins, always
                continue
            setattr(args, k, v)
            applied.append(k)
        print(f"[config] {args.config}: applied {len(applied)} option(s)"
              + (f"; overridden on the command line: "
                 f"{sorted(set(cfg) & set(typed))}"
                 if set(cfg) & set(typed) else ""))
    for name in ("source_dir", "output_dir"):
        if not getattr(args, name):
            raise SystemExit(
                f"[preprocess_stream] {name} is required: pass it positionally "
                f"(SRC OUT) or set '{name}' in the --config file.")
    return args


def _parse_ratios(spec) -> Tuple[float, float, float]:
    """'0.8,0.1,0.1' (or a YAML list) -> the three ratios, validated."""
    if isinstance(spec, (list, tuple)):
        parts = list(spec)
    else:
        parts = [p for p in str(spec).split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise SystemExit(f"[split] --split_ratios needs exactly 3 values "
                         f"(train,val,test), got {spec!r}.")
    try:
        r = tuple(float(x) for x in parts)
    except ValueError:
        raise SystemExit(f"[split] --split_ratios must be numbers, got {spec!r}.")
    if any(x < 0 for x in r):
        raise SystemExit(f"[split] --split_ratios must be >= 0, got {r}.")
    if not math.isclose(sum(r), 1.0, rel_tol=0, abs_tol=1e-6):
        raise SystemExit(f"[split] --split_ratios must sum to 1.0, got {r} "
                         f"(sum={sum(r)}).")
    return r


def _parse_wav_splits(spec) -> frozenset:
    """'none' | 'all' | 'val' | 'train,val' -> the set of splits to save."""
    if spec is None or spec is False:
        return frozenset()
    if spec is True:                      # save_wav: true in the YAML
        return frozenset(SPLIT_NAMES)
    if isinstance(spec, (list, tuple)):
        parts = [str(x).strip().lower() for x in spec]
    else:
        parts = [p.strip().lower() for p in str(spec).split(",") if p.strip()]
    if not parts or parts == ["none"]:
        return frozenset()
    if parts == ["all"]:
        return frozenset(SPLIT_NAMES)
    bad = sorted(set(parts) - set(SPLIT_NAMES))
    if bad:
        raise SystemExit(
            f"[wav] --save_wav got {bad}; expected 'none', 'all', or a "
            f"comma-separated subset of {list(SPLIT_NAMES)}.")
    return frozenset(parts)


def main():
    args = _resolve_args()

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

    split_ratios = _parse_ratios(args.split_ratios)
    stratify = not args.no_stratify
    wav_splits = _parse_wav_splits(args.save_wav)

    # ---- how sources are labelled (directory tree, or a metadata CSV) ----
    # Resolved HERE, before any model is loaded and before --split_only takes
    # its own path: the labels decide the output folders and the stratified
    # split, so both branches below must see exactly the same ones.
    label_resolver = build_label_resolver(args)
    labels_prov = labels_provenance(args, label_resolver)
    check_labels_provenance(out_root, labels_prov)

    # ---- split-only modes: decide the split and stop ----
    # Placed BEFORE the condition registry so neither mode loads CREPE/CLAP/DAC:
    # writing a split is a metadata operation and must stay one, otherwise
    # "give this finished dataset a split" would cost a model load per run.
    if args.import_legacy_split or args.split_only:
        if args.import_legacy_split:
            groups = import_legacy_split(out_root, latent_root, split_ratios,
                                         args.split_seed, stratify)
        else:
            print("Scanning source files ...")
            files = scan_audio_files(args.source_dir, args.single_class,
                                     args.class_name, label_resolver)
            if not files:
                raise SystemExit(f"[ERROR] No audio files under {args.source_dir}")
            print(f"  {len(files)} files")
            print_class_histogram(files)
            groups = resolve_splits(out_root, files, split_ratios,
                                    args.split_seed, stratify, args.resplit)
            # Record HOW these classes were decided, even though this mode
            # encodes nothing: the split is keyed by the output folder, so a
            # later full run in the other labelling mode would silently be
            # keyed differently. Existing source entries are preserved.
            write_source_manifest(out_root, load_source_manifest(out_root),
                                  {}, [], False, labels_prov)
        print(f"\nDONE (split only)\n  output: {out_root / SPLITS_NAME}")
        # The latents of a previous run are already on disk, so the samples per
        # split are countable here too -- and this is the cheap way to ask for
        # them: --split_only decodes nothing.
        print_chunk_counts(out_root, groups)
        return

    # ---- condition registry (frame + optional global selection) ----
    enabled_frame = None
    if args.conditions is not None:
        enabled_frame = [c.strip() for c in args.conditions.split(",") if c.strip()]
    enabled_global = None
    if args.global_conds is not None:
        enabled_global = [c.strip() for c in args.global_conds.split(",") if c.strip()]

    registry = None
    gpu_cond_names, cpu_cond_names = [], []
    chunk_cond_names, class_global_names = [], []
    if enabled_frame or enabled_global:
        from conditions import ConditionRegistry
        registry = ConditionRegistry(
            enabled_frame=enabled_frame if enabled_frame else [],
            enabled_global=enabled_global if enabled_global else [],
        )

        # ---- split by WHERE the value comes from (see the CONDITIONS header) --
        # Per-chunk conditions go through the stream; per-class ones are built
        # afterwards from their own assets. Selecting only frame conditions,
        # only globals, or any mixture needs no branch: one of the two lists
        # simply comes out empty.
        chunk_globals, class_global_names = _split_globals_by_stage(registry)
        chunk_cond_names = sorted(list(registry.frame_names) + chunk_globals)

        # ---- split the per-chunk conditions by WHERE they can run ----
        # GPU-capable ones (f0, rhythm) move to the MAIN process, alongside the
        # DAC batch: the card stays owned by one process, so the reason workers
        # were barred from CUDA is satisfied by construction rather than by a
        # rule. Everything else stays in the workers, on CPU.
        cond_device = args.cond_device or args.device
        if cond_device.startswith("cuda") and not torch.cuda.is_available():
            print("[cond] cond_device='cuda' but no GPU is available "
                  "-> condition extraction falls back to CPU.")
            cond_device = "cpu"

        gpu_capable, cpu_only = _split_extractors_by_device(registry)
        if cond_device.startswith("cuda"):
            gpu_cond_names, cpu_cond_names = gpu_capable, cpu_only
            _set_extractor_device(registry, gpu_cond_names, cond_device)
        else:
            # Everything on CPU: the extraction all happens worker-side, exactly
            # as before this split existed.
            gpu_cond_names, cpu_cond_names = [], sorted(gpu_capable + cpu_only)
            # And PIN them, which this branch used to leave undone. The pinning
            # below only ran with num_workers > 0, so "--cond_device cpu
            # --num_workers 0" left every extractor at its own default. That was
            # invisible while the only GPU-capable extractors defaulted to CPU
            # anyway (CrepeF0Extractor does, in CONDITION_CONFIG) -- but the
            # CLAP encoder defaults to CUDA, so it happily loaded onto the card
            # in a run that had just been told not to use it.
            _set_extractor_device(registry, cpu_cond_names, "cpu")

        if args.num_workers > 0:
            # The extractors that REMAIN in the workers must never touch CUDA:
            # one model per worker on a single card would contend with the DAC
            # for VRAM and time-slice the GPU between processes. Scoped to the
            # CPU-side group -- applying it to all of them, as it used to, would
            # put f0 straight back on CPU and undo the split above.
            _force_cpu_extractors(registry, cpu_cond_names)

        if chunk_cond_names:
            print(f"[cond] per-chunk conditions {chunk_cond_names}: "
                  f"GPU({cond_device})={gpu_cond_names or 'none'} in the main "
                  f"process | CPU={cpu_cond_names or 'none'} in the workers")
        if class_global_names:
            print(f"[cond] per-class conditions {class_global_names}: "
                  f"one sidecar per class, after the stream")
        if not chunk_cond_names and not class_global_names:
            print("[cond] a registry was built but it holds no condition "
                  "-> nothing to extract")

    # ---- scan ----
    print("Scanning source files ...")
    files = scan_audio_files(args.source_dir, args.single_class, args.class_name,
                             label_resolver)
    if not files:
        print(f"[ERROR] No audio files under {args.source_dir}")
        return
    classes = sorted({leaf for _, _, leaf, _, _ in files})
    print(f"  {len(files)} files, {len(classes)} classes")
    print_class_histogram(files)

    # ---- split: load or create, ALWAYS (it is part of the dataset) ----
    # Resolved even when no wavs are requested: the split is what the training
    # will read back, so a dataset produced by this script always carries one and
    # never depends on when someone first happened to ask for it.
    split_groups = resolve_splits(out_root, files, split_ratios, args.split_seed,
                                  stratify, args.resplit)

    # ---- which sources get a WAV ----
    # "all" travels as a sentinel rather than a set of every source: the workers
    # are separate processes and would each be shipped a copy of it.
    if not wav_splits:
        wav_sources = None
    elif set(wav_splits) == set(SPLIT_NAMES):
        wav_sources = "all"
    else:
        wav_sources = {
            rel for path, rel_parent, _leaf, src_hash, rel in files
            if split_groups.get(
                source_group_key(rel_parent, Path(path).name, src_hash)
            ) in wav_splits
        }
    if wav_splits:
        _n = len(files) if wav_sources == "all" else len(wav_sources)
        print(f"[wav] saving the per-chunk WAV of {sorted(wav_splits)} "
              f"-> {_n}/{len(files)} source(s)")

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
    if registry is not None and _split_globals_by_stage(registry)[0]:
        # The per-chunk globals are one vector for the whole chunk, so T does
        # not enter them -- but the chunk they summarise is the same one, and
        # saying so here keeps the two lines from looking contradictory.
        print(f"[conditions] per-chunk globals "
              f"{_split_globals_by_stage(registry)[0]}: one vector per chunk "
              f"(independent of T)")

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

    # ---- skip the sources that have nothing left to produce ----
    # Runs AFTER the audit (which needs the complete scan to spot changed and
    # removed sources) and only when the manifest can vouch for a source. With
    # --force there is nothing to skip: it means "redo what you encounter".
    if not args.force and prev_manifest:
        _run_cond_names = list(cpu_cond_names) + list(gpu_cond_names)
        n_before = len(files)
        files, n_skipped = filter_complete_sources(
            files, prev_manifest, changed_srcs,
            latent_root, cond_root, wav_root, n_frames_fixed,
            _run_cond_names, dac_enc is not None, wav_sources)
        if n_skipped:
            print(f"[resume] {n_skipped}/{n_before} source(s) already have every "
                  f"output this run would write -> not decoded at all.")
        if not files:
            # Deliberately NOT an early return: the class-level global conditions
            # below are not per-source, so a dataset whose chunks are all complete
            # can still be missing them.
            print("[resume] no source needs per-chunk work this run.")

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
        sr=args.sr, wav_sources=wav_sources, force=args.force,
        n_frames_fixed=n_frames_fixed,
        cpu_cond_names=cpu_cond_names, gpu_cond_names=gpu_cond_names,
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
    n_gcond = 0
    pending = []          # real chunks awaiting a full GPU batch
    produced = {}         # rel_source -> {size, mtime_ns, chunks} for the manifest
    # A batch is worth assembling if the main process owes it ANY GPU work: the
    # DAC encode, the GPU-side conditions, or both. With --skip_dac and f0 on the
    # GPU there is no encoder but there is still work, so the old "no encoder ->
    # drop everything" shortcut would have silently skipped the f0 of every chunk.
    _dac_on = not (args.skip_dac or dac_enc is None)
    _gpu_work = _dac_on or bool(gpu_cond_names)

    def _flush(items):
        nonlocal n_lat, n_gcond
        a, b = _process_gpu_batch(
            dac_enc if _dac_on else None, items, args.sr, n_frames_fixed,
            registry=registry, gpu_names=gpu_cond_names, force=args.force)
        n_lat += a
        n_gcond += b

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
            if not _gpu_work:
                pending.clear()                     # workers already wrote conds/wav
                continue
            # keep the GPU batches at exactly batch_size (the markers must not
            # shrink them), flushing the remainder after the stream ends.
            while len(pending) >= args.batch_size:
                _flush(pending[:args.batch_size])
                del pending[:args.batch_size]
        if pending and _gpu_work:
            _flush(pending)
            pending.clear()
    finally:
        if file_bar is not None:
            file_bar.close()
        # Persist whatever was observed, even on Ctrl-C / crash: a PARTIAL
        # manifest is strictly better than none (it still records the sources
        # actually handled). Written atomically via a temp file + replace.
        if produced or prev_manifest:
            n_src = write_source_manifest(out_root, prev_manifest, produced,
                                          removed_srcs, args.prune_orphans,
                                          labels_prov)
            print(f"[manifest] {out_root / MANIFEST_NAME}: {n_src} source(s) "
                  f"({len(produced)} seen this run)")
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ---- global conditions (class-level, after the stream) ----
    if class_global_names:
        extract_class_global_conditions(
            registry, out_root, classes, args.image_root, force=args.force
        )
    # The label vocabulary rides with the 'text' condition: it is what lets a
    # panel put words next to a stored CLAP vector. Written after the stream so
    # it is not paid for by a run that fails early.
    if "text" in _split_globals_by_stage(registry)[0]:
        write_text_label_vocab(registry, out_root,
                               getattr(args, "text_vocab", None),
                               force=args.force)
        # Then the per-chunk descriptions themselves. Reads back the vectors the
        # stream just wrote, so it also covers a run that only CHANGED the
        # vocabulary: no audio is touched and no model is loaded here.
        write_text_labels(out_root, cond_root,
                          n_terms=getattr(args, "text_labels_n", 2),
                          force=args.force,
                          text_extractor=registry.global_extractors.get("text"))

    print("\nDONE")
    print(f"  latents encoded this run: {n_lat}")
    if gpu_cond_names:
        print(f"  GPU per-chunk conditions {gpu_cond_names} extracted this run: "
              f"{n_gcond} chunk(s)")
    print("  CPU conditions / wav were written in-stream by the workers "
          "(counts not aggregated across processes).")
    print_chunk_counts(out_root, split_groups)
    print(f"  output: {out_root}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
