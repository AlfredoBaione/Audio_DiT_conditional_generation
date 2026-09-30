# Training for the Conditioned Audio DiT with Rectified Flow.
#
# Multi-modal conditioning:
#   - Frame-level (concatenated on the feature dimension at the input,
#     JASCO-style; see network_cond.py): f0, chroma, rhythm, energy
#   - Global (AdaLN added to timestep embedding): text (CLAP), image (CLIP)
#   - CFG dropout per-sample during training (drop-all / drop-frame /
#     drop-global / keep)
#   - Classifier-free guidance during validation (audio + metrics)
#
# Feature:
#   - ConditionedAudioDiT with AMP mixed precision
#   - EMA
#   - Conditioned audio generated every intervals.audio step on TensorBoard
#     (conditions taken from val-set samples)
#   - FD-DAC + KL (both directions) computed every intervals.metrics step on
#     conditioned samples, with reference pre-computed on the full validation
#     set (real validation latents, normalized space)
#   - Loss train + val on TensorBoard
#   - Configuration with OmegaConf YAML (configs/cond_default.yaml)
#
# Usage:
#   python training_cond.py
#   python training_cond.py --config configs/cond_default.yaml
#   python training_cond.py training.lr=2e-4 data.train_batch_size=16
#   python training_cond.py --run_name "cond_S_run1" model.kind=S
#   python training_cond.py --resume runs/cond_S_run1/checkpoints/checkpoint_step50000.pt
#
# RESUME BEHAVIOUR (important):
#   When you pass --resume, the script reads the configuration stored INSIDE the
#   checkpoint and uses it to rebuild the model, the conditioning selection
#   (enabled_frame / enabled_global) and the training setup automatically. You do
#   NOT need to re-pass model.kind, the enabled conditions, batch sizes, etc. -
#   they are restored from the checkpoint. Any CLI override you DO pass still
#   wins over the stored value (so you can deliberately change something on
#   resume if you really want to).

import os

# Keep transformers (CLAP/CLIP) on the PyTorch backend: never import TensorFlow
# (avoids protobuf/TF clashes with a conda base that ships TF, e.g. IRCAM tf2.18).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ============================================================
# CACHE / HOME REDIRECTION & VRAM OPTIMIZATION (Must run BEFORE importing torch)
# ------------------------------------------------------------
# 1. Force PyTorch to use expandable segments to drastically reduce VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. IRCAM Home redirection for DAC weights cache
#    DAC's dac.utils.download() resolves the weights path from Path.home(),
#    hardcoded, ignoring XDG_CACHE_HOME. Overriding HOME on the IRCAM machines
#    (detected by the local data path) points it to machine-local disk and
#    avoids the NFS PermissionError. On Windows / other systems HOME is left
#    untouched and DAC uses the platform default cache location.
_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    # TORCH_HOME is an ASSIGNMENT, not a setdefault: the IRCAM nodes already
    # export it, pointing inside the SHARED conda env
    # (.../envs/tf2.18/share/TORCH), which is read-only for us. torch.hub
    # prefers TORCH_HOME over XDG_CACHE_HOME, so the first download on a
    # machine with a cold cache (beat_this fetching its checkpoint) dies with
    # PermissionError. Only overwriting the variable fixes it.
    os.environ["TORCH_HOME"] = os.path.join(_cache, "torch")

import copy
import sys
import math
import json
import random
import argparse
from datetime import datetime
from pathlib import Path

# A console that cannot encode a character must not kill a training run. On
# Windows stdout defaults to the ANSI code page (cp1252), where a single
# non-ASCII character in a progress line raises UnicodeEncodeError and takes
# down the run from inside a print -- which is exactly how a metrics step was
# lost once. backslashreplace keeps the terminal's own encoding (so a UTF-8
# terminal, e.g. every IRCAM server, still prints the real characters) and only
# escapes what it cannot represent, instead of raising.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except Exception:
        pass          # not a real console (piped/captured): nothing to harden

import torch
# Enable TF32 on Ampere+ GPUs (e.g. RTX A4000). Same as facebookresearch/DiT:
# matmul/conv in TF32 mode -> roughly 2-3x faster than pure fp32 while keeping
# the same dynamic range as fp32 (no overflow risk, unlike fp16/AMP).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn.functional as F
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audio_dataset_npy import (
    DAC_LATENT_DIM, DAC_SAMPLE_RATE, frames_per_chunk,
)
from network_cond import ConditionedAudioDiT, TOKEN_DIM, check_ckpt_reinject_gate
from audio_dataset_cond import (
    build_conditioned_datasets, collate_conditioned, load_caption_table,
)
from conditions import (
    ConditionRegistry,
    CONDITION_CONFIG,
    make_null_frame_conditions, make_null_global_conditions,
)
from metrics import (
    precompute_latent_reference,
    compute_dac_metrics,
    precompute_audio_reference,
    compute_audio_mu_sigma,
    compute_fad,
    COND_METRICS,
)


# ======================
# DAC LOADER (singleton: load once, reuse for the whole run)
# ======================
# Loading the DAC model is slow and allocates a non-trivial amount of memory.
# The metrics / audio-preview / real-audio paths used to each load and free
# their own DAC every call; at frequent metrics steps that is wasteful and a
# source of fragmentation. Mirror the unconditional repo: load it ONCE on CPU
# and cache it for the whole run.
_DAC_MODEL  = None
_DAC_DEVICE = "cpu"


def set_dac_device(device: str):
    """
    Choose where the shared DAC decoder lives, from `metrics.dac_device`.

    WHY THE KNOB. The decoder is 76.7M params and decoding one 5-second clip
    costs, measured on an RTX 5050 laptop:

        CPU   2538 ms        CUDA   194 ms        (13x)

    At 128 samples that is 5.4 minutes of pure decoding per metrics step versus
    25 seconds, which on a short run dominates the wall clock. But the choice is
    NOT free on VRAM, and the number that matters is not the weights:

        weights      0.29 GB   (resident for the whole run)
        activations  0.69 GB   (peak, one 5-second clip at a time)
        peak         0.97 GB

    Nearly 1 GB, and it lands DURING the metrics step, when the model, the
    generations and the CREPE / beat_this / CLAP re-extraction are already
    resident. On a 24 GB card training an XL -- weights, grads, Adam states and
    the EMA shadow all live -- that spike is exactly the thing that kills a run
    at hour 40, and a slow metrics step is cheaper than losing days. Hence the
    DEFAULT IS "cpu": the historical behaviour, unchanged, for every run that
    does not ask otherwise. Turn it to "cuda" per machine, deliberately.

    NB: the values are not bit-identical across devices, so FD-DAC / FAD
    computed with a GPU decoder are not directly comparable with numbers
    produced by a CPU one. Comparable within an experiment (all runs on the same
    device), not across the switch.

    Must be called BEFORE the first get_dac(): the singleton is built once and
    is not moved afterwards, so a late call would silently do nothing.
    """
    global _DAC_DEVICE
    dev = str(device).strip().lower()
    if dev not in ("cpu", "cuda") and not dev.startswith("cuda:"):
        raise ValueError(
            f"metrics.dac_device must be 'cpu', 'cuda' or 'cuda:<n>', got "
            f"{device!r}."
        )
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"metrics.dac_device={device!r} but CUDA is not available. Set it to "
            f"'cpu' (the default) to decode on the CPU."
        )
    if _DAC_MODEL is not None and dev != _DAC_DEVICE:
        # Loud, not silent: a caller that asks for a device after the decoder
        # exists would otherwise believe it got one and measure on the other.
        raise RuntimeError(
            f"set_dac_device({device!r}) called after the DAC was already loaded "
            f"on {_DAC_DEVICE!r}. The device must be chosen before the first "
            f"get_dac()."
        )
    _DAC_DEVICE = dev


def get_dac():
    global _DAC_MODEL
    if _DAC_MODEL is None:
        import dac
        _DAC_MODEL = dac.DAC.load(dac.utils.download(model_type="44khz"))
        _DAC_MODEL.to(_DAC_DEVICE)
        _DAC_MODEL.eval()
        # The device is READ BACK from the model, not repeated as a literal:
        # this line used to say "CPU" as text, so changing where the decoder
        # lives would have left the log confidently reporting the old device.
        dev = next(_DAC_MODEL.parameters()).device
        print(f"[DAC] Model loaded once ({str(dev).upper()}) and cached for "
              f"the whole run.")
    return _DAC_MODEL


def decode_frames_to_wav(frames, normalizer, dac_model):
    """(n_frames, 72) NORMALIZED latent -> 1-D waveform on CPU.

    Single implementation, shared by the metrics step and the FAD reference: two
    copies of "denormalize, quantizer.from_latents, decode" would be two places
    to keep in sync, and a reference decoded differently from the generations
    would make the FAD compare the two decoders instead of the two
    distributions."""
    z = normalizer.denormalize(frames.T)
    z = z.unsqueeze(0).float().to(next(dac_model.parameters()).device)
    z_q, _, _ = dac_model.quantizer.from_latents(z)
    return dac_model.decode(z_q).squeeze().detach().cpu()


# ======================
# SPLIT / CACHE HELPERS  (new: split-less dataset + cache metadata validation)
# ======================
def _load_dataset_meta(latent_root):
    """dataset_meta.json is written by preprocess_stream.py at the dataset root
    (the parent of latents/). It records sr / chunk / acoustic params, i.e. HOW
    the latents were produced -- the fingerprint the cache must be tied to."""
    p = Path(latent_root).parent / "dataset_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _latent_file_list_hash(latent_root):
    """Deterministic hash of the latent file list (relative path + size + mtime).
    Detects added/removed/replaced .npy files that leave dataset_meta.json
    unchanged, so a stale normalizer / FD-DAC reference is never reused (#4)."""
    import hashlib
    root = Path(latent_root)
    # Hash INCREMENTALLY. Accumulating every entry in a list and then joining it
    # materialises the file list twice (the list of strings plus one giant blob)
    # -- roughly 2 GB of transient host RAM for a 5.8M-file corpus, right before
    # the heaviest startup phase. Feeding the digest entry by entry with the same
    # "\n" separator yields the SAME hash with flat memory.
    hasher = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*.npy")):
        st = p.stat()
        entry = f"{p.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}"
        if n:
            hasher.update(b"\n")        # separator BETWEEN entries, as join does
        hasher.update(entry.encode("utf-8"))
        n += 1
    return hasher.hexdigest(), n


def _splits_fingerprint(cfg):
    """Identity of the SPLIT the cache was computed against.

    The split now lives in the dataset (splits.json, written by
    preprocess_stream.py), so the fingerprint records the actual ASSIGNMENT --
    a digest of every source->split pair -- and not just the parameters that
    produced it. That is strictly tighter than what it replaced: growing the
    dataset adds sources to the train split with the parameters unchanged, and
    the normalizer fitted before that no longer describes the training data.

    A missing file is recorded as None rather than raised on: this runs before
    the datasets are built, and load_source_split() gives the actionable error.
    """
    import hashlib
    p = Path(cfg.paths.dataset_root).parent / "splits.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return {"unreadable": True}
    groups = payload.get("groups", {})
    digest = hashlib.sha1(
        json.dumps(groups, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "params": payload.get("params", {}),
        "source": payload.get("source"),
        "n_sources": len(groups),
        "assignment_sha1": digest,
    }


def _cache_fingerprint(cfg, n_frames):
    """Identity of the data the cached normalizer / FD-DAC reference depend on.
    The normalizer is fit on the TRAIN split and the FD reference on the VAL
    split, so the split itself is part of the fingerprint too."""
    flist_hash, flist_count = _latent_file_list_hash(cfg.paths.dataset_root)
    return {
        "latent_root": os.path.abspath(cfg.paths.dataset_root),
        "dataset_meta": _load_dataset_meta(cfg.paths.dataset_root),
        "duration_s": float(cfg.model.duration_s),
        "n_frames": int(n_frames),
        "latent_dim": int(DAC_LATENT_DIM),
        "latent_file_list_hash": flist_hash,
        "latent_file_count": flist_count,
        "split": _splits_fingerprint(cfg),
    }


def _validate_cache(cache_dir, fingerprint, guarded_files):
    """
    Tie the shared cache (normalizer.pt, fd_dac_ref_stats.pt) to the dataset it
    was computed on (report issue #3). Behaviour:
      * cache_meta.json present & matches   -> reuse the cache silently;
      * present & DIFFERS                    -> hard-fail (stale cache);
      * absent but cache files exist         -> hard-fail (unverifiable legacy
        cache; almost certainly stale after a preprocessing change);
      * absent and no cache files            -> write cache_meta.json (fresh).
    """
    meta_path = os.path.join(cache_dir, "cache_meta.json")
    present = [f for f in guarded_files if os.path.exists(f)]

    if os.path.exists(meta_path):
        try:
            old = json.loads(Path(meta_path).read_text())
        except Exception:
            old = None
        if old == fingerprint:
            print(f"[cache] verified against {meta_path} -> reusing cache.")
            return
        raise SystemExit(
            "[cache] STALE CACHE: the cached normalizer / FD-DAC reference in\n"
            f"  {cache_dir}\n"
            "were computed on a DIFFERENT dataset/duration/split than the current "
            "run, so reusing them would silently corrupt normalization and FD/KL.\n"
            f"  cached : {json.dumps(old,  sort_keys=True)}\n"
            f"  current: {json.dumps(fingerprint, sort_keys=True)}\n"
            "Point paths.cache_dir to a fresh directory, or delete "
            "normalizer.pt / fd_dac_ref_stats.pt / cache_meta.json there.")

    if present:
        raise SystemExit(
            "[cache] UNVERIFIABLE CACHE: found "
            f"{[os.path.basename(f) for f in present]} in\n  {cache_dir}\n"
            "but no cache_meta.json to tie them to a dataset. After the "
            "preprocessing refactor the latent statistics changed, so an old "
            "cache is almost certainly stale. Delete those files (and any "
            "cache_meta.json) or use a fresh paths.cache_dir.")

    os.makedirs(cache_dir, exist_ok=True)
    Path(meta_path).write_text(json.dumps(fingerprint, indent=2))
    print(f"[cache] fresh cache -> wrote {meta_path}")


def _npz_keys(npz):
    """Key list of ONE .npz (no array decompression), or None if unreadable."""
    try:
        with np.load(str(npz)) as d:
            return set(d.files)
    except Exception:
        return None


def _scan_frame_conditions(condition_root, io_workers=16, block=4096):
    """
    SPLIT-LESS full scan: over ALL .npz under condition_root (any depth), count
    how many exist and, per condition name, in how many the key is present. Reads
    only the .npz key list (no array decompression).
    Returns {"total": int, "present": {name: count}}.

    PARALLEL READS + a progress bar. Each file costs a disk round trip, not
    computation -- measured 10-16 ms per file the first time it is read against
    0.12 ms once it is in the OS cache -- and there is one file per chunk. Read
    one at a time that is minutes on a small dataset but HOURS on a corpus of
    hundreds of thousands of chunks, spent in silence with the GPU already
    locked and idle: on IRCAM that gets the run terminated before it starts
    (XL_chord_lakh on oban, 30 Sept 2026). `io_workers` threads cut the cold
    scan 8.7x on the laptop it was measured on (np.load releases the GIL during
    I/O); the COUNTS are the same, a count does not depend on the order the
    files are read in. There is no work here a GPU could take over -- the time
    is the disk.

    The paths are handed to the pool `block` at a time: executor.map() would
    otherwise turn the whole file list into futures up front, which is GBs of
    host RAM on a multi-million-file corpus.
    """
    from collections import Counter as _Counter
    from concurrent.futures import ThreadPoolExecutor
    from itertools import islice
    root = Path(condition_root) if condition_root else None
    if root is None or not root.exists():
        return {"total": 0, "present": {}}
    total = 0
    present = _Counter()
    paths = root.rglob("*.npz")
    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as ex:
        with tqdm(desc="Condition scan", unit="file") as pbar:
            while True:
                chunk = list(islice(paths, block))
                if not chunk:
                    break
                for keys in ex.map(_npz_keys, chunk):
                    if keys is None:
                        continue
                    total += 1
                    for k in keys:
                        present[k] += 1
                pbar.update(len(chunk))
    return {"total": total, "present": dict(present)}


# ======================
# CONFIG LOADING
# ======================
def _flatten_keys(d, prefix=""):
    """Yield the dotted keys of a nested dict (e.g. 'data.num_val_batches'), so a
    bad CLI override can be named precisely."""
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.append(kk)
            out.extend(_flatten_keys(v, kk))
    return out


def load_config():
    """
    Loads the config from YAML, applies override CLI in dotlist
    (e.g. training.lr=2e-4 data.train_batch_size=16) and handles --resume + --run_name.

    Returns:
        cfg: OmegaConf with the final config (CLI override already applied)
        run_name: string identifying the run (default timestamp)
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str,
                        default="configs/cond_default.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path checkpoint for resume (override YAML). "
                              "The model architecture, the conditioning "
                              "selection and the training config are restored "
                              "from the checkpoint automatically.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Directory name of the run. "
                              "Default: timestamp YYYY-MM-DD_HH-MM-SS")
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = OmegaConf.load(args.config)

    # If resuming, layer the checkpoint's stored config ON TOP of the YAML, so
    # the architecture / conditioning selection / training params match what was
    # actually used. Only the lightweight metadata is read here
    # (map_location='cpu'); the weights are reloaded later in the resume section.
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        _meta = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "config" in _meta and _meta["config"] is not None:
            ckpt_cfg = OmegaConf.create(_meta["config"])
            cfg = OmegaConf.merge(cfg, ckpt_cfg)
            print("[RESUME] Config restored from checkpoint "
                  f"(model.kind={cfg.model.kind}, "
                  f"frame_reinject_every={cfg.model.get('frame_reinject_every', 0)}, "
                  f"enabled_frame={cfg.conditioning.enabled_frame}, "
                  f"enabled_global={cfg.conditioning.enabled_global}, "
                  f"train_batch_size={cfg.data.train_batch_size}).")
        elif "model_kind" in _meta:
            # Older checkpoint without a full stored config: at least restore
            # the model kind, which MUST match to load the weights at all.
            cfg.model.kind = _meta["model_kind"]
            print(f"[RESUME] model.kind restored from checkpoint: {cfg.model.kind} "
                  "(older checkpoint without full config; other params come "
                  "from the YAML/CLI).")
        del _meta

    # THE INFLUENCE SET, ONE KNOB -> TWO (29 Sept 2026). A config written before
    # then -- the checkpoint's own on a resume, or an old dumped config.yaml
    # passed with --config -- carries sampling.n_influence_samples, which meant
    # BOTH halves. Its value becomes both new keys, so an old run resumes with
    # exactly the panels it had; on a resume it wins over the YAML's defaults
    # like every other checkpoint value. The old key is then removed, so the
    # dumped config.yaml holds only what the code reads. Before the CLI merge,
    # so a CLI value for the new keys still wins.
    _smp = cfg.get("sampling", None)
    if _smp is not None and "n_influence_samples" in _smp:
        _old = _smp.get("n_influence_samples")
        # 0 as well as null: the old reader was `... or 16`, so an old 0 ran
        # with 16 -- and that is what the resumed run must keep.
        _old = int(_old or 16)
        for _k in INFLUENCE_SET_KEYS.values():
            _smp[_k] = _old
        del _smp["n_influence_samples"]
        print(f"[config] sampling.n_influence_samples={_old} (older config) -> "
              f"n_influence_samples_valid={_old}, "
              f"n_influence_samples_probe={_old}")

    # CLI overrides win over everything (YAML + checkpoint config), but a typo in
    # a key must FAIL rather than silently create a phantom top-level entry while
    # the real parameter keeps its YAML value. `from_dotlist` parses each token,
    # and merging into a struct-locked config raises on any key that does not
    # already exist -- so `data_num_val_batches=8` (should be data.num_val_batches)
    # stops the run instead of being ignored.
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(unknown)
        OmegaConf.set_struct(cfg, True)
        try:
            cfg = OmegaConf.merge(cfg, cli_cfg)
        except Exception as e:
            base_keys = set(_flatten_keys(OmegaConf.to_container(cfg, resolve=False)))
            bad = [k for k in _flatten_keys(OmegaConf.to_container(cli_cfg))
                   if k not in base_keys]
            _hint = ""
            if "sampling.n_influence_samples" in bad:
                _hint = ("\n          sampling.n_influence_samples is now two "
                         "keys: sampling.n_influence_samples_valid and "
                         "sampling.n_influence_samples_probe.")
            raise SystemExit(
                f"[config] unknown CLI override key(s): {bad or [str(e)]}\n"
                f"          These do not exist in {args.config}. Check the "
                f"spelling and the dotted path (e.g. 'data.num_val_batches', not "
                f"'data_num_val_batches'). Nothing was run.{_hint}")
        OmegaConf.set_struct(cfg, False)

    # The two influence-set sizes, checked here so a bad value costs a second.
    # Validation must keep at least one sample: its panels and the audio preview
    # number their blocks off that list, and with none of it they would fall
    # back to a different spread of the validation set. The probe may be 0 (off).
    _smp = cfg.get("sampling", None)
    if _smp is not None:
        for _which, _lo in (("valid", 1), ("probe", 0)):
            _k = INFLUENCE_SET_KEYS[_which]
            _v = _smp.get(_k, None)
            if _v is not None and int(_v) < _lo:
                raise SystemExit(
                    f"[config] sampling.{_k} = {_v}, must be >= {_lo}"
                    f"{' (0 turns the probe off)' if _which == 'probe' else ''}"
                    f". Nothing was run.")

    # CFG DROPOUT BUCKETS: the three group probabilities must leave room for the
    # "keep both" case. They partition [0, 1) into disjoint ranges (see
    # apply_cfg_dropout), so if they sum to 1.0 or more the leftover mass is
    # zero and EVERY sample has something dropped: the model never sees full
    # conditioning and quietly trains towards the unconditional one, with no
    # error and nothing odd in the loss curve. The YAML has always said "must be
    # < 1.0"; nothing enforced it. Checked HERE, before the dataset, the model
    # and the GPU, so a bad config costs a second instead of a night.
    _p_all = float(cfg.conditioning.p_drop_all)
    _p_frm = float(cfg.conditioning.p_drop_frame)
    _p_gbl = float(cfg.conditioning.p_drop_global)
    _p_each = float(cfg.conditioning.get("p_drop_each_frame", 0.0))
    for _name, _val in (("p_drop_all", _p_all), ("p_drop_frame", _p_frm),
                        ("p_drop_global", _p_gbl),
                        ("p_drop_each_frame", _p_each)):
        if not 0.0 <= _val <= 1.0:
            raise SystemExit(
                f"[config] conditioning.{_name} = {_val} is outside [0, 1]. "
                f"These are probabilities. Nothing was run.")
    _p_sum = _p_all + _p_frm + _p_gbl
    if _p_sum >= 1.0:
        raise SystemExit(
            f"[config] conditioning.p_drop_all + p_drop_frame + p_drop_global "
            f"= {_p_all} + {_p_frm} + {_p_gbl} = {_p_sum:.3f}, which leaves "
            f"NOTHING for the 'keep everything' case: every training sample "
            f"would have some condition dropped and the model would never see "
            f"full conditioning. Their sum must stay below 1.0 (the default "
            f"0.10 + 0.05 + 0.05 = 0.20 leaves 80%). Nothing was run.")
    if _p_all <= 0.0 and (_p_frm <= 0.0 or _p_gbl <= 0.0):
        print(f"[config] WARNING: p_drop_all={_p_all}, p_drop_frame={_p_frm}, "
              f"p_drop_global={_p_gbl}. Classifier-free guidance needs a NULL "
              f"branch to extrapolate from, and the influence panel needs it as "
              f"its baseline: with no dropout on a branch, guidance_scale and "
              f"the delta column on that branch are meaningless.")
    del _p_all, _p_frm, _p_gbl, _p_each, _p_sum

    # CLI --resume prevails over YAML
    if args.resume is not None:
        cfg.paths.resume_from = args.resume

    # Run name: CLI --run_name > YAML paths.run_name > timestamp default
    if args.run_name is not None:
        run_name = args.run_name
    elif cfg.paths.get("run_name") is not None:
        run_name = cfg.paths.run_name
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Persist final value in cfg so it appears in the dumped config.yaml
    cfg.paths.run_name = run_name

    # Derivates
    cfg.data.effective_bs = cfg.data.train_batch_size * cfg.data.grad_accum

    return cfg, run_name


# ======================
# LR SCHEDULE (factory: gets num_steps and schedule via closure)
# ======================
def make_lr_lambda(num_steps: int, warmup_steps: int, decay_start_frac: float):
    decay_start = int(num_steps * decay_start_frac)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / (num_steps - decay_start)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)).item())

    return lr_lambda


# ======================
# T SAMPLING (receives t_min/t_max explicitly)
# ======================
def sample_logit_normal(batch_size, device, t_min, t_max, mean=0.0, std=1.0):
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u).clamp(t_min, t_max)


# ======================
# EMA
# ======================
class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def copy_from(self, model):
        """Hard-copy the live weights into the EMA shadow.

        Called ONCE when ema_start is reached. Without it the shadow would still
        hold the RANDOM INITIALISATION it was deepcopy'd from at step 0 (it is
        not updated before ema_start), and lerp with decay=0.9999 would then wash
        that noise out only with a ~6931-update half-life: the EMA would still be
        ~50% random init 7k steps after it starts being used for validation, the
        BEST-checkpoint decision, the previews and the metrics. Seeding from the
        live model makes the EMA a true average of TRAINED weights from the very
        first update.
        """
        ema_params = dict(self.model.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].copy_(p.data)
        ema_buffers = dict(self.model.named_buffers())
        for name, b in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(b.data)

    @torch.no_grad()
    def update(self, model):
        # Iterate over named_parameters (same as facebookresearch/DiT). Iterating
        # over names guarantees parameter correspondence by identifier rather
        # than by ordering. Equivalent for a deepcopy'd model, but more defensive.
        ema_params = dict(self.model.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].lerp_(p.data, 1.0 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# ======================
# CFG DROPOUT (per-sample, applied only during training)
# ======================
def apply_cfg_dropout(frame_cond, global_cond, device, global_configs, B,
                      p_drop_all, p_drop_frame, p_drop_global,
                      p_drop_each_frame=0.0, text_ctx_mask=None):
    """
    Per-sample CFG dropout, in two stages.

    STAGE 1 -- GROUP buckets. Each element of the batch flips one coin:
        p_drop_all     -> drop everything       (pure unconditional)
        p_drop_frame   -> drop only frame-level
        p_drop_global  -> drop only global
        remaining mass -> keep both branches

    These three are what classifier-free guidance extrapolates FROM: the
    all-null branch has to be a well-trained model in its own right, so its
    probability mass is reserved and is never diluted by stage 2.

    STAGE 2 -- PER-CONDITION dropout (`p_drop_each_frame`; 0.0 disables it and
    restores the stage-1-only behaviour exactly). On top of the buckets, EVERY
    frame condition then flips its OWN independent coin, so the model also sees
    the partial subsets: f0 alone, f0+energy, chroma alone, ...

    Without stage 2 the model only ever sees the frame conditions ALL present
    or ALL absent, and asking it for one condition at inference (nulls in the
    other slots) is out of distribution: the input says "conditioned" in one
    slot and "unconditional" in the others, a combination it was never trained
    to resolve. Stage 2 is what makes partial conditioning a capability of the
    model rather than an accident. It has to be decided BEFORE training -- it
    changes what the model learns and cannot be bolted on at sampling time.

    With P = p_drop_each_frame over N conditions, a sample in the "keep" bucket
    still holds all N with probability (1-P)^N, so P is not a small correction:
    at N=3, P=0.2 leaves all three standing only about half the time. Raising it
    buys subset coverage and costs joint-conditioning signal.

    This gives the model a cond/uncond mixture in EVERY batch, a much more
    stable training signal than per-batch dropout.
    """
    r = torch.rand(B, device=device)
    drop_all    = r < p_drop_all
    drop_frame  = (r >= p_drop_all) & (r < p_drop_all + p_drop_frame)
    drop_global = (r >= p_drop_all + p_drop_frame) & \
                  (r < p_drop_all + p_drop_frame + p_drop_global)

    drop_f = drop_all | drop_frame    # mask: samples with frame conds dropped
    drop_g = drop_all | drop_global   # mask: samples with global conds dropped

    # Frame-level: zero the selected rows (zero is the null for frame conds).
    # STAGE 2 lives here: each condition draws its OWN coin, which is what
    # produces the partial subsets. A sample already in drop_f stays fully
    # dropped either way, so the reserved all-null mass is untouched.
    if frame_cond:
        for k in frame_cond:
            drop_k = drop_f
            if p_drop_each_frame > 0.0:
                drop_k = drop_f | (torch.rand(B, device=device)
                                   < p_drop_each_frame)
            keep_mask = (~drop_k).view(B, 1, 1).to(frame_cond[k].dtype)
            frame_cond[k] = frame_cond[k] * keep_mask

    # Global: replace with null (zeros) where drop_g
    if global_cond:
        null_g = make_null_global_conditions(B, global_configs, device)
        for k in global_cond:
            keep_mask = (~drop_g).view(B, 1).to(global_cond[k].dtype)
            global_cond[k] = global_cond[k] * keep_mask \
                             + null_g[k] * (1 - keep_mask)

    # The cross-attention context follows the SAME coin as the pooled text.
    # It has to: they are two views of one condition, and dropping one while
    # keeping the other would build a branch -- "no vector but here are the
    # words" -- that the CFG null pass never produces and that guidance would
    # then extrapolate away from nothing.
    #
    # Only the MASK is dropped, not the tokens. A row masked to all-False is
    # what ConditionedAudioDiT reads as its learned null token, so the null of
    # this channel is a value the model owns rather than a tensor this function
    # has to invent -- and the tokens do not have to be re-written every step.
    if text_ctx_mask is not None:
        text_ctx_mask = text_ctx_mask & (~drop_g).view(B, 1)

    return frame_cond, global_cond, text_ctx_mask


# ======================
# LOSS
# ======================
VALIDATION_PROTOCOL = "fixed_subset_common_noise_sample_weighted_v2"


def compute_loss(model, batch, device, use_amp, t_min, t_max,
                 global_configs, p_drop_all, p_drop_frame, p_drop_global,
                 training=True, x0=None, t=None, p_drop_each_frame=0.0):
    frames, frame_cond, _labels, text_embs, image_embs, text_ctx = batch
    # NB: `labels` is discarded as conditioning (CLAP-text plays that role
    # better now). It is kept in the batch only as metadata for logging.

    x1 = frames.to(device).float()
    B = x1.shape[0]
    if x0 is None:
        x0 = torch.randn_like(x1)
    else:
        if tuple(x0.shape) != tuple(x1.shape):
            raise ValueError(
                f"fixed x0 has shape {tuple(x0.shape)}, expected {tuple(x1.shape)}")
        x0 = x0.to(device=device, dtype=x1.dtype, non_blocking=True)

    if t is None:
        t = sample_logit_normal(B, device, t_min, t_max)
    else:
        if t.ndim != 1 or t.shape[0] != B:
            raise ValueError(f"fixed t has shape {tuple(t.shape)}, expected ({B},)")
        t = t.to(device=device, dtype=x1.dtype, non_blocking=True)
    t_expand = t.view(B, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0

    # Move conditions to device
    fc = {k: v.to(device).float() for k, v in frame_cond.items()}
    gc = {}
    if "text" in global_configs:
        gc["text"] = text_embs.to(device)
    if "image" in global_configs:
        gc["image"] = image_embs.to(device)

    # The text SEQUENCE. Sent only when the model actually has a
    # cross-attention: on every other model it would be silently ignored, and
    # moving a (B, L, 768) tensor to the GPU every step to have it ignored is
    # bandwidth spent on nothing.
    ctx = ctx_mask = None
    if getattr(model, "text_cross_layers", None) and text_ctx is not None:
        ctx = text_ctx["tokens"].to(device, non_blocking=True)
        ctx_mask = text_ctx["mask"].to(device, non_blocking=True)

    # CFG dropout only at training time
    if training:
        fc, gc, ctx_mask = apply_cfg_dropout(
            fc, gc, device, global_configs, B,
            p_drop_all=p_drop_all,
            p_drop_frame=p_drop_frame,
            p_drop_global=p_drop_global,
            p_drop_each_frame=p_drop_each_frame,
            text_ctx_mask=ctx_mask,
        )

    with torch.amp.autocast('cuda', enabled=use_amp):
        pred = model(xt, t, frame_conditions=fc, global_conditions=gc,
                     text_context=ctx, text_context_mask=ctx_mask)
        loss = F.mse_loss(pred, target)
    return loss


@torch.no_grad()
def euler_sample_cfg(model, n_frames, device, steps, t_min, t_max, use_amp,
                      frame_cond, global_cond, guidance,
                      frame_dims, global_configs, gen_rng=None,
                      text_ctx=None):
    """
    Euler integrator with classifier-free guidance.
    Both `frame_cond` and `global_cond` are expected as batch=1 dicts on device.
    If `guidance` <= 1.0 or both conditioning sources are absent, a single
    forward pass per step is used.
    `gen_rng` (optional torch.Generator) fixes the initial-noise x0 so the metric
    FD/KL are comparable across checkpoints (mirrors the uncond metrics seed);
    None = free-running.

    `text_ctx` is {"tokens": (1, L, d), "mask": (1, L)} for the cross-attention,
    or None. The NULL branch always passes text_context=None, which the model
    resolves to its learned null token -- the same value the training dropout
    showed it, which is the only thing that makes the extrapolation below mean
    anything.
    """
    model.eval()
    x = torch.randn(1, n_frames, TOKEN_DIM, device=device, generator=gen_rng)
    dt = (t_max - t_min) / steps

    null_fc = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc = make_null_global_conditions(1, global_configs or {}, device)

    # NB: test the CONTENT, not `is not None`: with every condition disabled the
    # caller passes empty dicts ({}), which are "no conditioning" -- treating them
    # as present would engage CFG and burn two IDENTICAL forwards per step.
    # The context only exists for a model that has the sub-layer; anything else
    # would be ignored, and the null branch never sends one at all.
    ctx = ctx_mask = None
    if getattr(model, "text_cross_layers", None) and text_ctx is not None:
        ctx = text_ctx["tokens"].to(device)
        ctx_mask = text_ctx["mask"].to(device)

    has_cond = bool(frame_cond) or bool(global_cond) or (ctx is not None)
    use_cfg = (guidance > 1.0) and has_cond

    for i in range(steps):
        tv = t_min + i * dt
        t = torch.ones(1, device=device) * tv
        with torch.amp.autocast('cuda', enabled=use_amp):
            if use_cfg:
                fc = frame_cond if frame_cond else null_fc
                gc = global_cond if global_cond else null_gc
                v_c = model(x, t, frame_conditions=fc,      global_conditions=gc,
                            text_context=ctx, text_context_mask=ctx_mask)
                v_u = model(x, t, frame_conditions=null_fc, global_conditions=null_gc,
                            text_context=None)
                v = v_u + guidance * (v_c - v_u)
            else:
                v = model(x, t,
                          frame_conditions=frame_cond or null_fc,
                          global_conditions=global_cond or null_gc,
                          text_context=ctx, text_context_mask=ctx_mask)
        x = x + v.float() * dt

    return x[0].cpu()


@torch.no_grad()
def euler_sample_cfg_paired(model, n_frames, device, steps, t_min, t_max, use_amp,
                            frame_cond, global_cond, guidance,
                            frame_dims, global_configs, gen_rng=None,
                            text_ctx=None):
    """
    FUSED CFG sampler: produces the CONDITIONED and the UNCONDITIONAL samples for
    B samples at once, from the same initial noise, in ONE batch-3B forward per
    Euler step. The rows are three BRANCHES (not three samples):
        rows   0..B-1   = conditioned (real conditions)  -> the guided velocity
        rows   B..2B-1  = those same x with NULL conditions -> the CFG null branch
        rows  2B..3B-1  = unconditional (null everywhere)   -> the free velocity
    All three are REQUIRED by the math: v_guided = v_null + g*(v_cond - v_null)
    needs the first two, the uncond axis needs the third. B (samples per forward)
    is the only tunable part: B=1 -> batch 3, B=2 -> batch 6, ... Larger B means
    fewer, bigger forwards (faster on a GPU with headroom) but a proportionally
    higher activation peak.

    The conditions decide B: every tensor in `frame_cond`/`global_cond` must have
    batch B. With NO conditions at all, B cannot be inferred and pairing is
    pointless anyway -- the caller must not use this path.

    GENERIC / portable: the batch is built by iterating over EVERY active frame
    and global condition and concatenating it with its null, so this works
    unchanged for f0-only, f0+energy, CLAP-text, CLIP-image, or any future
    combination (driven by the run's condition dicts, nothing is hardcoded).

    Returns (cond_latents, uncond_latents): two lists of B tensors on CPU.
    Mathematically equivalent to separate euler_sample_cfg() calls from the same
    x0 -- but only up to CUDA op ordering, so the two paths agree to
    floating-point tolerance, not bit for bit. B does NOT affect which samples
    come out: the noise is drawn per-sample (see below), so spf is purely about
    how the work is packed into forwards.
    """
    model.eval()
    null_fc_1 = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc_1 = make_null_global_conditions(1, global_configs or {}, device)
    fc_c = frame_cond if frame_cond else {}
    gc_c = global_cond if global_cond else {}

    # B is dictated by the conditions handed in (they are what varies per sample)
    any_t = next(iter(fc_c.values()), None)
    if any_t is None:
        any_t = next(iter(gc_c.values()), None)
    if any_t is None:
        raise RuntimeError("euler_sample_cfg_paired needs at least one condition "
                           "to infer the batch size; use euler_sample_cfg instead.")
    B = int(any_t.shape[0])

    null_fc = {k: v.expand(B, *v.shape[1:]).contiguous() for k, v in null_fc_1.items()}
    null_gc = {k: v.expand(B, *v.shape[1:]).contiguous() for k, v in null_gc_1.items()}

    # x0 is drawn ONE SAMPLE AT A TIME and stacked, NOT as a single randn(B,...):
    # this way sample i gets the i-th draw of the generator whatever B is, so
    # changing metrics_samples_per_forward does NOT change the noise, the samples
    # or the metric values -- it only changes how the work is packed into
    # forwards. (A single randn(B,...) would consume the RNG differently for
    # different B and silently shift every number.)
    x0 = torch.stack([
        torch.randn(n_frames, TOKEN_DIM, device=device, generator=gen_rng)
        for _ in range(B)
    ])
    x_cond = x0.clone()
    x_unc = x0.clone()
    dt = (t_max - t_min) / steps

    # [cond | cfg-null | uncond-null] stacked on the batch dim, per condition key
    frame_batch = {n: torch.cat([fc_c.get(n, null_fc[n]), null_fc[n], null_fc[n]],
                                dim=0) for n in null_fc}
    global_batch = {n: torch.cat([gc_c.get(n, null_gc[n]), null_gc[n], null_gc[n]],
                                 dim=0) for n in null_gc}

    # The text sequence, stacked the same way. The tokens are repeated on all
    # three branches and only the MASK distinguishes them: rows B..3B-1 are
    # masked to all-False, which the model resolves to its learned null token.
    # Repeating the tokens rather than sending zeros keeps this tensor one
    # `cat` of one array -- and the mask is the single place where "this branch
    # has no text" is written, so the two null branches cannot drift apart.
    ctx_batch = ctx_mask_batch = None
    if getattr(model, "text_cross_layers", None) and text_ctx is not None:
        tok = text_ctx["tokens"].to(device)
        msk = text_ctx["mask"].to(device)
        if tok.shape[0] != B:
            raise ValueError(
                f"text_ctx has batch {tok.shape[0]}, the conditions say B={B}")
        ctx_batch = torch.cat([tok, tok, tok], dim=0)
        ctx_mask_batch = torch.cat(
            [msk, torch.zeros_like(msk), torch.zeros_like(msk)], dim=0)

    for i in range(steps):
        tv = t_min + i * dt
        t = torch.ones(3 * B, device=device) * tv
        xb = torch.cat([x_cond, x_cond, x_unc], dim=0)   # (3B, n_frames, TOKEN_DIM)
        with torch.amp.autocast('cuda', enabled=use_amp):
            vb = model(xb, t, frame_conditions=frame_batch,
                       global_conditions=global_batch,
                       text_context=ctx_batch,
                       text_context_mask=ctx_mask_batch)
        v_c    = vb[0:B].float()
        v_null = vb[B:2 * B].float()
        v_u    = vb[2 * B:3 * B].float()
        v_guided = v_null + guidance * (v_c - v_null)
        x_cond = x_cond + v_guided * dt
        x_unc = x_unc + v_u * dt

    return ([x_cond[b].cpu() for b in range(B)],
            [x_unc[b].cpu() for b in range(B)])


# ======================
# TENSORBOARD AUDIO PANELS: ONE BLOCK PER SAMPLE
# ======================
# The two TensorBoard groups that COLLECT one card per sample, outside the
# per-sample blocks. The dashboard groups on the text before the first '/', so
# these strings are the group headers exactly as they read on screen.
UNCOND_AUDIO_GROUP = "uncond generation"
REAL_AUDIO_GROUP = "ground truth"


def norm_wav(x):
    """
    A waveform as (1, L) float32 torch, peak-normalized -- what add_audio wants.

    Accepts numpy arrays, 1-D torch tensors and (1, L) torch tensors.

    The normalization is not cosmetic: a sonified condition is synthesized at a
    fixed low level while a generation is not, so without it the A/B between two
    cards would be between loudnesses as much as between contents.
    """
    a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    a = np.asarray(a, dtype=np.float32).reshape(1, -1)
    return torch.from_numpy(a / (np.abs(a).max() + 1e-8))


def audio_panel_tags(family, idx, active_conditions=(), suffix="",
                     conditioned=None):
    """
    The TensorBoard AUDIO tags of ONE sample -> {"conditions": {name: tag},
    "generation": tag, "generation_no_cond": tag, "real": tag}.

    Single source of truth for the audio window's layout, because the four
    places that log audio (the metrics step, the cheap preview between metrics
    steps, the step-0 real references, the f0 probe) all write into the same
    blocks and MUST agree on their names -- otherwise one sample would own two
    half-filled blocks instead of one.

    THREE layout facts drive every name here, and all three are properties of
    the dashboard rather than choices:

    1. Cards are grouped by the text BEFORE the first '/', and each group is its
       own headed, collapsible block. So the SAMPLE is that prefix. With every
       tag under a single 'Validation/' prefix the dashboard builds ONE grid
       holding every card of every sample, and a condition ends up separated
       from its own generation by a row break; one block per sample holds 2-5
       cards that stay together and are visibly walled off from the next sample.

    2. Inside a block cards are sorted alphabetically and flow into a grid that
       wraps every 2-3 cards depending on window width, so the numeric prefix
       is what fixes the listening order. That order is: the RECORDING the
       conditions were extracted from, then EVERY condition the generation was
       given, then the generation itself LAST.

          1  real_validation_XX         the recording (validation blocks only)
          2  f0_<family>_XX             the sonified f0 target, when f0 is on
          3..N <condition>_<family>_XX  the others, alphabetical
          N+1  generation_<family>_XX   the generation, after all of them

       A PROBE block has no recording -- its stimuli are synthesised, there is
       nothing they were extracted from -- so there the conditions start at 1.

       The generation is not named after any single condition. It used to be
       `generation_with_f0_XX`, which read as "conditioned on f0" on a run that
       also drove chroma and text -- the card is the generation of the WHOLE
       block, and the block header (and the condition cards above it) already
       say what went into it.

    3. The unconditional generation does NOT live in the per-sample block. It is
       COLLECTED into a group of its own, holding one card per sample, so that
       all of them can be heard as a grid of peers instead of one at a time
       inside separate collapsibles:

          uncond generation/uncond_<family>_XX

       The 'ground truth' group is what is LEFT of the same idea: it now holds
       the recordings of an UNCONDITIONED run only. In a conditioned run every
       recording sits at the top of its own validation block instead, next to
       the conditions that were extracted from it -- which is the comparison
       anyone actually makes, and it was being made across two windows.

       The family and the index in the CARD name are what ties a card back to
       the block it belongs to: 'uncond generation/uncond_validation_03' is the
       null twin of 'validation_03/4_generation_validation_03', drawn from the
       same noise. The group is flat -- no numeric prefix -- because nothing
       inside it is an ordered A/B against a neighbouring card.

    The card names repeat the family and index that the block header already
    shows. That redundancy is deliberate: a card read, filtered or screenshotted
    on its own still says what it is and which sample it belongs to.

    `family` is "probe" or "validation". `suffix` decorates the BLOCK header
    only (the probe prompt, the validation label). It is deliberately NOT put
    on the uncond card: "uncond_probe_00 [solo pipe organ ...]" reads as a
    generation driven by that prompt, which is the one thing it is not. The
    index is the cross-reference to the block (and to the f0 image of the same
    sample), where the prompt can be read. f0 leads the
    condition cards when it is active because it is the one most listened to
    against the generation; if it is off, the first condition alphabetically
    takes slot 1 and nothing else changes.

    With NO condition active at all (pure-unconditional run) the block
    generation IS the unconditional generation -- nothing was dropped to obtain
    it -- so it is filed straight into the uncond group and no per-sample block
    is created. Such a run's audio window is exactly the two collected groups.

    `conditioned` says whether ANYTHING conditioned this generation -- a frame
    condition OR a global one. It exists because `active_conditions` lists only
    the FRAME conditions (they are the ones that get a card of their own: a
    picture and a prompt are not waveforms), so a run conditioned on text/image
    ALONE looked identical here to a pure-unconditional one: the generation was
    filed into the uncond group, where the null generation of the same sample
    then landed on the very same tag at the very same step. Its default --
    "there is a frame condition" -- is what every caller meant before global-only
    runs existed, so passing nothing keeps the old behaviour exactly.
    """
    active = sorted(active_conditions)
    lead = "f0" if "f0" in active else (active[0] if active else None)
    if conditioned is None:
        conditioned = lead is not None
    block = f"{family}_{idx:02d}{suffix}"
    tail = f"{family}_{idx:02d}"

    # THE RECORDING LEADS THE BLOCK -- but only where one exists. A validation
    # panel is built FROM a real chunk: its conditions were extracted from that
    # recording, so the recording is where the listening starts, the conditions
    # are what was taken out of it, and the generation is what they produced. A
    # probe panel has no such recording (its stimuli are synthesised), and an
    # unconditioned run has no per-sample block at all -- there the recording
    # keeps its own collected group, which is the only way to hear real material
    # in a run with no panels.
    panel_real = (family == "validation" and conditioned)
    first = 2 if panel_real else 1

    # Then every condition card -- f0 leading when it is on -- and the
    # generation after the last of them, so a panel is listened to in the order
    # it was built: the recording, the stimuli taken from it, then what they
    # produced.
    ordered = ([lead] + [n for n in active if n != lead]) if lead else []
    conditions = {c: f"{block}/{j + first}_{c}_{tail}"
                  for j, c in enumerate(ordered)}
    gen_slot = len(ordered) + first

    uncond = f"{UNCOND_AUDIO_GROUP}/uncond_{tail}"
    real = (f"{block}/1_real_{tail}" if panel_real
            else f"{REAL_AUDIO_GROUP}/real_{tail}")

    return {
        "conditions": conditions,
        # Two cases:
        #   anything conditioned -> the generation owns the last card of the
        #     block, named after the block and NOT after one of the conditions
        #     (naming it after f0 hid chroma, rhythm and text from a reader of
        #     the card alone -- the very conditions the panel is there to show);
        #   nothing conditioned  -> the generation IS the uncond card, and the
        #     two keys deliberately name the same tag: the callers that write
        #     the null baseline are guarded on a condition being active, so the
        #     tag is still written exactly once.
        "generation": (f"{block}/{gen_slot}_generation_{tail}"
                       if conditioned else uncond),
        "generation_no_cond": uncond,
        "real": real,
    }
def subset_generation_tag(family, idx, label, suffix="", n_conditions=0,
                          has_real=False):
    """
    The audio card of ONE condition-subset generation, inside that sample's
    block. It takes the SAME slot as the reference generation -- the one right
    after the last condition card -- so every generation of the sample sits
    together at the end of the block and they sort among themselves by subset
    label. `n_conditions` is how many condition cards the block holds; pass the
    same count audio_panel_tags was given, or the ladder lands above the
    stimuli instead of below them. `has_real` says whether the recording takes
    slot 1 of that block (true for a conditioned validation panel), because that
    is what shifts every card below it down by one.
    """
    return (f"{family}_{idx:02d}{suffix}"
            f"/{int(n_conditions) + (2 if has_real else 1)}"
            f"_gen_{label}_{family}_{idx:02d}")


# The tags already written by THIS board's fixed cards, one per line. Lives
# beside the events files because that is what it describes.
FIXED_CARDS_FILE = ".fixed_cards_logged"


def fixed_card_pending(writer, tag):
    """True the FIRST time `tag` has to be written into this TensorBoard run
    directory, False every time after -- for the cards that never change.

    WHAT THIS IS FOR. A condition card is a STIMULUS: the sonified f0 the
    generation was asked for, the picture it was shown, the recording the
    conditions were extracted from. Those are read from disk, not generated, so
    they are identical at every step -- the validation panels are a FIXED set of
    samples (metrics_sample_positions) and the probe stimuli are fixed by
    construction. Re-logging them gave every one of those cards a step slider
    that walked through copies of the same picture, right beside the generation
    whose slider means something. Written once, the card has a single step and
    TensorBoard drops the control entirely (the audio and image loaders gate the
    <paper-slider> on `_hasMultipleSteps`, i.e. steps.length > 1). A card with
    one step is NOT empty at later steps: each card's slider is built from that
    card's OWN steps, so the loader keeps showing its only datum whatever the
    generation cards are showing.

    WHY A FILE, PER TAG. A resume reopens the same run directory with a NEW
    events file, and a second write would file two identical entries under one
    tag -- bringing the slider back with two copies of the same thing. So what
    has been written has to outlive the process. Per TAG rather than per family
    because the writers cover different panels: the cheap audio preview
    refreshes the first n_audio_samples blocks, the metrics step covers the
    whole influence set, and the real recordings are logged at startup. One flag
    for all of them would let whichever ran first close the door on the others.

    A resume into a FRESH run directory finds no file and writes every card
    again, which is what that board needs. Deleting the file re-writes them all
    on the next step -- which is also the answer if the panels are ever widened
    (n_influence_samples_valid / _probe or n_plot raised) on an existing board."""
    d = getattr(writer, "log_dir", None)
    if not d:
        return True                      # nowhere to remember: write it
    cache = getattr(fixed_card_pending, "_cache", None)
    if cache is None:
        cache = fixed_card_pending._cache = {}
    seen = cache.get(d)
    if seen is None:
        try:
            with open(os.path.join(d, FIXED_CARDS_FILE), encoding="utf-8") as fh:
                seen = {line.rstrip("\n") for line in fh}
        except OSError:
            seen = set()
        cache[d] = seen
    # A tag is one line of the file, so it must not contain one. Nothing builds
    # a tag with a newline in it today (the block suffixes are one-line captions
    # truncated to 48 chars); normalising here keeps a future one from silently
    # splitting into two entries that never match again.
    key = tag.replace("\n", " ")
    if key in seen:
        return False
    seen.add(key)
    try:
        with open(os.path.join(d, FIXED_CARDS_FILE), "a", encoding="utf-8") as fh:
            fh.write(key + "\n")
    except OSError:
        pass                             # unwritable dir: at worst they repeat
    return True


def resolve_influence_subsets(spec, active_names):
    """
    -> [(label, (condition names, ...))]: which CONDITION SUBSETS the metrics
    step generates and scores, in the order they will appear in the panel.

    `spec` is sampling.influence_subsets. Each entry is either an explicit list
    of condition names, or one of the keywords:

        "all"         -> every active condition (the reference row)
        "loo"         -> leave-one-out: N subsets, each missing one condition
        "singletons"  -> N subsets, each holding exactly one condition

    An empty/None spec returns [] and the metrics step behaves exactly as
    before: one conditioned pass, one null pass, one table.

    Names unknown to the run are a hard error rather than a silent skip -- a
    typo in a subset list would otherwise quietly measure something else. The
    empty subset is dropped instead: it is the unconditional pass, which the
    metrics step already generates and pairs against.

    Every subset is ordered by the run's canonical condition order (not by the
    order written in the YAML) so a label means the same thing whatever way it
    was spelled, and two spellings of one subset collapse to one generation.
    """
    active = list(active_names or [])
    if not spec or not active:
        return []
    out = []

    def add(label, names):
        picked = tuple(a for a in active if a in set(names))
        if not picked:
            return                       # empty subset == the uncond pass
        if any(lbl == label for lbl, _ in out):
            return                       # already asked for, in any spelling
        out.append((label, picked))

    def label_for(names):
        if len(names) == len(active):
            return "all"
        if len(names) == 1:
            return f"only_{names[0]}"
        if len(names) == len(active) - 1:
            missing = [a for a in active if a not in set(names)]
            return f"no_{missing[0]}"
        return "+".join(names)

    for entry in spec:
        if isinstance(entry, str):
            key = entry.strip().lower()
            if key == "all":
                add("all", active)
            elif key in ("loo", "leave_one_out"):
                for n in active:
                    rest = [a for a in active if a != n]
                    add(label_for(tuple(rest)), rest)
            elif key in ("singletons", "each"):
                for n in active:
                    add(f"only_{n}", [n])
            else:
                raise ValueError(
                    f"sampling.influence_subsets: unknown keyword '{entry}'. "
                    f"Use 'all', 'loo', 'singletons', or an explicit list of "
                    f"condition names from {active}.")
        else:
            names = [str(n) for n in entry]
            unknown = [n for n in names if n not in active]
            if unknown:
                raise ValueError(
                    f"sampling.influence_subsets: condition(s) {unknown} are "
                    f"not active in this run. Active: {active}.")
            picked = tuple(a for a in active if a in set(names))
            add(label_for(picked), picked)
    return out


# ======================
# WHICH GENERATIONS OWN A TENSORBOARD PANEL
# ======================
def metrics_sample_positions(n_samples, n_influence):
    """
    -> the positions of the metrics generation list that make up the INFLUENCE
    SET: the N validation samples that are scored for condition fidelity AND
    own a TensorBoard panel (a target-vs-generated image and an audio block).

    ONE list, deliberately. The influence table, the Images window and the Audio
    window are three views of the SAME samples: the number in the table is about
    the curve you are looking at and the audio you are hearing. Splitting them
    (score over 64, plot 4) meant the panels illustrated a number computed on
    samples you never saw.

    Single source of truth also across steps: the metrics step and the cheaper
    audio preview both write into the validation_XX/ block, so they MUST agree
    on which validation sample XX is -- otherwise the same block would hold two
    different recordings depending on which step last wrote to it.

    The spread is uniform over the WHOLE list, not its first N: the generation
    indices come from linspace(0, len(val)-1, n_samples), so a prefix lands on a
    tiny head of the validation set (with n_samples=1024 over a 165-sample val,
    the first 64 generations cover only 11 DISTINCT conditions, each re-drawn
    with different noise) and the mean would describe that head.
    """
    n_samples = int(n_samples or 0)
    n_fid = min(int(n_influence or 0), n_samples)
    if n_fid <= 0 or n_samples <= 0:
        return []
    return sorted(set(
        torch.linspace(0, n_samples - 1, n_fid).round().long().tolist()))


INFLUENCE_SET_KEYS = {"valid": "n_influence_samples_valid",
                      "probe": "n_influence_samples_probe"}


def influence_set_size(sampling_cfg, which="valid", default=16) -> int:
    """The N of one half of the influence set:
      which="valid" -> sampling.n_influence_samples_valid, how many validation
                       samples are scored, plotted and played;
      which="probe" -> sampling.n_influence_samples_probe, how many probe
                       stimuli (at most the size of a bank, 16).

    Two knobs since 29 Sept 2026 (one, n_influence_samples, before). Each half
    stays self-consistent -- its table rows describe exactly the samples its
    panels show -- and the two are read side by side across steps, like the
    training and the validation loss, never against each other.

    It is NOT n_metrics_samples: the distributional metrics (FD-DAC / KL / FAD)
    run on that much larger pool, because they estimate a covariance and want
    the samples; the influence set is what a human reads panel by panel, so it
    is small and fully shown.

    Only a MISSING key (or null) falls back to `default`. A 0 is honoured: it
    turns the probe off. The ranges are checked once in load_config."""
    if sampling_cfg is None:
        return int(default)
    v = sampling_cfg.get(INFLUENCE_SET_KEYS[which], None)
    return int(default) if v is None else int(v)


# ======================
# AUDIO PREVIEW (conditioned, into the per-sample panels)
# ======================
@torch.no_grad()
def generate_and_log_audio(
    model, normalizer, val_dataset, n_frames, step, writer, device,
    output_dir, n_samples, sampling_cfg, conditioning_cfg, use_amp,
    frame_dims, global_configs, prefix="EMA",
):
    """
    Cheap audio preview BETWEEN metrics steps, written into the SAME panels the
    metrics step uses (the validation_XX/ blocks): same validation samples, same
    tag names, only a finer cadence. The audio window therefore holds ONE family
    of blocks, and the step slider walks each block through training instead of
    scattering near-identical tags across the dashboard.

    It refreshes the condition cards and the `generation_validation_XX` card;
    the null generation and the real reference are added by the metrics step,
    which is the only place they are computed.

    `n_samples` is sampling.n_audio_samples: how many panels to refresh, capped
    by how many panels exist.
    """
    guidance = float(conditioning_cfg.guidance_scale)
    from condition_metrics import sonify_condition

    total = len(val_dataset)
    n_metrics = int(getattr(sampling_cfg, "n_metrics_samples", 512) or 512)
    # The influence set, then the PREFIX of it this preview refreshes: index XX
    # keeps meaning the same validation sample whether the block was last
    # written by the metrics step or by this cheaper preview.
    panel_pos = metrics_sample_positions(
        n_metrics, influence_set_size(sampling_cfg))[:max(0, int(n_samples))]
    # The metrics step generates from these val-dataset indices; the panel of
    # position p describes val_dataset[indices[p]].
    indices = torch.linspace(0, total - 1, n_metrics).long().tolist()
    if not panel_pos or not (frame_dims or global_configs):
        # NO conditioning at all (or no panels): fall back to a plain spread over
        # the validation set, still one panel per sample. A global-only run does
        # NOT come here -- it has the same panels as any other conditioned run,
        # and taking this branch made the preview refresh block validation_XX
        # with a different validation sample from the one the metrics step put
        # there.
        panel_pos = list(range(min(int(n_samples), total)))
        indices = torch.linspace(0, total - 1,
                                 max(1, min(int(n_samples), total))).long().tolist()
    panel_pos = panel_pos[:max(1, int(n_samples))]
    # ONE source of truth for the block names (see validation_panel_suffixes).
    _sfx = validation_panel_suffixes(val_dataset, sampling_cfg, frame_dims,
                                     global_configs)

    dac_model = get_dac()

    for k, p in enumerate(panel_pos):
        idx = indices[p] if p < len(indices) else indices[-1]
        (_frames_real, frame_cond_real, _label_idx, text_emb, image_emb,
         text_ctx) = val_dataset[idx]

        fc = {kk: v.unsqueeze(0).to(device).float()
              for kk, v in frame_cond_real.items()}
        gc = {}
        if "text" in global_configs:
            gc["text"] = text_emb.unsqueeze(0).to(device)
        if "image" in global_configs:
            gc["image"] = image_emb.unsqueeze(0).to(device)

        tctx = None
        if getattr(model, "text_cross_layers", None):
            tctx = {"tokens": text_ctx["tokens"].unsqueeze(0),
                    "mask":   text_ctx["mask"].unsqueeze(0)}
        gen = euler_sample_cfg(
            model, n_frames, device,
            steps=sampling_cfg.euler_steps,
            t_min=sampling_cfg.t_min,
            t_max=sampling_cfg.t_max,
            use_amp=use_amp,
            frame_cond=fc, global_cond=gc, guidance=guidance,
            frame_dims=frame_dims, global_configs=global_configs,
            text_ctx=tctx,
        )
        if not torch.isfinite(gen).all():
            continue

        tags = audio_panel_tags("validation", k, frame_cond_real.keys(),
                                suffix=_sfx.get(k, ""),
                                conditioned=bool(frame_dims or global_configs))

        # decode_frames_to_wav returns 1-D; unsqueeze back to (1, T), which is
        # what add_audio expects.
        waveform = decode_frames_to_wav(gen, normalizer, dac_model).unsqueeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)

        # Same cards as the metrics step, written to the SAME tags: the f0
        # target and the generation it produced are the first two cards of the
        # block, each on its own player.
        # The condition cards go in ONCE, at step 0: this panel is the same
        # validation sample at every step (metrics_sample_positions), so its
        # stimulus never changes and a slider over it would walk through copies
        # of one waveform. Only the generation below moves with training. The
        # guard is per tag because this preview refreshes only the first
        # n_audio_samples blocks while the metrics step covers the whole
        # influence set -- see fixed_card_pending.
        for cname, carr in sorted(frame_cond_real.items()):
            son = sonify_condition(cname, carr.cpu().numpy(), DAC_SAMPLE_RATE)
            if son is not None and fixed_card_pending(
                    writer, tags["conditions"][cname]):
                writer.add_audio(tags["conditions"][cname], norm_wav(son),
                                 global_step=0,
                                 sample_rate=DAC_SAMPLE_RATE)

        writer.add_audio(tags["generation"], wn,
                         global_step=step, sample_rate=DAC_SAMPLE_RATE)

        wav_path = os.path.join(
            output_dir, f"step{step:07d}_{prefix}_{k:02d}.wav"
        )
        sf.write(wav_path, waveform.squeeze().numpy(), DAC_SAMPLE_RATE)
    # dac_model is the shared singleton -> do not delete it.


# ======================
# LOG REAL AUDIO SAMPLES (once at startup, step=0)
# ======================
@torch.no_grad()
def log_real_audio_samples(val_dataset, normalizer, writer, n_samples,
                           sampling_cfg=None, frame_dims=None,
                           global_configs=None):
    """Logs the real audio of the PANEL samples at step 0, so the recording is
    audible from the start instead of only appearing at the first metrics step.

    CONDITIONED run: one card at the TOP of every validation block, beside the
    conditions that were extracted from that very recording. EVERY panel of the
    influence set gets one -- `n_samples` does not bound it, because a panel
    without its recording is a panel you cannot judge, and the cost is one DAC
    decode each, once per board.

    UNCONDITIONED run: there are no per-sample blocks at all, so the recordings
    go to the collected "ground truth" group and `n_samples`
    (sampling.n_audio_samples) is what says how many to log -- that case is the
    only reason the knob still exists."""
    dac_model = get_dac()

    total = len(val_dataset)
    panel_pos, indices = [], []
    # `frame_dims or global_configs`: same reason as generate_and_log_audio --
    # a global-only run has the ordinary panels, so the recording of block XX
    # must be the sample the metrics step will put in block XX.
    conditioned = bool(frame_dims or global_configs)
    if sampling_cfg is not None and conditioned:
        n_metrics = int(getattr(sampling_cfg, "n_metrics_samples", 512) or 512)
        panel_pos = metrics_sample_positions(
            n_metrics, influence_set_size(sampling_cfg))
        indices = torch.linspace(0, total - 1, n_metrics).long().tolist()
    if not panel_pos:
        panel_pos = list(range(min(int(n_samples), total)))
        indices = torch.linspace(0, total - 1,
                                 max(1, min(int(n_samples), total))).long().tolist()

    _sfx = validation_panel_suffixes(val_dataset, sampling_cfg, frame_dims,
                                     global_configs)
    written = 0
    for k, p in enumerate(panel_pos):
        # Checked BEFORE the decode: this runs at every startup, and on a resume
        # into the same run directory the card is already there. Writing it
        # again would put a second step-0 entry under one tag and give a
        # recording a step slider -- see fixed_card_pending.
        # `conditioned` decides WHERE the card goes: slot 1 of block XX, or the
        # collected group. It is the same flag audio_panel_tags is given
        # everywhere else, so all the writers agree on the one tag.
        tag = audio_panel_tags("validation", k, suffix=_sfx.get(k, ""),
                               conditioned=conditioned)["real"]
        if not fixed_card_pending(writer, tag):
            continue
        idx = indices[p] if p < len(indices) else indices[-1]
        # ConditionedAudioDataset returns a 5-tuple: take only the frames
        frames, _frame_cond, _label_idx, _text_emb, _image_emb, _ctx = val_dataset[idx]
        waveform = decode_frames_to_wav(frames, normalizer, dac_model).unsqueeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)

        writer.add_audio(tag, wn, global_step=0, sample_rate=DAC_SAMPLE_RATE)
        written += 1

    # dac_model is the shared singleton -> do not delete it.
    print(f"  {written} real audios logged on TensorBoard"
          + ("" if written == len(panel_pos)
             else f" ({len(panel_pos) - written} already on this board)"))


# ======================
# OUT-OF-THE-BOX JOINT PROBE (proof of concept)
# ======================
@torch.no_grad()
def run_joint_probe(probe_sets, model, normalizer, n_frames,
                    step, writer, device,
                    output_dir, use_amp, sampling_cfg, guidance,
                    frame_dims, global_configs, fidelity_evaluator,
                    dac_model, prefix, n_plot, n_audio,
                    metrics_seed=None, global_embedders=None):
    """
    Generate conditioned on the out-of-the-box probe stimuli of EVERY active
    condition AT ONCE, and report the result three ways.

    `probe_sets` is {condition name -> ConditionProbeSet}. Panel i drives every
    active condition with the i-th stimulus of its OWN bank: the f0 of a scale,
    the chroma of a triad, the energy of a crescendo, the beat grid of a 120 bpm
    pattern -- combined by index. The pairing is by index and therefore
    arbitrary, but it is DETERMINISTIC, so panel 03 means the same combination
    at every checkpoint and the curves stay comparable across steps.

    WHY JOINTLY. A model trained on a fixed condition set with
    `conditioning.p_drop_each_frame = 0.0` only ever sees TWO situations: all of
    its conditions present, or all absent. Probing such a model one condition at
    a time (the others nulled) asks it for a partial subset it was never trained
    to resolve, so the answer would describe the hole in the training
    distribution rather than the conditioning. The joint probe presents exactly
    the shape the model was trained on, which is what makes it readable.

    WHY THE COMBINATION IS NOT "ALIGNED". The stimuli come from DIFFERENT banks
    and are not mutually consistent (a rising scale under a static triad). That
    is deliberate. The probe is an out-of-the-box controllability check on
    unambiguous stimuli: it answers "does this conditioning move the generation
    at all", which the validation rows cannot answer alone -- on real material a
    condition is often not cleanly extractable (a smeared chromagram, a beat
    grid that does not exist, an f0 that fails on 3 samples out of 4) and a
    middling score there does not separate "the conditioning is weak" from "the
    target was ambiguous". The validation rows carry the aligned, in-corpus
    case; the probe carries the clean, artificial one. Both are needed and
    neither replaces the other.

    What it logs, all at `step` so the TensorBoard slider walks them together:
      * IMAGES  Validation/<cond>_probe_vs_gen_XX -- target vs re-extracted,
                one per active condition, titled with the stimulus it used
      * AUDIO   the probe_XX/ block: one card per condition holding the stimulus
                that condition was taken from, then the generation they jointly
                conditioned. The null generation goes to the "uncond generation"
                group with the others.
      * TEXT    a <cond>_probe row per condition for the Condition_influence
                table, returned to the caller as (influence, coverage) to be
                merged in. Nothing else: which stimulus each panel used is
                already in the title of that panel's own comparison image.

    The delta column needs a baseline, so the generation is PAIRED (with-cond
    and null from the same x0, one fused batch) exactly like the validation
    metrics -- the null generations cost nothing extra, they are already
    required by the CFG math.

    `fidelity_evaluator` is reused (reset first) rather than rebuilt: it carries
    the run's exact extractor configuration, and instantiating a second CREPE
    would risk the two drifting apart. The caller must therefore have already
    taken its per_sample()/coverage()/contours() copies for the validation rows.
    """
    from condition_metrics import pair_influence, pair_scalar
    # f0 keeps its own dedicated plot (log-Hz axis + voicing ribbon); the other
    # conditions are drawn by the generic plotter, which picks the form that
    # suits the shape (curve / two curves / paired heatmaps).
    from probe_conditions import plot_condition_comparison

    # Only conditions the MODEL actually has: a bank for a condition this run
    # does not use would be fed into a slot that does not exist.
    # Frame conditions and GLOBAL conditions are both driven, jointly, by the
    # same panel: panel i takes the i-th stimulus of every active bank. The two
    # are kept in separate name lists only because they enter the model through
    # different doors (concatenated at the input vs added to the AdaLN vector);
    # everything else below treats them alike, and a run without globals simply
    # has an empty second list.
    names = [c for c in (frame_dims or {}) if c in (probe_sets or {})]
    gnames = [c for c in (global_configs or {}) if c in (probe_sets or {})]
    if not names and not gnames:
        return {}, {}
    missing = [c for c in list(frame_dims or {}) + list(global_configs or {})
               if c not in (probe_sets or {})]
    if missing:
        # Not fatal, but it means those slots go in NULL and the probe is no
        # longer the in-distribution shape described above -- say so loudly.
        print(f"    [probe] WARNING: no bank for {missing}; those conditions "
              f"go in NULL, so this probe is a partial subset")

    # The panels are index-aligned across banks, so the count is the shortest.
    n_probe = min(len(probe_sets[c]) for c in names + gnames)
    if n_probe == 0:
        return {}, {}
    n_plot = max(0, min(int(n_plot), n_probe))

    targets = {c: [np.asarray(t, dtype=np.float32)
                   for t in probe_sets[c].targets] for c in names}
    gtargets = {c: [np.asarray(t, dtype=np.float32).reshape(-1)
                    for t in probe_sets[c].targets] for c in gnames}

    def _frame_cond(idxs):
        # Every configured name must be present (FrameConditionEncoder.forward),
        # so start from the null dict and fill the ones this probe drives.
        fc = make_null_frame_conditions(len(idxs), n_frames, frame_dims or {},
                                        device)
        for c in names:
            fc[c] = torch.from_numpy(
                np.stack([targets[c][i] for i in idxs])).to(device).float()
        return fc

    def _global_cond(idxs):
        """The global half of the same panel: (B, dim) per active global.
        Same null-then-fill contract as _frame_cond, so a global with no bank
        goes in as zeros rather than being absent from the dict."""
        gc = make_null_global_conditions(len(idxs), global_configs or {}, device)
        for c in gnames:
            gc[c] = torch.from_numpy(
                np.stack([gtargets[c][i] for i in idxs])).to(device).float()
        return gc

    # The cross-attention context of a group of probe stimuli: the PROMPT as a
    # token sequence, from the text bank itself. A model without the sub-layer
    # gets None and nothing is built.
    #
    # A probe bank with no token sequences is reported ONCE rather than passed
    # over: on such a run every probe generation would be driven by the null
    # token, the panels would still be drawn and the influence row would still
    # be computed, and nothing on screen would say that the prompt never
    # reached the model.
    _probe_ctx_warned = []

    def _text_ctx(idxs):
        if not getattr(model, "text_cross_layers", None):
            return None
        ps = (probe_sets or {}).get("text")
        if ps is None or "text" not in gnames:
            return None
        items = [ps.context(i) for i in idxs]
        if any(c is None for c in items):
            if not _probe_ctx_warned:
                _probe_ctx_warned.append(True)
                print("    [probe] WARNING: the text bank carries no token "
                      "sequences, so the cross-attention sees only its null "
                      "token on every probe. Delete the probe_text cache and "
                      "let it rebuild.")
            return None
        return {"tokens": torch.cat([c["tokens"] for c in items], dim=0),
                "mask":   torch.cat([c["mask"]   for c in items], dim=0)}

    # Same seeding contract as the validation metrics: a fixed generator makes
    # the probe curves comparable ACROSS checkpoints (what moves is the model,
    # not the noise). None = free-running.
    def _rng():
        if metrics_seed is None:
            return None
        g = torch.Generator(device=device)
        g.manual_seed(int(metrics_seed))
        return g

    spf = max(1, int(sampling_cfg.get("metrics_samples_per_forward", 1) or 1))
    # The probe's baseline is ITS OWN: n_probe generations without conditions,
    # from the same x0 as the conditioned ones. It has nothing to do with
    # sampling.metrics_uncond, which decides whether the DISTRIBUTIONAL metrics
    # (FD-DAC / KL / FAD) are also computed on the unconditioned branch over
    # n_metrics_samples. Tying the two, as this line used to, meant switching off
    # a 512-generation metric silently removed the delta column of the probe --
    # the very number the probe exists to produce.
    paired = guidance > 1.0

    cond_lat, null_lat = [], []
    gen_rng = _rng()
    if paired:
        for s in range(0, n_probe, spf):
            grp = list(range(s, min(s + spf, n_probe)))
            gc, gu = euler_sample_cfg_paired(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                use_amp=use_amp,
                frame_cond=_frame_cond(grp), global_cond=_global_cond(grp),
                guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=_text_ctx(grp),
            )
            cond_lat.extend(gc)
            null_lat.extend(gu)
    else:
        # No baseline available (guidance <= 1, so there is no CFG to fuse and
        # no unconditioned branch to compare against). The row is then reported
        # UNPAIRED: with-cond only, delta as n/a -- never as if a baseline had
        # been measured.
        for i in range(n_probe):
            cond_lat.append(euler_sample_cfg(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                use_amp=use_amp,
                frame_cond=_frame_cond([i]), global_cond=_global_cond([i]),
                guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=_text_ctx([i]),
            ))

    # ---- score + collect the curves, one decode per generation ----
    # add_sample receives EVERY condition of the panel, so one decode scores all
    # of them and the table gets a row per condition from a single pass.
    def _score(lat_list, keep):
        fidelity_evaluator.reset()
        fidelity_evaluator.keep_contours_for(range(n_plot) if keep else ())
        wavs = []
        # Per-sample similarity of the GENERATION to the global stimulus that
        # conditioned it: {global name: {panel index: cosine}}. Measured on the
        # same decoded waveform the frame conditions are re-extracted from, so
        # it costs one embedder forward and no extra generation.
        gsim = {c: {} for c in gnames if c in (global_embedders or {})}
        for i, lat in enumerate(lat_list):
            wav = decode_frames_to_wav(lat, normalizer, dac_model)
            wnp = wav.numpy()
            fidelity_evaluator.add_sample(
                wnp, DAC_SAMPLE_RATE, n_frames,
                {c: targets[c][i] for c in names}, sample_id=i)
            for c in gsim:
                try:
                    emb = global_embedders[c].embed(wnp, DAC_SAMPLE_RATE)
                    gsim[c][i] = float(np.dot(emb, gtargets[c][i]))
                except Exception as _e:
                    # One failed embedding loses one row of one panel, not the
                    # metrics step. Reported once per condition per pass.
                    if not gsim[c]:
                        print(f"    [probe] {c} similarity unavailable: "
                              f"{type(_e).__name__}: {_e}")
            wavs.append(wav if i < n_plot else None)
        return (fidelity_evaluator.per_sample(), fidelity_evaluator.coverage(),
                {c: fidelity_evaluator.contours(c) for c in names}, wavs, gsim)

    ps_cond, cov_cond, cont_cond, wavs_cond, gsim_cond = _score(cond_lat, keep=True)
    # Both defaults matter: with no null pass (metrics_uncond off, or guidance
    # <= 1) the audio loop below still asks for len(wavs_null).
    ps_null, wavs_null, gsim_null = {}, [], {}
    if null_lat:
        # The null pass contributes only its per-sample values: `attempted` and
        # the failure counts in the panel describe the CONDITIONED pass, which
        # is the one the row is about.
        ps_null, _cn, _ct, wavs_null, gsim_null = _score(null_lat, keep=False)

    # ---- IMAGES + AUDIO for the first n_plot panels ----
    # One TensorBoard PANEL per probe index: the stimuli it was conditioned on
    # and the generation sit under the SAME tag prefix, so the audio window
    # cannot separate a generation from the conditions that produced it.
    for i in range(n_plot):
        # The block name of THIS panel, decided ONCE and used by every tag it
        # owns -- the comparison images, the image card and the audio. A text
        # prompt cannot be a card, so it names the block instead; computing it
        # here rather than beside the audio (as it was) is what stops the Images
        # and Audio windows from filing the same panel under two names.
        _suffix = (f" [{probe_sets['text'].text(i)[:48]}]"
                   if "text" in gnames else "")
        _blk = f"probe_{i:02d}{_suffix}"

        # The score shown in each plot title is that condition's FIRST metric,
        # whatever it is called (f0/energy -> corr, chroma -> cosine,
        # rhythm -> beat_corr; under mir_eval f0 -> overall_accuracy, chroma /
        # chord -> chroma_accuracy): the probe must not hardcode a metric name
        # that only some conditions have.
        # Its name travels with it, so the f0 title says which number it is.
        for c in names:
            _mkeys = sorted(k for k in ps_cond if k.startswith(f"{c}/"))
            corr_map = ps_cond.get(_mkeys[0], {}) if _mkeys else {}
            _mname = _mkeys[0].partition("/")[2] if _mkeys else None
            if i in cont_cond.get(c, {}):
                tgt, gen = cont_cond[c][i]
                writer.add_image(
                    # Same block name as this panel's AUDIO tags, so the Images
                    # and Audio tabs collapse into the same per-sample sections
                    # instead of one flat list of every condition x every panel.
                    f"{_blk}/{c}_target_vs_gen",
                    plot_condition_comparison(
                        c, tgt, gen, kind="probe",
                        label=f"'{probe_sets[c].names[i]}'", step=step,
                        prefix=prefix, guidance=guidance,
                        score=corr_map.get(i), score_name=_mname),
                    global_step=step)

        # The GLOBAL stimuli of the same panel, in the same block.
        # An image IS the stimulus, so it is shown as-is -- there is no
        # "target vs re-extracted" pair to draw, because nothing re-extracts a
        # picture from audio. A text prompt is not an image at all: it rides in
        # the TITLE of the panel's other cards via `suffix` below, and its
        # adherence is a number, which belongs in the influence table.
        # Once per board, at step 0: the stimulus is a file on disk, chosen
        # before training starts, so it is the same picture at every probe step
        # and a slider over it would walk through copies of itself.
        for c in gnames:
            if c != "image":
                continue
            try:
                _arr = np.asarray(probe_sets[c].image(i)).transpose(2, 0, 1)
                # Guarded AFTER the load, so a picture that failed to open is
                # retried at the next probe step instead of being written off.
                _tag = f"{_blk}/image_condition"
                if fixed_card_pending(writer, _tag):
                    writer.add_image(_tag, _arr, global_step=0)
            except Exception as _e:
                print(f"    [probe] image card {i:02d} unavailable: "
                      f"{type(_e).__name__}: {_e}")

        # `names` holds the FRAME banks this panel drives; a probe over text or
        # image alone has none, and without `conditioned` its generation would
        # be filed as the unconditional one.
        tags = audio_panel_tags("probe", i, names, suffix=_suffix,
                                conditioned=bool(names or gnames))

        # The stimuli are written ONCE for the whole board, at step 0: they are
        # read from probe_sets, so they are the same sound at every probe step
        # and a card with one step carries no slider (fixed_card_pending
        # explains why a file, not a flag, is what remembers it across a
        # resume). They are written unconditionally, including when nothing was
        # generated: the card is what keeps a failed panel visible instead of
        # silently absent. Each is logged at its OWN bank's rate, which need not
        # be the DAC's.
        for c in names:
            if not fixed_card_pending(writer, tags["conditions"][c]):
                continue
            writer.add_audio(tags["conditions"][c],
                             norm_wav(probe_sets[c].wav(i)),
                             global_step=0, sample_rate=probe_sets[c].sr)
        if wavs_cond[i] is not None:
            writer.add_audio(tags["generation"], norm_wav(wavs_cond[i]),
                             global_step=step, sample_rate=DAC_SAMPLE_RATE)
            sf.write(os.path.join(_probe_dir(output_dir, step),
                                  f"probe_{i:02d}.wav"),
                     wavs_cond[i].numpy(), DAC_SAMPLE_RATE)
        # The null generation of the SAME panel, on its own card: it is the
        # baseline the influence row is computed against, and what it has to be
        # told apart from is the card beside it.
        if i < len(wavs_null) and wavs_null[i] is not None:
            # Card bounded by n_audio_samples, like its validation twin: the
            # "uncond generation" group is listening material, so it gets n per
            # family rather than one per panel. The .wav is written for every
            # panel regardless -- on disk the baseline of panel 07 has to exist
            # even when only the first few are worth a card.
            if i < n_audio:
                writer.add_audio(tags["generation_no_cond"], norm_wav(wavs_null[i]),
                                 global_step=step, sample_rate=DAC_SAMPLE_RATE)
            sf.write(os.path.join(_probe_dir(output_dir, step),
                                  f"probe_{i:02d}_uncond.wav"),
                     wavs_null[i].numpy(), DAC_SAMPLE_RATE)

    # NO "which stimuli each panel combines" TABLE.
    # There used to be a Validation/Probe_combinations text panel here, mapping
    # each panel index to the stimulus every condition contributed. It was
    # removed as redundant: each probe comparison image in the IMAGES window is
    # already TITLED with its own stimulus name (see the `label=` passed to the
    # plotters above), so the mapping is readable panel by panel where it is
    # actually being looked at. The table restated it in a second window, and
    # the only thing it added -- reading the whole mapping at a glance -- was
    # not worth a permanent text panel that had to be scrolled past at every
    # metrics step.

    # ---- the influence rows, renamed so they read as their own axis ----
    # pair_influence keys off the condition name; renaming to "<cond>_probe" is
    # what keeps a probe row from being mistaken for the validation row of the
    # same condition in the same table.
    inf, cov = pair_influence(ps_cond, ps_null, coverage_cond=cov_cond,
                              have_null=bool(null_lat))
    rows = {f"{c}_probe": inf[c] for c in names if inf.get(c)}

    # The GLOBAL conditions' rows, built the same way from a per-sample SCALAR
    # instead of a re-extracted curve: pair_scalar does for one number what
    # pair_influence does for a metric dict. Same delta, same baseline, same
    # table -- so "how much did the image move this generation" is read next to
    # "how much did f0 move it", on the same scale and from the same pass.
    _gmetric = {"text": "clap_sim", "image": "clip_sim"}
    for c in gnames:
        if c not in gsim_cond or not gsim_cond[c]:
            continue
        _cm, _nm, _dm, _npair = pair_scalar(
            gsim_cond[c], gsim_null.get(c, {}), have_null=bool(null_lat))
        key = _gmetric.get(c, "sim")
        rows[f"{c}_probe"] = {key: {"cond": _cm, "null": _nm, "delta": _dm}}
        cov[f"{c}_probe/{key}"] = {
            "valid": _npair,
            "attempted": len(gsim_cond[c]),
            "unpaired": (len(set(gsim_cond[c]) ^ set(gsim_null.get(c, {})))
                         if null_lat else 0),
        }

    if not rows:
        # Every re-extraction failed. Say so on the console rather than adding an
        # empty block that renders as no row at all -- "the probe is missing from
        # the table" and "the probe scored nothing" must not look the same.
        print("    [probe] no measurable generation -- no probe row this step")
        return {}, {}
    coverage = {}
    for c in names:
        coverage.update({k.replace(f"{c}/", f"{c}_probe/", 1): v
                         for k, v in cov.items() if k.startswith(f"{c}/")})
    # The global rows wrote their coverage under the final key already (they
    # were never named "<cond>/..."), so carry those entries over untouched.
    coverage.update({k: v for k, v in cov.items() if k.endswith("_probe/clap_sim")
                     or k.endswith("_probe/clip_sim")})
    return rows, coverage


def load_text_label_vocab(latent_root):
    """-> (phrases, (n, dim) embeddings) from the dataset, or (None, None).

    Written by preprocess_stream.py beside the conditions it labels. Missing is
    NORMAL -- a dataset preprocessed before this existed, or without --global
    text -- and simply means the panels show no words."""
    try:
        d = Path(latent_root).parent / "global_conditions"
        js, npy = d / "text_vocab.json", d / "text_vocab.npy"
        if not (js.exists() and npy.exists()):
            return None, None
        phrases = json.loads(js.read_text(encoding="utf-8"))["phrases"]
        return list(phrases), np.load(str(npy)).astype(np.float32)
    except Exception as e:
        print(f"[metrics] label vocabulary unreadable ({type(e).__name__}: {e})")
        return None, None


def load_text_captions(latent_root):
    """-> {chunk key: caption} from the dataset's text_labels.jsonl, or {}.

    The captions are written by preprocess_stream.py for EVERY chunk, so the
    panel does not decide how a sample is described -- it reads what the dataset
    says. That is the point of the file: the number of panels a machine can
    afford must not decide how much of the corpus gets described, and the words
    under a panel must be the same words the dataset records.

    Missing is NORMAL (a dataset built before the sidecar existed, or one
    without --global text) and the caller then falls back to computing the
    retrieval itself, as it always did.
    """
    try:
        p = Path(latent_root).parent / "global_conditions" / "text_labels.jsonl"
        if not p.exists():
            return {}
        out = {}
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("chunk"):
                    out[r["chunk"]] = r.get("caption") or r.get("class") or ""
        return out
    except Exception as e:
        print(f"[metrics] text_labels.jsonl unreadable "
              f"({type(e).__name__}: {e})")
        return {}


def load_caption_conditions(latent_root):
    """-> ({chunk key: row index}, (n_captions, dim) float32), or ({}, None).

    The CLAP TEXT embedding of each DISTINCT caption, written by
    preprocess_stream.py. It is what lets the validation generate FROM the
    description -- the vector a prompt would put in the text slot at inference --
    without CLAP ever being loaded in the training process, which is exactly the
    arrangement the dataset-side rewrite of 6 Sept was for.

    Distinct captions, not one row per chunk: with --text_labels_n 1 a four-class
    corpus has four of them.

    A thin view over audio_dataset_cond.load_caption_table, which is the single
    reader of that sidecar. It has to be the same one the DATASET uses: the two
    would otherwise be free to disagree about which caption belongs to which
    chunk, and the disagreement would show up as a validation number that
    describes a different description from the one in the panel header.
    """
    t = load_caption_table(latent_root)
    if not t or "emb" not in t:
        return {}, None
    return t.get("ids", {}), t["emb"]


def _chunk_key_of(val_dataset, cond_path):
    """The key text_labels.jsonl is indexed by: the chunk's path relative to
    conditions/, without the extension. None when it cannot be formed.

    NB the sidecar is keyed by PREPROCESSING chunk, which is the unit the CLAP
    vector belongs to. When the model's n_frames is shorter than a stored chunk,
    several training samples come out of that chunk -- and share its caption,
    correctly: they were conditioned on the very same vector.
    """
    try:
        if cond_path is None:
            return None
        root = Path(val_dataset.latent_root).parent / "conditions"
        return Path(cond_path).relative_to(root).with_suffix("").as_posix()
    except Exception:
        return None


def describe_validation_sample(val_dataset, ds_idx, captions=None,
                               phrases=None, vocab_emb=None, k=1):
    """One human-readable line for a validation sample: its CATEGORY, then the
    nearest phrases to its stored text vector, each with its cosine.

    The category is what the sample IS -- read off the dataset, exact. The
    phrases are a retrieval over a closed vocabulary and are shown with their
    cosine precisely so the two are not confused: a phrase at 0.12 is the least
    bad match in the list, not a description.

    TWO SOURCES, IN ORDER. `captions` is what preprocess_stream.py wrote for
    every chunk of the dataset (text_labels.jsonl); when it has this sample, its
    string is returned VERBATIM, so a panel says exactly what the dataset
    records and how many phrases it holds was decided once, at preprocessing.
    `phrases`/`vocab_emb` are the fallback that computes the same retrieval here
    -- the path every dataset used before that file existed, kept so older runs
    do not lose their captions. The two agree by construction; the fallback is
    the slower one, because it opens the chunk's .npz.

    Returns "" when there is nothing to say (no captions, no vocabulary, no text
    condition), so every caller can drop it into a string unconditionally.
    """
    try:
        if not (0 <= ds_idx < len(val_dataset.samples)):
            return ""
        npy_path, cond_path, _start, _lab, class_name = \
            val_dataset.samples[ds_idx]
        # The dataset's own answer, when it has one. Nothing is
        # recomputed here, so the words under a panel are the words
        # on disk, by construction rather than by coincidence.
        if captions:
            key = _chunk_key_of(val_dataset, cond_path)
            if key is not None and key in captions:
                return captions[key]
        if not phrases or vocab_emb is None or cond_path is None:
            return str(class_name)
        with np.load(str(cond_path)) as z:
            if "text" not in z:
                return str(class_name)
            vec = z["text"]
        from conditions import nearest_phrases
        near = nearest_phrases(vec, vocab_emb, phrases, k=k)
        if not near:
            return str(class_name)
        return (f"{class_name} · "
                + ", ".join(f"\"{p}\" ({c:+.2f})" for p, c in near))
    except Exception:
        return ""


def validation_panel_suffixes(val_dataset, sampling_cfg, frame_dims,
                              global_configs):
    """{panel index: block suffix} for the validation panels.

    THE POINT OF THIS FUNCTION IS THAT THERE IS ONLY ONE OF IT. The block name
    `validation_XX` is built in five places -- the metrics step, the cheap audio
    preview between metrics steps, the step-0 real references, the comparison
    images and the image card -- and TensorBoard groups by the text before the
    first '/'. A suffix computed in some of them and not the others would give
    one sample two half-filled blocks instead of one, which is precisely the
    failure audio_panel_tags spends a paragraph warning about. Every site calls
    THIS, with the inputs it already has, so they cannot disagree.

    The suffix names what the panel was conditioned on: the sample's CATEGORY
    (exact, from the dataset) and the nearest phrase to its stored CLAP vector
    with its cosine (a retrieval over a closed vocabulary -- CLAP has no
    decoder, so there is no true sentence to recover). Only ONE phrase: this
    goes in a block header, which stays readable only while it stays short.

    Empty for every panel when the text condition is off, so a run without it
    keeps exactly the block names it has always had.

    Cached per (dataset, config): it reads one .npz header per panel and is
    asked for the same answer at every metrics step.
    """
    if "text" not in (global_configs or {}) or val_dataset is None:
        return {}
    key = (id(val_dataset), id(sampling_cfg))
    cache = getattr(validation_panel_suffixes, "_cache", None)
    if cache is None:
        cache = validation_panel_suffixes._cache = {}
    if key in cache:
        return cache[key]

    # The SAME recipe the panels use to map an index to a validation sample:
    # position p of the metrics list describes val_dataset[indices[p]].
    total = len(val_dataset)
    n_metrics = int(getattr(sampling_cfg, "n_metrics_samples", 512) or 512)
    panel_pos = metrics_sample_positions(n_metrics,
                                         influence_set_size(sampling_cfg))
    indices = torch.linspace(0, total - 1, n_metrics).long().tolist()
    # `frame_dims or global_configs`, not `frame_dims`: a run conditioned only
    # on text/image still has an influence set and still numbers its panels off
    # metrics_sample_positions, so falling back here would make this header
    # describe a different validation sample from the one in the block.
    if not panel_pos or not (frame_dims or global_configs):
        n = max(1, min(influence_set_size(sampling_cfg), total))
        panel_pos = list(range(n))
        indices = torch.linspace(0, total - 1, n).long().tolist()

    # How many phrases a caption carries is decided ONCE, in the preprocessing
    # (--text_labels_n), and travels with the dataset. The vocabulary is loaded
    # only as a fallback, for datasets written before the sidecar existed.
    captions = load_text_captions(val_dataset.latent_root)
    phrases, vocab = ((None, None) if captions
                      else load_text_label_vocab(val_dataset.latent_root))
    out = {}
    for k, p in enumerate(panel_pos):
        idx = indices[p] if p < len(indices) else indices[-1]
        desc = describe_validation_sample(val_dataset, idx, captions,
                                          phrases, vocab)
        out[k] = f" [{desc}]" if desc else ""
    cache[key] = out
    return out


def _probe_dir(output_dir, step):
    d = os.path.join(output_dir, f"step_{step:07d}", "probe")
    os.makedirs(d, exist_ok=True)
    return d


# ======================
# METRICS EVALUATION (conditioned generation, FD-DAC + KL)
# ======================
@torch.no_grad()
def evaluate_and_log_metrics(
    model, normalizer, val_dataset, step, writer, device, output_dir,
    fd_dac_ref_stats, n_samples,
    sampling_cfg, conditioning_cfg, use_amp,
    frame_dims, global_configs,
    fidelity_evaluator=None, global_embedders=None, compute_uncond=True,
    prefix="EMA", metrics_seed=None, metrics_enabled=COND_METRICS,
    fad_embedder=None, fad_ref_stats=None, n_fad=0, fad_device="cuda",
    probe_sets=None, image_root=None,
):
    """
    Computes the full validation metric suite on a FIXED subset of the val set
    (deterministic linspace indices, so the curves are comparable across steps),
    in three independent axes:

      1. UNCONDITIONAL generation  -> Fd_dac_uncond / Kl_uncond
         Generated with NULL conditions (no CFG). This is the ONLY metric that
         is apples-to-apples comparable with the unconditional model: it answers
         "how well does this (conditioned-trained) model generate freely?".

      2. CONDITIONAL generation    -> Fd_dac_cond / Kl_cond
         Each sample generated from one specific validation condition (with CFG
         guidance). Distributional fidelity of the conditioned generations to
         the real data. NOT comparable with the unconditional model (the
         conditioning restricts the distribution).

      3. CONDITION INFLUENCE       -> Validation/Condition_influence (text panel)
         Paired delta: re-extract each condition from the conditioned AND the
         null generations, score adherence on both (f0 corr, chroma
         cosine, rhythm/energy correlation, text CLAP audio-text cosine), and
         report Δ = with-cond - null. Answers "how much does the condition pull
         the generation toward its target?". Consolidated into a single Markdown
         table (no separate scalar curves), built only from the conditions
         ACTIVE in the run, so it adapts automatically. `image` is scored
         through Wav2CLIP (audio embedded into CLIP's own space, the only way an
         audio-vs-image cosine exists); it falls back to an explicit "no
         audio->CLIP embedder" note when wav2clip is not installed, never to a
         fabricated zero.

    FD-DAC and KL (both directions, real||gen and gen||real) share the SAME real
    validation latent reference (fd_dac_ref_stats) in both the cond and uncond
    cases; the distributional metrics are latent-only (audio is decoded only for
    the influence re-extraction and the audio previews). `prefix`
    ("EMA" / "Model") tags the previews by the generating weights. Returns
    (fd_dac_cond, kl_cond_real_gen, kl_cond_gen_real) for the caller.
    """
    guidance = float(conditioning_cfg.guidance_scale)
    n_frames = val_dataset.n_frames
    total = len(val_dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()
    # ---- WHICH samples get the rich treatment (comparison plot + audio panel) ----
    # Resolved BEFORE the generation pass, because the REAL latent of those
    # samples has to be captured while it runs.
    frame_active = (fidelity_evaluator is not None and fidelity_evaluator.active)
    # WHICH global conditions can be SCORED this step: one that is active in
    # the run AND has an embedder able to place a generated waveform in the same
    # space as its stored condition. text -> CLAP's audio tower, image ->
    # Wav2CLIP. A global with no embedder is still USED to condition; it simply
    # gets no influence row, which is honest rather than a fabricated zero.
    # ---- validation conditioned on the DESCRIPTION, not on the chunk ------
    # With sampling.validation_text_from_caption the text slot of a validation
    # generation receives the CLAP TEXT embedding of that sample's description
    # (its class, plus the nearest phrases -- see --text_labels_n), instead of
    # the CLAP AUDIO embedding of the chunk it was extracted from. TRAINING is
    # untouched, and so is the validation LOSS: only the generations the metrics
    # and the panels are built from change, which is what makes the audio-text
    # similarity an audio-vs-TEXT number and the text influence a measure of how
    # much a written description moves the output.
    _cap_ids, _cap_emb = ({}, None)
    if (sampling_cfg is not None
            and bool(sampling_cfg.get("validation_text_from_caption", False))
            and "text" in (global_configs or {})):
        _cap_ids, _cap_emb = load_caption_conditions(val_dataset.latent_root)
        if _cap_emb is None:
            print("    [metrics] validation_text_from_caption is on but this "
                  "dataset carries no caption embeddings -- re-run the "
                  "preprocessing with --global text. Falling back to the "
                  "chunk's own CLAP vector.")
        else:
            print(f"    [metrics] validation text conditioned on the DESCRIPTION "
                  f"({_cap_emb.shape[0]} distinct caption(s))")

    def _text_vec_for(idx, text_emb):
        """The vector the text slot is GIVEN for validation sample `idx`.

        Returned to BOTH the conditioning and the scoring target, which is why
        one substitution here is the whole change: the similarity and the
        influence delta measure whatever conditioned the generation."""
        if _cap_emb is None:
            return text_emb
        try:
            key = _chunk_key_of(val_dataset, val_dataset.samples[idx][1])
            cid = _cap_ids.get(key)
            if cid is None:
                return text_emb
            return torch.from_numpy(_cap_emb[cid].copy())
        except Exception:
            return text_emb

    gsim_names = sorted(c for c in (global_configs or {})
                        if (global_embedders or {}).get(c) is not None)
    text_active  = "text" in gsim_names
    influence_active = frame_active or bool(gsim_names)

    n_val_save  = int(getattr(sampling_cfg, "n_val_save", 8) or 8)
    n_influence = influence_set_size(sampling_cfg)
    # The two COLLECTED audio groups -- "ground truth" (the recordings) and
    # "uncond generation" (the same model with no conditions) -- are listening
    # material, not diagnostics, so they have their own size: n_audio_samples.
    # They are a PREFIX of the influence set, so real_validation_03 is still the
    # recording that block validation_03 was conditioned from.
    n_audio = int(getattr(sampling_cfg, "n_audio_samples", 4) or 0)
    n_fid = min(n_influence, n_samples) if influence_active else 0
    # THE influence set: the same N validation samples are scored, plotted and
    # played. plot_ids IS fid_pos -- every scored sample owns a panel, so the
    # table, the Images window and the Audio window all describe one set.
    fid_pos = metrics_sample_positions(
        n_samples, n_influence if influence_active else 0)
    plot_ids = list(fid_pos) if frame_active else []
    # Block names for the validation panels, from the ONE shared source, so the
    # audio cards, the comparison images and the image card of a sample all land
    # in the same collapsible section.
    _sfx = validation_panel_suffixes(val_dataset, sampling_cfg, frame_dims,
                                     global_configs)
    n_log = min(2, n_samples)   # how many samples to log richly (audio/real)
    # fid_pos, not plot_ids: the REAL latent of every PANEL sample has to be
    # captured during the generation pass, and a global-only run has panels
    # without any plot_ids (no curve is re-extracted for text or image).
    log_ids = sorted(set(range(n_log)) | set(fid_pos))
    log_id_set = set(log_ids)
    n_keep = max(n_val_save, n_log)

    # ---- CONDITION SUBSETS (condition-combination influence) ----
    # Each subset is a full extra generation pass over n_samples, so the cost of
    # the metrics step is linear in how many are asked for. They are all scored
    # against the SAME null pass, which is what makes their deltas comparable.
    subset_specs = []
    if influence_active and frame_active:
        subset_specs = resolve_influence_subsets(
            sampling_cfg.get("influence_subsets", None), list(frame_dims or {}))

    ref_frames = (fd_dac_ref_stats["n_total"]
                  if fd_dac_ref_stats is not None else "n/a")
    print(f"\n  Compute metrics @ step {step}: {n_samples} generations "
          f"(cond guidance={guidance}"
          f"{', + uncond' if compute_uncond else ''}) "
          f"vs reference ({ref_frames} frames)...")

    # ---- generate n_samples latents, conditioned or unconditional ----
    # For the conditioned pass we also keep, for the first n_log samples, the
    # real latent (to decode the real audio) and the target condition.
    # The cross-attention context of a batch of panel samples. None on a model
    # without the sub-layer, so nothing is stacked or moved for a run that
    # cannot use it.
    _wants_ctx = bool(getattr(model, "text_cross_layers", None))

    def _stack_ctx(items):
        if not _wants_ctx or not items:
            return None
        return {"tokens": torch.stack([c["tokens"] for c in items]),
                "mask":   torch.stack([c["mask"]   for c in items])}

    def _generate(conditioned, subset=None):
        """`subset`: when given, only these frame conditions are handed to the
        model; the others are left out and the network zero-fills them, which is
        the same null the CFG dropout used at training time. The TARGETS kept
        for scoring stay the FULL set either way -- measuring a condition that
        was not given is exactly how its side effects show up."""
        lat_list = []
        targets = []        # paired frame conditions (cpu numpy); cond only
        real_frames = {}    # real latents of log_ids, keyed by generation index
        global_targets = [] # paired global conds (text/image emb); cond only
        # Dedicated, isolated RNG for the metric noise so FD/KL are comparable
        # across checkpoints (mirrors the uncond metrics seed). Re-seeded at the
        # start of EACH pass, so the cond and uncond generations start from the
        # SAME x0 stream (paired), and neither touches the global training RNG.
        gen_rng = None
        if metrics_seed is not None:
            gen_rng = torch.Generator(device=device)
            gen_rng.manual_seed(int(metrics_seed))
        for j, idx in enumerate(indices):
            (frames_real, frame_cond_real, _lab, text_emb, image_emb,
             text_ctx) = val_dataset[idx]
            text_emb = _text_vec_for(idx, text_emb)
            if conditioned:
                fc = {k: v.unsqueeze(0).to(device).float()
                      for k, v in frame_cond_real.items()
                      if subset is None or k in subset}
                gc = {}
                if "text" in global_configs:
                    gc["text"] = text_emb.unsqueeze(0).to(device)
                if "image" in global_configs:
                    gc["image"] = image_emb.unsqueeze(0).to(device)
                g = guidance
                targets.append({k: v.cpu().numpy()
                                for k, v in frame_cond_real.items()})
                global_targets.append({
                    "text":  text_emb.cpu().numpy()  if "text"  in global_configs else None,
                    "image": image_emb.cpu().numpy() if "image" in global_configs else None,
                })
                if j in log_id_set:
                    real_frames[j] = frames_real
                tctx = _stack_ctx([text_ctx])
            else:
                fc, gc, g, tctx = None, None, 1.0, None
            gen = euler_sample_cfg(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                use_amp=use_amp,
                frame_cond=fc, global_cond=gc, guidance=g,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=tctx,
            )
            lat_list.append(gen)
        return lat_list, targets, real_frames, global_targets

    def _generate_paired(spf):
        """Fused cond+uncond generation: each call to the paired sampler handles
        `spf` samples at once (batch 3*spf) and yields BOTH their conditioned and
        unconditional latents from the same x0. Only valid when CFG applies
        (guidance>1 and conditions present); the caller gates on that. Collects
        the same cond-side extras (targets / real_frames / global_targets) as the
        conditioned _generate."""
        cond_list, unc_list = [], []
        targets, real_frames, global_targets = [], {}, []
        gen_rng = None
        if metrics_seed is not None:
            gen_rng = torch.Generator(device=device)
            gen_rng.manual_seed(int(metrics_seed))
        for start in range(0, len(indices), spf):
            group = indices[start:start + spf]
            fcs, gcs, ctxs = [], [], []
            for j, idx in enumerate(group, start=start):
                (frames_real, frame_cond_real, _lab, text_emb, image_emb,
                 text_ctx) = val_dataset[idx]
                text_emb = _text_vec_for(idx, text_emb)
                fcs.append(frame_cond_real)
                gcs.append((text_emb, image_emb))
                ctxs.append(text_ctx)
                targets.append({k: v.cpu().numpy() for k, v in frame_cond_real.items()})
                global_targets.append({
                    "text":  text_emb.cpu().numpy()  if "text"  in global_configs else None,
                    "image": image_emb.cpu().numpy() if "image" in global_configs else None,
                })
                if j in log_id_set:
                    real_frames[j] = frames_real
            # stack the group's conditions -> batch `len(group)`
            fc = {k: torch.stack([d[k] for d in fcs]).to(device).float()
                  for k in (fcs[0].keys() if fcs else [])}
            gc = {}
            if "text" in global_configs:
                gc["text"] = torch.stack([t for t, _ in gcs]).to(device)
            if "image" in global_configs:
                gc["image"] = torch.stack([i for _, i in gcs]).to(device)
            gen_c, gen_u = euler_sample_cfg_paired(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                use_amp=use_amp,
                frame_cond=fc, global_cond=gc, guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=_stack_ctx(ctxs),
            )
            cond_list.extend(gen_c)
            unc_list.extend(gen_u)
        return cond_list, targets, real_frames, global_targets, unc_list

    # ---- DAC decode helpers (model loaded once, reused) ----
    def _decode_one(frames, dac_model):
        return decode_frames_to_wav(frames, normalizer, dac_model)

    def _decode(lat_list, dac_model):
        return [_decode_one(f, dac_model) for f in lat_list]

    has_ref = fd_dac_ref_stats is not None

    # ===== CONDITIONAL generation =====
    # Distributional metrics (FD-DAC + KL both directions) are latent-only and
    # share the SAME real reference (fd_dac_ref_stats). The DAC decode below is
    # needed ONLY for conditioning fidelity (re-extract from audio) and for the
    # rich audio logging -- NOT for the distributional metrics.
    # `sampling.metrics_samples_per_forward` (spf) is the ONE knob for the metrics
    # generation, and it maps directly onto VRAM:
    #   0 -> do not fuse: reference serial path (lowest peak, slowest)
    #   1 -> fuse the 3 CFG branches of 1 sample   -> batch 3
    #   N -> fuse N samples                        -> batch 3N (fastest, highest peak)
    # The 3 is structural (conditioned / cfg-null / unconditional are all required
    # by the CFG math), so it is never a tunable. Fusing needs a CFG to fuse, so
    # the serial path also runs automatically when guidance <= 1, uncond is off,
    # or no condition is active (pure-unconditional run).
    any_cond_active = bool(frame_dims) or bool(global_configs)
    spf = int(sampling_cfg.get("metrics_samples_per_forward", 1))
    # NOT gated on compute_uncond. Under CFG the unconditional velocity is
    # computed at every step anyway (that IS the CFG math), so integrating it
    # into a null LATENT is essentially free -- and that null is the baseline the
    # condition-influence delta is measured against. Tying it to metrics_uncond,
    # as this line used to, meant switching off a distributional metric silently
    # emptied the delta column of the influence table.
    # metrics_uncond now decides only whether the uncond DISTRIBUTIONAL metrics
    # (Fd_dac_uncond / Kl_uncond / Fad_vggish_uncond) are computed and logged.
    paired = (spf >= 1 and (guidance > 1.0) and any_cond_active)
    _unc_pre = None
    if paired:
        cond_lat, cond_targets, real_frames, cond_globals, _unc_pre = _generate_paired(spf)
    else:
        cond_lat, cond_targets, real_frames, cond_globals = _generate(conditioned=True)
        if compute_uncond and not any_cond_active:
            # No conditions active at all (pure-unconditional run): the
            # "conditioned" pass above already IS an unconditional generation
            # (null inputs, no CFG) and starts from the SAME metrics seed as the
            # uncond pass would -- so the two are identical sample by sample.
            # Reuse them instead of generating the same thing twice. This is
            # exact, not an approximation.
            _unc_pre = cond_lat
    cond_stack = torch.stack(cond_lat)
    if has_ref:
        _m = compute_dac_metrics(cond_stack, fd_dac_ref_stats,
                                 enabled=metrics_enabled, device=device)
        fd_dac_cond = _m["fd_dac"]
        kl_cond = {"kl_real_gen": _m["kl_real_gen"], "kl_gen_real": _m["kl_gen_real"]}
    else:
        fd_dac_cond = None
        kl_cond = {"kl_real_gen": None, "kl_gen_real": None}
    del cond_stack            # free the big stack right after FD/KL (latent-only)
    if device == "cuda":
        torch.cuda.empty_cache()

    # ===== UNCONDITIONAL generation (comparable to the unconditional model) =====
    # The null generations serve two roles: the uncond distributional metrics,
    # AND the baseline for the condition-INFLUENCE measure (how much closer to
    # the target the conditioned generation gets vs the unconditioned one).
    # frame_active / text_active / influence_active: hoisted above.

    fd_dac_uncond = None
    kl_uncond = {"kl_real_gen": None, "kl_gen_real": None}
    # The null latents themselves are needed by the INFLUENCE baseline whatever
    # metrics_uncond says; what metrics_uncond gates is scoring them
    # distributionally (an extra FD/KL pass, and n_fad more DAC decodes below).
    unc_lat = _unc_pre if _unc_pre is not None else []
    # The comment above is the INTENT; this is what enforces it. The fused
    # sampler hands the null latents over for free, so the intent held whenever
    # it ran -- but the SERIAL path (metrics_samples_per_forward=0) produces
    # none, and generating them only `if compute_uncond` meant that spf=0, the
    # documented escape hatch for an OOM at the metrics step, silently emptied
    # the Δ column of the ENTIRE validation influence table (the probe rows
    # survived, since run_joint_probe makes its own null pass -- which is
    # exactly what makes the loss easy to miss). The baseline is generated
    # whenever something can be scored against it, whatever metrics_uncond says.
    need_null_for_influence = influence_active and any_cond_active
    if not unc_lat and (compute_uncond or need_null_for_influence):
        if not compute_uncond:
            print("    generating the null pass for the influence baseline "
                  "(serial sampler: it is not produced by the CFG math here)...")
        unc_lat = _generate(conditioned=False)[0]
    if compute_uncond and unc_lat:
        unc_stack = torch.stack(unc_lat)
        if has_ref:
            _mu = compute_dac_metrics(unc_stack, fd_dac_ref_stats,
                                      enabled=metrics_enabled, device=device)
            fd_dac_uncond = _mu["fd_dac"]
            kl_uncond = {"kl_real_gen": _mu["kl_real_gen"],
                         "kl_gen_real": _mu["kl_gen_real"]}
        del unc_stack         # free
        if device == "cuda":
            torch.cuda.empty_cache()

    # ===== AUDIO PART: STREAMED, memory-flat (this is the OOM fix) =====
    # FD-DAC/KL above are latent-only over ALL n_samples (cheap). The AUDIO part
    # (DAC decode + re-extraction via CREPE/librosa + CLAP) is the
    # RAM-hungry step. Instead of decoding ALL generations into a big list and
    # re-extracting from all of them at once (which OOM-kills the process at
    # large n_metrics_samples), we STREAM it: decode ONE generation -> re-extract
    # its descriptors -> ACCUMULATE the metric -> DISCARD the waveform. Peak RAM
    # is therefore independent of how many samples we score, so
    # n_influence_samples_valid can be raised freely -- the only cost of
    # raising it is TIME (the extractor
    # runs once per sample). We still HOLD the first n_keep decoded waveforms,
    # which are needed for the disk dump (n_val_save) and the TB previews (n_log).
    # n_val_save / n_influence / n_fid / n_keep were resolved above, before
    # the generation pass: the audio panels need to know WHICH samples they
    # describe in time to capture their real latent.

    dac_model = get_dac()    # load-once singleton (shared across the whole run)

    # ---- how many samples the SIMILARITY SCALARS are averaged over ----
    # Their own knob, separate from n_metrics_samples and from
    # n_influence_samples_valid, because they answer to different constraints:
    # FD-DAC/KL estimate a covariance and want hundreds of LATENTS (cheap); the
    # influence set is read panel by panel and is small; these two are the mean
    # of one scalar per sample, which converges fast BUT forces a DAC decode
    # plus an embedder forward for every sample it touches. 128 is close to 512
    # in precision at a quarter of the cost.
    n_sim = int(sampling_cfg.get("n_similarity_samples", 128) or 0)
    sim_pos = (metrics_sample_positions(n_samples, n_sim)
               if (n_sim > 0 and gsim_names) else [])

    def _stream_audio(lat_list, with_sim=False):
        """Decode generations ONE AT A TIME: re-extract frame-condition fidelity
        and CLAP audio<->text similarity on a UNIFORM subset of n_fid positions
        (accumulated, the waveform then discarded), and RETURN the first n_keep
        waveforms (held for the disk dump / TB previews). Peak memory is
        O(n_keep), not O(len).

        The fidelity positions are spread over the WHOLE list rather than being
        its first n_fid: the generation indices come from
        linspace(0, len(val)-1, n_samples), so a prefix of n_fid lands on a tiny
        head of the validation set (with n_samples=1024 over a 165-sample val,
        the first 64 generations cover only 11 DISTINCT conditions, each re-drawn
        with different noise). The mean would then describe that head, not the
        validation set. Spreading costs nothing: same number of decodes, same
        CREPE work -- only WHICH samples are scored changes.
        """
        n_lat = len(lat_list)
        # fid_pos was resolved once, before the generation pass, from n_samples;
        # both passes score identically-sized lists, so the positions match and
        # pair_influence can compare sample by sample.
        pos = [p for p in fid_pos if p < n_lat]
        fid_set = set(pos)
        # The panel samples are KEPT even when they fall outside the first
        # n_keep: their waveform is what the audio panel plays next to the image
        # of the same sample. Cost: at most N extra waveforms in RAM.
        keep_set = set(range(min(n_keep, n_lat))) | {p for p in fid_pos if p < n_lat}
        # The similarity scalars are averaged over their OWN, larger set. Those
        # positions are decoded and embedded but NOT re-extracted: running CREPE
        # over 128 samples to feed a cosine would be the expensive half of the
        # metrics step for a number that does not use it.
        sim_set = {p for p in sim_pos if p < n_lat} if with_sim else set()
        if frame_active:
            fidelity_evaluator.reset()
            # Retain the raw curves of the panel positions, for the TensorBoard
            # comparison plots. plot_ids is deterministic, so these are the SAME
            # validation samples at every metrics step and the plots can be read
            # as a time series.
            fidelity_evaluator.keep_contours_for(plot_ids)
        gsims, kept = {c: {} for c in gsim_names}, {}
        for i in sorted(fid_set | keep_set | sim_set):  # decode each index ONCE
            wav = _decode_one(lat_list[i], dac_model)  # decode ONE
            if i in fid_set or i in sim_set:
                wn = wav.numpy()
                if frame_active and i in fid_set:
                    # sample_id=i: the conditioned and the null pass score the
                    # SAME positions of the same list, so tagging with i is what
                    # lets pair_influence compare them sample by sample instead
                    # of mean against mean.
                    fidelity_evaluator.add_sample(
                        wn, DAC_SAMPLE_RATE, n_frames, cond_targets[i],
                        sample_id=i)
                # Every scorable global, the same way: embed the generation
                # into that condition's space and take the cosine with the
                # condition it was given. One decode already paid for; this
                # adds one embedder forward per condition.
                for c in gsim_names:
                    t = cond_globals[i].get(c) if i < len(cond_globals) else None
                    if t is None:
                        continue
                    try:
                        emb = global_embedders[c].embed(wn, DAC_SAMPLE_RATE)
                        gsims[c][i] = float(np.dot(emb, np.asarray(t).reshape(-1)))
                    except Exception as _e:
                        if not gsims[c]:
                            print(f"    [metrics] {c} similarity unavailable: "
                                  f"{type(_e).__name__}: {_e}")
            if i in keep_set:
                kept[i] = wav              # keyed by index: the kept set is not
                                           # a prefix any more
            # else: wav is dropped here -> RAM stays flat regardless of n_fid
        # PER-SAMPLE values, not means: the two passes are averaged only AFTER
        # being intersected (pair_influence). Coverage travels with them --
        # reporting a score without saying how many samples reached it is how
        # failures disappear from the numbers.
        per  = fidelity_evaluator.per_sample() if frame_active else {}
        cov  = fidelity_evaluator.coverage() if frame_active else {}
        # ALL conditions, not just f0: the comparison images are drawn for
        # every active condition, so the curves of every one have to come back.
        cont = fidelity_evaluator.contours() if frame_active else {}
        return per, gsims, kept, len(pos), cov, cont

    if influence_active:
        print(f"    measuring condition influence on "
              f"{min(n_fid, len(cond_lat))} generations "
              f"(uniformly spread, streamed, memory-flat)...")
    # with_sim=True ONLY here: the similarity CURVES are averaged over
    # sampling.n_similarity_samples generations, which is a bigger set than the
    # influence one and lives on the conditioned pass alone. Without this flag
    # sim_set stayed empty and the curves were averaged over whatever positions
    # the two sets happened to share -- 4 samples out of 128 with the default
    # config, and never more than n_influence_samples_valid. The null pass is NOT
    # asked for them: it would double the decode+embed cost to widen a delta
    # that is deliberately read on the influence set (see below).
    ps_cond, gsim_cond, cond_wavs, n_fid_used, cov_cond, cont_cond = \
        _stream_audio(cond_lat, with_sim=True)
    ps_null, gsim_null, unc_wavs = {}, {}, {}
    have_null = False
    if unc_lat:
        if not any_cond_active:
            # Pure-unconditional run: unc_lat IS cond_lat (reused above), so the
            # decoded waveforms are identical -- reuse them instead of running the
            # DAC decoder (CPU-bound) a second time over the same latents.
            unc_wavs = cond_wavs
        else:
            ps_null, gsim_null, unc_wavs, _, _, _ = _stream_audio(unc_lat)
            have_null = True

    # ===== PER-SUBSET GENERATIONS (condition-combination influence) =====
    # One extra generation pass per subset, each scored against the SAME null
    # pass computed above -- a shared baseline is what lets the rows of the
    # matrix be compared with one another. The "all" subset is not regenerated:
    # the conditioned pass above already IS it, bit for bit.
    # Latents are dropped as soon as a subset has been scored, so peak memory
    # does not grow with the number of subsets (only the time does).
    # (label, per_sample frame metrics, coverage, per_sample GLOBAL similarities).
    # The globals are carried too: every subset pass already embeds its
    # generations into CLAP/CLIP space (that costs nothing extra -- the waveform
    # is decoded anyway for the frame re-extraction), and dropping that dict, as
    # this used to, is what made the text/image rows disappear from the panel
    # the moment influence_subsets was switched on.
    subset_entries = []
    subset_wavs = {}             # label -> {sample id: waveform} for the panels
    if subset_specs and have_null:
        _full = tuple(frame_dims or {})
        for _lab, _names in subset_specs:
            if _names == _full:
                subset_entries.append((_lab, ps_cond, cov_cond, gsim_cond))
                subset_wavs[_lab] = cond_wavs
                continue
            print(f"    subset '{_lab}' [{'+'.join(_names)}]: "
                  f"{n_samples} generations...")
            _slat, _st, _srf, _sg = _generate(conditioned=True,
                                              subset=set(_names))
            _sps, _scl, _swav, _sn, _scov, _scont = _stream_audio(_slat)
            subset_entries.append((_lab, _sps, _scov, _scl))
            subset_wavs[_lab] = _swav
            del _slat
            if device == "cuda":
                torch.cuda.empty_cache()
    elif subset_specs:
        print("    [subsets] skipped: they are deltas against the null pass, "
              "and no null pass was generated (sampling.metrics_uncond=false "
              "or guidance <= 1).")

    # ===== FAD (VGGish) on the DECODED generations =====
    # Latent-only metrics (FD-DAC / KL) score the DAC latent space; the FAD
    # scores the AUDIO, through an embedder trained on real recordings, which is
    # what the controllable-music literature reports. It therefore costs a DAC
    # decode + a VGGish forward per sample, on top of everything above -- that is
    # what sampling.n_fad_samples bounds. The statistics are accumulated as
    # running sums (compute_audio_mu_sigma), so the peak memory does not grow
    # with the sample count: raising it costs TIME, not RAM.
    fad_cond = fad_uncond = None
    if fad_embedder is not None and fad_ref_stats is not None and cond_lat:
        n_fad_use = min(int(n_fad or 0), len(cond_lat))
        if n_fad_use > 0:
            fad_pos = sorted(set(
                torch.linspace(0, len(cond_lat) - 1, n_fad_use)
                .round().long().tolist()))

            def _fad_clips(lat_list):
                for i in fad_pos:
                    yield (_decode_one(lat_list[i], dac_model).view(1, 1, -1),
                           DAC_SAMPLE_RATE)

            print(f"    FAD-VGGish on {len(fad_pos)} generations "
                  f"(decode + embed, streamed)...")
            _mu, _sig, _nv = compute_audio_mu_sigma(
                _fad_clips(cond_lat), len(fad_pos), fad_embedder,
                device=fad_device, desc="FAD cond")
            fad_cond = compute_fad(_mu, _sig, fad_ref_stats, device=fad_device)
            print(f"    FAD-VGGish cond: {len(fad_pos)} clips -> {_nv} embedding "
                  f"vectors (128-D)")
            if compute_uncond and unc_lat:
                if not any_cond_active:
                    # pure-unconditional run: the two lists ARE the same latents
                    fad_uncond = fad_cond
                else:
                    _mu, _sig, _ = compute_audio_mu_sigma(
                        _fad_clips(unc_lat), len(fad_pos), fad_embedder,
                        device=fad_device, desc="FAD uncond")
                    fad_uncond = compute_fad(_mu, _sig, fad_ref_stats,
                                             device=fad_device)
            del _mu, _sig
            if device == "cuda":
                torch.cuda.empty_cache()

    del cond_lat, unc_lat    # latents no longer needed
    if device == "cuda":
        torch.cuda.empty_cache()

    # ===== CONDITION INFLUENCE (delta: with-cond vs null, PAIRED) =====
    # influence[cond_name][metric] = {"cond":.., "null":.., "delta":..}.
    # delta>0 means the condition pulled the generation toward its target. Built
    # only from the conditions ACTIVE in this run (registry-driven).
    #
    # All three columns are averaged over the SAME samples: those where the
    # re-extraction produced a finite value on BOTH the conditioned and the null
    # generation. Averaging each pass over "whatever survived in that pass" and
    # subtracting would compare two means computed on two different sample sets,
    # so a moving delta could come entirely from moving denominators. The
    # coverage column reports the paired count and how many samples the pairing
    # had to drop.
    from condition_metrics import pair_influence, pair_scalar
    influence, cov_paired = {}, {}
    if frame_active:
        influence, cov_paired = pair_influence(
            ps_cond, ps_null, coverage_cond=cov_cond, have_null=have_null)

    # ---- the GLOBAL conditions' rows: one paired scalar each ----
    # text  -> cosine in CLAP's space (its audio tower embeds the generation,
    #          the stored condition is a CLAP vector of the source chunk)
    # image -> cosine in CLIP's space (Wav2CLIP embeds the generation, the
    #          stored condition is the CLIP vector of the picture)
    # Both are the SAME shape of number as an f0 correlation: with-cond, null,
    # and the delta between them on the same samples -- so they sit in the same
    # table and are read the same way.
    _gmetric = {"text": "clap_sim", "image": "clip_sim"}
    # The INFLUENCE SET, and only it. The conditioned pass now also embeds the
    # (larger) similarity set for the scalar curves, but the null pass does not,
    # so restricting here is what keeps `attempted` describing the samples this
    # row is actually about instead of reporting a hundred phantom unpaired ones.
    _fid_set = set(fid_pos)

    def _global_rows(gsims, into_inf, into_cov):
        """Add one paired row per scorable global condition to (influence,
        coverage). Used for the reference pass AND for every subset, so a
        subset's table carries text/image exactly like the frame conditions."""
        for c in gsim_names:
            _gc = {i: v for i, v in (gsims or {}).get(c, {}).items()
                   if i in _fid_set}
            if not _gc:
                continue
            _cm, _nm, _dm, _npair = pair_scalar(
                _gc, gsim_null.get(c, {}), have_null=have_null)
            key = _gmetric.get(c, "sim")
            into_inf[c] = {key: {"cond": _cm, "null": _nm, "delta": _dm}}
            into_cov[f"{c}/{key}"] = {
                "valid": _npair,
                "attempted": len(_gc),
                "unpaired": (len(set(_gc) ^ set(gsim_null.get(c, {})))
                             if have_null else 0),
            }

    _global_rows(gsim_cond, influence, cov_paired)

    # Each subset is paired against the SAME null pass as the reference above,
    # so every row of the matrix is a delta over one shared baseline and the
    # rows can be read against each other.
    subset_tables = []
    for _lab, _sps, _scov, _sgsim in subset_entries:
        _sinf, _scovp = pair_influence(_sps, ps_null, coverage_cond=_scov,
                                       have_null=have_null)
        _global_rows(_sgsim, _sinf, _scovp)
        subset_tables.append((_lab, _sinf, _scovp))

    # A global that is CONDITIONING the run but cannot be scored (no embedder
    # installed) keeps a row saying exactly that, so its absence from the table
    # is never mistaken for an influence of zero.
    for c in (global_configs or {}):
        if c in influence or c in gsim_names:
            continue
        influence[c] = {_gmetric.get(c, "sim"): {
            "cond": None, "null": None, "delta": None,
            "note": ("no audio->CLIP embedder: pip install wav2clip"
                     if c == "image" else "no embedder for this condition"),
        }}

    # ===== COMPARISON PLOTS, ONE PER ACTIVE CONDITION (IMAGES panel) =====
    # Target vs the same quantity re-extracted from the generation it
    # conditioned. The re-extraction ALREADY happened inside _stream_audio (it
    # is what produces the table's rows); the evaluator was merely asked to keep
    # the curves for these few samples instead of only the scalar they collapse
    # into, so these plots cost NO extra generation and no extra extractor pass.
    #
    # Every active condition gets the same treatment -- f0_valid_vs_gen_XX,
    # energy_valid_vs_gen_XX, chroma_valid_vs_gen_XX, rhythm_valid_vs_gen_XX --
    # and each has a matching <cond>_probe_vs_gen_XX from the probe, so an
    # ablation run on any single condition is read exactly the way the f0 one is.
    if cont_cond and plot_ids:
        from probe_conditions import plot_condition_comparison
        for _cname in sorted(frame_dims or {}):
            _cc = cont_cond.get(_cname, {})
            if not _cc:
                continue
            # The score in the title is the condition's first metric, whatever
            # it is named (corr / cosine / beat_corr / overall_accuracy /
            # chroma_accuracy), and its name goes with it for the f0 title.
            _mk = sorted(k for k in ps_cond if k.startswith(f"{_cname}/"))
            _corr = ps_cond.get(_mk[0], {}) if _mk else {}
            _mname = _mk[0].partition("/")[2] if _mk else None
            for _j, _sid in enumerate(plot_ids):
                if _sid not in _cc:
                    continue
                _tgt, _gen = _cc[_sid]
                writer.add_image(
                    f"validation_{_j:02d}{_sfx.get(_j, '')}"
                    f"/{_cname}_target_vs_gen",
                    plot_condition_comparison(
                        _cname, _tgt, _gen, kind="valid",
                        label=f"validation sample #{_sid}",
                        step=step, prefix=prefix, guidance=guidance,
                        score=_corr.get(_sid), score_name=_mname),
                    global_step=step)

    # ---- the IMAGE condition of the same validation panels ----
    # Into the block that already exists, beside the frame-condition plots of
    # the same sample -- not a new family. An image has no "target vs
    # re-extracted" pair to draw (nothing re-extracts a picture from audio), so
    # the card IS the picture the generation was conditioned on.
    # The file has to be re-opened from the raw image folder: the dataset holds
    # only embeddings, and an embedding cannot be turned back into a picture.
    # Without a readable image_root the card is skipped and the rest of the
    # panel is unaffected.
    if "image" in (global_configs or {}) and plot_ids and image_root:
        from PIL import Image as _PILImage
        _shown, _failed = 0, None
        for _j, _sid in enumerate(plot_ids):
            _ref = val_dataset.image_file_for(indices[_sid]) \
                if _sid < len(indices) else None
            if _ref is None:
                continue
            _cls, _fname = _ref
            try:
                _p = Path(image_root) / _cls / _fname
                _im = np.asarray(_PILImage.open(_p).convert("RGB"))
                # Once per board, at step 0: panel _j is the same validation
                # sample at every metrics step, so this is the same file every
                # time. Guarded AFTER the open, so a picture that failed to load
                # is retried at the next step instead of being written off.
                _tag = (f"validation_{_j:02d}{_sfx.get(_j, '')}"
                        f"/image_condition")
                if fixed_card_pending(writer, _tag):
                    writer.add_image(_tag, _im.transpose(2, 0, 1),
                                     global_step=0)
                _shown += 1
            except Exception as _e:
                _failed = f"{_cls}/{_fname}: {type(_e).__name__}: {_e}"
        if _failed is not None and _shown == 0:
            print(f"    [metrics] image cards unavailable ({_failed}); "
                  f"is paths.image_root still pointing at the right folder?")

    # ===== OUT-OF-THE-BOX JOINT PROBE (all active conditions at once) =====
    # Runs AFTER the validation influence dicts have been read out of the
    # evaluator (per_sample/coverage/contours all return copies), because the
    # probe resets the same evaluator to score its own generations.
    #
    # ONE call, not one per condition: the panel drives every active condition
    # together, which is the shape a model trained with p_drop_each_frame = 0.0
    # actually saw. See run_joint_probe for why the stimuli are combined by
    # index and deliberately not mutually aligned.
    probe_influence, probe_cov = {}, {}
    # Runs when ANYTHING active can be scored on it: a frame condition through
    # the fidelity evaluator, or a global one through its embedder. Gating this
    # on frame_active alone -- as it did -- silently skipped the whole probe in
    # a run conditioned only on image and/or text, which is exactly the run the
    # global conditions were added for.
    if probe_sets and (frame_active or gsim_names):
        # Collected separately as well: the matrix branch below renders from
        # `subset_tables`, which never looks at `influence`, so probe rows added
        # only there would vanish whenever subsets and probes are both on.
        _paired = 'paired' if guidance > 1.0 else 'unpaired'
        # EVERY condition the probe actually drives -- frame and global alike.
        # Listing only the frame ones (as this did) made the log claim a probe
        # over ['f0'] while it was also driving the image and text banks.
        _pnames = [c for c in list(frame_dims or {}) + list(global_configs or {})
                   if c in probe_sets]
        _npanel = min([len(probe_sets[c]) for c in _pnames], default=0)
        if _npanel:
            print(f"    joint probe: {_npanel} panels over {_pnames} "
                  f"({_paired})...")
            try:
                probe_influence, probe_cov = run_joint_probe(
                    probe_sets,
                    model=model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer, device=device,
                    output_dir=output_dir, use_amp=use_amp,
                    sampling_cfg=sampling_cfg,
                    guidance=guidance, frame_dims=frame_dims,
                    global_configs=global_configs,
                    fidelity_evaluator=fidelity_evaluator, dac_model=dac_model,
                    prefix=prefix,
                    n_plot=influence_set_size(sampling_cfg, "probe"),
                    n_audio=n_audio,
                    metrics_seed=metrics_seed,
                    global_embedders=global_embedders,
                )
                influence.update(probe_influence)
                cov_paired.update(probe_cov)
            except Exception as _e:
                # The probe is a diagnostic bolted onto the metrics step; a
                # failure (a missing probe wav, an OOM on its extra generations)
                # must not take down a training run that is otherwise fine.
                # Reported, not swallowed silently -- a probe that stops
                # appearing without a reason in the log is worse than no probe.
                print(f"    [probe] SKIPPED at step {step}: "
                      f"{type(_e).__name__}: {_e}")
                probe_influence, probe_cov = {}, {}
            if device == "cuda":
                torch.cuda.empty_cache()

    # ===== LOG SCALARS (distributional quality only; influence -> panel) =====
    # The tag scheme follows the RUN MODE, so each dashboard matches its project:
    #   * conditioned run  -> this project's two-axis scheme (Fd_dac_cond vs
    #     Fd_dac_uncond, Kl_cond/* vs Kl_uncond/*): the comparison is the point.
    #   * pure-unconditional run (no conditions active) -> there is only ONE
    #     distribution to score (cond and uncond generations are literally the
    #     same samples), so log a SINGLE axis under the EXACT tags of the
    #     unconditional project: Fd_dac / Kl_real_gen / Kl_gen_real.
    if any_cond_active:
        if fd_dac_cond is not None:
            writer.add_scalar("Validation/Metrics/Fd_dac_cond", fd_dac_cond, step)
        if kl_cond["kl_real_gen"] is not None:
            writer.add_scalar("Validation/Metrics/Kl_cond/real_gen",
                              kl_cond["kl_real_gen"], step)
            writer.add_scalar("Validation/Metrics/Kl_cond/gen_real",
                              kl_cond["kl_gen_real"], step)
        if fd_dac_uncond is not None:
            writer.add_scalar("Validation/Metrics/Fd_dac_uncond", fd_dac_uncond, step)
        if kl_uncond["kl_real_gen"] is not None:
            writer.add_scalar("Validation/Metrics/Kl_uncond/real_gen",
                              kl_uncond["kl_real_gen"], step)
            writer.add_scalar("Validation/Metrics/Kl_uncond/gen_real",
                              kl_uncond["kl_gen_real"], step)
        if fad_cond is not None:
            writer.add_scalar("Validation/Metrics/Fad_vggish_cond", fad_cond, step)
        if fad_uncond is not None:
            writer.add_scalar("Validation/Metrics/Fad_vggish_uncond", fad_uncond, step)
    else:
        # unconditional run: fd_dac_cond/kl_cond ARE the unconditional numbers
        if fd_dac_cond is not None:
            writer.add_scalar("Validation/Metrics/Fd_dac", fd_dac_cond, step)
        if kl_cond["kl_real_gen"] is not None:
            writer.add_scalar("Validation/Metrics/Kl_real_gen",
                              kl_cond["kl_real_gen"], step)
            writer.add_scalar("Validation/Metrics/Kl_gen_real",
                              kl_cond["kl_gen_real"], step)
        if fad_cond is not None:
            writer.add_scalar("Validation/Metrics/Fad_vggish", fad_cond, step)

    # ===== GLOBAL-CONDITION SIMILARITY (scalar curves, beside FD/KL/FAD) =====
    # How close the generation lands to the global condition it was given:
    #   Audio_text_similarity  -- cosine in CLAP's space
    #   Audio_image_similarity -- cosine in CLIP's space, via Wav2CLIP
    # These are SCALARS, not panel rows, because they are the two numbers you
    # watch as a curve across training the way you watch FD-DAC -- unlike the
    # influence table, which is a paired delta read at a step. They are averaged
    # over sampling.n_similarity_samples generations (their own knob; see where
    # sim_pos is built) rather than over the small influence set, so the curve
    # is smooth enough to show a trend.
    # A condition with no embedder logs nothing at all: an absent curve says
    # "not measured", a curve pinned at zero would say "no similarity".
    _scalar_name = {"text": "Audio_text_similarity",
                    "image": "Audio_image_similarity"}
    for _c in gsim_names:
        _vals = [v for i, v in gsim_cond.get(_c, {}).items() if i in set(sim_pos)]
        if not _vals:
            continue
        writer.add_scalar(f"Validation/Metrics/{_scalar_name.get(_c, _c)}",
                          float(np.mean(_vals)), step)
        print(f"  [{_c}]    similarity: {float(np.mean(_vals)):.4f} "
              f"over {len(_vals)} samples")

    # ===== CONDITION-INFLUENCE PANEL (consolidated text table) =====
    # All per-condition adherence/influence lives HERE now, as a single table,
    # NOT as separate scalar curves. TensorBoard keeps a per-step history of the
    # text, so the step slider walks the panel across training.
    # The table holds the validation rows and the probe rows side by side, and
    # the probe ones already say `<cond>_probe`. So the validation ones say
    # `<cond>_validation` instead of the bare condition name: they ARE the
    # validation panels of the Images and Audio windows (metrics_sample_positions
    # -- same samples), and a reader should not have to know that "no suffix"
    # meant validation. Display only: the dicts the rest of the step reads keep
    # the plain name, which is what the subset matrix matches its labels against.
    def _row_label(name):
        return name if name in probe_influence else f"{name}_validation"

    _inf_named = {_row_label(k): v for k, v in influence.items()}
    _cov_named = {}
    for _k, _v in (cov_paired or {}).items():
        # "<cond>/<metric>" -- the condition name never contains a '/'.
        _c, _, _m = _k.partition("/")
        _cov_named[f"{_row_label(_c)}/{_m}"] = _v

    if influence:
        from condition_metrics import (format_influence_panel,
                                       format_influence_matrix,
                                       format_influence_legend)
        # the influence is measured on n_fid_used generations, NOT on the
        # n_metrics_samples used for FD/KL: reporting the latter would claim
        # a sample size that was never used for these numbers.
        if subset_tables:
            # Subsets requested: the panel becomes the delta MATRIX (one row per
            # combination) with the detailed tables underneath. The single-table
            # form below is what a run with no subsets keeps.
            panel_md = format_influence_matrix(
                subset_tables, step=step, prefix=prefix,
                guidance=guidance, n_samples=n_fid_used,
                extra=probe_influence, extra_coverage=probe_cov,
                # The subsets vary the FRAME conditions only: text and image are
                # handed to the model in every row, so their cells are never a
                # side effect and must not be marked as one.
                always_given=set(gsim_names),
            )
        else:
            panel_md = format_influence_panel(
                _inf_named, step=step, prefix=prefix,
                guidance=guidance,
                n_samples=n_fid_used,
                coverage=_cov_named,
            )
        writer.add_text("Validation/Condition_influence", panel_md, step)
        # Log the explanatory legend ONCE, on its own tag. Pinned to step 0 so it
        # reads as a one-time preamble, not tied to a metrics step (TensorBoard
        # text always carries SOME step; 0 is the most neutral).
        if not getattr(evaluate_and_log_metrics, "_legend_logged", False):
            writer.add_text("Validation/Condition_influence_legend",
                            format_influence_legend(), 0)
            evaluate_and_log_metrics._legend_logged = True

    # ===== console summary =====
    def _f(x):
        return f"{x:.4f}" if x is not None else "n/a"
    print(f"  [cond]   FD-DAC: {_f(fd_dac_cond)} | "
          f"KL(real||gen): {_f(kl_cond['kl_real_gen'])} | "
          f"KL(gen||real): {_f(kl_cond['kl_gen_real'])}"
          + (f" | FAD-VGGish: {_f(fad_cond)}" if fad_cond is not None else ""))
    if compute_uncond:
        print(f"  [uncond] FD-DAC: {_f(fd_dac_uncond)} | "
              f"KL(real||gen): {_f(kl_uncond['kl_real_gen'])} | "
              f"KL(gen||real): {_f(kl_uncond['kl_gen_real'])}"
              + (f" | FAD-VGGish: {_f(fad_uncond)}" if fad_uncond is not None else ""))
    if influence:
        parts = []
        for cname, metrics in _inf_named.items():
            for m, vals in metrics.items():
                d = vals.get("delta")
                parts.append(f"{cname}/{m} delta={_f(d)}")
        if parts:
            print("  [influence] " + " | ".join(parts))

    # ===== SAVE VALIDATION ARTIFACTS TO DISK (one dir per step, one sub-dir per generation) =====
    # output_dir/step_{step}/generation_{i}/ contains, for generation i:
    #   conditions.npz   - the EXACT input conditions used (f0, energy, ...)
    #   cond_{name}.wav  - AUDIBLE rendering of each condition (f0 as a sine
    #                      contour, energy as an amplitude-modulated tone) so one
    #                      can hear how it maps into the conditioned audio
    #   cond.wav         - the conditioned generation
    #   uncond.wav       - the unconditioned (null) generation, same index
    #   real.wav         - the reference audio (panel samples only)
    # How many generations are dumped is sampling.n_val_save; the samples that
    # own a TensorBoard panel are always dumped too, even when they fall outside
    # it, so what you hear on TB has a file on disk next to it.
    def _to_np(wav):
        return wav.numpy() if wav.dim() == 1 else wav.squeeze().numpy()

    from condition_metrics import sonify_condition

    # cond_wavs is keyed by GENERATION INDEX (not a prefix any more): it holds
    # the first n_keep decoded waveforms plus the panel samples.
    save_ids = sorted({i for i in cond_wavs if i < n_val_save}
                      | {i for i in plot_ids if i in cond_wavs})
    step_dir = os.path.join(output_dir, f"step_{step:07d}")

    def _gen_dir(i):
        d = os.path.join(step_dir, f"generation_{i:03d}")
        os.makedirs(d, exist_ok=True)
        return d

    for i in save_ids:
        gdir = _gen_dir(i)
        if not any_cond_active:
            # Unconditional run: one generation per dir, nothing to sonify and no
            # cond/uncond pair to compare (they would be the same waveform).
            sf.write(os.path.join(gdir, "generated.wav"),
                     _to_np(cond_wavs[i]), DAC_SAMPLE_RATE)
            continue
        sf.write(os.path.join(gdir, "cond.wav"),
                 _to_np(cond_wavs[i]), DAC_SAMPLE_RATE)
        if i < len(cond_targets) and cond_targets[i]:
            np.savez(os.path.join(gdir, "conditions.npz"), **cond_targets[i])
            for cname, carr in cond_targets[i].items():
                son = sonify_condition(cname, carr, DAC_SAMPLE_RATE)
                if son is not None:
                    sf.write(os.path.join(gdir, f"cond_{cname}.wav"),
                             son, DAC_SAMPLE_RATE)
        if i in unc_wavs:
            sf.write(os.path.join(gdir, "uncond.wav"),
                     _to_np(unc_wavs[i]), DAC_SAMPLE_RATE)
    print(f"    saved {len(save_ids)} validation generations "
          + ("(per-generation dirs: cond+uncond+conditions+sonified) to "
             if any_cond_active else "(per-generation dirs: generated+real) to ")
          + step_dir)

    # ===== AUDIO PANELS on TensorBoard =====
    # ONE BLOCK per validation sample -- audio_panel_tags builds the names and
    # explains the layout -- holding one card per audio:
    #   1_f0_validation_XX           - the sonified f0 target
    #   2..N_<condition>_validation_XX - energy, chroma, ... alphabetical
    #   N+1_generation_validation_XX - the generation all of them produced
    # The null generation and the real recording of the same sample are NOT in
    # this block: they go to the collected groups, one card per sample --
    #   uncond generation/uncond_validation_XX
    #   ground truth/real_validation_XX
    # Validation/f0_valid_vs_gen_XX is the f0 picture of the same sample, same XX.
    # Everything is peak-normalized: these are meant to be A/B'd by ear, and a
    # sonified condition is written at a fixed low level, so without this the
    # comparison would be between loudnesses as much as between contents.
    def _log_audio(wav, tag):
        writer.add_audio(tag, norm_wav(wav), global_step=step,
                         sample_rate=DAC_SAMPLE_RATE)

    # The panel samples are the ones the images describe, enumerated in the SAME
    # order: block validation_03 must be the same validation sample in the Audio
    # tab and in the Images tab. Filtering the list here (as this used to do)
    # renumbered the audio whenever one generation was missing, so a block could
    # end up holding the audio of one sample and the curves of another.
    # fid_pos, NOT plot_ids: plot_ids is empty in a run conditioned only on
    # text/image (nothing re-extracts a curve from audio for those), and falling
    # back to "the first n_log generations" then numbered the audio panels off a
    # different list from the one validation_panel_suffixes and the cheap audio
    # preview use -- so block validation_01 held the audio of one validation
    # sample, the header of a second and, at the preview step, a third.
    # With NO condition active at all there is no influence set, and the
    # fallback (the first n_log generations) is the right and only answer.
    panel_ids = list(fid_pos) or [i for i in sorted(cond_wavs)][:n_log]

    for k, sid in enumerate(panel_ids):
        if sid not in cond_wavs:
            continue                     # index k stays tied to fid_pos
        has_targets = any_cond_active and sid < len(cond_targets)
        tags = audio_panel_tags("validation", k,
                                cond_targets[sid] if has_targets else (),
                                suffix=_sfx.get(k, ""),
                                conditioned=any_cond_active)

        # The stimulus cards, ONCE per board at step 0 (fixed_card_pending):
        # block XX is the same validation sample at every metrics step, so its
        # conditions are the same waveform every time and re-logging them only
        # gave a fixed card a step slider. The generation cards below keep
        # theirs -- those are what actually changes.
        if has_targets:
            for cname, carr in sorted(cond_targets[sid].items()):
                son = sonify_condition(cname, carr, DAC_SAMPLE_RATE)
                if son is not None and fixed_card_pending(
                        writer, tags["conditions"][cname]):
                    writer.add_audio(tags["conditions"][cname], norm_wav(son),
                                     global_step=0,
                                     sample_rate=DAC_SAMPLE_RATE)

        _log_audio(cond_wavs[sid], tags["generation"])

        # One extra card per condition SUBSET, in the same block, so the whole
        # combination ladder of one validation sample is played side by side.
        # "all" is skipped: the card above already is that generation.
        for _lab, _names in subset_specs:
            if _lab == "all":
                continue
            _w = subset_wavs.get(_lab, {}).get(sid)
            if _w is not None:
                # SAME suffix as the block above. Without it these cards fell
                # into a bare `validation_XX/` group while every other card of
                # the sample sat under `validation_XX [<label>]/`, so with the
                # text condition on, the combination ladder this card exists for
                # was split across two blocks in the dashboard.
                _log_audio(_w, subset_generation_tag(
                    "validation", k, _lab, suffix=_sfx.get(k, ""),
                    n_conditions=len(cond_targets[sid]) if has_targets else 0,
                    # The recording takes slot 1 of a conditioned validation
                    # block, so the whole ladder below it shifts down by one.
                    has_real=any_cond_active))

        # With no condition active "generation" already IS the uncond tag
        # (audio_panel_tags maps both keys to it), so the guard is what keeps
        # the same card from being written twice for one step.
        if any_cond_active and k < n_audio and sid in unc_wavs:
            _log_audio(unc_wavs[sid], tags["generation_no_cond"])

        # ---------- REAL ----------
        # The card goes to the collected "ground truth" group, bounded by
        # n_audio_samples like its uncond twin; the .wav on disk is written for
        # every dumped generation regardless (it is what makes a dumped
        # generation listenable against its source).
        # A RECORDING, so it is the most fixed card of all: written once at
        # step 0 -- usually by log_real_audio_samples at startup, and here for
        # whatever panel that pass did not reach. The decode still runs every
        # time: the .wav on disk below needs it.
        # NOT bounded by n_audio in a conditioned run: the card is slot 1 of
        # THIS block, so every panel owns one. The bound survives only for the
        # unconditioned case, where the recordings go to a collected group and
        # n_audio_samples is what says how many of them to hear.
        if sid in real_frames:
            real_wav = _decode_one(real_frames[sid], dac_model)
            if ((any_cond_active or k < n_audio)
                    and fixed_card_pending(writer, tags["real"])):
                writer.add_audio(tags["real"], norm_wav(real_wav),
                                 global_step=0, sample_rate=DAC_SAMPLE_RATE)
            sf.write(os.path.join(_gen_dir(sid), "real.wav"),
                     real_wav.numpy(), DAC_SAMPLE_RATE)

    # dac_model is the shared singleton (get_dac) -> do NOT delete it; just free
    # any CUDA scratch from the metrics step.
    if device == "cuda":
        torch.cuda.empty_cache()

    # Backward-compatible return: the conditioned FD-DAC + KL (both directions).
    return fd_dac_cond, kl_cond["kl_real_gen"], kl_cond["kl_gen_real"]


# ======================
# METRICS ADAPTER
# ======================
# metrics.py expects "slim" datasets with:
#   - __getitem__(idx) -> (frames, label_idx)             (2-tuple)
#   - .samples[idx]    -> (npy_path, start, label_idx)    (3-tuple)
#   - .n_frames, .idx_to_label
# The ConditionedAudioDataset returns 6-tuples in both cases (because of
# frame_conds, text_emb, image_emb, text_ctx). This adapter exposes the "slim view"
# without duplicating data or touching metrics.py.
class MetricsAdapter:
    """Slim view of ConditionedAudioDataset compatible with metrics.py."""

    def __init__(self, cond_dataset):
        self._ds = cond_dataset
        self.n_frames = cond_dataset.n_frames
        self.idx_to_label = cond_dataset.idx_to_label
        # samples slim: (npy_path, start, label_idx) - drop cond_path and class_name
        self.samples = [
            (npy, start, label)
            for (npy, _cond, start, label, _class) in cond_dataset.samples
        ]

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        frames, _frame_cond, label_idx, _text, _image, _ctx = self._ds[idx]
        return frames, label_idx


# ======================
# DATALOADER
# ======================
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


# ======================
# RNG STATE (resume continues the RNG streams instead of restarting from the seed)
# ======================
def _capture_rng_state(data_generator=None):
    """Snapshot of every RNG stream so a --resume continues them instead of
    restarting from the seed: python, numpy, torch CPU, torch CUDA, and the
    DataLoader shuffle generator. Mirrors training.py.

    NOT a bit-exact resume, and it cannot be: `infinite_loader` starts a FRESH
    epoch, so the sampler draws a NEW permutation from the restored generator
    while the interrupted run was mid-way through a permutation drawn earlier.
    Measured: the training loss diverges from the FIRST step after a resume,
    while weights, optimizer, scheduler and EMA are restored exactly. The
    resumed run is statistically equivalent, never byte-identical. Making it
    exact needs a step-indexed permutation or a stateful sampler
    (torchdata.StatefulDataLoader), neither of which is used here."""
    state = {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def _restore_rng_state(state, data_generator=None):
    """Restore the RNG snapshot saved by _capture_rng_state. Best-effort: old
    checkpoints (no rng_state) or a different GPU count fall back to the freshly
    seeded RNG with a warning instead of crashing. Mirrors training.py."""
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        if data_generator is not None and state.get("data_generator") is not None:
            data_generator.set_state(state["data_generator"])
        print("[SEED] RNG streams restored from checkpoint (x0, t, CFG dropout). "
              "Batch ORDER restarts on a fresh epoch, so a resumed run is "
              "statistically equivalent, not bit-identical.")
    except Exception as e:
        print(f"[SEED] WARNING: could not fully restore RNG state ({type(e).__name__}: "
              f"{e}); continuing with the freshly seeded RNG.")


# ======================
# CHECKPOINT HELPER
# ======================
def build_ckpt_data(model, ema, optimizer, scheduler, scaler, step,
                    val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                    frame_cond_dims, frame_cond_out_dims, global_configs,
                    data_generator=None, ema_ready=False):
    """
    Assemble the conditioned checkpoint dict. The full `config` is stored so a
    later --resume can rebuild the exact same model / conditioning / training
    setup without the user having to re-pass model.kind, the enabled conditions,
    batch sizes, etc. `model_kind` and the per-condition dims are also kept as
    top-level fields for backward compatibility with sampling_cond.py /
    test_cond.py (which read them directly from the checkpoint). `rng_state`
    stores every RNG stream so --resume continues them (x0, t, CFG dropout)
    rather than restarting from the seed -- see _capture_rng_state for why the
    batch order is the one thing that does NOT resume exactly. Mirrors
    training.py.
    """
    data = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "step":                 step,
        "val_loss":             val_loss,
        "best_val_loss":        best_val_loss,
        "model_kind":           cfg.model.kind,
        # Architecture parameter, top-level like model_kind for the same reason:
        # sampling_cond.py / test_cond.py rebuild the model from these fields,
        # not from `config`, and they must rebuild the SAME state_dict layout.
        # Absent in checkpoints written before this option existed -> those
        # readers default it to 0, which is what those runs actually were.
        "frame_reinject_every": int(cfg.model.get("frame_reinject_every", 0)),
        # Same contract for the cross-attention: it adds four tensors per
        # selected block plus the null token, so a reader that rebuilds the
        # model has to know the stride. The WIDTH is not stored -- it is read
        # off the weights themselves (network_cond.ckpt_text_ctx_dim), which
        # is the one source that cannot be out of date.
        "text_cross_every":     int(cfg.model.get("text_cross_every", 0)),
        "config":               OmegaConf.to_container(cfg, resolve=True),
        "label_map":            label_map,
        "frame_cond_dims":      frame_cond_dims,
        "frame_cond_out_dims":  frame_cond_out_dims,
        "global_configs":       global_configs,
        "n_frames":             n_frames,
        "run_name":             run_name,
        "validation_protocol":  VALIDATION_PROTOCOL,
        "rng_state":            _capture_rng_state(data_generator),
    }
    if cfg.training.use_ema and ema is not None:
        data["ema_state_dict"] = ema.state_dict()
        # REAL state, passed in by the caller: it is True only once the shadow
        # has actually been seeded from the live weights. Deriving it from
        # `step >= ema_start` would lie if the run died AT ema_start before the
        # seeding ran (last_step is set at the top of the iteration), producing a
        # checkpoint that claims a trained EMA while holding the random init.
        data["ema_ready"] = bool(ema_ready)
    return data


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    cfg, run_name = load_config()
    print(f"[RUN NAME] {run_name}")

    # Where the shared DAC decoder will live. Decided HERE, before anything can
    # call get_dac(), because the decoder is a load-once singleton and is never
    # moved afterwards. .get() keeps a config written before this option existed
    # on the historical behaviour (CPU). See set_dac_device for the measurements.
    set_dac_device(cfg.metrics.get("dac_device", "cpu"))

    # ======================
    # RUN DIRECTORY (self-contained) + CACHE DIRECTORY (shared)
    # ======================
    run_dir   = os.path.join(cfg.paths.runs_dir, run_name)
    ckpt_dir  = os.path.join(run_dir, "checkpoints")
    audio_dir = os.path.join(run_dir, "audio")
    cache_dir = cfg.paths.cache_dir
    os.makedirs(run_dir,   exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Paths derived from the above directories
    normalizer_path   = os.path.join(cache_dir, "normalizer.pt")
    fd_dac_cache_path = os.path.join(cache_dir, "fd_dac_ref_stats.pt")
    # The FAD reference depends on HOW it was built (real wavs vs DAC-decoded
    # val latents), so the mode is part of the file name: switching
    # metrics.fad_reference must not silently reuse the other one's statistics.
    _fad_ref_mode = str((cfg.get("metrics", None) or {}).get("fad_reference", "wav"))
    fad_cache_path = os.path.join(cache_dir, f"fad_vggish_ref_{_fad_ref_mode}.pt")

    # Cache safety (report #3): tie the shared normalizer / FD-DAC reference to
    # the dataset + duration + split they were computed on, so a stale cache from
    # a different preprocessing / split can never be silently reused.
    _n_frames_fp = frames_per_chunk(cfg.paths.dataset_root, cfg.model.duration_s)
    _validate_cache(cache_dir, _cache_fingerprint(cfg, _n_frames_fp),
                    guarded_files=[normalizer_path, fd_dac_cache_path,
                                   fad_cache_path])

    # Config's dump (with CLI override already applied) in the run dir
    config_dump_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(cfg, config_dump_path)
    print(f"[CONFIG DUMP] {config_dump_path}")
    print(f"[RUN DIR]     {run_dir}")
    print(f"[CACHE DIR]   {cache_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    # Global run seed: makes x0, the LogitNormal t-sampling, the per-sample CFG
    # dropout and the DataLoader shuffle reproducible. Set null in the YAML to
    # disable (free-running RNG). Mirrors training.py.
    run_seed = cfg.training.get("seed", None)
    data_generator = None
    if run_seed is not None:
        run_seed = int(run_seed)
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        data_generator = torch.Generator()      # for the train loader shuffle
        data_generator.manual_seed(run_seed)
        print(f"[SEED] Global training seed = {run_seed}")

    # Metric-generation seed: fixes the metric noise (x0) so FD/KL are comparable
    # across checkpoints, via a DEDICATED Generator inside the metrics eval that
    # never touches the global training RNG. null = free-running. Mirrors uncond.
    metrics_cfg = cfg.get("metrics", None)
    metrics_seed = (metrics_cfg.get("seed", None) if metrics_cfg is not None else None)
    validation_seed = int(cfg.data.get("validation_seed", 12345))
    validation_shuffle = bool(cfg.data.get("validation_shuffle", True))

    # Which distributional metrics to compute, mirroring the unconditional
    # project's `metrics.enabled` registry. A metric listed here is an EXPLICIT
    # request: asking for one this pipeline cannot produce is a HARD ERROR at
    # startup (not a silent skip), so you never train for hours believing a
    # metric is on when it is not.
    metrics_enabled = list(
        metrics_cfg.get("enabled", list(COND_METRICS))
        if metrics_cfg is not None else list(COND_METRICS))
    _unknown = [m for m in metrics_enabled if m not in COND_METRICS]
    if _unknown:
        raise SystemExit(
            f"[metrics] metrics.enabled contains {_unknown}, which the CONDITIONED "
            f"pipeline does not provide. Available: {list(COND_METRICS)}. "
            f"(fad_encodec needs the Encodec embedder and is not wired here; "
            f"fad_vggish is.)")
    print(f"[metrics] enabled: {metrics_enabled or '(none: distributional metrics off)'}")

    # Which metrics score the condition-influence rows: ours or mir_eval's
    # (metrics.influence_family, see the YAML). Checked HERE, at startup, so a
    # misspelt value stops the run before the references are built. .get()
    # keeps a config written before the option existed on ours.
    from condition_metrics import INFLUENCE_FAMILIES
    influence_family = str(
        metrics_cfg.get("influence_family", "influence_metrics")
        if metrics_cfg is not None else "influence_metrics")
    if influence_family not in INFLUENCE_FAMILIES:
        raise SystemExit(
            f"[metrics] metrics.influence_family='{influence_family}' is not a "
            f"valid choice. Use 'influence_metrics' (ours) or "
            f"'mir_influence_metrics' (mir_eval, for the conditions it covers).")
    # The value at which a chroma / chord pitch class counts as ON for the mir
    # rows (see the YAML). Same startup check; .get() -> 0.5 for older configs.
    mir_threshold = float(
        metrics_cfg.get("mir_threshold", 0.5)
        if metrics_cfg is not None else 0.5)
    if not 0.0 < mir_threshold <= 1.0:
        raise SystemExit(
            f"[metrics] metrics.mir_threshold={mir_threshold} is not in (0, 1]: "
            f"it is a fraction of the frame's loudest chroma class, and a "
            f"crema probability for chord.")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")

    # ======================
    # CONDITION REGISTRY
    # ======================
    # The active set of conditions (which frame extractors, which global
    # encoders, their dims) comes from CONDITION_CONFIG in conditions.py.
    # The class list is derived later from the split's label_to_idx (the dataset
    # is split-less on disk, so there is no train/ folder to probe here).

    # ------------------------------------------------------------------
    # Per-run selection of which FRAME conditions to activate, driven by
    # conditioning.enabled_frame in the YAML, with these rules:
    #   * null / absent  -> train with EXACTLY the frame conditions that have
    #                       actually been EXTRACTED to disk (the .npz contents),
    #                       intersected with the enabled pool in CONDITION_CONFIG.
    #   * explicit list  -> train with that subset. The .npz are NOT scanned at
    #                       startup: that the files hold what was asked of the
    #                       preprocessing is the preprocessing's job, and a
    #                       sample whose .npz lacks a requested condition is
    #                       refused by the dataset when it is loaded (RAISE
    #                       under strict_conditions -- no silent zero-fill).
    # Global conditions (text / image) are NOT stored in the .npz, so
    # enabled_global keeps its previous semantics (null = all enabled in config).
    # The dataset is split-less, so the scan is a single pass over ALL .npz.
    #
    # WHY THE EXPLICIT LIST IS NOT SCANNED (30 Sept 2026). The scan opens every
    # .npz, one per chunk, with the GPU already locked and idle, and IRCAM
    # terminates a job whose locked GPU does nothing for 15 minutes. On Lakh
    # (more than 757,000 chunks) XL_chord_lakh was killed twice inside it
    # before training a single step. With an explicit list it decides nothing:
    # it only repeated, up front, the check the dataset makes on every sample.
    # What is given up is WHEN a bad file is found -- at the sample that hits
    # it instead of at startup -- and, with strict_conditions=false, the
    # startup count of the files that will be zero-filled.
    # ------------------------------------------------------------------
    enabled_pool = {name for name, c in CONDITION_CONFIG["frame_level"].items()
                    if c.get("enabled", False)}

    enabled_f = cfg.conditioning.get("enabled_frame",  None)
    enabled_g = cfg.conditioning.get("enabled_global", None)
    # OmegaConf converts YAML null -> None, YAML list -> ListConfig.
    if enabled_f is not None:
        enabled_f = list(enabled_f)
    if enabled_g is not None:
        enabled_g = list(enabled_g)

    if enabled_f is None:
        cond_scan = _scan_frame_conditions(cfg.paths.condition_root)
    else:
        cond_scan = {"total": 0, "present": {}}     # explicit list: no scan
    # "available" = present in EVERY .npz (usable as a full condition).
    available_all = {n for n, c in cond_scan["present"].items()
                     if cond_scan["total"] > 0 and c == cond_scan["total"]}
    available = available_all & enabled_pool

    if enabled_f is None:
        # Default: train with whatever is present in ALL train .npz.
        enabled_f = sorted(available)
        print(f"[conditions] enabled_frame not set -> using the frame conditions "
              f"present in ALL train .npz: {enabled_f}")
        if not enabled_f:
            raise RuntimeError(
                f"No frame conditions found on disk under "
                f"{cfg.paths.condition_root}. Extract them first, e.g.:\n"
                f"  python extract_conditions.py <dataset_root> "
                f"--conditions f0 --device cuda")
    elif enabled_f:
        print(f"[conditions] enabled_frame={enabled_f} given explicitly -> the "
              f".npz are not scanned at startup; each sample is checked when it "
              f"is loaded (training.strict_conditions).")

    # ---- U4: STRICT FULL validation of the requested conditions ----
    # Every requested condition must be present in EVERY .npz; otherwise some
    # samples would be silently zero-filled and the model would train on NULL
    # conditions without any error. The dataset is split-less, so this is one
    # full scan over all .npz. strict=True (default) hard-fails.
    # Runs on the counts of the scan above, so only when enabled_frame was
    # null; for an explicit list the same rule is enforced per sample.
    strict_conditions = bool(cfg.training.get("strict_conditions", True))
    if enabled_f and cfg.paths.condition_root and cond_scan["total"] > 0:
        problems = []
        for name in enabled_f:
            cnt = int(cond_scan["present"].get(name, 0))
            miss = cond_scan["total"] - cnt
            status = "OK" if miss == 0 else f"MISSING in {miss}/{cond_scan['total']}"
            print(f"[conditions] '{name}': "
                  f"{cnt}/{cond_scan['total']} present  [{status}]")
            if miss > 0:
                problems.append((name, miss, cond_scan["total"]))
        if problems:
            lines = "\n".join(f"    - {nm}: missing in {mi}/{to} .npz"
                              for nm, mi, to in problems)
            msg = (f"[conditions] Some requested conditions are NOT present in every "
                   f".npz:\n{lines}\n  Re-run preprocess_stream.py / "
                   f"extract_conditions.py for them.")
            if strict_conditions:
                raise RuntimeError(
                    msg + "\n(training.strict_conditions=True: refusing to train "
                          "with samples that would fall back to NULL conditions. "
                          "Set training.strict_conditions=false to allow it.)")
            print(msg + "\n[conditions] WARNING (strict=false): affected samples "
                        "will use NULL (zero) conditions.")

    registry = ConditionRegistry(
        enabled_frame  = enabled_f,
        enabled_global = enabled_g,
    )
    print(f"Condition registry: {registry}\n")

    FRAME_COND_DIMS     = registry.frame_cond_dims
    FRAME_COND_OUT_DIMS = registry.frame_cond_out_dims
    GLOBAL_CONFIGS      = registry.global_cond_configs
    print(f"Frame cond dims:     {FRAME_COND_DIMS}")
    print(f"Frame cond out dims: {FRAME_COND_OUT_DIMS}")
    print(f"Global cond configs: {GLOBAL_CONFIGS}\n")

    # ======================
    # DATA
    # ======================
    print("Loading conditioned datasets...")
    cond_root = cfg.paths.condition_root if Path(cfg.paths.condition_root).exists() else None
    img_root  = cfg.paths.image_root     if Path(cfg.paths.image_root).exists()     else None

    train_dataset, val_dataset, test_dataset, normalizer, label_map, split_info = \
        build_conditioned_datasets(
            latent_root=cfg.paths.dataset_root,
            condition_root=cond_root,
            image_root=img_root,
            duration_s=cfg.model.duration_s,
            normalizer_path=(normalizer_path
                             if os.path.exists(normalizer_path) else None),
            registry=registry,
            preload=False,
            strict_conditions=strict_conditions,
            # WHAT THE TEXT SLOT IS FED -- the chunk's own CLAP AUDIO vector,
            # its caption's CLAP TEXT vector, or a per-sample mix. A property
            # of the data, so it is resolved here and the model never learns
            # which of the two it is looking at.
            text_source=cfg.conditioning.get("text_source", "audio"),
            text_mix_p=cfg.conditioning.get("text_mix_p", 0.5),
            # The split is READ from the dataset's splits.json, written by
            # preprocess_stream.py. There is nothing to configure here any more:
            # ratios/seed/stratification are decided once, with the dataset.
            splits_path=cfg.paths.get("splits_path", None),
        )

    n_classes = len(label_map)
    print(f"\nDetected {n_classes} classes: {list(label_map.keys())}")
    print(f"[split] files  -> train {split_info['file_counts']['train']} | "
          f"val {split_info['file_counts']['val']} | "
          f"test {split_info['file_counts']['test']}")
    if split_info["manifest_path"]:
        print(f"[split] read from: {split_info['manifest_path']}")

    # Save the normalizer in the cache_dir
    if not os.path.exists(normalizer_path):
        normalizer.save(normalizer_path)

    n_workers = int(cfg.data.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.train_batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=(device == "cuda"),
        persistent_workers=(n_workers > 0),
        drop_last=True, collate_fn=collate_conditioned,
        generator=data_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.val_batch_size,
        shuffle=validation_shuffle,
        num_workers=n_workers, pin_memory=(device == "cuda"),
        persistent_workers=False,
        drop_last=False, collate_fn=collate_conditioned,
        generator=torch.Generator().manual_seed(validation_seed),
    )

    train_iter = infinite_loader(train_loader)

    # Fixed validation protocol: cache the same deterministic batches once, then
    # pair every batch with a dedicated, fixed (x0, t). Both the live model and
    # EMA receive these exact inputs at every validation step, so their losses and
    # the curve across checkpoints differ only because the weights changed.
    requested_val_batches = int(cfg.data.num_val_batches)
    if requested_val_batches <= 0:
        raise ValueError(
            f"data.num_val_batches must be > 0, got {requested_val_batches}.")
    fixed_val_batches = []
    for vb in val_loader:
        fixed_val_batches.append(vb)
        if len(fixed_val_batches) >= requested_val_batches:
            break
    del val_loader
    if not fixed_val_batches:
        raise RuntimeError("Validation dataset produced no batches.")

    val_rng = torch.Generator(device="cpu")
    val_rng.manual_seed(validation_seed)
    fixed_val_inputs = []
    for vb in fixed_val_batches:
        val_frames = vb[0]
        fixed_x0 = torch.randn(
            val_frames.shape, dtype=val_frames.dtype, generator=val_rng)
        u = torch.randn(val_frames.shape[0], generator=val_rng)
        fixed_t = torch.sigmoid(u).clamp(
            float(cfg.sampling.t_min), float(cfg.sampling.t_max))
        fixed_val_inputs.append((fixed_x0, fixed_t))
    n_fixed_val_samples = sum(int(vb[0].shape[0]) for vb in fixed_val_batches)
    print(f"[validation] fixed protocol={VALIDATION_PROTOCOL} | "
          f"shuffle={validation_shuffle} | seed={validation_seed} | "
          f"batches={len(fixed_val_batches)} | "
          f"samples={n_fixed_val_samples}")

    # ======================
    # PRE-COMPUTATION REFERENCE STATS (one time only, cached in cache_dir)
    # ======================
    # Reference stats are computed on REAL validation samples (same as in the
    # unconditional repo). The conditioned model is evaluated against the
    # same target distribution. MetricsAdapter exposes the slim view that
    # metrics.py expects.
    print("\nPre-computation of reference statistics for the metrics...")
    metrics_val_ds = MetricsAdapter(val_dataset)

    # Single real-latent reference (mu + full covariance) shared by BOTH
    # FD-DAC and the KL divergence, in the normalized latent space -- exactly
    # as in the unconditional repo. No audio-domain reference is needed anymore
    # (FAD/Encodec has been removed). Built ONLY if a metric that needs it is
    # enabled (mirrors the uncond build_references(enabled, ...) behaviour).
    if metrics_enabled:
        fd_dac_ref_stats = precompute_latent_reference(
            metrics_val_ds,
            cache_path=fd_dac_cache_path,
        )
        print(f"Reference stats ready: FD-DAC + KL on "
              f"{fd_dac_ref_stats['n_total']} latent frames "
              f"({len(val_dataset)} val samples)\n")
    else:
        fd_dac_ref_stats = None
        print("Reference stats SKIPPED: no distributional metric enabled "
              "(metrics.enabled is empty).\n")

    # ---- FAD (VGGish): embedder + real reference, only if requested ----
    # Everything here is skipped unless 'fad_vggish' is in metrics.enabled, so a
    # run that does not ask for it pays nothing and loads no VGGish weights.
    fad_embedder = None
    fad_ref_stats = None
    n_fad = 0
    fad_device = "cpu"
    if "fad_vggish" in metrics_enabled:
        from metrics import VGGishEmbedder
        fad_device = str(cfg.metrics.get("fad_device", "cuda"))
        if fad_device.startswith("cuda") and not torch.cuda.is_available():
            print("[metrics] fad_device='cuda' but no GPU is available "
                  "-> FAD falls back to CPU.")
            fad_device = "cpu"
        n_fad = int(cfg.sampling.get("n_fad_samples", 512) or 0)
        fad_embedder = VGGishEmbedder(device=fad_device)
        print(f"[metrics] FAD-VGGish enabled | device={fad_device} | "
              f"n_fad_samples={n_fad} | reference={_fad_ref_mode}")

        # The reference is the REAL validation audio. The dataset is split-less
        # on disk, so the wav/ tree holds train, val AND test together: the file
        # list is derived from the VAL SPLIT, never by globbing the directory,
        # or the "real" distribution would include the test set.
        _val_npy = sorted({str(smp[0]) for smp in val_dataset.samples})
        if _fad_ref_mode == "wav":
            _lat_root = Path(cfg.paths.dataset_root)
            _wav_root = Path(cfg.paths.wav_root)
            _val_wavs = [_wav_root / Path(f).relative_to(_lat_root).with_suffix(".wav")
                         for f in _val_npy]
            _missing = [w for w in _val_wavs if not w.exists()]
            if _missing:
                raise SystemExit(
                    f"[metrics] fad_vggish with metrics.fad_reference='wav' needs the "
                    f"real validation wavs, but {len(_missing)}/{len(_val_wavs)} are "
                    f"missing under {_wav_root} (first: {_missing[0]}).\n"
                    f"  Either re-run preprocess_stream.py with --save_wav (it does "
                    f"NOT re-encode the latents), or set metrics.fad_reference="
                    f"'decoded' to build the reference by decoding the validation "
                    f"latents through DAC instead. NB: 'decoded' compares two "
                    f"DAC-decoded distributions, which isolates the model from the "
                    f"codec but is NOT comparable with published FAD values.")
            fad_ref_stats = precompute_audio_reference(
                _val_wavs, fad_embedder, cache_path=fad_cache_path,
                device=fad_device)
        elif _fad_ref_mode == "decoded":
            _dac = get_dac()

            def _real_clips():
                for i in range(len(val_dataset)):
                    frames = val_dataset[i][0]
                    yield (decode_frames_to_wav(frames, normalizer, _dac)
                           .view(1, 1, -1), DAC_SAMPLE_RATE)

            if os.path.exists(fad_cache_path):
                print(f"[Audio ref] loading cache: {fad_cache_path}")
                fad_ref_stats = torch.load(fad_cache_path, map_location="cpu",
                                           weights_only=False)
            else:
                print(f"[Audio ref] embedding {len(val_dataset)} DAC-decoded val "
                      f"latents (metrics.fad_reference='decoded')...")
                _mu, _sig, _n = compute_audio_mu_sigma(
                    _real_clips(), len(val_dataset), fad_embedder,
                    device=fad_device, desc="FAD ref")
                fad_ref_stats = {"mu": _mu.cpu(), "sigma": _sig.cpu(), "n_total": _n}
                _tmp = fad_cache_path + ".tmp"      # atomic publish, as elsewhere
                torch.save(fad_ref_stats, _tmp)
                os.replace(_tmp, fad_cache_path)
                print(f"[Audio ref] cache saved: {fad_cache_path}")
        else:
            raise SystemExit(
                f"[metrics] metrics.fad_reference='{_fad_ref_mode}' is not a valid "
                f"choice. Use 'wav' (real validation wavs, comparable with the "
                f"literature) or 'decoded' (validation latents decoded through "
                f"DAC, no wavs needed).")
        print(f"FAD reference ready: {fad_ref_stats['n_total']} embedding "
              f"vectors (128-D)\n")

    # Conditioning-influence evaluators (validation only, never affect training):
    #   - frame conditions: re-extract the enabled frame conditions from the
    #     generations and compare, paired, to the input ones (f0 corr,
    #     chroma cosine, rhythm/energy correlation -- or mir_eval's metrics
    #     for the conditions it covers, with metrics.influence_family).
    #   - text (CLAP): the audio side of the same CLAP checkpoint, to score
    #     audio<->text adherence; loaded lazily ONLY if 'text' is active.
    # The per-condition influence (delta vs null) is consolidated in the
    # Validation/Condition_influence text panel.
    from condition_metrics import ConditionFidelityEvaluator
    # Device for the re-extraction (CREPE-full / beat_this / CLAP-audio) at the
    # metrics step: "cuda" (default) or "cpu", from metrics.fidelity_device.
    # It does not change the extracted values, only speed. "cpu" exists purely as
    # an escape hatch if the metrics step runs out of VRAM (there the model and
    # the DAC decoder are both resident, and CREPE-full on top can overflow a
    # card with little margin).
    _fid_device = str(cfg.metrics.get("fidelity_device", "cuda"))
    if _fid_device.startswith("cuda") and not torch.cuda.is_available():
        print("[metrics] fidelity_device='cuda' but no GPU is available "
              "-> falling back to CPU for the re-extraction.")
        _fid_device = "cpu"
    fidelity_evaluator = ConditionFidelityEvaluator(
        enabled_frame=list(FRAME_COND_DIMS.keys()),
        device=_fid_device,
        registry=registry,   # #15: re-extract with the run's exact extractor config
        family=influence_family,
        mir_threshold=mir_threshold,
    )
    for _name, _extractor in fidelity_evaluator.extractors.items():
        _actual_device = getattr(
            _extractor, "device", getattr(_extractor, "_device", "cpu"))
        _batch = getattr(_extractor, "batch_size", None)
        _batch_msg = f" | batch_size={_batch}" if _batch is not None else ""
        # `metrics=` is the family that condition actually got: under
        # mir_influence_metrics a condition mir_eval does not cover keeps ours.
        # A scorer bound to the ON threshold (mir chroma / chord) says which.
        _thr = getattr(fidelity_evaluator.fidelity_fns[_name], "keywords",
                       {}).get("threshold")
        _thr_msg = f" (ON at >= {_thr})" if _thr is not None else ""
        print(f"[metrics] extractor={_name} | device={_actual_device}{_batch_msg}"
              f" | metrics={fidelity_evaluator.families[_name]}{_thr_msg}")
    # ---- ONE EMBEDDER PER SCORABLE GLOBAL CONDITION ----
    # Each maps a GENERATED waveform into the space its condition lives in, so
    # the two can be compared: that cosine is what the influence row and the
    # similarity scalars are made of. The dict is the whole extension point --
    # a third global condition needs an entry here and nothing else.
    global_embedders = {}
    if "text" in GLOBAL_CONFIGS:
        from conditions import ClapAudioEmbedder
        # match the CLAP checkpoint used by the text condition
        clap_model_name = CONDITION_CONFIG["global"]["text"]["kwargs"].get(
            "model_name", "laion/clap-htsat-unfused")
        global_embedders["text"] = ClapAudioEmbedder(model_name=clap_model_name,
                                                     device=_fid_device)
        print(f"Text-influence (CLAP audio) enabled: {clap_model_name}")
    if "image" in GLOBAL_CONFIGS:
        # Wav2CLIP: audio distilled INTO CLIP's space, which is the only way an
        # audio-vs-image cosine exists at all (CLIP and CLAP are different
        # spaces). Optional: without it the image still CONDITIONS the model,
        # it just has no influence row -- so a missing package degrades the
        # diagnostics, never the training.
        try:
            from conditions import Wav2ClipAudioEmbedder
            _w2c = Wav2ClipAudioEmbedder(device=_fid_device)
            _w2c._load()                       # fail now, not mid-metrics-step
            global_embedders["image"] = _w2c
            print("Image-influence (Wav2CLIP audio->CLIP space) enabled")
        except Exception as _e:
            print(f"Image-influence DISABLED: {type(_e).__name__}: {_e}")

    # ---- OUT-OF-THE-BOX PROBE SETS (proof of concept) ----
    # A bank is built for EVERY condition active in this run, and the panels
    # then drive them all together (see run_joint_probe). Each bank is built
    # ONCE and cached (keyed by the stimuli, the chunk geometry and the
    # extractor's parameters), the same contract as the normalizer and the
    # FD-DAC reference -- shared by every run over the same setup, rebuilt
    # automatically if any of those change.
    #
    # This is the CONTROLLABILITY instrument. On unambiguous synthetic stimuli
    # it answers whether the conditioning of this run moves the generation at
    # all -- the question the validation rows cannot answer on their own,
    # because on real material a condition is often not cleanly extractable and
    # a middling score there does not separate weak conditioning from an
    # ambiguous target. It costs one paired generation pass per panel, whatever
    # the number of conditions, because the panel drives them jointly.
    #
    # sampling.n_influence_samples_probe is the probe half of the influence set:
    # that many stimuli per bank, all of them scored, plotted and played (the
    # validation half has its own knob, n_influence_samples_valid). Fewer than a
    # bank holds -> build_condition_probe_set picks a FIXED random subset, the
    # same at every step and in every run; more -> the whole bank, said so.
    # 0 -> no probe at all.
    probe_sets = {}
    _n_probes = influence_set_size(cfg.sampling, "probe")
    print(f"[influence] validation: "
          f"{influence_set_size(cfg.sampling, 'valid')} samples | probe: "
          f"{_n_probes if _n_probes > 0 else 'off'}"
          f"{' stimuli per bank' if _n_probes > 0 else ''}")
    # RNG GUARD. The probe banks are built HERE, before ConditionedAudioDiT is
    # constructed further down, and building the f0 bank runs torchcrepe, which
    # draws from the global torch-CPU and numpy generators (chroma/energy/rhythm
    # draw nothing -- their synthesis uses local np.random.default_rng(seed)).
    # Left unguarded, a COLD cache (bank built) and a WARM one (bank loaded from
    # disk) hand the model two different RNG states, so the same config and the
    # same training.seed produce two DIFFERENT initialisations -- measured on
    # five runs: all 54 tensors differ, max|dW| = 1.08. Snapshotting around the
    # whole loop makes the bank RNG-neutral, so the init depends only on the
    # seed and ablation runs stay comparable whatever the cache state.
    _rng_guard = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    # EVERY condition active in the run gets a bank -- the frame ones and the
    # globals alike. probe_conditions.py holds a bank and a synthesizer for
    # each; what differs is only which registry the extractor comes from and
    # what a stimulus IS (a waveform for a frame condition, a .png or a prompt
    # for a global). A run without globals iterates a shorter list and behaves
    # exactly as before.
    _probe_names = list(FRAME_COND_DIMS) + list(GLOBAL_CONFIGS)
    for _cond in _probe_names:
        if _n_probes <= 0:
            continue
        try:
            from probe_conditions import (build_condition_probe_set,
                                          text_probe_bank)
            # One builder for all four. The only per-condition thing left is
            # WHERE the cache lives: f0 keeps its historical directory (and the
            # paths.f0_probe_dir override) so an existing f0 cache is still a
            # hit; the others sit next to it under cache_dir.
            if _cond == "f0":
                _probe_root = (cfg.paths.get("f0_probe_dir", None)
                               or os.path.join(cfg.paths.cache_dir, "f0_probe"))
            else:
                _probe_root = os.path.join(cfg.paths.cache_dir,
                                           f"probe_{_cond}")
            # WHICH extractor builds the bank. NOT registry.frame_extractors:
            # that object is the one CONDITION_CONFIG pins to device="cpu" (the
            # device the preprocessing DataLoader workers need), and building the
            # f0 bank with it runs CREPE-full on CPU with batch_size=512 -- a
            # multi-GB RSS spike per stimulus, sixteen times in a row, which is
            # enough to get the job SIGKILLed part way through the bank. That
            # death is invisible here: it is not a Python exception, so the
            # except below never sees it and the run simply disappears.
            # The fidelity evaluator already holds a shallow COPY of the very
            # same extractor, moved to metrics.fidelity_device, with every
            # value-bearing parameter identical -- so building the bank with it
            # is the same extraction, on the GPU, at no extra cost. Measured:
            # 16 stimuli in 7s, 1.4 GB VRAM (the model is not built yet here).
            # It is not BIT-identical to the CPU one -- CREPE's convolutions are
            # not, ~1.4e-3 mean on the normalized pitch, corr 0.9997 -- which is
            # orders of magnitude below anything the influence panel resolves.
            # SAFE FOR THE CACHES: probe_conditions._fingerprint excludes the
            # device on purpose ("speed, never values"), so a bank built on
            # either device stays a hit for the other, and every existing cache
            # keeps working. Falls back to the registry object if the evaluator
            # has no extractor for this condition.
            #
            # A GLOBAL condition has no entry in either of those: its bank is
            # built by the run's own CLAP-text / CLIP-image encoder, straight
            # from registry.global_extractors, so the probe targets live in
            # exactly the space the model was conditioned in.
            if _cond in GLOBAL_CONFIGS:
                _probe_extractor = registry.global_extractors[_cond]
            else:
                _probe_extractor = fidelity_evaluator.extractors.get(
                    _cond, registry.frame_extractors[_cond])
            _probe_dev = getattr(_probe_extractor, "device",
                                 getattr(_probe_extractor, "_device", "cpu"))
            # The TEXT bank follows the dataset: captions that are single
            # labels are probed with those same labels, richer captions with
            # the descriptions (probe_conditions.text_probe_bank). Every other
            # condition keeps its own bank, exactly as before.
            _bank = None
            if _cond == "text":
                _bank, _why = text_probe_bank(
                    load_caption_table(val_dataset.latent_root),
                    n_panels=_n_probes)
                print(f"[text-probe] {_why}")
            _ps = build_condition_probe_set(
                _cond, _probe_root, val_dataset.n_frames,
                extractor=_probe_extractor,
                n_probes=_n_probes,
                duration_s=float(cfg.model.duration_s),
                sr=DAC_SAMPLE_RATE,
                bank=_bank,
            )
            probe_sets[_cond] = _ps
            print(f"{_cond} probe: {len(_ps)} elementary stimuli, "
                  f"all plotted on TB | device={_probe_dev} | dir={_ps.dir}")
        except Exception as e:
            # A probe that cannot be built must not take the training run down
            # with it: it is a diagnostic, not part of the objective.
            print(f"[{_cond}-probe] disabled -- could not build it: {e}")
    # Close the RNG guard opened before the loop: whatever the probe banks drew
    # (or did not draw) is rolled back, so the model init below sees the state
    # left by the seeding block and nothing else.
    torch.set_rng_state(_rng_guard[0])
    np.random.set_state(_rng_guard[1])
    random.setstate(_rng_guard[2])
    if _rng_guard[3] is not None:
        torch.cuda.set_rng_state_all(_rng_guard[3])
    # The global EXTRACTORS (CLAP-text, CLIP-vision) exist in a training run for
    # one job only: encoding the probe banks, above. That job is done, and on a
    # warm cache it never even started -- reading `dim` no longer builds them
    # (conditions._projection_dim_from_config). Release them here so a cold
    # cache does not leave a second checkpoint resident on the training GPU for
    # the rest of the run. What SCORES the global conditions at validation is a
    # different object (global_embedders: the CLAP audio tower and Wav2CLIP),
    # which is deliberately untouched.
    for _ext in getattr(registry, "global_extractors", {}).values():
        if hasattr(_ext, "unload"):
            try:
                _ext.unload()
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if probe_sets:
        print()

    metrics_uncond = bool(cfg.sampling.get("metrics_uncond", True))
    print(f"Conditioning influence: frame={list(FRAME_COND_DIMS.keys()) or 'none'}"
          f"{' + text(CLAP)' if 'text' in global_embedders else ''}"
          f"{' + image(Wav2CLIP)' if 'image' in global_embedders else ''}"
          f"{' + image(n/a)' if ('image' in GLOBAL_CONFIGS and 'image' not in global_embedders) else ''} "
          f"| uncond metrics: {'on' if metrics_uncond else 'off'}\n")

    # ======================
    # MODEL + EMA
    # ======================
    # cfg.model.kind and cfg.conditioning.* already reflect the checkpoint on
    # resume (restored in load_config), so the model is rebuilt with the correct
    # architecture and conditioning layout automatically - no need to re-pass
    # them on the command line.
    # frame_reinject_every: architecture parameter (it changes the state_dict),
    # defaulted to 0 with .get() so a YAML written before this option existed
    # keeps building exactly the model it used to.
    FRAME_REINJECT_EVERY = int(cfg.model.get("frame_reinject_every", 0))

    # TEXT CROSS-ATTENTION. Architecture parameter like the one above, and
    # like it, .get() so a YAML written before it existed builds exactly the
    # model it used to.
    #
    # The WIDTH is not a setting: it is the width of one token state of the
    # text encoder that wrote the dataset, and the dataset is the only thing
    # that can state it. Reading it off a config field would let a run be
    # configured for 768 and fed 512 -- the model would build, the load would
    # pass, and the first forward would raise somewhere unhelpful.
    TEXT_CROSS_EVERY = int(cfg.model.get("text_cross_every", 0))
    TEXT_CTX_DIM = int(getattr(train_dataset, "_ctx_dim", 0) or 0)
    if TEXT_CROSS_EVERY > 0 and "text" in GLOBAL_CONFIGS and TEXT_CTX_DIM <= 0:
        raise RuntimeError(
            "model.text_cross_every > 0 but this dataset carries no caption "
            "TOKEN sequences (global_conditions/text_labels_tok.npy), so the "
            "cross-attention would attend to the null token at every step and "
            "learn nothing -- silently, which is why this refuses to start "
            "instead of warning.\n"
            "  Re-run the preprocessing on the SAME output dir with "
            "--global_conds text: it rewrites only the sidecar, reads back the CLAP "
            "vectors already on disk and does not touch a single audio file."
        )
    print(f"[MODEL] Building ConditionedAudioDiT-{cfg.model.kind} "
          f"| frame={list(FRAME_COND_DIMS)} | global={list(GLOBAL_CONFIGS)} "
          f"| frame_reinject_every={FRAME_REINJECT_EVERY} "
          f"| text_cross_every={TEXT_CROSS_EVERY}")
    model = ConditionedAudioDiT(
        kind=cfg.model.kind,
        drop=cfg.model.get("drop", 0.0),
        frame_cond_dims=FRAME_COND_DIMS,
        frame_cond_out_dims=FRAME_COND_OUT_DIMS,
        global_cond_configs=GLOBAL_CONFIGS,
        frame_reinject_every=FRAME_REINJECT_EVERY,
        text_cross_every=TEXT_CROSS_EVERY,
        text_ctx_dim=TEXT_CTX_DIM,
    ).to(device)
    # EMA is optional, controlled by cfg.training.use_ema. When disabled,
    # validation/audio/metrics use the live model directly (no shadow copy).
    ema = EMAModel(model, decay=cfg.training.ema_decay) if cfg.training.use_ema else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    lr_lambda = make_lr_lambda(
        num_steps=cfg.training.num_steps,
        warmup_steps=cfg.training.warmup_steps,
        decay_start_frac=cfg.training.decay_start_frac,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda', enabled=cfg.training.use_amp)

    # SummaryWriter points directly to the run directory
    writer = SummaryWriter(run_dir)

    # Log the config in TB (tab "Text"). The split COMPOSITION (file/chunk counts
    # per split) is nested under data.split.composition, so it shows up INSIDE the
    # config panel's `data` section -- one window, not a separate Dataset/split
    # panel.
    _cfg_log = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(_cfg_log, False)
    if "data" not in _cfg_log:
        _cfg_log.data = {}
    if _cfg_log.data.get("split", None) is None:
        _cfg_log.data.split = {}
    # NB: no chunk_counts here -- with duration_s equal to the preprocessing chunk
    # length each latent file yields exactly ONE training chunk, so it would just
    # repeat file_counts.
    # The split is not a training parameter any more, so what gets logged is
    # what was READ: the file it came from and the parameters it was created
    # with. That is what makes a TensorBoard run self-describing about its own
    # val/test sets without having to go and open the dataset.
    _cfg_log.data.split.composition = {
        "file_counts":  dict(split_info["file_counts"]),
        "n_classes":    int(split_info["n_classes"]),
        "splits_file":  split_info["manifest_path"],
        "params":       dict(split_info.get("params", {})),
    }

    # Parameter counts, nested under the config's `model` section (same panel) and
    # also logged as scalars so they can be compared across runs in TensorBoard.
    _n_params_total = sum(p.numel() for p in model.parameters())
    _n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if "model" not in _cfg_log:
        _cfg_log.model = {}
    _cfg_log.model.n_params_total     = int(_n_params_total)
    _cfg_log.model.n_params_total_M   = round(_n_params_total / 1e6, 2)
    _cfg_log.model.n_params_trainable_M = round(_n_params_train / 1e6, 2)

    writer.add_text(
        "config",
        "```yaml\n" + OmegaConf.to_yaml(_cfg_log) + "\n```",
        global_step=0,
    )
    # NB: the parameter counts live in the `config` panel (model.n_params_*) and
    # in the console line below -- NOT as scalars: they are constants, and a
    # single flat point at step 0 only clutters the Scalars dashboard.
    print(f"[params] total={_n_params_total/1e6:.2f}M | "
          f"trainable={_n_params_train/1e6:.2f}M")

    best_val_loss = float("inf")
    start_step = 0

    # ======================
    # WARM START (paths.init_from)
    # ======================
    # A NEW run that begins from the weights of an old one. Not a resume: the
    # step counter, the optimizer, the scheduler and the RNG streams all start
    # from scratch, and only the model (and the EMA shadow) are seeded.
    #
    # WHAT IT IS FOR, concretely: turning the cross-attention on. Its output
    # projection is zero-initialised, so a model built with it computes exactly
    # the same function as the model without it -- the run can therefore
    # continue from a checkpoint that never had one, instead of paying for the
    # steps already spent. `--resume` cannot do that: it refuses an architecture
    # mismatch, and rightly, because it also restores an optimizer state whose
    # moments are indexed by parameter.
    #
    # THE ONLY KEYS ALLOWED TO BE MISSING are the ones the new architecture
    # adds. Anything else means the two models disagree about something that is
    # NOT zero-initialised -- a different kind, other conditions, another
    # frame_reinject stride -- and silently leaving those at their random init
    # while reporting a warm start is exactly the failure this checks for.
    init_from = cfg.paths.get("init_from", None)
    if init_from and cfg.paths.resume_from:
        raise RuntimeError(
            "paths.init_from and paths.resume_from are both set. They are "
            "different things: resume continues a run (same architecture, same "
            "optimizer, same step), init_from STARTS one from another run's "
            "weights. Pick one.")
    if init_from:
        if not os.path.exists(init_from):
            raise FileNotFoundError(f"paths.init_from: {init_from} not found")
        print(f"Warm start from: {init_from}")
        _ck = torch.load(init_from, map_location="cpu", weights_only=False)
        check_ckpt_reinject_gate(_ck, init_from)
        _sd = (_ck.get("ema_state_dict") if _ck.get("ema_ready", True)
               and "ema_state_dict" in _ck else _ck.get("model_state_dict"))
        if not isinstance(_sd, dict):
            raise RuntimeError(f"{init_from} carries no model weights")
        _res = model.load_state_dict(_sd, strict=False)
        _new = tuple(["cross_attn.", "text_null", "norm_cross."])
        _bad = [k for k in _res.missing_keys if not any(n in k for n in _new)]
        if _bad or _res.unexpected_keys:
            raise RuntimeError(
                f"{init_from} does not fit this model.\n"
                f"  missing and NOT part of the new cross-attention: "
                f"{_bad[:8]}{' ...' if len(_bad) > 8 else ''}\n"
                f"  present in the checkpoint but not in this model: "
                f"{list(_res.unexpected_keys)[:8]}"
                f"{' ...' if len(_res.unexpected_keys) > 8 else ''}\n"
                f"A warm start may only ADD zero-initialised weights. Rebuild "
                f"this run with the same model.kind, conditions and "
                f"frame_reinject_every as the checkpoint.")
        print(f"  {len(_sd) - len(_res.missing_keys)} tensor(s) loaded, "
              f"{len(_res.missing_keys)} left at their zero/random init "
              f"(the new cross-attention). Step counter starts at 0.")
        if ema is not None:
            # The shadow starts AS the loaded weights, not as the random init it
            # was deepcopy'd from: otherwise everything the EMA is used for --
            # validation, the best-checkpoint decision, the previews -- would be
            # reading noise while the live model is already trained.
            ema.copy_from(model)
        del _ck, _sd

    # ======================
    # RESUME
    # ======================
    resume_from = cfg.paths.resume_from
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from: {resume_from}")
        # Load the checkpoint on CPU first, NOT directly on the GPU.
        # With map_location=device the whole checkpoint (model + EMA + the AdamW
        # optimizer state, which is ~2x the model size) is pushed onto the GPU in
        # one shot, on top of the already-allocated model/EMA/optimizer. That
        # instantaneous spike can exceed the VRAM and raise CUDA OutOfMemory at
        # resume even when training-from-zero fits. Loading on CPU and letting
        # load_state_dict copy tensors into the (already on-GPU) modules avoids
        # keeping a second GPU copy of the checkpoint alive during the load.
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)

        # Defensive check: the model we built must match the checkpoint. The
        # architecture (kind) and the conditioning layout are restored from the
        # checkpoint config in load_config(), so normally they already agree; if
        # a CLI override forced a mismatch we stop here with a clear message
        # instead of a wall of size-mismatch errors.
        ckpt_kind = ckpt.get("model_kind", None)
        if ckpt_kind is not None and ckpt_kind != cfg.model.kind:
            raise RuntimeError(
                f"Checkpoint was trained with model.kind='{ckpt_kind}' but the "
                f"model was built as '{cfg.model.kind}'. They must match to "
                f"resume. (Normally the kind is restored automatically from the "
                f"checkpoint; if you passed model.kind on the command line, "
                f"remove it or set it to '{ckpt_kind}'.)"
            )
        # Same class of mismatch as `kind`: the re-injection adds one tensor per
        # selected block to the state_dict, so a checkpoint trained without it
        # simply has no weights to put there (and one trained WITH it has
        # weights with nowhere to go). Caught here with an explanation instead
        # of as a wall of "Missing key(s) in state_dict: frame_reinject.1...".
        # Checkpoints written before this option existed have no such key ->
        # default 0, which is exactly what they were trained as.
        ckpt_cross = int(ckpt.get("text_cross_every", 0))
        if ckpt_cross != TEXT_CROSS_EVERY:
            raise RuntimeError(
                f"Checkpoint was trained with model.text_cross_every="
                f"{ckpt_cross} but the model was built with "
                f"{TEXT_CROSS_EVERY}. They must match to resume: the "
                f"cross-attention adds four tensors per selected block plus "
                f"the learned null token, so the two state_dicts cannot be "
                f"put into one another. To START a NEW run with a different "
                f"value, leave paths.resume_from empty -- a zero-initialised "
                f"cross-attention computes the same function as none at all, "
                f"so nothing is lost by warm-starting from this checkpoint "
                f"with paths.init_from instead, if the run supports it."
            )
        ckpt_reinject = int(ckpt.get("frame_reinject_every", 0))
        if ckpt_reinject != FRAME_REINJECT_EVERY:
            raise RuntimeError(
                f"Checkpoint was trained with model.frame_reinject_every="
                f"{ckpt_reinject} but the model was built with "
                f"{FRAME_REINJECT_EVERY}. They must match to resume: the "
                f"per-block frame re-injection changes the weights themselves, "
                f"not just a training setting. (Normally the value is restored "
                f"automatically from the checkpoint config; if you passed "
                f"model.frame_reinject_every on the command line, remove it or "
                f"set it to {ckpt_reinject}. To START a NEW run with a "
                f"different value, do not use --resume: this is a different "
                f"architecture and needs its own run.)"
            )
        # ...and the same check one level finer: `frame_reinject_every`
        # matching is no longer enough to pin the state_dict down, because a
        # run trained with re-injection BEFORE the per-condition gate existed
        # has the projections and none of the gate tensors. That passes every
        # test above and then dies inside load_state_dict as a raw wall of
        # missing keys, so it is named here instead.
        check_ckpt_reinject_gate(ckpt, cfg.paths.resume_from)

        ckpt_frame_dims = ckpt.get("frame_cond_dims", None)
        if ckpt_frame_dims is not None and dict(ckpt_frame_dims) != dict(FRAME_COND_DIMS):
            raise RuntimeError(
                f"Checkpoint frame conditions {dict(ckpt_frame_dims)} do not match "
                f"the ones built for this run {dict(FRAME_COND_DIMS)}. They must "
                f"match to resume. (Normally conditioning.enabled_frame is restored "
                f"from the checkpoint; if you overrode it on the command line, "
                f"remove the override.)"
            )

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        # NB: scheduler.load_state_dict() already restores last_epoch. Do NOT
        # also set scheduler.last_epoch = ckpt["step"]: it would add a 1-step
        # offset to the LR curve on resume. Same fix as in training.py.
        if cfg.training.use_ema:
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            else:
                # Old checkpoint without EMA: start a fresh shadow copy from
                # the loaded live weights.
                ema = EMAModel(model, decay=cfg.training.ema_decay)
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        # Restore every RNG stream so the run continues them (x0, t, CFG
        # dropout) instead of restarting from the seed. The batch ORDER does not
        # resume: a fresh epoch draws a new permutation (see
        # _capture_rng_state). Best-effort for old checkpoints without rng_state.
        _restore_rng_state(ckpt.get("rng_state"), data_generator)
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        saved_val_protocol = ckpt.get("validation_protocol", None)
        if saved_val_protocol != VALIDATION_PROTOCOL:
            print("[validation] Checkpoint best_val_loss was measured with a "
                  "different/stochastic protocol; resetting best_val_loss so it "
                  "is not compared with the new fixed validation curve.")
            best_val_loss = float("inf")
        print(f"  -> Step {start_step} | best_val_loss: {best_val_loss:.6f}")
        writer.add_text("resumed_from", resume_from, global_step=start_step)

        # After loading the optimizer state from a CPU checkpoint, the AdamW
        # buffers (exp_avg / exp_avg_sq) may still live on CPU. Move them to the
        # GPU explicitly so the first optimizer.step() doesn't hit a device
        # mismatch. Done tensor-by-tensor (gradual), not in one big push.
        if device == "cuda":
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        # Carry the EMA readiness across the resume, BEFORE the checkpoint is
        # freed. Legacy checkpoints have no such field: fall back to the old
        # semantics (the shadow was being updated past ema_start) rather than
        # re-seeding and discarding a legitimate average.
        _resumed_ema_ready = bool(
            ckpt.get("ema_ready", start_step > cfg.training.ema_start))

        # Free the CPU copy of the checkpoint and clear any cached GPU blocks
        # left over from the load before the training loop starts.
        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        _resumed_ema_ready = False
        print("Training from zero.")

    # ======================
    # INFO
    # ======================
    n_frames = train_dataset.n_frames
    print(f"\n{'='*60}")
    print(f"Training on {device} | ConditionedAudioDiT-{cfg.model.kind}")
    print(f"Steps: {cfg.training.num_steps} | "
          f"Effective Batch: {cfg.data.effective_bs}")
    print(f"LR: {cfg.training.lr} | "
          f"EMA: {'on (decay=' + str(cfg.training.ema_decay) + ')' if cfg.training.use_ema else 'off'} | "
          f"AMP: {cfg.training.use_amp}")
    print(f"Sequence: {n_frames} frame = {n_frames} token of dim {TOKEN_DIM}")
    print(f"Train: {len(train_dataset)} chunk | Val: {len(val_dataset)} chunk")
    print(f"Audio every {cfg.intervals.audio} step | "
          f"Metrics every {cfg.intervals.metrics} step")
    if fd_dac_ref_stats is not None:
        print(f"Metrics: {cfg.sampling.n_metrics_samples} generated vs "
              f"{fd_dac_ref_stats['n_total']} reference frames")
    else:
        print("Metrics: distributional metrics DISABLED (metrics.enabled: [])")
    print(f"DAC decoder device: {_DAC_DEVICE}"
          + ("  (~2.5 s per 5 s clip; 'cuda' is ~13x faster but adds a ~1 GB "
             "peak at the metrics step)" if _DAC_DEVICE == "cpu" else
             "  (~0.2 s per 5 s clip; ~1 GB peak at the metrics step)"))
    # .get: a config written before per-condition dropout existed stays valid
    # and keeps its exact previous behaviour (0.0 disables stage 2).
    P_DROP_EACH_FRAME = float(cfg.conditioning.get("p_drop_each_frame", 0.0))
    print(f"CFG dropout: all={cfg.conditioning.p_drop_all} "
          f"frame={cfg.conditioning.p_drop_frame} "
          f"global={cfg.conditioning.p_drop_global} "
          f"each_frame={P_DROP_EACH_FRAME}"
          + ("  (partial subsets ON)" if P_DROP_EACH_FRAME > 0 else ""))
    print(f"CFG guidance scale (validation): {cfg.conditioning.guidance_scale}")
    _reinj_txt = (f"every {FRAME_REINJECT_EVERY} block(s)"
                  if FRAME_REINJECT_EVERY > 0 else "OFF (input concat only)")
    print(f"Frame re-injection (per block): {_reinj_txt}")
    print(f"DATASET_ROOT:   {cfg.paths.dataset_root}")
    print(f"WAV_ROOT:       {cfg.paths.wav_root}")
    print(f"CONDITION_ROOT: {cfg.paths.condition_root}")
    print(f"IMAGE_ROOT:     {cfg.paths.image_root}")
    print(f"RUN DIR:        {run_dir}")
    print(f"{'='*60}\n")

    # Real reference audio, logged ONCE at step 0 (it never changes during
    # training, unlike the generations), so it is audible from the start instead
    # of only appearing at the first metrics step. In a conditioned run it lands
    # at the TOP of every validation block, beside the conditions extracted from
    # it (validation_XX/1_real_validation_XX); n_audio_samples does not bound it
    # there, only in the unconditioned case, where it fills the 'ground truth'
    # group instead.
    log_real_audio_samples(
        val_dataset=val_dataset,
        normalizer=normalizer,
        writer=writer,
        n_samples=cfg.sampling.n_audio_samples,
        sampling_cfg=cfg.sampling,
        frame_dims=FRAME_COND_DIMS,
        global_configs=GLOBAL_CONFIGS,
    )

    # ======================
    # Real audio / conditions are logged inside evaluate_and_log_metrics at every
    # metrics step: the sonified conditions and the conditioned generation into
    # the per-sample block, the null generation into the "uncond generation"
    # group and the recording into "ground truth". All of them walk with the
    # TensorBoard step slider (the reals are also written once at step 0, so the
    # ground-truth group is populated before the first metrics step).

    # ======================
    # TRAIN LOOP
    # ======================
    val_loss = None
    pbar = tqdm(range(start_step, cfg.training.num_steps),
                initial=start_step, total=cfg.training.num_steps,
                desc="Training", unit="step")
    last_step = start_step
    # Real EMA state: True only once the shadow holds TRAINED weights (it is
    # seeded from the live model when ema_start is reached). Persisted in the
    # checkpoint and carried across resumes, so it can never be inferred wrongly.
    ema_ready = _resumed_ema_ready

    try:
        for step in pbar:
            last_step = step
            model.train()

            accum_loss = 0.0
            for _ in range(cfg.data.grad_accum):
                batch = next(train_iter)
                loss = compute_loss(
                    model, batch, device,
                    use_amp=cfg.training.use_amp,
                    t_min=cfg.sampling.t_min,
                    t_max=cfg.sampling.t_max,
                    global_configs=GLOBAL_CONFIGS,
                    p_drop_all=cfg.conditioning.p_drop_all,
                    p_drop_frame=cfg.conditioning.p_drop_frame,
                    p_drop_global=cfg.conditioning.p_drop_global,
                    p_drop_each_frame=P_DROP_EACH_FRAME,
                    training=True,
                ) / cfg.data.grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item()

                del loss, batch

            scaler.unscale_(optimizer)
            # If grad_clip > 0 we clip and get back the pre-clip total L2 norm.
            # If grad_clip <= 0 (or None) we pass `inf` as max_norm, which never
            # clips but still returns the total norm so we can log it.
            _clip = cfg.training.grad_clip
            max_norm = _clip if (_clip is not None and _clip > 0) else float('inf')
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if cfg.training.use_ema and step >= cfg.training.ema_start:
                if not ema_ready:
                    # Seed the shadow with the CURRENT (trained) weights the first
                    # time it goes live. Until now it still held the random init
                    # it was deepcopy'd from at step 0, and lerping that away
                    # takes ~7k steps per halving -- during which the EMA is what
                    # validation, best-checkpoint selection and the metrics use.
                    # Driven by the STATE, not by `step == ema_start`, so a resume
                    # landing past ema_start with an unseeded shadow still gets
                    # seeded instead of averaging noise forever.
                    ema.copy_from(model)
                    ema_ready = True
                    pbar.write(f"  -> EMA seeded from the live model @ step {step}")
                else:
                    ema.update(model)

            writer.add_scalar("Train/Loss", accum_loss, step)
            writer.add_scalar("Train/Learning rate",
                              scheduler.get_last_lr()[0], step)
            # Pre-clip gradient norm: gives an early warning of instability
            # (sudden spikes mean the model is approaching a NaN regime).
            writer.add_scalar("Train/Grad_norm", grad_norm.item(), step)
            pbar.set_postfix(loss=f"{accum_loss:.4f}",
                              lr=f"{scheduler.get_last_lr()[0]:.1e}")

            # ======================
            # VALIDATION (loss)
            # ======================
            if step % cfg.intervals.val == 0:
                model.eval()

                if device == "cuda":
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    val_loss_sum = 0.0
                    ema_val_loss_sum = 0.0
                    val_sample_count = 0
                    ema_active = (cfg.training.use_ema
                                  and step >= cfg.training.ema_start)
                    for vb, (fixed_x0, fixed_t) in zip(
                            fixed_val_batches, fixed_val_inputs):
                        batch_sample_count = int(vb[0].shape[0])
                        vl = compute_loss(
                            model, vb, device,
                            use_amp=cfg.training.use_amp,
                            t_min=cfg.sampling.t_min,
                            t_max=cfg.sampling.t_max,
                            global_configs=GLOBAL_CONFIGS,
                            p_drop_all=cfg.conditioning.p_drop_all,
                            p_drop_frame=cfg.conditioning.p_drop_frame,
                            p_drop_global=cfg.conditioning.p_drop_global,
                            p_drop_each_frame=P_DROP_EACH_FRAME,
                            training=False,
                            x0=fixed_x0,
                            t=fixed_t,
                        ).item()
                        val_loss_sum += vl * batch_sample_count
                        val_sample_count += batch_sample_count

                        if ema_active:
                            evl = compute_loss(
                                ema.model, vb, device,
                                use_amp=cfg.training.use_amp,
                                t_min=cfg.sampling.t_min,
                                t_max=cfg.sampling.t_max,
                                global_configs=GLOBAL_CONFIGS,
                                p_drop_all=cfg.conditioning.p_drop_all,
                                p_drop_frame=cfg.conditioning.p_drop_frame,
                                p_drop_global=cfg.conditioning.p_drop_global,
                                p_drop_each_frame=P_DROP_EACH_FRAME,
                                training=False,
                                x0=fixed_x0,
                                t=fixed_t,
                            ).item()
                            ema_val_loss_sum += evl * batch_sample_count

                    val_loss = val_loss_sum / val_sample_count
                    ema_val_loss = val_loss
                    if ema_active:
                        ema_val_loss = ema_val_loss_sum / val_sample_count
                        writer.add_scalar("Validation/Loss_ema", ema_val_loss, step)

                writer.add_scalar("Validation/Loss", val_loss, step)

                if device == "cuda":
                    torch.cuda.empty_cache()

                ema_str = (f" | EMA Val {ema_val_loss:.6f}"
                           if cfg.training.use_ema and step >= cfg.training.ema_start else "")
                pbar.write(f"Step {step:7d} | Train {accum_loss:.6f} | "
                           f"Val {val_loss:.6f}{ema_str} | "
                           f"LR {scheduler.get_last_lr()[0]:.2e}")

                # Best model: compare on EMA val loss if active, else on plain val loss.
                check_loss = (ema_val_loss
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else val_loss)
                if check_loss < best_val_loss:
                    best_val_loss = check_loss
                    save_path = os.path.join(ckpt_dir, f"best_model_step{step}.pt")
                    ckpt_data = build_ckpt_data(
                        model, ema, optimizer, scheduler, scaler, step,
                        val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                        FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS,
                        data_generator=data_generator, ema_ready=ema_ready)
                    torch.save(ckpt_data, save_path)
                    for old in Path(ckpt_dir).glob("best_model_step*.pt"):
                        if old.resolve() != Path(save_path).resolve():
                            old.unlink()
                    pbar.write(f"  -> Best model: {save_path}")

            # ======================
            # AUDIO PREVIEW (optional, conditioned-only, separate from metrics)
            # Audio preview every intervals.audio steps, written into the
            # SAME validation_XX/ audio blocks the metrics step uses: it
            # refreshes the condition and the with-cond generation, while
            # the "uncond generation" and "ground truth" groups are filled
            # by the metrics step. Skipped when the two cadences coincide,
            # so a card never gets two values for the same step.
            if (step > 0 and step % cfg.intervals.audio == 0
                    and step % cfg.intervals.metrics != 0):
                pbar.write(f"\n  Audio preview step {step}...")
                gen_model = (ema.model
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else model)
                generate_and_log_audio(
                    model=gen_model, normalizer=normalizer,
                    val_dataset=val_dataset,
                    n_frames=n_frames, step=step, writer=writer,
                    device=device, output_dir=audio_dir,
                    n_samples=cfg.sampling.n_audio_samples,
                    sampling_cfg=cfg.sampling,
                    conditioning_cfg=cfg.conditioning,
                    use_amp=cfg.training.use_amp,
                    frame_dims=FRAME_COND_DIMS,
                    global_configs=GLOBAL_CONFIGS,
                    prefix=("EMA"
                            if cfg.training.use_ema and step >= cfg.training.ema_start
                            else "Model"),
                )
                pbar.write(f"  Audio preview logged (step {step})\n")
                model.train()

            # ======================
            # PERIODICAL CHECKPOINT
            # ------------------------------------------------------------
            # Saved BEFORE the metrics on purpose. The metrics step can run for a
            # long time (generation + DAC decode + condition re-extraction) and is
            # the most likely place to die (CUDA OOM, the lab watchdog, node
            # crash, SIGKILL). Checkpointing first means such a death costs the
            # metrics pass, never the training progress -- and with
            # intervals.ckpt == intervals.metrics there is nothing to gain by
            # waiting for the evaluation to finish.
            # ======================
            if step % cfg.intervals.ckpt == 0 and step > 0:
                p = os.path.join(ckpt_dir, f"checkpoint_step{step}.pt")
                ckpt_data = build_ckpt_data(
                    model, ema, optimizer, scheduler, scaler, step,
                    val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                    FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS,
                        data_generator=data_generator, ema_ready=ema_ready)
                torch.save(ckpt_data, p)
                pbar.write(f"  -> Checkpoint: {p}")

                # Keep only the last N periodic checkpoints (best and last are
                # not touched: they use different name prefixes). Mirrors
                # training.py.
                keep_n = cfg.intervals.get("keep_last_n_ckpts", 4)
                periodic_ckpts = sorted(
                    Path(ckpt_dir).glob("checkpoint_step*.pt"),
                    key=lambda x: int(x.stem.replace("checkpoint_step", "")),
                )
                for old in periodic_ckpts[:-keep_n]:
                    old.unlink()
                    pbar.write(f"  -> Removed old periodic checkpoint: {old.name}")

            # ======================
            # METRICS (FD-DAC + KL) on conditioned generations
            # ======================
            if step > 0 and step % cfg.intervals.metrics == 0:
                gen_model = (ema.model
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else model)
                fd_dac_cond, kl_cond_rg, kl_cond_gr = evaluate_and_log_metrics(
                    model=gen_model,
                    normalizer=normalizer,
                    val_dataset=val_dataset,
                    step=step,
                    writer=writer,
                    device=device,
                    output_dir=audio_dir,
                    fd_dac_ref_stats=fd_dac_ref_stats,
                    n_samples=cfg.sampling.n_metrics_samples,
                    sampling_cfg=cfg.sampling,
                    conditioning_cfg=cfg.conditioning,
                    use_amp=cfg.training.use_amp,
                    frame_dims=FRAME_COND_DIMS,
                    global_configs=GLOBAL_CONFIGS,
                    fidelity_evaluator=fidelity_evaluator,
                    global_embedders=global_embedders,
                    compute_uncond=metrics_uncond,
                    prefix=("EMA"
                            if cfg.training.use_ema and step >= cfg.training.ema_start
                            else "Model"),
                    metrics_seed=metrics_seed,
                    metrics_enabled=metrics_enabled,
                    fad_embedder=fad_embedder,
                    fad_ref_stats=fad_ref_stats,
                    n_fad=n_fad,
                    fad_device=fad_device,
                    probe_sets=probe_sets,
                    image_root=cfg.paths.get("image_root", None),
                )
                # Report ONLY the metrics that were actually requested: with
                # metrics.enabled=["fd_dac"] the KL values are None (and vice
                # versa), so formatting them unconditionally would raise a
                # TypeError and kill the run at the first metrics step.
                _parts = []
                if fd_dac_cond is not None:
                    _parts.append(f"FD-DAC={fd_dac_cond:.4f}")
                if kl_cond_rg is not None:
                    _parts.append(f"KL(real||gen)={kl_cond_rg:.4f}")
                if kl_cond_gr is not None:
                    _parts.append(f"KL(gen||real)={kl_cond_gr:.4f}")
                if _parts:
                    pbar.write("  Metrics [cond]: " + " | ".join(_parts) + "\n")
                model.train()

    finally:
        # Always try to save the last checkpoint, whatever killed the loop
        # (Ctrl+C, normal end, or an exception such as CUDA OutOfMemory).
        #
        # IMPORTANT: when the loop dies from a CUDA OOM (e.g. at the metrics
        # step), the GPU is full, so the naive save below can ITSELF fail -
        # building state_dict() and running torch.save touch the GPU, and there
        # may be no memory left. That is the most likely reason a previous run
        # died WITHOUT leaving a checkpoint_last. So we save defensively:
        #   1. first attempt: normal save (fast path, Ctrl+C / clean end)
        #   2. if it fails: free the CUDA cache, move model+EMA+optimizer to CPU,
        #      and retry the save entirely from CPU (no GPU allocation needed).
        last_path = os.path.join(ckpt_dir, f"checkpoint_last_step{last_step}.pt")

        def _try_save():
            ckpt_data = build_ckpt_data(
                model, ema, optimizer, scheduler, scaler, last_step,
                val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS,
                        data_generator=data_generator, ema_ready=ema_ready)
            torch.save(ckpt_data, last_path)

        saved = False
        try:
            _try_save()
            saved = True
            print(f"\n  -> Last checkpoint saved: {last_path}")
        except Exception as e_gpu:
            print(f"\n  [WARN] Normal checkpoint save failed ({type(e_gpu).__name__}: "
                  f"{e_gpu}). Retrying from CPU after freeing GPU memory...")
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
                # Move everything off the GPU so the save needs no VRAM.
                model.to("cpu")
                if cfg.training.use_ema and ema is not None:
                    ema.model.to("cpu")
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()
                if device == "cuda":
                    torch.cuda.empty_cache()
                _try_save()
                saved = True
                print(f"  -> Last checkpoint saved from CPU: {last_path}")
            except Exception as e_cpu:
                print(f"  [ERROR] Could not save the last checkpoint even from CPU "
                      f"({type(e_cpu).__name__}: {e_cpu}). "
                      f"The most recent usable checkpoint is the latest "
                      f"best_model_step*.pt / checkpoint_step*.pt in {ckpt_dir}.")

        try:
            pbar.close()
            writer.close()
        except Exception:
            pass
        print("Training concluded." if saved else "Training ended (see warnings above).")
