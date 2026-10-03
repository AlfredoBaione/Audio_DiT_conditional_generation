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
#   - FD-DAC + KL (both directions) computed every intervals.metrics step on
#     conditioned samples, with reference pre-computed on the full validation
#     set (real validation latents, normalized space); on the SAME generations,
#     the Condition_influence panel (with-cond / null / delta per metric)
#   - Listening panels on TensorBoard at every metrics step: test-set samples
#     and probe stimuli, plus the unconditioned generations (also refreshed
#     every intervals.audio step)
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
from datetime import datetime, timedelta
from contextlib import nullcontext
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
import torch.distributed as dist
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
    compute_mu_sigma,
    compute_fad,
    compute_frechet_distance,
    gaussian_kl_fullcov,
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


def merge_cli_overrides(cfg, dotlist, where):
    """Merge CLI dotlist overrides (e.g. training.lr=2e-4) into `cfg` and return
    the result. They win over everything, but a typo in a key must FAIL rather
    than silently create a phantom top-level entry while the real parameter
    keeps its config value: `from_dotlist` parses each token, and merging into
    a struct-locked config raises on any key that does not already exist -- so
    `data_num_val_batches=8` (should be data.num_val_batches) stops the run
    instead of being ignored. `where` names the config in the message. Shared
    by training_cond.load_config and test_cond.load_config."""
    if not dotlist:
        return cfg
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    OmegaConf.set_struct(cfg, True)
    try:
        cfg = OmegaConf.merge(cfg, cli_cfg)
    except Exception as e:
        base_keys = set(_flatten_keys(OmegaConf.to_container(cfg, resolve=False)))
        bad = [k for k in _flatten_keys(OmegaConf.to_container(cli_cfg))
               if k not in base_keys]
        # A key that existed once gets a pointer to what replaced it.
        notes = dict.fromkeys(
            OBSOLETE_SAMPLING_KEYS[k.partition(".")[2]] for k in bad
            if k.startswith("sampling.")
            and k.partition(".")[2] in OBSOLETE_SAMPLING_KEYS)
        _hint = "".join(f"\n          {n}" for n in notes)
        raise SystemExit(
            f"[config] unknown CLI override key(s): {bad or [str(e)]}\n"
            f"          These do not exist in {where}. Check the "
            f"spelling and the dotted path (e.g. 'data.num_val_batches', not "
            f"'data_num_val_batches'). Nothing was run.{_hint}")
    OmegaConf.set_struct(cfg, False)
    return cfg


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

    # Keys that no longer exist (the influence-set sizes, n_similarity_samples)
    # are removed here with a log line saying what replaced them, and the panel
    # keys are put in when missing. Before the CLI merge, so a CLI value still
    # wins. The same function reads test_cond.py's config.
    drop_obsolete_sampling_keys(cfg)

    # CLI overrides win over everything (YAML + checkpoint config); a typo in a
    # key stops the run (merge_cli_overrides).
    cfg = merge_cli_overrides(cfg, unknown, args.config)

    # The two panel counts, checked here so a bad value costs a second. 0 turns
    # that family of panels off.
    _smp = cfg.get("sampling", None)
    if _smp is not None:
        for _k in PANEL_KEYS.values():
            _v = _smp.get(_k, None)
            if _v is not None and int(_v) < 0:
                raise SystemExit(
                    f"[config] sampling.{_k} = {_v}, must be >= 0 (0 = no "
                    f"panels of that family). Nothing was run.")

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
              f"branch to extrapolate from: with no dropout on a branch, "
              f"guidance_scale on that branch is meaningless.")
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
    # getattr(model, "module", model): under multi-GPU the training model is
    # wrapped in DistributedDataParallel, which does not expose the attributes
    # of the network inside it.
    if (getattr(getattr(model, "module", model), "text_cross_layers", None)
            and text_ctx is not None):
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
                            text_ctx=None, x0=None):
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

    `x0`: (B, n_frames, TOKEN_DIM) starting noise drawn by the caller, or None
    to draw it here (the multi-GPU metrics draw it themselves: see
    evaluate_and_log_metrics).

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
    if x0 is None:
        x0 = torch.stack([
            torch.randn(n_frames, TOKEN_DIM, device=device, generator=gen_rng)
            for _ in range(B)
        ])
    else:
        x0 = x0.to(device)
        if tuple(x0.shape) != (B, n_frames, TOKEN_DIM):
            raise ValueError(f"x0 has shape {tuple(x0.shape)}, expected "
                             f"{(B, n_frames, TOKEN_DIM)}")
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


def uncond_card_tag(k):
    """The k-th 'uncond generation' card: uncond_00, uncond_01, ... Just a
    number -- a generation with NO conditions carries nothing of any sample."""
    return f"{UNCOND_AUDIO_GROUP}/uncond_{int(k):02d}"


def audio_panel_tags(family, idx, active_conditions=(), suffix="",
                     conditioned=None):
    """
    The TensorBoard AUDIO tags of ONE sample -> {"conditions": {name: tag},
    "generation": tag, "real": tag}.

    Single source of truth for the audio window's layout, because the places
    that log audio (the metrics step, the step-0 real references, the probe)
    all write into the same blocks and groups and MUST agree on their names --
    otherwise one sample would own two half-filled blocks instead of one.

    TWO layout facts drive every name here, and both are properties of the
    dashboard rather than choices:

    1. Cards are grouped by the text BEFORE the first '/', and each group is its
       own headed, collapsible block. So the SAMPLE is that prefix. With every
       tag under a single prefix the dashboard builds ONE grid holding every
       card of every sample, and a condition ends up separated from its own
       generation by a row break; one block per sample holds 2-5 cards that
       stay together and are visibly walled off from the next sample.

    2. Inside a block cards are sorted alphabetically and flow into a grid that
       wraps every 2-3 cards depending on window width, so the numeric prefix
       is what fixes the listening order. That order is: the RECORDING the
       conditions were extracted from, then EVERY condition the generation was
       given, then the generation itself LAST.

          1  real_test_XX               the recording (test blocks only)
          2  f0_<family>_XX             the sonified f0 target, when f0 is on
          3..N <condition>_<family>_XX  the others, alphabetical
          N+1  generation_<family>_XX   the generation, after all of them

       A PROBE block has no recording -- its stimuli are synthesised, there is
       nothing they were extracted from -- so there the conditions start at 1.

       The generation is not named after any single condition: the card is the
       generation of the WHOLE block, and the block header (and the condition
       cards above it) already say what went into it.

    The generations WITHOUT conditions are not in any block: they are the
    'uncond generation' group (uncond_card_tag), sampling.n_audio_samples cards
    written by generate_and_log_uncond_cards. The 'ground truth' group holds
    the recordings of an UNCONDITIONED run only, which has no blocks at all; in
    a conditioned run every recording sits at the top of its own test block,
    next to the conditions that were extracted from it.

    The card names INSIDE a block repeat the family and index that the block
    header already shows. That redundancy is deliberate: a card read, filtered
    or screenshotted on its own still says what it is and which sample it
    belongs to.

    `family` is "test" or "probe" for the blocks; "test" with
    conditioned=False names the ground-truth recordings of an unconditioned
    run (real_test_XX). `suffix` decorates the BLOCK header only (the probe prompt, the test
    sample's caption). f0 leads the condition cards when it is active because
    it is the one most listened to against the generation; if it is off, the
    first condition alphabetically takes slot 1 and nothing else changes.

    `conditioned` says whether ANYTHING conditioned this generation -- a frame
    condition OR a global one. It exists because `active_conditions` lists only
    the FRAME conditions (they are the ones that get a card of their own: a
    picture and a prompt are not waveforms), so a run conditioned on text/image
    ALONE would otherwise look like an unconditioned one. Its default -- "there
    is a frame condition" -- is what the callers meant before global-only runs
    existed.
    """
    active = sorted(active_conditions)
    lead = "f0" if "f0" in active else (active[0] if active else None)
    if conditioned is None:
        conditioned = lead is not None
    block = f"{family}_{idx:02d}{suffix}"
    tail = f"{family}_{idx:02d}"

    # THE RECORDING LEADS THE BLOCK -- but only where one exists. A test panel
    # is built FROM a real chunk: its conditions were extracted from that
    # recording, so the recording is where the listening starts, the conditions
    # are what was taken out of it, and the generation is what they produced. A
    # probe panel has no such recording (its stimuli are synthesised), and an
    # unconditioned run has no per-sample block at all -- there the recording
    # keeps its own collected group, which is the only way to hear real material
    # in a run with no panels.
    panel_real = (family == "test" and conditioned)
    first = 2 if panel_real else 1

    # Then every condition card -- f0 leading when it is on -- and the
    # generation after the last of them, so a panel is listened to in the order
    # it was built: the recording, the stimuli taken from it, then what they
    # produced.
    ordered = ([lead] + [n for n in active if n != lead]) if lead else []
    conditions = {c: f"{block}/{j + first}_{c}_{tail}"
                  for j, c in enumerate(ordered)}
    gen_slot = len(ordered) + first

    real = (f"{block}/1_real_{tail}" if panel_real
            else f"{REAL_AUDIO_GROUP}/real_{tail}")

    return {
        "conditions": conditions,
        "generation": f"{block}/{gen_slot}_generation_{tail}",
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
    slot 1 of that block (true for a test panel), because that is what shifts
    every card below it down by one.
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
    they are identical at every step -- the test panels are a FIXED set of
    samples (test_panel_indices) and the probe stimuli are fixed by
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
    because the writers cover different cards: the metrics step writes the
    condition cards of the panels, and the real recordings are logged at
    startup. One flag for all of them would let whichever ran first close the
    door on the others.

    A resume into a FRESH run directory finds no file and writes every card
    again, which is what that board needs. Deleting the file re-writes them all
    on the next step -- which is also the answer if the panels are ever widened
    (n_test_panels / n_probe_panels raised) on an existing board."""
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
    before: one conditioned pass, and the table's own columns only.

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
# THE LISTENING PANELS: how many, and which samples
# ======================
# sampling.n_test_panels / n_probe_panels, since 2 Oct 2026. They replace the
# two halves of the old influence set (n_influence_samples_valid / _probe),
# which also decided which generations the Condition_influence table was
# computed on. The table now uses ALL the n_metrics_samples validation
# generations (the ones FD-DAC / KL / FAD use), and the panels are listening
# and looking material only: nothing measured on them is in the table.
PANEL_KEYS = {"test": "n_test_panels", "probe": "n_probe_panels"}
PANEL_DEFAULT = 8

# sampling keys that no longer exist -> what took their place. A config still
# holding one (an older YAML, a checkpoint's own config) has it removed at
# startup, with this line in the log; on the command line it stops the run.
_PANELS_NOTE = (
    "The Condition_influence table is computed on all the "
    "sampling.n_metrics_samples generations; the listening panels are "
    "sampling.n_test_panels (test samples) and sampling.n_probe_panels "
    "(probe stimuli).")
OBSOLETE_SAMPLING_KEYS = {
    "n_influence_samples": _PANELS_NOTE,
    "n_influence_samples_valid": _PANELS_NOTE,
    "n_influence_samples_probe": _PANELS_NOTE,
    "n_similarity_samples": (
        "The text / image similarity is a column of the Condition_influence "
        "table, on the same generations as FD-DAC / KL / FAD; it has no "
        "curve of its own any more."),
}


def drop_obsolete_sampling_keys(cfg):
    """Remove from cfg.sampling every key of OBSOLETE_SAMPLING_KEYS, saying
    what replaced it, and put in the panel keys with their default when they
    are missing (so that a CLI override of them works on an older config, and
    the dumped config names them). Shared by training_cond.load_config and
    test_cond.load_config, so the two read a config the same way."""
    smp = cfg.get("sampling", None)
    if smp is None:
        return
    gone = [k for k in OBSOLETE_SAMPLING_KEYS if k in smp]
    for k in gone:
        del smp[k]
    for note in dict.fromkeys(OBSOLETE_SAMPLING_KEYS[k] for k in gone):
        keys = [k for k in gone if OBSOLETE_SAMPLING_KEYS[k] == note]
        print(f"[config] sampling.{', sampling.'.join(keys)}: no longer read. "
              f"{note}")
    for k in PANEL_KEYS.values():
        if smp.get(k, None) is None:
            smp[k] = PANEL_DEFAULT


def panel_count(sampling_cfg, which) -> int:
    """How many panels of one family the metrics step shows:
      which="test"  -> sampling.n_test_panels, samples of the TEST split;
      which="probe" -> sampling.n_probe_panels, probe stimuli (at most the
                       size of a bank, 16).
    0 = none of that family. A missing key falls back to PANEL_DEFAULT
    (load_config puts both keys in, so that only happens to a config built by
    hand)."""
    if sampling_cfg is None:
        return PANEL_DEFAULT
    v = sampling_cfg.get(PANEL_KEYS[which], None)
    return PANEL_DEFAULT if v is None else max(0, int(v))


def test_panel_indices(n_test, n_panels):
    """-> the test_dataset indices of the test panels: `n_panels` samples
    spread evenly over the WHOLE test split (linspace, like the validation
    list), so they cover its classes and sources instead of its first files.
    The same samples at every metrics step: block test_03 is always the same
    recording, and its slider only moves because the model did."""
    n = min(max(0, int(n_panels or 0)), max(0, int(n_test or 0)))
    if n <= 0:
        return []
    return torch.linspace(0, int(n_test) - 1, n).long().tolist()


# ======================
# THE UNCOND CARDS: every metrics step AND every intervals.audio
# ======================
@torch.no_grad()
def generate_and_log_uncond_cards(
    model, normalizer, n_frames, step, writer, device, output_dir, n_cards,
    sampling_cfg, use_amp, frame_dims, global_configs, metrics_seed=None,
    prefix="EMA",
):
    """
    The 'uncond generation' group: `n_cards` (sampling.n_audio_samples)
    generations with NO condition, uncond_00 .. uncond_<n-1>, and nothing else.

    ONE noise stream. Card k starts from the k-th draw of ONE generator seeded
    with metrics.seed, so the n cards are n different generations by
    construction. Until 2 Oct 2026 they were collected from two passes, the
    validation panels' and then the probe's, each of which restarted the
    generator from the seed: uncond_00 and uncond_04 were the same draw, hence
    the same audio, where eight different generations had been asked for.

    THE SAME FUNCTION writes them at the metrics step and at every
    intervals.audio, so a card starts from the same noise at every step and
    its slider shows one generation evolving with the weights. The generator
    is a local one: the global training RNG is not touched. With metrics.seed
    unset the cards free-run, like every other metric generation.
    """
    n_cards = max(0, int(n_cards or 0))
    if n_cards <= 0:
        return
    dac_model = get_dac()
    gen_rng = None
    if metrics_seed is not None:
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(int(metrics_seed))
    for k in range(n_cards):
        gen = euler_sample_cfg(
            model, n_frames, device,
            steps=sampling_cfg.euler_steps,
            t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
            use_amp=use_amp,
            frame_cond=None, global_cond=None, guidance=1.0,
            frame_dims=frame_dims, global_configs=global_configs,
            gen_rng=gen_rng,
        )
        if not torch.isfinite(gen).all():
            print(f"    [uncond] uncond_{k:02d}: non-finite generation, "
                  f"not logged")
            continue
        wav = decode_frames_to_wav(gen, normalizer, dac_model)
        writer.add_audio(uncond_card_tag(k), norm_wav(wav), global_step=step,
                         sample_rate=DAC_SAMPLE_RATE)
        sf.write(os.path.join(output_dir,
                              f"step{step:07d}_{prefix}_uncond_{k:02d}.wav"),
                 wav.numpy(), DAC_SAMPLE_RATE)
    # dac_model is the shared singleton -> do not delete it.


# ======================
# LOG REAL AUDIO SAMPLES (once at startup, step=0)
# ======================
@torch.no_grad()
def log_real_audio_samples(test_dataset, normalizer, writer,
                           n_samples, sampling_cfg=None, frame_dims=None,
                           global_configs=None):
    """Logs the real recordings at step 0, so they are audible from the start
    instead of only appearing at the first metrics step. They are always TEST
    recordings, in the training and in test_cond.py alike.

    CONDITIONED run: one card at the TOP of every TEST block
    (test_XX/1_real_test_XX), beside the conditions that were extracted from
    that very recording -- one per test panel (sampling.n_test_panels),
    decoded from the test sample's own latent.

    UNCONDITIONED run: there are no blocks at all, so `n_samples`
    (sampling.n_audio_samples) recordings of `test_dataset`, spread evenly
    over it, go to the collected 'ground truth' group (real_test_XX) -- the
    only real material such a run has to listen to."""
    dac_model = get_dac()
    # `frame_dims or global_configs`: a global-only run has the ordinary
    # panels, so its recordings belong at the top of the test blocks too.
    conditioned = bool(frame_dims or global_configs)
    if conditioned:
        ds, family = test_dataset, "test"
        indices = test_panel_indices(len(ds) if ds is not None else 0,
                                     panel_count(sampling_cfg, "test"))
        sfx = panel_suffixes(ds, indices, global_configs)
    else:
        ds, family = test_dataset, "test"
        total = len(ds) if ds is not None else 0
        n = max(0, min(int(n_samples or 0), total))
        indices = (torch.linspace(0, total - 1, n).long().tolist()
                   if n else [])
        sfx = {}

    written = 0
    for k, idx in enumerate(indices):
        # Checked BEFORE the decode: this runs at every startup, and on a resume
        # into the same run directory the card is already there. Writing it
        # again would put a second step-0 entry under one tag and give a
        # recording a step slider -- see fixed_card_pending.
        tag = audio_panel_tags(family, k, suffix=sfx.get(k, ""),
                               conditioned=conditioned)["real"]
        if not fixed_card_pending(writer, tag):
            continue
        # ConditionedAudioDataset returns a 6-tuple: take only the frames
        frames = ds[idx][0]
        waveform = decode_frames_to_wav(frames, normalizer, dac_model).unsqueeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)

        writer.add_audio(tag, wn, global_step=0, sample_rate=DAC_SAMPLE_RATE)
        written += 1

    # dac_model is the shared singleton -> do not delete it.
    print(f"  {written} real audios logged on TensorBoard"
          + ("" if written == len(indices)
             else f" ({len(indices) - written} already on this board)"))


# ======================
# OUT-OF-THE-BOX JOINT PROBE (proof of concept)
# ======================
@torch.no_grad()
def run_joint_probe(probe_sets, model, normalizer, n_frames,
                    step, writer, device,
                    output_dir, use_amp, sampling_cfg, guidance,
                    frame_dims, global_configs, fidelity_evaluator,
                    dac_model, prefix, n_plot,
                    metrics_seed=None):
    """
    Generate conditioned on the out-of-the-box probe stimuli of EVERY active
    condition AT ONCE, and show the result to be listened to and looked at.
    Nothing measured here goes into the Condition_influence table: since 2 Oct
    2026 that table is computed on the validation generations alone.

    `probe_sets` is {condition name -> ConditionProbeSet}. Panel i drives every
    active condition with the i-th stimulus of its OWN bank: the f0 of a scale,
    the chroma of a triad, the energy of a crescendo, the beat grid of a 120 bpm
    pattern -- combined by index. The pairing is by index and therefore
    arbitrary, but it is DETERMINISTIC, so panel 03 means the same combination
    at every checkpoint and the panels stay comparable across steps.

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
    at all", which real material cannot answer alone -- there a condition is
    often not cleanly extractable (a smeared chromagram, a beat grid that does
    not exist, an f0 that fails on 3 samples out of 4). The validation table
    and the test panels carry the aligned, in-corpus case; the probe carries
    the clean, artificial one.

    What it logs, all at `step` so the TensorBoard slider walks them together:
      * IMAGES  probe_XX/<cond>_target_vs_gen -- target vs re-extracted, one
                per active condition, titled with the stimulus it used
      * AUDIO   the probe_XX/ block: one card per condition holding the
                stimulus that condition was taken from, then the generation
                they jointly conditioned.
    No generation without conditions is made here: those are the 'uncond
    generation' cards, sampling.n_audio_samples of them, all from one noise
    stream (generate_and_log_uncond_cards).

    `fidelity_evaluator` is reused (reset first) rather than rebuilt: it carries
    the run's exact extractor configuration, and instantiating a second CREPE
    would risk the two drifting apart. Here it only re-extracts the curves the
    images draw (and the per-sample score in their title). The caller must
    therefore have already taken its copies of the validation results.
    """
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
        return
    missing = [c for c in list(frame_dims or {}) + list(global_configs or {})
               if c not in (probe_sets or {})]
    if missing:
        # Not fatal, but it means those slots go in NULL and the probe is no
        # longer the in-distribution shape described above -- say so loudly.
        print(f"    [probe] WARNING: no bank for {missing}; those conditions "
              f"go in NULL, so this probe is a partial subset")

    # The panels are index-aligned across banks, so the count is the shortest.
    n_probe = min(len(probe_sets[c]) for c in names + gnames)
    n_plot = max(0, min(int(n_plot), n_probe))
    if n_plot == 0:
        return

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
    # token, the panels would still be drawn, and nothing on screen would say
    # that the prompt never reached the model.
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
    # the probe panels comparable ACROSS checkpoints (what moves is the model,
    # not the noise). None = free-running.
    gen_rng = None
    if metrics_seed is not None:
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(int(metrics_seed))

    spf = max(1, int(sampling_cfg.get("metrics_samples_per_forward", 1) or 1))
    cond_lat = []
    if guidance > 1.0:
        # The fused sampler, as for the validation generations: the guided
        # generation of `spf` panels per forward. Its second output, the
        # generation without conditions, is not used here.
        for s in range(0, n_plot, spf):
            grp = list(range(s, min(s + spf, n_plot)))
            gc, _gu = euler_sample_cfg_paired(
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
    else:
        # No CFG to fuse (guidance <= 1): one plain conditioned pass per panel.
        for i in range(n_plot):
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

    # ---- one decode per generation: the curves for the images, the audio ----
    # add_sample receives EVERY condition of the panel, so one decode
    # re-extracts all of them.
    fidelity_evaluator.reset()
    fidelity_evaluator.keep_contours_for(range(n_plot))
    wavs = []
    for i, lat in enumerate(cond_lat):
        wav = decode_frames_to_wav(lat, normalizer, dac_model)
        if names:
            fidelity_evaluator.add_sample(
                wav.numpy(), DAC_SAMPLE_RATE, n_frames,
                {c: targets[c][i] for c in names}, sample_id=i)
        wavs.append(wav)
    ps_cond = fidelity_evaluator.per_sample() if names else {}
    cont_cond = ({c: fidelity_evaluator.contours(c) for c in names}
                 if names else {})

    # ---- IMAGES + AUDIO, one block per probe panel ----
    # The stimuli a panel was conditioned on and its generation sit under the
    # SAME tag prefix, so the audio window cannot separate a generation from
    # the conditions that produced it.
    for i in range(n_plot):
        # The block name of THIS panel, decided ONCE and used by every tag it
        # owns -- the comparison images, the image card and the audio. A text
        # prompt cannot be a card, so it names the block instead; computing it
        # here rather than beside the audio is what stops the Images and Audio
        # windows from filing the same panel under two names.
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
        # the TITLE of the panel's other cards via `suffix` below.
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
        # image alone has none, and `conditioned` says it is still conditioned.
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
        writer.add_audio(tags["generation"], norm_wav(wavs[i]),
                         global_step=step, sample_rate=DAC_SAMPLE_RATE)
        sf.write(os.path.join(_panel_dir(output_dir, step, "probe"),
                              f"probe_{i:02d}.wav"),
                 wavs[i].numpy(), DAC_SAMPLE_RATE)

    # NO "which stimuli each panel combines" TABLE: each probe comparison image
    # is TITLED with its own stimulus name (the `label=` passed to the plotters
    # above), so the mapping is readable panel by panel where it is looked at.


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
    """One human-readable line for a sample of `val_dataset` -- any split: the
    training calls it on the TEST set, for the header of the test panels
    (panel_suffixes) -- its CATEGORY, then the nearest phrases to its stored
    text vector, each with its cosine.

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


def panel_suffixes(dataset, ds_indices, global_configs):
    """{panel index: block suffix} for the panels built from real samples of
    `dataset` (the test panels): panel k is dataset[ds_indices[k]].

    THE POINT OF THIS FUNCTION IS THAT THERE IS ONLY ONE OF IT. The block name
    `test_XX` is built in several places -- the metrics step, the step-0 real
    references, the comparison images and the image card -- and TensorBoard
    groups by the text before the first '/'. A suffix computed in some of them
    and not the others would give one sample two half-filled blocks instead of
    one, which is precisely the failure audio_panel_tags warns about. Every
    site calls THIS, with the inputs it already has, so they cannot disagree.

    The suffix names what the panel was conditioned on: the sample's CATEGORY
    (exact, from the dataset) and the nearest phrase to its stored CLAP vector
    with its cosine (a retrieval over a closed vocabulary -- CLAP has no
    decoder, so there is no true sentence to recover). Only ONE phrase: this
    goes in a block header, which stays readable only while it stays short.

    Empty for every panel when the text condition is off, so a run without it
    keeps plain block names.

    Cached per (dataset, indices): it reads one .npz header per panel and is
    asked for the same answer at every metrics step.
    """
    if "text" not in (global_configs or {}) or dataset is None:
        return {}
    key = (id(dataset), tuple(int(i) for i in ds_indices))
    cache = getattr(panel_suffixes, "_cache", None)
    if cache is None:
        cache = panel_suffixes._cache = {}
    if key in cache:
        return cache[key]

    # How many phrases a caption carries is decided ONCE, in the preprocessing
    # (--text_labels_n), and travels with the dataset. The vocabulary is loaded
    # only as a fallback, for datasets written before the sidecar existed.
    captions = load_text_captions(dataset.latent_root)
    phrases, vocab = ((None, None) if captions
                      else load_text_label_vocab(dataset.latent_root))
    out = {}
    for k, idx in enumerate(ds_indices):
        desc = describe_validation_sample(dataset, int(idx), captions,
                                          phrases, vocab)
        out[k] = f" [{desc}]" if desc else ""
    cache[key] = out
    return out


def _panel_dir(output_dir, step, family):
    """runs/<run>/audio/step_<step>/<family>/: the .wav of the panels of one
    family (test / probe) at one metrics step."""
    d = os.path.join(output_dir, f"step_{step:07d}", family)
    os.makedirs(d, exist_ok=True)
    return d


# ======================
# THE TEST PANELS: listening and looking, not measuring
# ======================
@torch.no_grad()
def run_test_panels(test_dataset, ds_indices, model, normalizer, n_frames,
                    step, writer, device, output_dir, use_amp, sampling_cfg,
                    guidance, frame_dims, global_configs, fidelity_evaluator,
                    dac_model, prefix, metrics_seed=None, text_vec_for=None,
                    subset_specs=(), image_root=None):
    """
    The TEST panels: real samples of the TEST split (`ds_indices`, from
    test_panel_indices), generated with their own conditions at every metrics
    step and shown to be listened to and looked at. Nothing measured on them
    enters the Condition_influence table, which is computed on the validation
    generations.

    Per panel, block test_XX[ caption]:
      * AUDIO   1_real_test_XX, the recording the conditions were extracted
                from (written once, at startup, by log_real_audio_samples, and
                here only for a card that pass did not reach); then one
                sonified card per frame condition (once per board); then the
                generation, at every metrics step. With
                sampling.influence_subsets, one more generation per subset,
                after it, so the whole ladder of a sample is heard side by side.
      * IMAGES  <cond>_target_vs_gen: the condition given vs the one
                re-extracted from the generation, one per frame condition;
                image_condition: the picture, in an image-conditioned run.
    Every generation is also written to runs/<run>/audio/step_<step>/test/.

    Same generation contract as the validation pass: the guidance and the CFG
    of the run, the text slot from the caption when
    sampling.validation_text_from_caption is on (`text_vec_for`), and the
    initial noise from a generator seeded with metrics.seed (panel j <- draw
    j), so a panel's slider moves only because the weights did.

    `fidelity_evaluator` is reset and reused, as in the probe: the caller must
    already hold its copies of the validation results.
    """
    from probe_conditions import plot_condition_comparison
    from condition_metrics import sonify_condition

    n = len(ds_indices)
    if n == 0:
        return
    names = list(frame_dims or {})
    gnames = list(global_configs or {})
    sfx = panel_suffixes(test_dataset, ds_indices, global_configs)
    wants_ctx = bool(getattr(model, "text_cross_layers", None))

    # (frames, frame conditions, text vector, image vector, text context)
    items = []
    for idx in ds_indices:
        frames_real, fcond, _lab, text_emb, image_emb, text_ctx = \
            test_dataset[idx]
        if text_vec_for is not None:
            text_emb = text_vec_for(test_dataset, idx, text_emb)
        items.append((frames_real, fcond, text_emb, image_emb, text_ctx))

    def _fc(grp, subset=None):
        # A subset hands the model only its own conditions; the others are
        # left out and the network fills them with the null, as in training.
        return {k: torch.stack([items[j][1][k] for j in grp]).to(device).float()
                for k in names if subset is None or k in subset}

    def _gc(grp):
        gc = {}
        if "text" in gnames:
            gc["text"] = torch.stack([items[j][2] for j in grp]).to(device)
        if "image" in gnames:
            gc["image"] = torch.stack([items[j][3] for j in grp]).to(device)
        return gc

    def _ctx(grp):
        if not wants_ctx:
            return None
        return {"tokens": torch.stack([items[j][4]["tokens"] for j in grp]),
                "mask":   torch.stack([items[j][4]["mask"] for j in grp])}

    spf = int(sampling_cfg.get("metrics_samples_per_forward", 1))

    def _generate(subset=None):
        gen_rng = None
        if metrics_seed is not None:
            gen_rng = torch.Generator(device=device)
            gen_rng.manual_seed(int(metrics_seed))
        out = []
        if spf >= 1 and guidance > 1.0:
            # The fused sampler, like the validation pass; its second output
            # (the generation without conditions) is not used here.
            for s in range(0, n, spf):
                grp = list(range(s, min(s + spf, n)))
                gen_c, _gen_u = euler_sample_cfg_paired(
                    model, n_frames, device,
                    steps=sampling_cfg.euler_steps,
                    t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                    use_amp=use_amp,
                    frame_cond=_fc(grp, subset), global_cond=_gc(grp),
                    guidance=guidance,
                    frame_dims=frame_dims, global_configs=global_configs,
                    gen_rng=gen_rng, text_ctx=_ctx(grp),
                )
                out.extend(gen_c)
        else:
            for j in range(n):
                out.append(euler_sample_cfg(
                    model, n_frames, device,
                    steps=sampling_cfg.euler_steps,
                    t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                    use_amp=use_amp,
                    frame_cond=_fc([j], subset), global_cond=_gc([j]),
                    guidance=guidance,
                    frame_dims=frame_dims, global_configs=global_configs,
                    gen_rng=gen_rng, text_ctx=_ctx([j]),
                ))
        return out

    # ---- generate, decode once, re-extract the curves the images draw ----
    lat = _generate()
    use_ev = bool(names) and fidelity_evaluator is not None \
        and fidelity_evaluator.active
    if use_ev:
        fidelity_evaluator.reset()
        fidelity_evaluator.keep_contours_for(range(n))
    wavs = []
    for j, z in enumerate(lat):
        wav = decode_frames_to_wav(z, normalizer, dac_model)
        if use_ev:
            fidelity_evaluator.add_sample(
                wav.numpy(), DAC_SAMPLE_RATE, n_frames,
                {k: v.cpu().numpy() for k, v in items[j][1].items()},
                sample_id=j)
        wavs.append(wav)
    ps = fidelity_evaluator.per_sample() if use_ev else {}
    cont = fidelity_evaluator.contours() if use_ev else {}

    # ---- the combination ladder: one generation per condition subset ----
    # "all" is the generation above. Same noise draws, so every rung of a
    # sample starts from the same x0 and differs only by what it was given.
    ladder = {}
    _full = tuple(names)
    for _lab, _names in (subset_specs or ()):
        if tuple(_names) == _full:
            continue
        ladder[_lab] = [decode_frames_to_wav(z, normalizer, dac_model)
                        for z in _generate(subset=set(_names))]

    # ---- one block per panel: IMAGES then AUDIO ----
    pdir = _panel_dir(output_dir, step, "test")
    img_failed, img_shown = None, 0
    for k in range(n):
        blk = f"test_{k:02d}{sfx.get(k, '')}"

        # Target vs re-extracted, per frame condition. The score in the title
        # is the condition's first metric, whatever it is named, and its name
        # goes with it for the f0 title -- as in the probe.
        for c in sorted(names):
            if k not in cont.get(c, {}):
                continue
            mk = sorted(key for key in ps if key.startswith(f"{c}/"))
            score_map = ps.get(mk[0], {}) if mk else {}
            tgt, gen = cont[c][k]
            writer.add_image(
                f"{blk}/{c}_target_vs_gen",
                plot_condition_comparison(
                    c, tgt, gen, kind="test",
                    label=f"test sample #{ds_indices[k]}",
                    step=step, prefix=prefix, guidance=guidance,
                    score=score_map.get(k),
                    score_name=mk[0].partition("/")[2] if mk else None),
                global_step=step)

        # The picture the generation was conditioned on, once per board: the
        # dataset holds only its embedding, so the file is re-opened from the
        # raw image folder. Without a readable image_root the card is skipped
        # and the rest of the panel is unaffected.
        if "image" in gnames and image_root:
            ref = test_dataset.image_file_for(ds_indices[k])
            if ref is not None:
                try:
                    from PIL import Image as _PILImage
                    im = np.asarray(_PILImage.open(
                        Path(image_root) / ref[0] / ref[1]).convert("RGB"))
                    tag = f"{blk}/image_condition"
                    if fixed_card_pending(writer, tag):
                        writer.add_image(tag, im.transpose(2, 0, 1),
                                         global_step=0)
                    img_shown += 1
                except Exception as e:
                    img_failed = f"{ref[0]}/{ref[1]}: {type(e).__name__}: {e}"

        tags = audio_panel_tags("test", k, names, suffix=sfx.get(k, ""),
                                conditioned=True)
        # The stimuli, ONCE per board at step 0 (fixed_card_pending): block
        # test_XX is the same test sample at every metrics step, so its
        # conditions are the same waveform every time.
        for cname, carr in sorted(items[k][1].items()):
            if cname not in tags["conditions"]:
                continue
            son = sonify_condition(cname, carr.cpu().numpy(), DAC_SAMPLE_RATE)
            if son is not None and fixed_card_pending(
                    writer, tags["conditions"][cname]):
                writer.add_audio(tags["conditions"][cname], norm_wav(son),
                                 global_step=0, sample_rate=DAC_SAMPLE_RATE)
        # The recording: normally already written at startup.
        if fixed_card_pending(writer, tags["real"]):
            writer.add_audio(
                tags["real"],
                norm_wav(decode_frames_to_wav(items[k][0], normalizer,
                                              dac_model)),
                global_step=0, sample_rate=DAC_SAMPLE_RATE)
        writer.add_audio(tags["generation"], norm_wav(wavs[k]),
                         global_step=step, sample_rate=DAC_SAMPLE_RATE)
        sf.write(os.path.join(pdir, f"test_{k:02d}.wav"), wavs[k].numpy(),
                 DAC_SAMPLE_RATE)
        for _lab, _w in ladder.items():
            writer.add_audio(
                subset_generation_tag("test", k, _lab, suffix=sfx.get(k, ""),
                                      n_conditions=len(names), has_real=True),
                norm_wav(_w[k]), global_step=step, sample_rate=DAC_SAMPLE_RATE)
            sf.write(os.path.join(pdir, f"test_{k:02d}_gen_{_lab}.wav"),
                     _w[k].numpy(), DAC_SAMPLE_RATE)
    if img_failed is not None and img_shown == 0:
        print(f"    [test panels] image cards unavailable ({img_failed}); "
              f"is paths.image_root still pointing at the right folder?")


# ======================
# THE TEXT SLOT OF A METRICS GENERATION
# ======================
def caption_text_vec_fn(latent_root, sampling_cfg, global_configs):
    """-> f(dataset, idx, text_emb): the vector the text slot is GIVEN for
    sample `idx` of `dataset` (validation or test, scored or in a panel).

    With sampling.validation_text_from_caption the text slot receives the CLAP
    TEXT embedding of the sample's description (its class, plus the nearest
    phrases -- see --text_labels_n), instead of the CLAP AUDIO embedding of the
    chunk it was extracted from. Both were computed by the preprocessing, and
    the caption sidecar covers every chunk of every split. TRAINING is
    untouched, and so is the validation LOSS: only the generations the metrics
    and the panels are built from change, which is what makes the table's
    `text/clap_sim` an audio-vs-TEXT cosine.

    The vector is BOTH the conditioning and the target that column is measured
    against, which is why this one substitution is the whole change. Without
    the option (or without caption embeddings in the dataset, said once) the
    function hands back the dataset's own vector."""
    cap_ids, cap_emb = ({}, None)
    if (sampling_cfg is not None
            and bool(sampling_cfg.get("validation_text_from_caption", False))
            and "text" in (global_configs or {})):
        cap_ids, cap_emb = load_caption_conditions(latent_root)
        if cap_emb is None:
            print("    [metrics] validation_text_from_caption is on but this "
                  "dataset carries no caption embeddings -- re-run the "
                  "preprocessing with --global text. Falling back to the "
                  "chunk's own CLAP vector.")
        else:
            print(f"    [metrics] text conditioned on the DESCRIPTION "
                  f"({cap_emb.shape[0]} distinct caption(s))")

    def text_vec_for(dataset, idx, text_emb):
        if cap_emb is None:
            return text_emb
        try:
            key = _chunk_key_of(dataset, dataset.samples[idx][1])
            cid = cap_ids.get(key)
            if cid is None:
                return text_emb
            return torch.from_numpy(cap_emb[cid].copy())
        except Exception:
            return text_emb
    return text_vec_for


# ======================
# WHAT IS THERE TO LISTEN TO AND LOOK AT
# ======================
@torch.no_grad()
def log_listening_panels(model, normalizer, step, writer, device, output_dir,
                         sampling_cfg, conditioning_cfg, use_amp, frame_dims,
                         global_configs, fidelity_evaluator, test_dataset,
                         probe_sets, n_frames, prefix="EMA", metrics_seed=None,
                         image_root=None, text_vec_for=None):
    """
    The listening (and looking) half of the metrics step: the TEST panels
    (run_test_panels), the PROBE panels (run_joint_probe) and the 'uncond
    generation' cards (generate_and_log_uncond_cards). Nothing measured here
    goes into the Condition_influence table.

    The training calls it at the end of every metrics step; test_cond.py calls
    it too -- after its own metrics, or alone when it is asked for no metrics
    -- so the two windows are filled by the same code.

    Must run AFTER the table row has been read out of the evaluator, because
    the panels reset the same evaluator to re-extract their curves.
    `text_vec_for` is caption_text_vec_fn's function; None builds it here.
    """
    guidance = float(conditioning_cfg.guidance_scale)
    any_cond_active = bool(frame_dims) or bool(global_configs)
    dac_model = get_dac()
    # The combination ladder of the test panels follows the subsets the table
    # measures (sampling.influence_subsets), when there are frame conditions
    # to subset.
    subset_specs = []
    if fidelity_evaluator is not None and fidelity_evaluator.active:
        subset_specs = resolve_influence_subsets(
            sampling_cfg.get("influence_subsets", None), list(frame_dims or {}))

    # ===== THE TEST PANELS =====
    if any_cond_active and test_dataset is not None:
        _t_idx = test_panel_indices(len(test_dataset),
                                    panel_count(sampling_cfg, "test"))
        if _t_idx:
            if text_vec_for is None:
                text_vec_for = caption_text_vec_fn(
                    test_dataset.latent_root, sampling_cfg, global_configs)
            print(f"    test panels: {len(_t_idx)} samples of the test split...")
            try:
                run_test_panels(
                    test_dataset, _t_idx, model=model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer, device=device,
                    output_dir=output_dir, use_amp=use_amp,
                    sampling_cfg=sampling_cfg, guidance=guidance,
                    frame_dims=frame_dims, global_configs=global_configs,
                    fidelity_evaluator=fidelity_evaluator, dac_model=dac_model,
                    prefix=prefix, metrics_seed=metrics_seed,
                    text_vec_for=text_vec_for, subset_specs=subset_specs,
                    image_root=image_root)
            except Exception as _e:
                # Listening material bolted onto the metrics step: a failure
                # must not take down a training run that is otherwise fine.
                # Reported, not swallowed.
                print(f"    [test panels] SKIPPED at step {step}: "
                      f"{type(_e).__name__}: {_e}")
            if device == "cuda":
                torch.cuda.empty_cache()

    # ===== OUT-OF-THE-BOX JOINT PROBE =====
    # ONE call, not one per condition: the panel drives every active condition
    # together, which is the shape a model trained with p_drop_each_frame = 0.0
    # actually saw. See run_joint_probe for why the stimuli are combined by
    # index and deliberately not mutually aligned.
    if probe_sets:
        _pnames = [c for c in list(frame_dims or {}) + list(global_configs or {})
                   if c in probe_sets]
        _npanel = min(panel_count(sampling_cfg, "probe"),
                      min([len(probe_sets[c]) for c in _pnames], default=0))
        if _npanel:
            print(f"    joint probe: {_npanel} panels over {_pnames}...")
            try:
                run_joint_probe(
                    probe_sets,
                    model=model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer, device=device,
                    output_dir=output_dir, use_amp=use_amp,
                    sampling_cfg=sampling_cfg,
                    guidance=guidance, frame_dims=frame_dims,
                    global_configs=global_configs,
                    fidelity_evaluator=fidelity_evaluator, dac_model=dac_model,
                    prefix=prefix, n_plot=_npanel,
                    metrics_seed=metrics_seed,
                )
            except Exception as _e:
                # Same contract as the test panels: reported, never fatal.
                print(f"    [probe] SKIPPED at step {step}: "
                      f"{type(_e).__name__}: {_e}")
            if device == "cuda":
                torch.cuda.empty_cache()

    # ===== THE UNCOND CARDS =====
    # sampling.n_audio_samples generations without conditions, from one noise
    # stream -- the same cards, from the same noise, that every intervals.audio
    # rewrites between the metrics steps of a training.
    generate_and_log_uncond_cards(
        model=model, normalizer=normalizer, n_frames=n_frames, step=step,
        writer=writer, device=device, output_dir=output_dir,
        n_cards=int(sampling_cfg.get("n_audio_samples", 4) or 0),
        sampling_cfg=sampling_cfg, use_amp=use_amp,
        frame_dims=frame_dims, global_configs=global_configs,
        metrics_seed=metrics_seed, prefix=prefix)

    # dac_model is the shared singleton (get_dac) -> do NOT delete it; just free
    # any CUDA scratch.
    if device == "cuda":
        torch.cuda.empty_cache()


# ======================
# METRICS EVALUATION (conditioned generation, FD-DAC + KL + the table)
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
    probe_sets=None, image_root=None, test_dataset=None,
    tag_prefix="Validation", split_name="validation", report=None,
    dist_info=None,
):
    """
    The metrics step, on a FIXED subset of `val_dataset` (deterministic linspace
    indices, so the numbers are comparable across steps). In the training that
    is the validation set; test_cond.py calls this same function with the TEST
    set, `tag_prefix="Test"` (Test/Metrics/*, Test/Condition_influence),
    and `split_name="test"` (the rows of the table read `<cond>_test`).
    `report`, when a dict, is filled with every number of the step (test_cond
    writes it to a JSON file).

    MULTI-GPU (`dist_info` with world > 1): EVERY rank calls this function.
    Position j of the n_samples is generated, decoded and measured by rank
    j % N, from the same starting noise it has on one GPU (each rank draws the
    whole noise stream in order and keeps its own positions' draws). The
    latent sums, the VGGish sums and the per-sample values are gathered on
    rank 0, which computes FD-DAC / KL / FAD and the table and writes them;
    each rank writes the dumps of its own positions; the listening panels are
    rank 0's alone. With one process (dist_info None) nothing changes.

      1. UNCONDITIONAL generation  -> Fd_dac_uncond / Kl_uncond
         Generated with NULL conditions (no CFG); scored only with
         sampling.metrics_uncond. The ONLY metrics that are apples-to-apples
         comparable with the unconditional model.

      2. CONDITIONAL generation    -> Fd_dac_cond / Kl_cond (and FAD)
         Each sample generated from one specific validation condition (with CFG
         guidance). Distributional fidelity of the conditioned generations to
         the real data. NOT comparable with the unconditional model (the
         conditioning restricts the distribution).

      3. CONDITION INFLUENCE       -> Validation/Condition_influence (text)
         On the SAME generations as 2: every active condition is re-extracted
         from each of them and compared with the condition it was given --
         ours or mir_eval's metrics (metrics.influence_family) for the frame
         conditions, a CLAP / CLIP cosine for text / image. The same is
         measured on the NULL generations: same samples, same starting noise,
         no condition, scored against the same targets. ONE panel per metrics
         step: a row per (condition, metric), columns with-cond / null /
         Δ = with-cond - null / valid/used, each a mean over the generations
         measurable on both sides (pair_influence, format_influence_panel;
         format_influence_matrix when sampling.influence_subsets is on).

    Then what is there to be listened to and looked at, and is NOT in the
    table: the TEST panels (run_test_panels), the PROBE panels
    (run_joint_probe) and the 'uncond generation' cards
    (generate_and_log_uncond_cards).

    EACH GENERATION IS DECODED ONCE. The waveform serves the re-extraction,
    the CLAP / CLIP embedding, the VGGish embedding of the FAD and the dump to
    disk, and is then dropped, so the peak memory does not grow with
    n_samples. FD-DAC and KL (both directions, real||gen and gen||real) are
    latent-only and share the SAME real validation latent reference
    (fd_dac_ref_stats) in both the cond and uncond cases. `prefix`
    ("EMA" / "Model") tags the outputs by the generating weights. Returns
    (fd_dac_cond, kl_cond_real_gen, kl_cond_gen_real) for the caller.
    """
    guidance = float(conditioning_cfg.guidance_scale)
    n_frames = val_dataset.n_frames
    total = len(val_dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()
    # Multi-GPU: position j belongs to rank j % N. With one process every
    # position is this one's.
    _di = dist_info if (dist_info is not None and dist_info.enabled) else None
    is_main = _di is None or _di.is_main

    def _mine(j):
        return _di is None or (j % _di.world) == _di.rank

    frame_active = (fidelity_evaluator is not None and fidelity_evaluator.active)
    # The vector the text slot is given: the description's, with
    # sampling.validation_text_from_caption (caption_text_vec_fn).
    _text_vec_for = caption_text_vec_fn(val_dataset.latent_root, sampling_cfg,
                                        global_configs)

    # WHICH global conditions can be MEASURED: one that is active in the run
    # AND has an embedder able to place a generated waveform in the same space
    # as its stored condition. text -> CLAP's audio tower, image -> Wav2CLIP. A
    # global with no embedder still CONDITIONS; its column just reads n/a.
    gsim_names = sorted(c for c in (global_configs or {})
                        if (global_embedders or {}).get(c) is not None)
    scoring_active = frame_active or bool(gsim_names)
    any_cond_active = bool(frame_dims) or bool(global_configs)

    # The generations dumped to disk at every metrics step
    # (runs/<run>/audio/step_<step>/generation_<i>/): the first n_val_save of
    # the list. Their REAL latent is captured during the generation pass, so
    # every dump carries the recording it was conditioned from.
    n_val_save = int(getattr(sampling_cfg, "n_val_save", 8) or 8)
    dump_ids = list(range(min(n_val_save, int(n_samples))))
    dump_set = set(dump_ids)

    # ---- CONDITION SUBSETS ----
    # Each subset is a full extra generation pass over n_samples, measured like
    # the reference one, so the cost of the metrics step is linear in how many
    # are asked for. Its columns sit in the same table, headed by its label.
    subset_specs = []
    if scoring_active and frame_active:
        subset_specs = resolve_influence_subsets(
            sampling_cfg.get("influence_subsets", None), list(frame_dims or {}))

    ref_frames = (fd_dac_ref_stats["n_total"]
                  if fd_dac_ref_stats is not None else "n/a")
    print(f"\n  Compute metrics @ step {step}: {n_samples} generations "
          f"(cond guidance={guidance}"
          f"{', + uncond' if compute_uncond else ''}) "
          f"vs reference ({ref_frames} frames)...")

    # The cross-attention context of a batch of samples. None on a model
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
        for measuring stay the FULL set either way -- measuring a condition that
        was not given is exactly how its side effects show up."""
        lat_list = []
        targets = []        # paired frame conditions (cpu numpy); cond only
        real_frames = {}    # real latents of the dumped generations
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
            if not _mine(j):
                # Another rank's sample. Its noise is drawn all the same, with
                # the shape euler_sample_cfg draws it, so every later sample
                # gets the draw it has on one GPU.
                if gen_rng is not None:
                    torch.randn(1, n_frames, TOKEN_DIM, device=device,
                                generator=gen_rng)
                lat_list.append(None)
                if conditioned:
                    targets.append(None)
                    global_targets.append(None)
                continue
            (frames_real, frame_cond_real, _lab, text_emb, image_emb,
             text_ctx) = val_dataset[idx]
            text_emb = _text_vec_for(val_dataset, idx, text_emb)
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
                if j in dump_set:
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

    def _generate_paired_dist(spf):
        """_generate_paired on several GPUs: the same samples from the same
        noise, but this rank generates only its own positions, `spf` at a
        time. The whole noise stream is drawn here in position order, one draw
        per sample as euler_sample_cfg_paired draws it, and only this rank's
        draws are kept. The lists are aligned on the positions, with None where
        the sample is another rank's."""
        n = len(indices)
        cond_list, unc_list = [None] * n, [None] * n
        targets, real_frames, global_targets = [None] * n, {}, [None] * n
        gen_rng = None
        if metrics_seed is not None:
            gen_rng = torch.Generator(device=device)
            gen_rng.manual_seed(int(metrics_seed))
        x0_of = {}
        for j in range(n):
            x = torch.randn(n_frames, TOKEN_DIM, device=device, generator=gen_rng)
            if _mine(j):
                x0_of[j] = x
        own = [j for j in range(n) if _mine(j)]
        for start in range(0, len(own), spf):
            grp = own[start:start + spf]
            fcs, gcs, ctxs = [], [], []
            for j in grp:
                idx = indices[j]
                (frames_real, frame_cond_real, _lab, text_emb, image_emb,
                 text_ctx) = val_dataset[idx]
                text_emb = _text_vec_for(val_dataset, idx, text_emb)
                fcs.append(frame_cond_real)
                gcs.append((text_emb, image_emb))
                ctxs.append(text_ctx)
                targets[j] = {k: v.cpu().numpy() for k, v in frame_cond_real.items()}
                global_targets[j] = {
                    "text":  text_emb.cpu().numpy()  if "text"  in global_configs else None,
                    "image": image_emb.cpu().numpy() if "image" in global_configs else None,
                }
                if j in dump_set:
                    real_frames[j] = frames_real
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
                x0=torch.stack([x0_of.pop(j) for j in grp]),
            )
            for j, c, u in zip(grp, gen_c, gen_u):
                cond_list[j], unc_list[j] = c, u
        return cond_list, targets, real_frames, global_targets, unc_list

    def _generate_paired(spf):
        """Fused cond+uncond generation: each call to the paired sampler handles
        `spf` samples at once (batch 3*spf) and yields BOTH their conditioned and
        unconditional latents from the same x0. Only valid when CFG applies
        (guidance>1 and conditions present); the caller gates on that. Collects
        the same cond-side extras (targets / real_frames / global_targets) as the
        conditioned _generate."""
        if _di is not None:
            return _generate_paired_dist(spf)
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
                text_emb = _text_vec_for(val_dataset, idx, text_emb)
                fcs.append(frame_cond_real)
                gcs.append((text_emb, image_emb))
                ctxs.append(text_ctx)
                targets.append({k: v.cpu().numpy() for k, v in frame_cond_real.items()})
                global_targets.append({
                    "text":  text_emb.cpu().numpy()  if "text"  in global_configs else None,
                    "image": image_emb.cpu().numpy() if "image" in global_configs else None,
                })
                if j in dump_set:
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

    has_ref = fd_dac_ref_stats is not None

    # ===== CONDITIONAL generation =====
    # Distributional metrics (FD-DAC + KL both directions) are latent-only and
    # share the SAME real reference (fd_dac_ref_stats).
    # `sampling.metrics_samples_per_forward` (spf) is the ONE knob for the metrics
    # generation, and it maps directly onto VRAM:
    #   0 -> do not fuse: reference serial path (lowest peak, slowest)
    #   1 -> fuse the 3 CFG branches of 1 sample   -> batch 3
    #   N -> fuse N samples                        -> batch 3N (fastest, highest peak)
    # The 3 is structural (conditioned / cfg-null / unconditional are all required
    # by the CFG math), so it is never a tunable. Fusing needs a CFG to fuse, so
    # the serial path also runs automatically when guidance <= 1 or no condition
    # is active (pure-unconditional run).
    spf = int(sampling_cfg.get("metrics_samples_per_forward", 1))
    # The fused sampler hands over the generations WITHOUT conditions for free
    # (the unconditional velocity is computed at every step anyway: that IS the
    # CFG math), from the same x0 as the conditioned ones. They are the null
    # column of the table, and feed sampling.metrics_uncond and the uncond.wav
    # of the dump.
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
    if _di is not None:
        # Spread over the ranks: the sums of every rank's latents, added up.
        _m = (dist_dac_metrics(cond_lat, fd_dac_ref_stats, metrics_enabled,
                               device, _di) if has_ref else {})
        fd_dac_cond = _m.get("fd_dac")
        kl_cond = {"kl_real_gen": _m.get("kl_real_gen"),
                   "kl_gen_real": _m.get("kl_gen_real")}
    else:
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
    fd_dac_uncond = None
    kl_uncond = {"kl_real_gen": None, "kl_gen_real": None}
    unc_lat = _unc_pre if _unc_pre is not None else []
    # The serial sampler (metrics_samples_per_forward=0, or no CFG) produces no
    # generation without conditions: it is made here, from the same metrics
    # seed, when the table needs its null column or the uncond metrics ask for
    # it. NOT gated on metrics_uncond alone: that would silently empty the null
    # and Δ columns whenever the serial path runs.
    need_null = scoring_active and any_cond_active
    if not unc_lat and (compute_uncond or need_null):
        unc_lat = _generate(conditioned=False)[0]
    if compute_uncond and unc_lat and _di is not None:
        if has_ref:
            _mu = dist_dac_metrics(unc_lat, fd_dac_ref_stats, metrics_enabled,
                                   device, _di)
            fd_dac_uncond = _mu.get("fd_dac")
            kl_uncond = {"kl_real_gen": _mu.get("kl_real_gen"),
                         "kl_gen_real": _mu.get("kl_gen_real")}
        if device == "cuda":
            torch.cuda.empty_cache()
    elif compute_uncond and unc_lat:
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

    # ===== AUDIO: every generation decoded ONCE, streamed, memory-flat =====
    # FD-DAC/KL above are latent-only. Everything else needs the audio: the
    # re-extraction (CREPE / beat_this / crema / chroma), the CLAP / CLIP
    # embedding, the VGGish embedding of the FAD, the dump. One loop serves
    # them all from the same decode -- decode ONE generation, hand it to each
    # consumer that wants that position, keep it only if it is dumped, drop it.
    # Peak RAM is therefore independent of how many samples are measured; the
    # cost of measuring all of them is TIME (one extractor pass per generation).
    dac_model = get_dac()    # load-once singleton (shared across the whole run)

    fad_on = fad_embedder is not None and fad_ref_stats is not None
    n_fad_use = min(int(n_fad or 0), len(cond_lat)) if fad_on else 0
    fad_pos = (set(torch.linspace(0, len(cond_lat) - 1, n_fad_use)
                   .round().long().tolist()) if n_fad_use > 0 else set())

    def _stream_audio(lat_list, measure, fad, keep, desc):
        """Decode, ONCE each, every generation of `lat_list` that something
        needs: `measure` -> all of them are re-extracted and embedded (the
        table); `fad` -> the FAD positions feed the VGGish running sums;
        `keep` -> the dumped positions are returned. Returns (per-sample
        values, coverage, global cosines, kept waveforms, FAD sums)."""
        n_lat = len(lat_list)
        need = set(range(n_lat)) if measure else set()
        if fad:
            need |= {p for p in fad_pos if p < n_lat}
        if keep:
            need |= {p for p in dump_ids if p < n_lat}
        need = {p for p in need if _mine(p)}      # multi-GPU: this rank's only
        if measure and frame_active:
            fidelity_evaluator.reset()
            fidelity_evaluator.keep_contours_for(())
        gsims = {c: {} for c in gsim_names} if measure else {}
        kept = {}
        # The VGGish statistics as running sums, with exactly the operations,
        # the dtype and the order of metrics._audio_clips_to_mu_sigma: the FAD
        # value is the one the separate pass used to give.
        fst = {"sx": None, "sxx": None, "n": 0}
        for i in tqdm(sorted(need), desc=desc, leave=False,
                      disable=not is_main):
            wav = decode_frames_to_wav(lat_list[i], normalizer, dac_model)
            if measure:
                wn = wav.numpy()
                if frame_active:
                    # sample_id=i: the generation's position in the list.
                    fidelity_evaluator.add_sample(
                        wn, DAC_SAMPLE_RATE, n_frames, cond_targets[i],
                        sample_id=i)
                # Every measurable global, the same way: embed the generation
                # into that condition's space and take the cosine with the
                # condition it was given.
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
            if fad and i in fad_pos:
                e = fad_embedder.embed(wav.view(1, 1, -1), DAC_SAMPLE_RATE).to(
                    device=fad_device, dtype=torch.float64)
                if fst["sx"] is None:
                    d = e.shape[-1]
                    fst["sx"] = torch.zeros(d, dtype=torch.float64, device=fad_device)
                    fst["sxx"] = torch.zeros(d, d, dtype=torch.float64,
                                             device=fad_device)
                fst["sx"] = fst["sx"] + e.sum(dim=0)
                fst["sxx"] = fst["sxx"] + e.T @ e
                fst["n"] = fst["n"] + e.shape[0]
                del e
            if keep and i in dump_set:
                kept[i] = wav
            # else: wav is dropped here -> RAM stays flat
        if _di is not None:
            # Multi-GPU: rank 0 collects what every rank measured, so that its
            # evaluator, its cosines and its FAD sums end up as if it had
            # measured every position itself. The same calls on every rank.
            if measure and frame_active:
                _st = dist_all_gather_object(
                    fidelity_evaluator.export_state(), _di)
                if is_main:
                    for _r, _s in enumerate(_st):
                        if _r != _di.rank:
                            fidelity_evaluator.merge_state(_s)
            if measure:
                for _r, _g in enumerate(dist_all_gather_object(gsims, _di)):
                    if is_main and _r != _di.rank:
                        for _c, _d in _g.items():
                            gsims.setdefault(_c, {}).update(_d)
            if fad:
                _mine_f = (None if fst["sx"] is None else
                           (fst["sx"].cpu(), fst["sxx"].cpu(), fst["n"]))
                _all_f = dist_all_gather_object(_mine_f, _di)
                if is_main:
                    _got = [f for f in _all_f if f is not None]
                    if _got:
                        fst = {"sx": sum(f[0] for f in _got).to(fad_device),
                               "sxx": sum(f[1] for f in _got).to(fad_device),
                               "n": sum(f[2] for f in _got)}
            if not is_main:
                return ({}, {}, {}, kept, {"sx": None, "sxx": None, "n": 0})
        # PER-SAMPLE values and their coverage: a mean is only meaningful with
        # the count of generations that reached it.
        per = fidelity_evaluator.per_sample() if (measure and frame_active) else {}
        cov = fidelity_evaluator.coverage() if (measure and frame_active) else {}
        return per, cov, gsims, kept, fst

    print("    decoding each generation once: "
          + (f"all {len(cond_lat)} measured for the table" if scoring_active
             else "nothing to measure")
          + (f", {len(fad_pos)} for FAD" if fad_pos else "")
          + "...")
    per_cond, cov_cond, gsim_cond, cond_wavs, fad_c = _stream_audio(
        cond_lat, measure=scoring_active, fad=fad_on, keep=True,
        desc="metrics: decode + measure")
    unc_wavs, fad_u = {}, None
    # The null side of the table: the generations without conditions, measured
    # against the SAME targets (cond_targets / cond_globals of the same index).
    per_null, gsim_null, have_null = {}, {}, False
    if unc_lat:
        if not any_cond_active:
            # Pure-unconditional run: unc_lat IS cond_lat (reused above), so the
            # decoded waveforms and the FAD sums are identical -- reuse them
            # instead of running the DAC decoder twice over the same latents.
            unc_wavs, fad_u = cond_wavs, fad_c
        else:
            per_null, _, gsim_null, unc_wavs, fad_u = _stream_audio(
                unc_lat, measure=need_null, fad=(fad_on and compute_uncond),
                keep=True,
                desc=("metrics: decode + measure (no conditions)" if need_null
                      else "metrics: decode (no conditions)"))
            have_null = need_null

    # ===== PER-SUBSET GENERATIONS (condition-combination rows) =====
    # One extra generation pass per subset, measured like the reference one.
    # The "all" subset is not regenerated: the conditioned pass above already
    # IS it, bit for bit, so its row reuses that pass. Latents are dropped as
    # soon as a subset has been measured, so peak memory does not grow with the
    # number of subsets (only the time does).
    subset_entries = []          # (label, condition names, per-sample, coverage, globals)
    _full = tuple(frame_dims or {})
    for _lab, _names in subset_specs:
        if _names == _full:
            subset_entries.append((_lab, _names, per_cond, cov_cond, gsim_cond))
            continue
        print(f"    subset '{_lab}' [{'+'.join(_names)}]: "
              f"{n_samples} generations...")
        _slat = _generate(conditioned=True, subset=set(_names))[0]
        _sper, _scov, _sg, _, _ = _stream_audio(
            _slat, measure=True, fad=False, keep=False,
            desc=f"metrics: subset {_lab}")
        subset_entries.append((_lab, _names, _sper, _scov, _sg))
        del _slat
        if device == "cuda":
            torch.cuda.empty_cache()

    # ===== FAD (VGGish), from the sums accumulated while streaming =====
    # Latent-only metrics (FD-DAC / KL) score the DAC latent space; the FAD
    # scores the AUDIO, through an embedder trained on real recordings, which
    # is what the controllable-music literature reports. sampling.n_fad_samples
    # says how many generations it is computed on.
    fad_cond = fad_uncond = None
    if fad_c is not None and fad_c["n"] > 0:
        _mu, _sig, _ = compute_mu_sigma(fad_c["sx"], fad_c["sxx"], fad_c["n"])
        fad_cond = compute_fad(_mu, _sig, fad_ref_stats, device=fad_device)
        print(f"    FAD-VGGish cond: {len(fad_pos)} clips -> {fad_c['n']} "
              f"embedding vectors (128-D)")
        if compute_uncond and unc_lat:
            if not any_cond_active:
                # pure-unconditional run: the two lists ARE the same latents
                fad_uncond = fad_cond
            elif fad_u is not None and fad_u["n"] > 0:
                _mu, _sig, _ = compute_mu_sigma(fad_u["sx"], fad_u["sxx"],
                                                fad_u["n"])
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

    def _no_metric_free(d):
        return {k: v for k, v in (d or {}).items()
                if not k.endswith("/<no metric>")}

    influence, cov_paired = {}, {}
    if frame_active and scoring_active:
        influence, cov_paired = pair_influence(
            _no_metric_free(per_cond), _no_metric_free(per_null),
            coverage_cond=_no_metric_free(cov_cond), have_null=have_null)

    # ---- the GLOBAL conditions' rows: one paired scalar each ----
    # text  -> cosine in CLAP's space (its audio tower embeds the generation,
    #          the stored condition is a CLAP vector of the source chunk)
    # image -> cosine in CLIP's space (Wav2CLIP embeds the generation, the
    #          stored condition is the CLIP vector of the picture)
    # Both are the SAME shape of number as an f0 correlation: with-cond, null,
    # and the delta between them on the same samples -- so they sit in the same
    # table and are read the same way.
    _gmetric = {"text": "clap_sim", "image": "clip_sim"}

    def _global_rows(gsims, into_inf, into_cov):
        """Add one paired row per scorable global condition to (influence,
        coverage). Used for the reference pass AND for every subset, so a
        subset's table carries text/image exactly like the frame conditions."""
        for c in gsim_names:
            _gc = (gsims or {}).get(c, {})
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

    if scoring_active:
        _global_rows(gsim_cond, influence, cov_paired)

    # Each subset is paired against the SAME null pass as the reference above,
    # so every row of the matrix is a delta over one shared baseline and the
    # rows can be read against each other.
    subset_tables = []
    for _lab, _names, _sps, _scov, _sgsim in subset_entries:
        _sinf, _scovp = pair_influence(
            _no_metric_free(_sps), _no_metric_free(per_null),
            coverage_cond=_no_metric_free(_scov), have_null=have_null)
        _global_rows(_sgsim, _sinf, _scovp)
        subset_tables.append((_lab, _sinf, _scovp))

    # A global that is CONDITIONING the run but cannot be scored (no embedder
    # installed) keeps a row saying exactly that, so its absence from the table
    # is never mistaken for an influence of zero.
    if scoring_active:
        for c in (global_configs or {}):
            if c in influence or c in gsim_names:
                continue
            influence[c] = {_gmetric.get(c, "sim"): {
                "cond": None, "null": None, "delta": None,
                "note": ("no audio->CLIP embedder: pip install wav2clip"
                         if c == "image" else "no embedder for this condition"),
            }}

    # ONE panel per metrics step: a row per (condition, metric), columns
    # with-cond / null / Δ / valid/used. Logged as text at `step`, so the step
    # slider walks the panel across training. The rows say
    # `<cond>_validation` (`<cond>_test` in test_cond.py): they are measured on
    # that split. Display only: the dicts above keep the plain name, which is
    # what the subset matrix matches its labels against.
    def _row_label(name):
        return f"{name}_{split_name}"

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
        if subset_tables:
            # Subsets requested: the panel becomes the delta MATRIX (one row per
            # combination) with the detailed tables underneath. The single-table
            # form below is what a run with no subsets keeps.
            panel_md = format_influence_matrix(
                subset_tables, step=step, prefix=prefix,
                guidance=guidance, n_samples=int(n_samples),
                # The subsets vary the FRAME conditions only: text and image are
                # handed to the model in every row, so their cells are never a
                # side effect and must not be marked as one.
                always_given=set(gsim_names),
            )
        else:
            panel_md = format_influence_panel(
                _inf_named, step=step, prefix=prefix,
                guidance=guidance,
                n_samples=int(n_samples),
                coverage=_cov_named,
            )
        writer.add_text(f"{tag_prefix}/Condition_influence", panel_md, step)
        # Log the explanatory legend ONCE per board, on its own tag. Pinned to
        # step 0 so it reads as a one-time preamble, not tied to a metrics step
        # (TensorBoard text always carries SOME step; 0 is the most neutral).
        _legend_tag = f"{tag_prefix}/Condition_influence_legend"
        if fixed_card_pending(writer, _legend_tag):
            writer.add_text(_legend_tag, format_influence_legend(), 0)

    # ===== LOG SCALARS (FD-DAC, KL, FAD; every conditioning measure -> the table) =====
    # The tag scheme follows the RUN MODE, so each dashboard matches its project:
    #   * conditioned run  -> this project's two-axis scheme (Fd_dac_cond vs
    #     Fd_dac_uncond, Kl_cond/* vs Kl_uncond/*): the comparison is the point.
    #   * pure-unconditional run (no conditions active) -> there is only ONE
    #     distribution to score (cond and uncond generations are literally the
    #     same samples), so log a SINGLE axis under the EXACT tags of the
    #     unconditional project: Fd_dac / Kl_real_gen / Kl_gen_real.
    # The text / image similarity is NOT a curve here: since 2 Oct 2026 it is a
    # column of the table, over the same generations as everything else.
    _m = f"{tag_prefix}/Metrics"
    if any_cond_active:
        if fd_dac_cond is not None:
            writer.add_scalar(f"{_m}/Fd_dac_cond", fd_dac_cond, step)
        if kl_cond["kl_real_gen"] is not None:
            writer.add_scalar(f"{_m}/Kl_cond/real_gen",
                              kl_cond["kl_real_gen"], step)
            writer.add_scalar(f"{_m}/Kl_cond/gen_real",
                              kl_cond["kl_gen_real"], step)
        if fd_dac_uncond is not None:
            writer.add_scalar(f"{_m}/Fd_dac_uncond", fd_dac_uncond, step)
        if kl_uncond["kl_real_gen"] is not None:
            writer.add_scalar(f"{_m}/Kl_uncond/real_gen",
                              kl_uncond["kl_real_gen"], step)
            writer.add_scalar(f"{_m}/Kl_uncond/gen_real",
                              kl_uncond["kl_gen_real"], step)
        if fad_cond is not None:
            writer.add_scalar(f"{_m}/Fad_vggish_cond", fad_cond, step)
        if fad_uncond is not None:
            writer.add_scalar(f"{_m}/Fad_vggish_uncond", fad_uncond, step)
    else:
        # unconditional run: fd_dac_cond/kl_cond ARE the unconditional numbers
        if fd_dac_cond is not None:
            writer.add_scalar(f"{_m}/Fd_dac", fd_dac_cond, step)
        if kl_cond["kl_real_gen"] is not None:
            writer.add_scalar(f"{_m}/Kl_real_gen",
                              kl_cond["kl_real_gen"], step)
            writer.add_scalar(f"{_m}/Kl_gen_real",
                              kl_cond["kl_gen_real"], step)
        if fad_cond is not None:
            writer.add_scalar(f"{_m}/Fad_vggish", fad_cond, step)

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
        # One line per (condition, metric): with-cond / null / delta and how
        # many samples the three are paired over, out of how many were
        # measured. ASCII only: a Windows console (cp1252) cannot print a
        # Greek delta.
        print("  [influence] with-cond / null / delta (paired/measured)")
        for cname, metrics in _inf_named.items():
            for m, vals in metrics.items():
                cv = _cov_named.get(f"{cname}/{m}", {})
                d = vals.get("delta")
                print(f"    {cname}/{m}: {_f(vals.get('cond'))} / "
                      f"{_f(vals.get('null'))} / "
                      + (f"{d:+.4f}" if d is not None else "n/a")
                      + f"  ({cv.get('valid', 0)}/{cv.get('attempted', 0)})")

    # Every number of the step, for a caller that keeps them (test_cond.py's
    # JSON). The training passes nothing.
    if report is not None:
        report.update({
            "n_generations": int(n_samples),
            "fd_dac_cond": fd_dac_cond,
            "kl_cond_real_gen": kl_cond["kl_real_gen"],
            "kl_cond_gen_real": kl_cond["kl_gen_real"],
            "fd_dac_uncond": fd_dac_uncond,
            "kl_uncond_real_gen": kl_uncond["kl_real_gen"],
            "kl_uncond_gen_real": kl_uncond["kl_gen_real"],
            "fad_vggish_cond": fad_cond,
            "fad_vggish_uncond": fad_uncond,
            "influence": _inf_named,
            "influence_coverage": _cov_named,
        })

    # ===== SAVE THE SCORED GENERATIONS TO DISK (one dir per step, one sub-dir per generation) =====
    # output_dir/step_{step}/generation_{i}/ contains, for generation i:
    #   conditions.npz   - the EXACT input conditions used (f0, energy, ...)
    #   cond_{name}.wav  - AUDIBLE rendering of each condition (f0 as a sine
    #                      contour, energy as an amplitude-modulated tone) so one
    #                      can hear how it maps into the conditioned audio
    #   cond.wav         - the conditioned generation
    #   uncond.wav       - the generation without conditions, same index (when
    #                      the step produced one)
    #   real.wav         - the recording the conditions were extracted from
    # How many generations are dumped is sampling.n_val_save (the first ones of
    # the list).
    def _to_np(wav):
        return wav.numpy() if wav.dim() == 1 else wav.squeeze().numpy()

    from condition_metrics import sonify_condition

    step_dir = os.path.join(output_dir, f"step_{step:07d}")

    def _gen_dir(i):
        d = os.path.join(step_dir, f"generation_{i:03d}")
        os.makedirs(d, exist_ok=True)
        return d

    save_ids = [i for i in dump_ids if i in cond_wavs]   # multi-GPU: this rank's
    for i in save_ids:
        gdir = _gen_dir(i)
        if i in real_frames:
            sf.write(os.path.join(gdir, "real.wav"),
                     _to_np(decode_frames_to_wav(real_frames[i], normalizer,
                                                 dac_model)),
                     DAC_SAMPLE_RATE)
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
    _n_saved = (int(dist_sum([len(save_ids)], _di)[0]) if _di is not None
                else len(save_ids))
    print(f"    saved {_n_saved} {split_name} generations "
          + ("(per-generation dirs: cond+uncond+conditions+sonified+real) to "
             if any_cond_active else "(per-generation dirs: generated+real) to ")
          + step_dir)

    # ===== WHAT IS THERE TO LISTEN TO AND LOOK AT (not in the table) =====
    # AFTER the table row has been read out of the evaluator (per_sample /
    # coverage return copies), because the panels reset the same evaluator to
    # re-extract their curves. Multi-GPU: rank 0 alone.
    if not is_main:
        return fd_dac_cond, kl_cond["kl_real_gen"], kl_cond["kl_gen_real"]
    log_listening_panels(
        model=model, normalizer=normalizer, step=step, writer=writer,
        device=device, output_dir=output_dir, sampling_cfg=sampling_cfg,
        conditioning_cfg=conditioning_cfg, use_amp=use_amp,
        frame_dims=frame_dims, global_configs=global_configs,
        fidelity_evaluator=fidelity_evaluator, test_dataset=test_dataset,
        probe_sets=probe_sets, n_frames=n_frames, prefix=prefix,
        metrics_seed=metrics_seed, image_root=image_root,
        text_vec_for=_text_vec_for)

    # Backward-compatible return: the conditioned FD-DAC + KL (both directions).
    return fd_dac_cond, kl_cond["kl_real_gen"], kl_cond["kl_gen_real"]


# ======================
# SETUP SHARED WITH test_cond.py: what measures, and the probe banks
# ======================
def read_influence_settings(metrics_cfg):
    """(influence_family, mir_threshold) from metrics.*: which metrics are the
    columns of the Condition_influence table (ours or mir_eval's) and the
    value at which a chroma / chord pitch class counts as ON for the mir
    metrics. Checked at startup, so a misspelt value stops the run before the
    references are built; .get() keeps a config written before the options
    existed on ours / 0.5. Shared with test_cond.py."""
    from condition_metrics import INFLUENCE_FAMILIES
    influence_family = str(
        metrics_cfg.get("influence_family", "influence_metrics")
        if metrics_cfg is not None else "influence_metrics")
    if influence_family not in INFLUENCE_FAMILIES:
        raise SystemExit(
            f"[metrics] metrics.influence_family='{influence_family}' is not a "
            f"valid choice. Use 'influence_metrics' (ours) or "
            f"'mir_influence_metrics' (mir_eval, for the conditions it covers).")
    mir_threshold = float(
        metrics_cfg.get("mir_threshold", 0.5)
        if metrics_cfg is not None else 0.5)
    if not 0.0 < mir_threshold <= 1.0:
        raise SystemExit(
            f"[metrics] metrics.mir_threshold={mir_threshold} is not in (0, 1]: "
            f"it is a fraction of the frame's loudest chroma class, and a "
            f"crema probability for chord.")
    return influence_family, mir_threshold


def build_condition_scorers(cfg, frame_dims, global_configs, registry,
                            influence_family, mir_threshold):
    """-> (fidelity_evaluator, global_embedders): what MEASURES the
    conditions of a generation -- the re-extraction of the frame conditions
    and one embedder per scorable global condition. Built here, the same way,
    for the training and for test_cond.py, so a test number is measured
    exactly like its validation twin."""
    # Conditioning-influence evaluators (evaluation only, never affect training):
    #   - frame conditions: re-extract the enabled frame conditions from the
    #     generations and compare them, sample by sample, with the input ones
    #     (f0 corr, chroma cosine, rhythm/energy correlation -- or mir_eval's
    #     metrics for the conditions it covers, with metrics.influence_family).
    #   - text (CLAP): the audio side of the same CLAP checkpoint, to score
    #     audio<->text adherence; loaded lazily ONLY if 'text' is active.
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
        enabled_frame=list(frame_dims.keys()),
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
    # the two can be compared: that cosine is what the table's text / image
    # column is made of. The dict is the whole extension point -- a third
    # global condition needs an entry here and nothing else.
    global_embedders = {}
    if "text" in global_configs:
        from conditions import ClapAudioEmbedder
        # match the CLAP checkpoint used by the text condition
        clap_model_name = CONDITION_CONFIG["global"]["text"]["kwargs"].get(
            "model_name", "laion/clap-htsat-unfused")
        global_embedders["text"] = ClapAudioEmbedder(model_name=clap_model_name,
                                                     device=_fid_device)
        print(f"Text-influence (CLAP audio) enabled: {clap_model_name}")
    if "image" in global_configs:
        # Wav2CLIP: audio distilled INTO CLIP's space, which is the only way an
        # audio-vs-image cosine exists at all (CLIP and CLAP are different
        # spaces). Optional: without it the image still CONDITIONS the model,
        # its column in the table just reads n/a -- so a missing package
        # degrades the diagnostics, never the training.
        try:
            from conditions import Wav2ClipAudioEmbedder
            _w2c = Wav2ClipAudioEmbedder(device=_fid_device)
            _w2c._load()                       # fail now, not mid-metrics-step
            global_embedders["image"] = _w2c
            print("Image-influence (Wav2CLIP audio->CLIP space) enabled")
        except Exception as _e:
            print(f"Image-influence DISABLED: {type(_e).__name__}: {_e}")
    return fidelity_evaluator, global_embedders


def build_probe_sets(cfg, registry, fidelity_evaluator, dataset,
                     frame_dims, global_configs):
    """{condition: ConditionProbeSet} for every condition active in the
    model, sampling.n_probe_panels stimuli each, built (or read from the
    cache under paths.cache_dir) the same way for the training and for
    test_cond.py -- so the test shows the very probe panels the training
    shows. `dataset` gives the chunk geometry and the caption sidecar."""
    # ---- OUT-OF-THE-BOX PROBE SETS (proof of concept) ----
    # A bank is built for EVERY condition active in this run, and the panels
    # then drive them all together (see run_joint_probe). Each bank is built
    # ONCE and cached (keyed by the stimuli, the chunk geometry and the
    # extractor's parameters), the same contract as the normalizer and the
    # FD-DAC reference -- shared by every run over the same setup, rebuilt
    # automatically if any of those change.
    #
    # This is the CONTROLLABILITY instrument. On unambiguous synthetic stimuli
    # it lets one HEAR and SEE whether the conditioning of this run moves the
    # generation at all -- which real material cannot show on its own, because
    # there a condition is often not cleanly extractable. It costs one
    # generation per panel, whatever the number of conditions, because the
    # panel drives them jointly. Nothing measured on it is in the table.
    #
    # sampling.n_probe_panels: that many stimuli per bank, all of them plotted
    # and played. Fewer than a bank holds -> build_condition_probe_set picks a
    # FIXED random subset, the same at every step and in every run; more -> the
    # whole bank, said so. 0 -> no probe at all.
    probe_sets = {}
    _n_probes = panel_count(cfg.sampling, "probe")
    # RNG GUARD. In the training the probe banks are built before
    # ConditionedAudioDiT is constructed, and building the f0 bank runs torchcrepe, which
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
    _probe_names = list(frame_dims) + list(global_configs)
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
            # orders of magnitude below anything the probe images resolve.
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
            if _cond in global_configs:
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
                    load_caption_table(dataset.latent_root),
                    n_panels=_n_probes)
                print(f"[text-probe] {_why}")
            _ps = build_condition_probe_set(
                _cond, _probe_root, dataset.n_frames,
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
    # (or did not draw) is rolled back, so the training's model init, which
    # comes after, sees the state left by the seeding block and nothing else.
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
    return probe_sets


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
def infinite_loader(loader, sampler=None, first_epoch=0):
    """Batches forever. Multi-GPU: `sampler` is the DistributedSampler, told
    the epoch at every pass so each pass is a new permutation (without
    set_epoch every epoch would repeat the first); `first_epoch` = the resume
    step, so a resumed run does not replay epoch 0's order."""
    epoch = int(first_epoch)
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


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
        if dist.is_available() and dist.is_initialized():
            # Multi-GPU: this process's own GPU only. get_rng_state_all would
            # touch every visible GPU, i.e. the other ranks' too.
            state["torch_cuda_current"] = torch.cuda.get_rng_state()
        else:
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
            if dist.is_available() and dist.is_initialized():
                torch.cuda.set_rng_state(state["torch_cuda"][0])
            else:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        elif "torch_cuda_current" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["torch_cuda_current"])
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
# MULTI-GPU (DistributedDataParallel)
# ======================
# One process per GPU, started by `launch_training_cond.py --num-gpus N` (or by
# torchrun): RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT come in
# through the environment. Without them -- `python training_cond.py`, or the
# launcher with one GPU -- init_distributed() returns a single-process DistInfo
# and none of the code below runs: that is the single-GPU training, unchanged.
#
# Rank 0 does some work alone while the others wait for it inside a collective:
# the references at startup, the listening panels of the metrics step, the
# checkpoints. The default NCCL time-out (10 min in recent PyTorch) would stop
# the run there, so it is raised; a rank that really dies is caught by the
# launcher, which stops the others.
DIST_TIMEOUT_MIN = 180


class DistInfo:
    """Where this process sits in the run. world == 1: the single process."""

    def __init__(self, rank=0, world=1, local_rank=0, backend=None):
        self.rank = int(rank)
        self.world = int(world)
        self.local_rank = int(local_rank)
        self.backend = backend

    @property
    def enabled(self):
        return self.world > 1

    @property
    def is_main(self):
        return self.rank == 0


def init_distributed():
    """Join the process group when WORLD_SIZE > 1; otherwise do nothing.

    GPUs: one per rank, NCCL (Linux). Without CUDA the processes run on the
    CPU over gloo -- far too slow to train, it exists to run the multi-process
    code on a machine where DDP cannot use the GPU: on Windows NCCL does not
    exist, and DDP with CUDA tensors over gloo crashes (measured 3 Oct 2026,
    torch 2.11: access violation at the first backward, whatever the DDP
    options). Hide the GPU (CUDA_VISIBLE_DEVICES=-1) to get that mode."""
    world = int(os.environ.get("WORLD_SIZE", "1") or 1)
    if world <= 1:
        return DistInfo()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    timeout = timedelta(minutes=DIST_TIMEOUT_MIN)
    if not torch.cuda.is_available():
        dist.init_process_group(backend="gloo", timeout=timeout)
        return DistInfo(rank, world, 0, "gloo")
    if not dist.is_nccl_available() or sys.platform == "win32":
        raise RuntimeError(
            f"[rank {rank}] multi-GPU training needs NCCL, which this "
            f"PyTorch / platform does not have ({sys.platform}). Run it on a "
            f"Linux server, or with the GPU hidden (CUDA_VISIBLE_DEVICES=-1) to "
            f"test the multi-process code on the CPU.")
    n_dev = torch.cuda.device_count()
    if local_rank >= n_dev:
        raise RuntimeError(
            f"[rank {rank}] LOCAL_RANK={local_rank} but only {n_dev} GPU(s) are "
            f"visible (CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}): one GPU per rank.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timeout)
    return DistInfo(rank, world, local_rank, "nccl")


def dist_barrier(di):
    if not di.enabled:
        return
    if di.backend == "nccl":
        dist.barrier(device_ids=[di.local_rank])
    else:
        dist.barrier()


def dist_all_gather_object(obj, di):
    """[obj of rank 0, obj of rank 1, ...] on every rank."""
    if not di.enabled:
        return [obj]
    out = [None] * di.world
    dist.all_gather_object(out, obj)
    return out


def dist_broadcast_object(obj, di, src=0):
    """Rank `src`'s obj, on every rank."""
    if not di.enabled:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def dist_sum_tensors(tensors, di):
    """The element-wise sum over the ranks of each tensor (new tensors, on the
    device each came from). gloo reduces on the CPU."""
    if not di.enabled:
        return list(tensors)
    out = []
    for t in tensors:
        x = (t.detach().clone() if di.backend == "nccl"
             else t.detach().cpu().clone())
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        out.append(x.to(t.device))
    return out


def dist_sum(values, di):
    """The sum over the ranks of a list of numbers (float64)."""
    if not di.enabled:
        return [float(v) for v in values]
    dev = f"cuda:{di.local_rank}" if di.backend == "nccl" else "cpu"
    t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.cpu().tolist()


def dist_broadcast_tensor_(t, di, src=0):
    """Overwrite `t` in place with rank `src`'s values."""
    if not di.enabled:
        return
    if di.backend == "nccl":
        dist.broadcast(t, src=src)
    else:
        x = t.detach().cpu().clone()
        dist.broadcast(x, src=src)
        t.copy_(x.to(t.device))


def dist_dac_metrics(lat_list, ref_stats, enabled, device, di, block_size=16):
    """compute_dac_metrics for a pass whose generations are spread over the
    ranks (None where a position is another rank's). Each rank sums its own
    latent frames as metrics._generated_mu_sigma does (float64, blocks of
    `block_size` samples), the sums are added up over the ranks, and rank 0
    fits mu / covariance from them and computes the same metrics. Every rank
    must call it; only rank 0 gets numbers, the others a dict of None."""
    out = {"fd_dac": None, "kl_real_gen": None, "kl_gen_real": None}
    want_fd = "fd_dac" in enabled
    want_kl = "kl_dac" in enabled
    if not (want_fd or want_kl):
        return out
    lats = [x for x in lat_list if x is not None]
    sum_x = torch.zeros(TOKEN_DIM, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(TOKEN_DIM, TOKEN_DIM, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, len(lats), block_size):
        block = torch.stack(lats[start:start + block_size])
        block = block.reshape(-1, TOKEN_DIM).to(device=device, dtype=torch.float64)
        sum_x = sum_x + block.sum(dim=0)
        sum_xx = sum_xx + block.T @ block
        count = count + block.shape[0]
        del block
    sum_x, sum_xx, count = dist_sum_tensors([sum_x, sum_xx, count], di)
    if not di.is_main:
        return out
    mu_gen, sigma_gen, _ = compute_mu_sigma(sum_x, sum_xx, count)
    mu_ref = ref_stats["mu"].to(device)
    sigma_ref = ref_stats["sigma"].to(device)
    if want_fd:
        out["fd_dac"] = float(
            compute_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen).item())
    if want_kl:
        out["kl_real_gen"] = gaussian_kl_fullcov(mu_ref, sigma_ref, mu_gen, sigma_gen)
        out["kl_gen_real"] = gaussian_kl_fullcov(mu_gen, sigma_gen, mu_ref, sigma_ref)
    return out


class NullWriter:
    """What every rank but 0 is given instead of a SummaryWriter: it accepts
    every call and writes nothing (log_dir None, so fixed_card_pending keeps
    no file either). TensorBoard is rank 0's."""
    log_dir = None

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    cfg, run_name = load_config()
    # MULTI-GPU: one process per GPU (launch_training_cond.py --num-gpus N, or
    # torchrun). With one process DIST is the single process and changes
    # nothing below.
    DIST = init_distributed()
    if DIST.enabled:
        # A run name made of the start time can differ by a second between the
        # processes: rank 0's is the run's.
        run_name = dist_broadcast_object(run_name, DIST)
        cfg.paths.run_name = run_name
        if not DIST.is_main:
            # The console is rank 0's. Until the run directory exists the
            # others print nothing; then they write runs/<run>/rank<R>.log.
            sys.stdout = open(os.devnull, "w")
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
    if DIST.enabled and not DIST.is_main:
        sys.stdout = open(os.path.join(run_dir, f"rank{DIST.rank}.log"), "w",
                          buffering=1, encoding="utf-8",
                          errors="backslashreplace")

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
    if DIST.enabled and not DIST.is_main:
        dist_barrier(DIST)          # rank 0 checks / creates the cache first
    _validate_cache(cache_dir, _cache_fingerprint(cfg, _n_frames_fp),
                    guarded_files=[normalizer_path, fd_dac_cache_path,
                                   fad_cache_path])
    if DIST.enabled and DIST.is_main:
        dist_barrier(DIST)

    # Config's dump (with CLI override already applied) in the run dir
    config_dump_path = os.path.join(run_dir, "config.yaml")
    if DIST.is_main:
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

    # Which metrics are the columns of the Condition_influence table (ours or
    # mir_eval's) and the mir ON threshold, checked HERE, at startup.
    influence_family, mir_threshold = read_influence_settings(metrics_cfg)

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
    if DIST.enabled and not DIST.is_main:
        dist_barrier(DIST)          # rank 0 fits and saves the normalizer first
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
    if DIST.is_main and not os.path.exists(normalizer_path):
        normalizer.save(normalizer_path)
    if DIST.enabled and DIST.is_main:
        dist_barrier(DIST)

    n_workers = int(cfg.data.get("num_workers", 4))
    train_sampler = None
    if DIST.enabled:
        # Multi-GPU: each rank reads its own share of every epoch's
        # permutation; data.train_batch_size is PER GPU.
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=DIST.world, rank=DIST.rank,
            shuffle=True, seed=int(run_seed if run_seed is not None else 0),
            drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.data.train_batch_size,
            sampler=train_sampler,
            num_workers=n_workers, pin_memory=(device == "cuda"),
            persistent_workers=(n_workers > 0),
            drop_last=True, collate_fn=collate_conditioned,
        )
    else:
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
    if metrics_enabled and not DIST.is_main:
        fd_dac_ref_stats = None     # rank 0's, broadcast below
    elif metrics_enabled:
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
        if not DIST.is_main:
            pass                    # rank 0's reference, broadcast below
        elif _fad_ref_mode == "wav":
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
        if DIST.is_main:
            print(f"FAD reference ready: {fad_ref_stats['n_total']} embedding "
                  f"vectors (128-D)\n")
    if DIST.enabled:
        # The references are small (a mean and a covariance each): every rank
        # gets rank 0's, so all of them take the same decisions in the
        # metrics step (FD-DAC / FAD on or off).
        fd_dac_ref_stats, fad_ref_stats = dist_broadcast_object(
            (fd_dac_ref_stats, fad_ref_stats), DIST)

    # What measures the conditions of a generation (build_condition_scorers):
    # the re-extraction of the frame conditions and one embedder per scorable
    # global. The means over the n_metrics_samples validation generations make
    # the Validation/Condition_influence table: one row per metrics step.
    fidelity_evaluator, global_embedders = build_condition_scorers(
        cfg, FRAME_COND_DIMS, GLOBAL_CONFIGS, registry,
        influence_family, mir_threshold)

    # ---- OUT-OF-THE-BOX PROBE SETS (build_probe_sets) ----
    _n_probes = panel_count(cfg.sampling, "probe")
    if FRAME_COND_DIMS or GLOBAL_CONFIGS:
        _n_test_panels = len(test_panel_indices(
            len(test_dataset), panel_count(cfg.sampling, "test")))
        print(f"[panels] the table: all {int(cfg.sampling.n_metrics_samples)} "
              f"validation generations of the metrics step | to listen to: "
              f"test {_n_test_panels or 'off'}"
              + (" (the test split is empty)"
                 if panel_count(cfg.sampling, "test") and not len(test_dataset)
                 else "")
              + f" | probe {_n_probes if _n_probes > 0 else 'off'}")
    # The probe panels are rank 0's alone (listening material).
    probe_sets = (build_probe_sets(cfg, registry, fidelity_evaluator,
                                   val_dataset, FRAME_COND_DIMS,
                                   GLOBAL_CONFIGS)
                  if DIST.is_main else None)
    if probe_sets:
        print()

    metrics_uncond = bool(cfg.sampling.get("metrics_uncond", True))

    # The 'uncond generation' cards: sampling.n_audio_samples generations
    # without conditions, card k from the k-th draw of one noise stream
    # (generate_and_log_uncond_cards), written by the metrics step and by every
    # intervals.audio in between.
    n_uncond_cards = max(0, int(cfg.sampling.get("n_audio_samples", 4) or 0))
    print(f"Uncond cards: {n_uncond_cards}"
          + (f" (uncond_00..uncond_{n_uncond_cards - 1:02d}), refreshed every "
             f"intervals.audio and every intervals.metrics from the same noise"
             if n_uncond_cards else "")
          + "\n")
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

    # SummaryWriter points directly to the run directory (rank 0's alone)
    writer = SummaryWriter(run_dir) if DIST.is_main else NullWriter()

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
    # MULTI-GPU: the model every rank trains, the EMA, the noise, the data
    # ======================
    train_model = model
    if DIST.enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP
        _on_gpu = (device == "cuda")
        train_model = DDP(
            model,
            device_ids=[DIST.local_rank] if _on_gpu else None,
            output_device=DIST.local_rank if _on_gpu else None,
            gradient_as_bucket_view=True,
            # text_null (cross-attention) enters the graph only when the batch
            # holds a sample whose text was dropped: without this DDP stops at
            # the first batch that holds none.
            find_unused_parameters=bool(getattr(model, "text_cross_layers", None)))
        # DDP has just given every rank rank 0's live weights; the EMA shadow
        # gets rank 0's too (each process built its own random init).
        if ema is not None:
            for _t in list(ema.model.parameters()) + list(ema.model.buffers()):
                dist_broadcast_tensor_(_t.data, DIST)
        # Each rank its own noise (x0, t, CFG dropout): the same seed on every
        # rank would draw the same noise for different samples.
        if run_seed is not None and not DIST.is_main:
            _rs = run_seed + 1000003 * DIST.rank + start_step
            random.seed(_rs)
            np.random.seed(_rs % (2 ** 32))
            torch.manual_seed(_rs)
        train_iter = infinite_loader(train_loader, train_sampler,
                                     first_epoch=start_step)
        print(f"[multi-GPU] rank {DIST.rank} of {DIST.world} on "
              f"{f'cuda:{DIST.local_rank}' if _on_gpu else 'cpu'} | backend "
              f"{DIST.backend} | batch per optimizer step: "
              f"{cfg.data.effective_bs} x {DIST.world} processes = "
              f"{cfg.data.effective_bs * DIST.world}")

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
    # at the TOP of every test block, beside the conditions extracted from it
    # (test_XX/1_real_test_XX); in the unconditioned case n_audio_samples
    # test recordings fill the 'ground truth' group instead (real_test_XX).
    if DIST.is_main:
        log_real_audio_samples(
            test_dataset=test_dataset,
            normalizer=normalizer,
            writer=writer,
            n_samples=cfg.sampling.n_audio_samples,
            sampling_cfg=cfg.sampling,
            frame_dims=FRAME_COND_DIMS,
            global_configs=GLOBAL_CONFIGS,
        )

    # ======================
    # The rest of the listening material is written by evaluate_and_log_metrics
    # at every metrics step: the test and probe blocks (their stimuli once, the
    # generations at every step) and the "uncond generation" cards, which
    # intervals.audio also refreshes in between.

    # ======================
    # TRAIN LOOP
    # ======================
    val_loss = None
    pbar = tqdm(range(start_step, cfg.training.num_steps),
                initial=start_step, total=cfg.training.num_steps,
                desc="Training", unit="step", disable=not DIST.is_main)
    last_step = start_step
    # Real EMA state: True only once the shadow holds TRAINED weights (it is
    # seeded from the live model when ema_start is reached). Persisted in the
    # checkpoint and carried across resumes, so it can never be inferred wrongly.
    ema_ready = _resumed_ema_ready
    # True only when the loop has run to its last step (see the end of the
    # finally block below).
    loop_completed = False

    try:
        for step in pbar:
            last_step = step
            model.train()

            accum_loss = 0.0
            for _micro in range(cfg.data.grad_accum):
                batch = next(train_iter)
                # Multi-GPU: the gradients are averaged over the GPUs once per
                # optimizer step, at the last micro-batch, not at every one.
                _sync = (train_model.no_sync()
                         if DIST.enabled and _micro < cfg.data.grad_accum - 1
                         else nullcontext())
                with _sync:
                    loss = compute_loss(
                        train_model, batch, device,
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

            if DIST.enabled:
                # The loss of the whole optimizer step: the mean over the GPUs.
                accum_loss = dist_sum([accum_loss], DIST)[0] / DIST.world
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
                    for _vbi, (vb, (fixed_x0, fixed_t)) in enumerate(zip(
                            fixed_val_batches, fixed_val_inputs)):
                        # Multi-GPU: batch b is rank b % N's; the sums are
                        # added up below.
                        if DIST.enabled and _vbi % DIST.world != DIST.rank:
                            continue
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

                    if DIST.enabled:
                        val_loss_sum, ema_val_loss_sum, val_sample_count = \
                            dist_sum([val_loss_sum, ema_val_loss_sum,
                                      val_sample_count], DIST)
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
                if check_loss < best_val_loss and not DIST.is_main:
                    best_val_loss = check_loss      # rank 0 writes the file
                elif check_loss < best_val_loss:
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
            # UNCOND PREVIEW every intervals.audio steps: the 'uncond generation'
            # cards only -- the same ones the metrics step writes, by the same
            # function, from the same noise. The test and probe panels are
            # written by the metrics step alone. Skipped when the two cadences
            # coincide, so a card never gets two values for the same step.
            if (step > 0 and step % cfg.intervals.audio == 0
                    and step % cfg.intervals.metrics != 0 and n_uncond_cards
                    and DIST.is_main):
                pbar.write(f"\n  Uncond preview step {step}...")
                gen_model = (ema.model
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else model)
                generate_and_log_uncond_cards(
                    model=gen_model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer,
                    device=device, output_dir=audio_dir,
                    n_cards=n_uncond_cards,
                    sampling_cfg=cfg.sampling,
                    use_amp=cfg.training.use_amp,
                    frame_dims=FRAME_COND_DIMS,
                    global_configs=GLOBAL_CONFIGS,
                    metrics_seed=metrics_seed,
                    prefix=("EMA"
                            if cfg.training.use_ema and step >= cfg.training.ema_start
                            else "Model"),
                )
                pbar.write(f"  Uncond preview logged (step {step})\n")
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
            if step % cfg.intervals.ckpt == 0 and step > 0 and DIST.is_main:
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
                    test_dataset=test_dataset,
                    dist_info=DIST,
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
        loop_completed = True

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
            if not DIST.is_main:
                return                  # rank 0 writes the checkpoints
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
            if DIST.is_main:
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
        # Multi-GPU: the process group is closed only at the normal end of the
        # loop, where every rank gets here together. After a Ctrl+C or an
        # error the other ranks can be waiting inside a collective for this
        # one, and destroy_process_group() then waits for them: the process
        # never exits (seen on 3 Oct 2026 on guzheng: stuck right after "Last
        # checkpoint saved"). Exiting without it is safe: the operating system
        # releases the GPU and the NCCL resources together with the process.
        if DIST.enabled and loop_completed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        print("Training concluded." if saved else "Training ended (see warnings above).")
