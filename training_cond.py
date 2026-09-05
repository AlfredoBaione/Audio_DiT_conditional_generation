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
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_IRCAM_LOCAL, ".cache"))

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
from network_cond import ConditionedAudioDiT, TOKEN_DIM
from audio_dataset_cond import (
    build_conditioned_datasets, collate_conditioned,
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
_DAC_MODEL = None


def get_dac():
    global _DAC_MODEL
    if _DAC_MODEL is None:
        import dac
        _DAC_MODEL = dac.DAC.load(dac.utils.download(model_type="44khz"))
        _DAC_MODEL.to("cpu")
        _DAC_MODEL.eval()
        print("[DAC] Model loaded once (CPU) and cached for the whole run.")
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


def _scan_frame_conditions(condition_root):
    """
    SPLIT-LESS full scan: over ALL .npz under condition_root (any depth), count
    how many exist and, per condition name, in how many the key is present. Reads
    only the .npz key list (no array decompression), so it stays cheap.
    Returns {"total": int, "present": {name: count}}.
    """
    from collections import Counter as _Counter
    root = Path(condition_root) if condition_root else None
    if root is None or not root.exists():
        return {"total": 0, "present": {}}
    total = 0
    present = _Counter()
    for npz in root.rglob("*.npz"):
        try:
            with np.load(str(npz)) as d:
                keys = set(d.files)
        except Exception:
            continue
        total += 1
        for k in keys:
            present[k] += 1
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
            raise SystemExit(
                f"[config] unknown CLI override key(s): {bad or [str(e)]}\n"
                f"          These do not exist in {args.config}. Check the "
                f"spelling and the dotted path (e.g. 'data.num_val_batches', not "
                f"'data_num_val_batches'). Nothing was run.")
        OmegaConf.set_struct(cfg, False)

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
                      p_drop_each_frame=0.0):
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

    return frame_cond, global_cond


# ======================
# LOSS
# ======================
VALIDATION_PROTOCOL = "fixed_subset_common_noise_sample_weighted_v2"


def compute_loss(model, batch, device, use_amp, t_min, t_max,
                 global_configs, p_drop_all, p_drop_frame, p_drop_global,
                 training=True, x0=None, t=None, p_drop_each_frame=0.0):
    frames, frame_cond, _labels, text_embs, image_embs = batch
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

    # CFG dropout only at training time
    if training:
        fc, gc = apply_cfg_dropout(
            fc, gc, device, global_configs, B,
            p_drop_all=p_drop_all,
            p_drop_frame=p_drop_frame,
            p_drop_global=p_drop_global,
            p_drop_each_frame=p_drop_each_frame,
        )

    with torch.amp.autocast('cuda', enabled=use_amp):
        pred = model(xt, t, frame_conditions=fc, global_conditions=gc)
        loss = F.mse_loss(pred, target)
    return loss


@torch.no_grad()
def euler_sample_cfg(model, n_frames, device, steps, t_min, t_max, use_amp,
                      frame_cond, global_cond, guidance,
                      frame_dims, global_configs, gen_rng=None):
    """
    Euler integrator with classifier-free guidance.
    Both `frame_cond` and `global_cond` are expected as batch=1 dicts on device.
    If `guidance` <= 1.0 or both conditioning sources are absent, a single
    forward pass per step is used.
    `gen_rng` (optional torch.Generator) fixes the initial-noise x0 so the metric
    FD/KL are comparable across checkpoints (mirrors the uncond metrics seed);
    None = free-running.
    """
    model.eval()
    x = torch.randn(1, n_frames, TOKEN_DIM, device=device, generator=gen_rng)
    dt = (t_max - t_min) / steps

    null_fc = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc = make_null_global_conditions(1, global_configs or {}, device)

    # NB: test the CONTENT, not `is not None`: with every condition disabled the
    # caller passes empty dicts ({}), which are "no conditioning" -- treating them
    # as present would engage CFG and burn two IDENTICAL forwards per step.
    has_cond = bool(frame_cond) or bool(global_cond)
    use_cfg = (guidance > 1.0) and has_cond

    for i in range(steps):
        tv = t_min + i * dt
        t = torch.ones(1, device=device) * tv
        with torch.amp.autocast('cuda', enabled=use_amp):
            if use_cfg:
                fc = frame_cond if frame_cond else null_fc
                gc = global_cond if global_cond else null_gc
                v_c = model(x, t, frame_conditions=fc,      global_conditions=gc)
                v_u = model(x, t, frame_conditions=null_fc, global_conditions=null_gc)
                v = v_u + guidance * (v_c - v_u)
            else:
                v = model(x, t,
                          frame_conditions=frame_cond or null_fc,
                          global_conditions=global_cond or null_gc)
        x = x + v.float() * dt

    return x[0].cpu()


@torch.no_grad()
def euler_sample_cfg_paired(model, n_frames, device, steps, t_min, t_max, use_amp,
                            frame_cond, global_cond, guidance,
                            frame_dims, global_configs, gen_rng=None):
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

    for i in range(steps):
        tv = t_min + i * dt
        t = torch.ones(3 * B, device=device) * tv
        xb = torch.cat([x_cond, x_cond, x_unc], dim=0)   # (3B, n_frames, TOKEN_DIM)
        with torch.amp.autocast('cuda', enabled=use_amp):
            vb = model(xb, t, frame_conditions=frame_batch,
                       global_conditions=global_batch)
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


def audio_panel_tags(family, idx, active_conditions=(), suffix=""):
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
       wraps every 2-3 cards depending on window width. Only the first two
       positions are therefore side by side at EVERY width -- so slots 1 and 2
       are the f0 target and the generation conditioned on it, the one A/B these
       panels exist for.

          1  f0_<family>_XX                     the sonified f0 target
          2  generation_with_f0_<family>_XX     the generation it conditioned
          3+ <condition>_<family>_XX            energy, chroma, ... alphabetical

    3. The unconditional generation and the real recording do NOT live in the
       per-sample block. Each is COLLECTED into a group of its own, holding one
       card per sample, so that all of them can be heard as a grid of peers
       instead of one at a time inside separate collapsibles:

          uncond generation/uncond_<family>_XX<suffix>
          ground truth/real_<family>_XX

       The family and the index in the CARD name are what ties a card back to
       the block it belongs to: 'uncond generation/uncond_validation_03' is the
       null twin of 'validation_03/2_generation_with_f0_validation_03', drawn
       from the same noise, and 'ground truth/real_validation_03' is the
       recording that block's conditions were extracted from. These groups are
       flat -- no numeric prefix -- because nothing inside them is an ordered
       A/B against a neighbouring card.

    The card names repeat the family and index that the block header already
    shows. That redundancy is deliberate: a card read, filtered or screenshotted
    on its own still says what it is and which sample it belongs to.

    `family` is "probe" or "validation". `suffix` decorates the BLOCK header
    (the probe melody's name) and, since the collected groups have no block
    header to carry it, the uncond CARD name as well; the index stays the
    cross-reference to the f0 image of the same sample either way. If f0 is not
    among the active conditions the first one alphabetically leads instead, and
    the generation card is named after IT -- the panel never claims an f0
    pairing that was not part of the run.

    With NO condition active at all (pure-unconditional run) the block
    generation IS the unconditional generation -- nothing was dropped to obtain
    it -- so it is filed straight into the uncond group and no per-sample block
    is created. Such a run's audio window is exactly the two collected groups.
    """
    active = sorted(active_conditions)
    lead = "f0" if "f0" in active else (active[0] if active else None)
    block = f"{family}_{idx:02d}{suffix}"
    tail = f"{family}_{idx:02d}"

    conditions = {}
    if lead is not None:
        conditions[lead] = f"{block}/1_{lead}_{tail}"
    for j, cname in enumerate(n for n in active if n != lead):
        conditions[cname] = f"{block}/{3 + j}_{cname}_{tail}"

    uncond = f"{UNCOND_AUDIO_GROUP}/uncond_{tail}{suffix}"
    real = f"{REAL_AUDIO_GROUP}/real_{tail}"

    return {
        "conditions": conditions,
        # No lead -> nothing was conditioned, so the generation IS the uncond
        # card and the two keys deliberately name the same tag: the callers that
        # write the null baseline are guarded on a condition being active, so
        # the tag is still written exactly once.
        "generation": (f"{block}/2_generation_with_{lead}_{tail}"
                       if lead is not None else uncond),
        "generation_no_cond": uncond,
        "real": real,
    }
def subset_generation_tag(family, idx, label, suffix=""):
    """
    The audio card of ONE condition-subset generation, inside that sample's
    block. Slot 2 like the reference generation, so every generation of the
    sample sits together right after the condition cards and they sort among
    themselves by subset label.
    """
    return f"{family}_{idx:02d}{suffix}/2_gen_{label}_{family}_{idx:02d}"


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


def influence_set_size(sampling_cfg, default=16) -> int:
    """sampling.n_influence_samples -- the N of the influence set.

    ONE knob for: how many validation samples and how many probe stimuli are
    evaluated, plotted and played. It is NOT n_metrics_samples: the
    distributional metrics (FD-DAC / KL / FAD) run on that much larger pool,
    because they estimate a covariance and want the samples; the influence set
    is what a human reads panel by panel, so it is small and fully shown."""
    if sampling_cfg is None:
        return int(default)
    return int(sampling_cfg.get("n_influence_samples", default) or default)


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

    It refreshes the condition cards and `2_generation_with_f0_validation_XX`;
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
    if not panel_pos or not frame_dims:
        # No frame conditioning (or no panels): fall back to a plain spread over
        # the validation set, still one panel per sample.
        panel_pos = list(range(min(int(n_samples), total)))
        indices = torch.linspace(0, total - 1,
                                 max(1, min(int(n_samples), total))).long().tolist()
    panel_pos = panel_pos[:max(1, int(n_samples))]

    dac_model = get_dac()

    for k, p in enumerate(panel_pos):
        idx = indices[p] if p < len(indices) else indices[-1]
        _frames_real, frame_cond_real, _label_idx, text_emb, image_emb = val_dataset[idx]

        fc = {kk: v.unsqueeze(0).to(device).float()
              for kk, v in frame_cond_real.items()}
        gc = {}
        if "text" in global_configs:
            gc["text"] = text_emb.unsqueeze(0).to(device)
        if "image" in global_configs:
            gc["image"] = image_emb.unsqueeze(0).to(device)

        gen = euler_sample_cfg(
            model, n_frames, device,
            steps=sampling_cfg.euler_steps,
            t_min=sampling_cfg.t_min,
            t_max=sampling_cfg.t_max,
            use_amp=use_amp,
            frame_cond=fc, global_cond=gc, guidance=guidance,
            frame_dims=frame_dims, global_configs=global_configs,
        )
        if not torch.isfinite(gen).all():
            continue

        tags = audio_panel_tags("validation", k, frame_cond_real.keys())

        # decode_frames_to_wav returns 1-D; unsqueeze back to (1, T), which is
        # what add_audio expects.
        waveform = decode_frames_to_wav(gen, normalizer, dac_model).unsqueeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)

        # Same cards as the metrics step, written to the SAME tags: the f0
        # target and the generation it produced are the first two cards of the
        # block, each on its own player.
        for cname, carr in sorted(frame_cond_real.items()):
            son = sonify_condition(cname, carr.cpu().numpy(), DAC_SAMPLE_RATE)
            if son is not None:
                writer.add_audio(tags["conditions"][cname], norm_wav(son),
                                 global_step=step,
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
                           sampling_cfg=None, frame_dims=None):
    """Logs the real audio of the PANEL samples at step 0, into the same
    "ground truth" group the metrics step writes, so the reference is audible
    from the start instead of only appearing at the first metrics step. The
    card name carries the sample index, which is what ties it back to the
    validation_XX block and to the f0 image of the same XX."""
    dac_model = get_dac()

    total = len(val_dataset)
    panel_pos, indices = [], []
    if sampling_cfg is not None and frame_dims:
        n_metrics = int(getattr(sampling_cfg, "n_metrics_samples", 512) or 512)
        panel_pos = metrics_sample_positions(
            n_metrics, influence_set_size(sampling_cfg))[:max(0, int(n_samples))]
        indices = torch.linspace(0, total - 1, n_metrics).long().tolist()
    if not panel_pos:
        panel_pos = list(range(min(int(n_samples), total)))
        indices = torch.linspace(0, total - 1,
                                 max(1, min(int(n_samples), total))).long().tolist()

    for k, p in enumerate(panel_pos):
        idx = indices[p] if p < len(indices) else indices[-1]
        # ConditionedAudioDataset returns a 5-tuple: take only the frames
        frames, _frame_cond, _label_idx, _text_emb, _image_emb = val_dataset[idx]
        waveform = decode_frames_to_wav(frames, normalizer, dac_model).unsqueeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)

        writer.add_audio(
            audio_panel_tags("validation", k)["real"], wn,
            global_step=0, sample_rate=DAC_SAMPLE_RATE,
        )

    # dac_model is the shared singleton -> do not delete it.
    print(f"  {len(panel_pos)} real audios logged on TensorBoard")


# ======================
# OUT-OF-THE-BOX JOINT PROBE (proof of concept)
# ======================
@torch.no_grad()
def run_joint_probe(probe_sets, model, normalizer, n_frames,
                    step, writer, device,
                    output_dir, use_amp, sampling_cfg, guidance,
                    frame_dims, global_configs, fidelity_evaluator,
                    dac_model, prefix, n_plot, n_audio,
                    metrics_seed=None):
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
                merged in, plus a Probe_combinations panel saying which stimuli
                each panel puts together.

    The delta column needs a baseline, so the generation is PAIRED (with-cond
    and null from the same x0, one fused batch) exactly like the validation
    metrics -- the null generations cost nothing extra, they are already
    required by the CFG math.

    `fidelity_evaluator` is reused (reset first) rather than rebuilt: it carries
    the run's exact extractor configuration, and instantiating a second CREPE
    would risk the two drifting apart. The caller must therefore have already
    taken its per_sample()/coverage()/contours() copies for the validation rows.
    """
    from condition_metrics import pair_influence
    # f0 keeps its own dedicated plot (log-Hz axis + voicing ribbon); the other
    # conditions are drawn by the generic plotter, which picks the form that
    # suits the shape (curve / two curves / paired heatmaps).
    from probe_conditions import plot_condition_comparison

    # Only conditions the MODEL actually has: a bank for a condition this run
    # does not use would be fed into a slot that does not exist.
    names = [c for c in (frame_dims or {}) if c in (probe_sets or {})]
    if not names:
        return {}, {}
    missing = [c for c in (frame_dims or {}) if c not in (probe_sets or {})]
    if missing:
        # Not fatal, but it means those slots go in NULL and the probe is no
        # longer the in-distribution shape described above -- say so loudly.
        print(f"    [probe] WARNING: no bank for {missing}; those conditions "
              f"go in NULL, so this probe is a partial subset")

    # The panels are index-aligned across banks, so the count is the shortest.
    n_probe = min(len(probe_sets[c]) for c in names)
    if n_probe == 0:
        return {}, {}
    n_plot = max(0, min(int(n_plot), n_probe))

    targets = {c: [np.asarray(t, dtype=np.float32)
                   for t in probe_sets[c].targets] for c in names}

    def _frame_cond(idxs):
        # Every configured name must be present (FrameConditionEncoder.forward),
        # so start from the null dict and fill the ones this probe drives.
        fc = make_null_frame_conditions(len(idxs), n_frames, frame_dims or {},
                                        device)
        for c in names:
            fc[c] = torch.from_numpy(
                np.stack([targets[c][i] for i in idxs])).to(device).float()
        return fc

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
                frame_cond=_frame_cond(grp), global_cond={}, guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng,
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
                frame_cond=_frame_cond([i]), global_cond={}, guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng,
            ))

    # ---- score + collect the curves, one decode per generation ----
    # add_sample receives EVERY condition of the panel, so one decode scores all
    # of them and the table gets a row per condition from a single pass.
    def _score(lat_list, keep):
        fidelity_evaluator.reset()
        fidelity_evaluator.keep_contours_for(range(n_plot) if keep else ())
        wavs = []
        for i, lat in enumerate(lat_list):
            wav = decode_frames_to_wav(lat, normalizer, dac_model)
            fidelity_evaluator.add_sample(
                wav.numpy(), DAC_SAMPLE_RATE, n_frames,
                {c: targets[c][i] for c in names}, sample_id=i)
            wavs.append(wav if i < n_plot else None)
        return (fidelity_evaluator.per_sample(), fidelity_evaluator.coverage(),
                {c: fidelity_evaluator.contours(c) for c in names}, wavs)

    ps_cond, cov_cond, cont_cond, wavs_cond = _score(cond_lat, keep=True)
    # Both defaults matter: with no null pass (metrics_uncond off, or guidance
    # <= 1) the audio loop below still asks for len(wavs_null).
    ps_null, wavs_null = {}, []
    if null_lat:
        # The null pass contributes only its per-sample values: `attempted` and
        # the failure counts in the panel describe the CONDITIONED pass, which
        # is the one the row is about.
        ps_null, _cn, _ct, wavs_null = _score(null_lat, keep=False)

    # ---- IMAGES + AUDIO for the first n_plot panels ----
    # One TensorBoard PANEL per probe index: the stimuli it was conditioned on
    # and the generation sit under the SAME tag prefix, so the audio window
    # cannot separate a generation from the conditions that produced it.
    for i in range(n_plot):
        # The score shown in each plot title is that condition's FIRST metric,
        # whatever it is called (f0/energy -> corr, chroma -> cosine,
        # rhythm -> beat_corr): the probe must not hardcode a metric name that
        # only some conditions have.
        for c in names:
            _mkeys = sorted(k for k in ps_cond if k.startswith(f"{c}/"))
            corr_map = ps_cond.get(_mkeys[0], {}) if _mkeys else {}
            if i in cont_cond.get(c, {}):
                tgt, gen = cont_cond[c][i]
                writer.add_image(
                    # Same block name as this panel's AUDIO tags, so the Images
                    # and Audio tabs collapse into the same per-sample sections
                    # instead of one flat list of every condition x every panel.
                    f"probe_{i:02d}/{c}_target_vs_gen",
                    plot_condition_comparison(
                        c, tgt, gen, kind="probe",
                        label=f"'{probe_sets[c].names[i]}'", step=step,
                        prefix=prefix, guidance=guidance,
                        score=corr_map.get(i)),
                    global_step=step)

        tags = audio_panel_tags("probe", i, names)

        # The stimuli are re-logged at every probe step even though they never
        # change -- the audio slider shows the value AT the selected step, so a
        # card written once would be empty at every later step. They are logged
        # unconditionally, including when nothing was generated: the card is
        # what keeps a failed panel visible instead of silently absent. Each is
        # logged at its OWN bank's rate, which need not be the DAC's.
        for c in names:
            writer.add_audio(tags["conditions"][c],
                             norm_wav(probe_sets[c].wav(i)),
                             global_step=step, sample_rate=probe_sets[c].sr)
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

    # ---- which stimuli each panel combines ----
    # The tags carry only the index, so the mapping has to be written somewhere
    # readable; without it "probe_03" is unidentifiable in the audio window.
    combo = ["| Panel | " + " | ".join(f"`{c}`" for c in names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for i in range(n_probe):
        combo.append(f"| **{i:02d}** | "
                     + " | ".join(probe_sets[c].names[i] for c in names) + " |")
    writer.add_text("Validation/Probe_combinations",
                    "**Out-of-the-box probe panels** - each row is one joint "
                    "stimulus set, combined BY INDEX across the per-condition "
                    "banks and deliberately not mutually aligned.\n\n"
                    + "\n".join(combo), global_step=step)

    # ---- the influence rows, renamed so they read as their own axis ----
    # pair_influence keys off the condition name; renaming to "<cond>_probe" is
    # what keeps a probe row from being mistaken for the validation row of the
    # same condition in the same table.
    inf, cov = pair_influence(ps_cond, ps_null, coverage_cond=cov_cond,
                              have_null=bool(null_lat))
    rows = {f"{c}_probe": inf[c] for c in names if inf.get(c)}
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
    return rows, coverage


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
    fidelity_evaluator=None, clap_audio_embedder=None, compute_uncond=True,
    prefix="EMA", metrics_seed=None, metrics_enabled=COND_METRICS,
    fad_embedder=None, fad_ref_stats=None, n_fad=0, fad_device="cuda",
    probe_sets=None,
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
         ACTIVE in the run, so it adapts automatically. Image (CLIP) is shown as
         n/a until an audio-visual model (e.g. Wav2CLIP) is wired in.

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
    text_active  = ("text" in (global_configs or {})) and (clap_audio_embedder is not None)
    influence_active = frame_active or text_active

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
    n_log = min(2, n_samples)   # how many samples to log richly (audio/real)
    log_ids = sorted(set(range(n_log)) | set(plot_ids))
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
            frames_real, frame_cond_real, _lab, text_emb, image_emb = val_dataset[idx]
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
            else:
                fc, gc, g = None, None, 1.0
            gen = euler_sample_cfg(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                use_amp=use_amp,
                frame_cond=fc, global_cond=gc, guidance=g,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng,
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
            fcs, gcs = [], []
            for j, idx in enumerate(group, start=start):
                frames_real, frame_cond_real, _lab, text_emb, image_emb = val_dataset[idx]
                fcs.append(frame_cond_real)
                gcs.append((text_emb, image_emb))
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
                gen_rng=gen_rng,
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
    if compute_uncond:
        if not unc_lat:
            unc_lat = _generate(conditioned=False)[0]
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
    # is therefore independent of how many samples we score, so n_influence_samples
    # can be raised freely -- the only cost of raising it is TIME (the extractor
    # runs once per sample). We still HOLD the first n_keep decoded waveforms,
    # which are needed for the disk dump (n_val_save) and the TB previews (n_log).
    # n_val_save / n_influence / n_fid / n_keep were resolved above, before
    # the generation pass: the audio panels need to know WHICH samples they
    # describe in time to capture their real latent.

    dac_model = get_dac()    # load-once singleton (shared across the whole run)

    def _stream_audio(lat_list):
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
        keep_set = set(range(min(n_keep, n_lat))) | {p for p in plot_ids if p < n_lat}
        if frame_active:
            fidelity_evaluator.reset()
            # Retain the raw curves of the panel positions, for the TensorBoard
            # comparison plots. plot_ids is deterministic, so these are the SAME
            # validation samples at every metrics step and the plots can be read
            # as a time series.
            fidelity_evaluator.keep_contours_for(plot_ids)
        clap_sims, kept = {}, {}
        for i in sorted(fid_set | keep_set):           # decode each index ONCE
            wav = _decode_one(lat_list[i], dac_model)  # decode ONE
            if i in fid_set:
                wn = wav.numpy()
                if frame_active:
                    # sample_id=i: the conditioned and the null pass score the
                    # SAME positions of the same list, so tagging with i is what
                    # lets pair_influence compare them sample by sample instead
                    # of mean against mean.
                    fidelity_evaluator.add_sample(
                        wn, DAC_SAMPLE_RATE, n_frames, cond_targets[i],
                        sample_id=i)
                if text_active:
                    t = cond_globals[i].get("text") if i < len(cond_globals) else None
                    if t is not None:
                        emb = clap_audio_embedder.embed(wn, DAC_SAMPLE_RATE)
                        clap_sims[i] = float(np.dot(emb, t))
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
        return per, clap_sims, kept, len(pos), cov, cont

    if influence_active:
        print(f"    measuring condition influence on "
              f"{min(n_fid, len(cond_lat))} generations "
              f"(uniformly spread, streamed, memory-flat)...")
    ps_cond, clap_cond, cond_wavs, n_fid_used, cov_cond, cont_cond = \
        _stream_audio(cond_lat)
    ps_null, clap_null, unc_wavs = {}, {}, {}
    have_null = False
    if unc_lat:
        if not any_cond_active:
            # Pure-unconditional run: unc_lat IS cond_lat (reused above), so the
            # decoded waveforms are identical -- reuse them instead of running the
            # DAC decoder (CPU-bound) a second time over the same latents.
            unc_wavs = cond_wavs
        else:
            ps_null, clap_null, unc_wavs, _, _, _ = _stream_audio(unc_lat)
            have_null = True

    # ===== PER-SUBSET GENERATIONS (condition-combination influence) =====
    # One extra generation pass per subset, each scored against the SAME null
    # pass computed above -- a shared baseline is what lets the rows of the
    # matrix be compared with one another. The "all" subset is not regenerated:
    # the conditioned pass above already IS it, bit for bit.
    # Latents are dropped as soon as a subset has been scored, so peak memory
    # does not grow with the number of subsets (only the time does).
    subset_entries = []          # [(label, per_sample, coverage)]
    subset_wavs = {}             # label -> {sample id: waveform} for the panels
    if subset_specs and have_null:
        _full = tuple(frame_dims or {})
        for _lab, _names in subset_specs:
            if _names == _full:
                subset_entries.append((_lab, ps_cond, cov_cond))
                subset_wavs[_lab] = cond_wavs
                continue
            print(f"    subset '{_lab}' [{'+'.join(_names)}]: "
                  f"{n_samples} generations...")
            _slat, _st, _srf, _sg = _generate(conditioned=True,
                                              subset=set(_names))
            _sps, _scl, _swav, _sn, _scov, _scont = _stream_audio(_slat)
            subset_entries.append((_lab, _sps, _scov))
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

    # Each subset is paired against the SAME null pass as the reference above,
    # so every row of the matrix is a delta over one shared baseline and the
    # rows can be read against each other.
    subset_tables = []
    for _lab, _sps, _scov in subset_entries:
        _sinf, _scovp = pair_influence(_sps, ps_null, coverage_cond=_scov,
                                       have_null=have_null)
        subset_tables.append((_lab, _sinf, _scovp))
    if text_active:
        c_sim, n_sim, d_sim, n_pair = pair_scalar(
            clap_cond, clap_null, have_null=have_null)
        influence["text"] = {"clap_sim": {
            "cond": c_sim, "null": n_sim, "delta": d_sim}}
        cov_paired["text/clap_sim"] = {
            "valid": n_pair,
            "attempted": len(clap_cond),
            "unpaired": (len(set(clap_cond) ^ set(clap_null)) if have_null else 0),
        }

    # ---- image (CLIP): no direct audio<->CLIP metric available ----
    # Measuring image-condition influence on AUDIO needs an audio-visual model
    # in a shared space (e.g. Wav2CLIP / ImageBind). Until one is wired in, the
    # row is reported as not-available so the panel layout stays complete.
    if "image" in (global_configs or {}):
        influence["image"] = {"clip_sim": {
            "cond": None, "null": None, "delta": None,
            "note": "needs audio-visual model (e.g. Wav2CLIP)",
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
            # it is named (corr / cosine / beat_corr).
            _mk = sorted(k for k in ps_cond if k.startswith(f"{_cname}/"))
            _corr = ps_cond.get(_mk[0], {}) if _mk else {}
            for _j, _sid in enumerate(plot_ids):
                if _sid not in _cc:
                    continue
                _tgt, _gen = _cc[_sid]
                writer.add_image(
                    f"validation_{_j:02d}/{_cname}_target_vs_gen",
                    plot_condition_comparison(
                        _cname, _tgt, _gen, kind="valid",
                        label=f"validation sample #{_sid}",
                        step=step, prefix=prefix, guidance=guidance,
                        score=_corr.get(_sid)),
                    global_step=step)

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
    if probe_sets and frame_active:
        # Collected separately as well: the matrix branch below renders from
        # `subset_tables`, which never looks at `influence`, so probe rows added
        # only there would vanish whenever subsets and probes are both on.
        _paired = 'paired' if guidance > 1.0 else 'unpaired'
        _pnames = [c for c in (frame_dims or {}) if c in probe_sets]
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
                    prefix=prefix, n_plot=n_influence, n_audio=n_audio,
                    metrics_seed=metrics_seed,
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

    # ===== CONDITION-INFLUENCE PANEL (consolidated text table) =====
    # All per-condition adherence/influence lives HERE now, as a single table,
    # NOT as separate scalar curves. TensorBoard keeps a per-step history of the
    # text, so the step slider walks the panel across training.
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
            )
        else:
            panel_md = format_influence_panel(
                influence, step=step, prefix=prefix,
                guidance=guidance,
                n_samples=n_fid_used,
                coverage=cov_paired,
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
        for cname, metrics in influence.items():
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
    #   1_f0_validation_XX                 - the sonified f0 target
    #   2_generation_with_f0_validation_XX - the generation it conditioned
    #   3+_<condition>_validation_XX       - energy, chroma, ...
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
    # With no condition active there are no contours and no plot_ids, so fall
    # back to the first n_log generations: the window then holds real vs
    # generated and nothing else.
    panel_ids = list(plot_ids) or [i for i in sorted(cond_wavs)][:n_log]

    for k, sid in enumerate(panel_ids):
        if sid not in cond_wavs:
            continue                     # index k stays tied to plot_ids
        has_targets = any_cond_active and sid < len(cond_targets)
        tags = audio_panel_tags("validation", k,
                                cond_targets[sid] if has_targets else ())

        if has_targets:
            for cname, carr in sorted(cond_targets[sid].items()):
                son = sonify_condition(cname, carr, DAC_SAMPLE_RATE)
                if son is not None:
                    writer.add_audio(tags["conditions"][cname], norm_wav(son),
                                     global_step=step,
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
                _log_audio(_w, subset_generation_tag("validation", k, _lab))

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
        if sid in real_frames:
            real_wav = _decode_one(real_frames[sid], dac_model)
            if k < n_audio:
                _log_audio(real_wav, tags["real"])
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
# The ConditionedAudioDataset returns 5-tuples in both cases (because of
# frame_conds, text_emb, image_emb). This adapter exposes the "slim view"
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
        frames, _frame_cond, label_idx, _text, _image = self._ds[idx]
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
    #   * explicit list  -> train with that subset; every requested condition
    #                       MUST be present in the extracted .npz, otherwise we
    #                       RAISE (no silent zero-fill of an un-extracted cond).
    # Global conditions (text / image) are NOT stored in the .npz, so
    # enabled_global keeps its previous semantics (null = all enabled in config).
    # The dataset is split-less, so the scan is a single pass over ALL .npz.
    # ------------------------------------------------------------------
    enabled_pool = {name for name, c in CONDITION_CONFIG["frame_level"].items()
                    if c.get("enabled", False)}
    cond_scan = _scan_frame_conditions(cfg.paths.condition_root)
    # "available" = present in EVERY .npz (usable as a full condition).
    available_all = {n for n, c in cond_scan["present"].items()
                     if cond_scan["total"] > 0 and c == cond_scan["total"]}
    extracted = set(cond_scan["present"].keys())   # any presence (for messages)
    available = available_all & enabled_pool

    enabled_f = cfg.conditioning.get("enabled_frame",  None)
    enabled_g = cfg.conditioning.get("enabled_global", None)
    # OmegaConf converts YAML null -> None, YAML list -> ListConfig.
    if enabled_f is not None:
        enabled_f = list(enabled_f)
    if enabled_g is not None:
        enabled_g = list(enabled_g)

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
    else:
        # Explicit subset: every requested condition MUST be extracted.
        missing = [c for c in enabled_f if c not in extracted]
        if missing:
            raise RuntimeError(
                f"conditioning.enabled_frame requests {missing}, but these are NOT "
                f"present in the extracted .npz under {cfg.paths.condition_root}.\n"
                f"  Extracted/available: {sorted(extracted)}\n"
                f"  Extract the missing ones first:\n"
                f"    python extract_conditions.py <dataset_root> "
                f"--conditions {','.join(missing)} --device cuda")

    # ---- U4: STRICT FULL validation of the requested conditions ----
    # Every requested condition must be present in EVERY .npz; otherwise some
    # samples would be silently zero-filled and the model would train on NULL
    # conditions without any error. The dataset is split-less, so this is one
    # full scan over all .npz. strict=True (default) hard-fails.
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
    #     chroma cosine, rhythm/energy correlation).
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
    )
    for _name, _extractor in fidelity_evaluator.extractors.items():
        _actual_device = getattr(
            _extractor, "device", getattr(_extractor, "_device", "cpu"))
        _batch = getattr(_extractor, "batch_size", None)
        _batch_msg = f" | batch_size={_batch}" if _batch is not None else ""
        print(f"[metrics] extractor={_name} | device={_actual_device}{_batch_msg}")
    clap_audio_embedder = None
    if "text" in GLOBAL_CONFIGS:
        from conditions import ClapAudioEmbedder
        # match the CLAP checkpoint used by the text condition
        clap_model_name = CONDITION_CONFIG["global"]["text"]["kwargs"].get(
            "model_name", "laion/larger_clap_music")
        clap_audio_embedder = ClapAudioEmbedder(model_name=clap_model_name,
                                                device=_fid_device)
        print(f"Text-influence (CLAP audio) enabled: {clap_model_name}")

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
    # sampling.n_influence_samples is the SINGLE knob of the influence set: N
    # probe stimuli and N validation samples, all of them scored, plotted and
    # played. It replaces the old trio n_probes / n_cond_plot / (the former,
    # larger) n_influence_samples, which let the table describe one set of
    # samples while the panels showed another.
    probe_sets = {}
    _n_probes = influence_set_size(cfg.sampling)
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
    for _cond in FRAME_COND_DIMS:
        if _n_probes <= 0:
            continue
        try:
            from probe_conditions import build_condition_probe_set
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
            _probe_extractor = fidelity_evaluator.extractors.get(
                _cond, registry.frame_extractors[_cond])
            _probe_dev = getattr(_probe_extractor, "device",
                                 getattr(_probe_extractor, "_device", "cpu"))
            _ps = build_condition_probe_set(
                _cond, _probe_root, val_dataset.n_frames,
                extractor=_probe_extractor,
                n_probes=_n_probes,
                duration_s=float(cfg.model.duration_s),
                sr=DAC_SAMPLE_RATE,
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
    if probe_sets:
        print()

    metrics_uncond = bool(cfg.sampling.get("metrics_uncond", True))
    print(f"Conditioning influence: frame={list(FRAME_COND_DIMS.keys()) or 'none'}"
          f"{' + text(CLAP)' if clap_audio_embedder is not None else ''}"
          f"{' + image(n/a)' if 'image' in GLOBAL_CONFIGS else ''} "
          f"| uncond metrics: {'on' if metrics_uncond else 'off'}\n")

    # ======================
    # MODEL + EMA
    # ======================
    # cfg.model.kind and cfg.conditioning.* already reflect the checkpoint on
    # resume (restored in load_config), so the model is rebuilt with the correct
    # architecture and conditioning layout automatically - no need to re-pass
    # them on the command line.
    print(f"[MODEL] Building ConditionedAudioDiT-{cfg.model.kind} "
          f"| frame={list(FRAME_COND_DIMS)} | global={list(GLOBAL_CONFIGS)}")
    model = ConditionedAudioDiT(
        kind=cfg.model.kind,
        drop=cfg.model.get("drop", 0.0),
        frame_cond_dims=FRAME_COND_DIMS,
        frame_cond_out_dims=FRAME_COND_OUT_DIMS,
        global_cond_configs=GLOBAL_CONFIGS,
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
    # .get: a config written before per-condition dropout existed stays valid
    # and keeps its exact previous behaviour (0.0 disables stage 2).
    P_DROP_EACH_FRAME = float(cfg.conditioning.get("p_drop_each_frame", 0.0))
    print(f"CFG dropout: all={cfg.conditioning.p_drop_all} "
          f"frame={cfg.conditioning.p_drop_frame} "
          f"global={cfg.conditioning.p_drop_global} "
          f"each_frame={P_DROP_EACH_FRAME}"
          + ("  (partial subsets ON)" if P_DROP_EACH_FRAME > 0 else ""))
    print(f"CFG guidance scale (validation): {cfg.conditioning.guidance_scale}")
    print(f"DATASET_ROOT:   {cfg.paths.dataset_root}")
    print(f"WAV_ROOT:       {cfg.paths.wav_root}")
    print(f"CONDITION_ROOT: {cfg.paths.condition_root}")
    print(f"IMAGE_ROOT:     {cfg.paths.image_root}")
    print(f"RUN DIR:        {run_dir}")
    print(f"{'='*60}\n")

    # Real reference audio, logged ONCE at step 0 (it never changes during
    # training, unlike the generations). Mirrors the unconditional project, and
    # means the reference is audible from the start instead of only appearing at
    # the first metrics step. Tags: ground truth/real_validation_XX.
    log_real_audio_samples(
        val_dataset=val_dataset,
        normalizer=normalizer,
        writer=writer,
        n_samples=cfg.sampling.n_audio_samples,
        sampling_cfg=cfg.sampling,
        frame_dims=FRAME_COND_DIMS,
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
                    clap_audio_embedder=clap_audio_embedder,
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
