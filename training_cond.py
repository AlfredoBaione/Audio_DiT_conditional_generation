# Training for the Conditioned Audio DiT with Rectified Flow.
#
# Multi-modal conditioning:
#   - Frame-level (concatenated on the feature dimension at the input,
#     JASCO-style; see network_cond.py): melody, chroma, rhythm
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
import math
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from io import BytesIO

import torch
# Enable TF32 on Ampere+ GPUs (e.g. RTX A4000). Same as facebookresearch/DiT:
# matmul/conv in TF32 mode -> roughly 2-3x faster than pure fp32 while keeping
# the same dynamic range as fp32 (no overflow risk, unlike fp16/AMP).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn.functional as F
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    DAC_METRICS,
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


# ======================
# SPLIT / CACHE HELPERS  (new: split-less dataset + cache metadata validation)
# ======================
def _split_param(cfg, key, default):
    """Read data.split.<key> from the YAML, with a default."""
    d = cfg.data.get("split", None) if hasattr(cfg, "data") else None
    if d is None:
        return default
    return d.get(key, default)


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
    entries = []
    n = 0
    for p in sorted(root.rglob("*.npy")):
        st = p.stat()
        entries.append(f"{p.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}")
        n += 1
    h = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return h, n


def _cache_fingerprint(cfg, n_frames):
    """Identity of the data the cached normalizer / FD-DAC reference depend on.
    The normalizer is fit on the TRAIN split and the FD reference on the VAL
    split, so the split parameters are part of the fingerprint too."""
    flist_hash, flist_count = _latent_file_list_hash(cfg.paths.dataset_root)
    return {
        "latent_root": os.path.abspath(cfg.paths.dataset_root),
        "dataset_meta": _load_dataset_meta(cfg.paths.dataset_root),
        "duration_s": float(cfg.model.duration_s),
        "n_frames": int(n_frames),
        "latent_dim": int(DAC_LATENT_DIM),
        "latent_file_list_hash": flist_hash,
        "latent_file_count": flist_count,
        "split": {
            "ratios": list(_split_param(cfg, "ratios", [0.8, 0.1, 0.1])),
            "seed": int(_split_param(cfg, "seed", 42)),
            "group_by_source": bool(_split_param(cfg, "group_by_source", True)),
            "stratify_by_class": bool(_split_param(cfg, "stratify_by_class", True)),
        },
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
                      p_drop_all, p_drop_frame, p_drop_global):
    """
    Per-sample CFG dropout. Each element of the batch flips its own coin:
        p_drop_all     -> drop everything       (pure unconditional)
        p_drop_frame   -> drop only frame-level
        p_drop_global  -> drop only global
        remaining mass -> keep both branches

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

    # Frame-level: zero the selected rows (zero is the null for frame conds)
    if frame_cond:
        for k in frame_cond:
            keep_mask = (~drop_f).view(B, 1, 1).to(frame_cond[k].dtype)
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
def compute_loss(model, batch, device, use_amp, t_min, t_max,
                 global_configs, p_drop_all, p_drop_frame, p_drop_global,
                 training=True):
    frames, frame_cond, _labels, text_embs, image_embs = batch
    # NB: `labels` is discarded as conditioning (CLAP-text plays that role
    # better now). It is kept in the batch only as metadata for logging.

    x1 = frames.to(device).float()
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = sample_logit_normal(B, device, t_min, t_max)
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
        )

    with torch.amp.autocast('cuda', enabled=use_amp):
        pred = model(xt, t, frame_conditions=fc, global_conditions=gc)
        loss = F.mse_loss(pred, target)
    return loss


# ======================
# AUDIO/SPECTROGRAM UTILITIES
# ======================
def plot_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    import PIL.Image as Image
    import torchvision
    img = torchvision.transforms.ToTensor()(Image.open(buf))
    buf.close()
    return img


def make_spectrogram(waveform, sr, title=""):
    import torchaudio
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512
    )(waveform.cpu().float())
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(spec_db[0].numpy(), aspect='auto', origin='lower',
              cmap='viridis', vmin=-80, vmax=0)
    ax.set_title(title)
    ax.set_xlabel("Frame"); ax.set_ylabel("Mel Bin")
    plt.colorbar(ax.images[0], ax=ax, label="dB")
    img = plot_to_image(fig)
    plt.close(fig)
    return img


@torch.no_grad()
def euler_sample_cfg(model, n_frames, device, steps, t_min, t_max, use_amp,
                      frame_cond, global_cond, guidance,
                      frame_dims, global_configs, gen_rng=None, x0=None):
    """
    Euler integrator with classifier-free guidance.
    Both `frame_cond` and `global_cond` are expected as batch=1 dicts on device.
    If `guidance` <= 1.0 or both conditioning sources are absent, a single
    forward pass per step is used.
    `gen_rng` (optional torch.Generator) fixes the initial-noise x0 so the metric
    FD/KL are comparable across checkpoints (mirrors the uncond metrics seed);
    None = free-running.
    `x0` (optional) supplies the initial noise directly, bypassing gen_rng -- used
    by verify_paired_sampler.py to run this reference path from exactly the noise
    the fused sampler drew.
    """
    model.eval()
    x = (torch.randn(1, n_frames, TOKEN_DIM, device=device, generator=gen_rng)
         if x0 is None else x0.clone())
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
    unchanged for f0-only, melody+energy, CLAP-text, CLIP-image, or any future
    combination (driven by the run's condition dicts, nothing is hardcoded).

    Returns (cond_latents, uncond_latents): two lists of B tensors on CPU.
    Mathematically equivalent to separate euler_sample_cfg() calls from the same
    x0 -- but only up to CUDA op ordering, so A/B-verify before trusting it
    (verify_paired_sampler.py). B does NOT affect which samples come out: the
    noise is drawn per-sample (see below), so spf is purely about how the work is
    packed into forwards.
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
# AUDIO GENERATION (conditioned, from val-set samples)
# ======================
@torch.no_grad()
def generate_and_log_audio(
    model, normalizer, val_dataset, n_frames, step, writer, device,
    output_dir, n_samples, sampling_cfg, conditioning_cfg, use_amp,
    frame_dims, global_configs, prefix="EMA",
):
    """
    Generates `n_samples` CONDITIONED audios by taking conditions from
    val-set samples, decodes with DAC on CPU, logs on TensorBoard.
    Mirrors the unconditional repo's `generate_and_log_audio`: tags are
    Validation/Audio_generated_{prefix}_{i:02d} and
    Validation/Spectrogram_generated_{prefix}_{i:02d}, where {prefix} is
    "EMA" or "Model" depending on which weights produced the sample.
    """
    guidance = float(conditioning_cfg.guidance_scale)

    # Pick val-set indices to source the conditions from
    total = len(val_dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()

    # Pre-generate latents
    generated_frames = []
    sample_class_names = []
    for idx in indices:
        _frames_real, frame_cond_real, label_idx, text_emb, image_emb = val_dataset[idx]
        class_name = val_dataset.idx_to_label.get(label_idx, str(label_idx))
        sample_class_names.append(class_name)

        # Build batch=1 conditions on device
        fc = {k: v.unsqueeze(0).to(device).float() for k, v in frame_cond_real.items()}
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
        generated_frames.append(gen)

    # Decode with DAC on CPU (shared singleton, loaded once for the whole run)
    dac_model = get_dac()

    for i, gen in enumerate(generated_frames):
        if not torch.isfinite(gen).all():
            continue
        z = gen.T
        z = normalizer.denormalize(z)
        z_in = z.unsqueeze(0).float()
        z_q, _, _ = dac_model.quantizer.from_latents(z_in)   # (1,72,T) -> (1,1024,T)
        waveform = dac_model.decode(z_q).squeeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)

        writer.add_audio(
            f"Validation/Audio_generated_{prefix}_{i:02d}", wn,
            global_step=step, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(
            waveform, DAC_SAMPLE_RATE,
            f"{prefix} sample {i} ({sample_class_names[i]}) - "
            f"step {step} - guidance={guidance}",
        )
        writer.add_image(
            f"Validation/Spectrogram_generated_{prefix}_{i:02d}",
            spec_img, global_step=step,
        )

        wav_path = os.path.join(
            output_dir, f"step{step:07d}_{prefix}_{i:02d}.wav"
        )
        sf.write(wav_path, waveform.squeeze().numpy(), DAC_SAMPLE_RATE)
    # dac_model is the shared singleton -> do not delete it.


# ======================
# LOG REAL AUDIO SAMPLES (once at startup, step=0)
# ======================
@torch.no_grad()
def log_real_audio_samples(val_dataset, normalizer, writer, n_samples):
    """Logs real audio from the val dataset for comparison on TensorBoard."""
    dac_model = get_dac()

    total = len(val_dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()

    for i, idx in enumerate(indices):
        # ConditionedAudioDataset returns a 5-tuple: take only frames + label
        frames, _frame_cond, label_idx, _text_emb, _image_emb = val_dataset[idx]
        class_name = val_dataset.idx_to_label.get(label_idx, str(label_idx))
        z = frames.T
        z = normalizer.denormalize(z)
        z_in = z.unsqueeze(0).float()
        z_q, _, _ = dac_model.quantizer.from_latents(z_in)   # (1,72,T) -> (1,1024,T)
        waveform = dac_model.decode(z_q).squeeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)

        writer.add_audio(
            f"Validation/Audio_real_{i:02d}", wn,
            global_step=0, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(
            waveform, DAC_SAMPLE_RATE,
            f"Real sample {i} ({class_name})",
        )
        writer.add_image(
            f"Validation/Spectrogram_real_{i:02d}",
            spec_img, global_step=0,
        )

    # dac_model is the shared singleton -> do not delete it.
    print(f"  {n_samples} real audios logged on TensorBoard")


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
    prefix="EMA", metrics_seed=None, metrics_enabled=DAC_METRICS,
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
         null generations, score adherence on both (melody RPA/RCA, chroma
         cosine, rhythm/energy correlation, text CLAP audio-text cosine), and
         report Δ = with-cond - null. Answers "how much does the condition pull
         the generation toward its target?". Consolidated into a single Markdown
         table (no separate scalar curves), built only from the conditions
         ACTIVE in the run, so it adapts automatically. Image (CLIP) is shown as
         n/a until an audio-visual model (e.g. Wav2CLIP) is wired in.

    FD-DAC and KL (both directions, real||gen and gen||real) share the SAME real
    validation latent reference (fd_dac_ref_stats) in both the cond and uncond
    cases; the distributional metrics are latent-only (audio is decoded only for
    the influence re-extraction and the audio/spectrogram previews). `prefix`
    ("EMA" / "Model") tags the previews by the generating weights. Returns
    (fd_dac_cond, kl_cond_real_gen, kl_cond_gen_real) for the caller.
    """
    guidance = float(conditioning_cfg.guidance_scale)
    n_frames = val_dataset.n_frames
    total = len(val_dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()
    n_log = min(2, n_samples)   # how many samples to log richly (audio/specs/rolls)

    ref_frames = (fd_dac_ref_stats["n_total"]
                  if fd_dac_ref_stats is not None else "n/a")
    print(f"\n  Compute metrics @ step {step}: {n_samples} generations "
          f"(cond guidance={guidance}"
          f"{', + uncond' if compute_uncond else ''}) "
          f"vs reference ({ref_frames} frames)...")

    # ---- generate n_samples latents, conditioned or unconditional ----
    # For the conditioned pass we also keep, for the first n_log samples, the
    # real latent (to decode the real audio) and the target condition.
    def _generate(conditioned):
        lat_list = []
        targets = []        # paired frame conditions (cpu numpy); cond only
        real_frames = []    # real latents of the first n_log samples; cond only
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
                      for k, v in frame_cond_real.items()}
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
                if j < n_log:
                    real_frames.append(frames_real)
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
        targets, real_frames, global_targets = [], [], []
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
                if j < n_log:
                    real_frames.append(frames_real)
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
        z = normalizer.denormalize(frames.T)
        z_q, _, _ = dac_model.quantizer.from_latents(z.unsqueeze(0).float())
        wav = dac_model.decode(z_q).squeeze()
        return wav.cpu()

    def _decode(lat_list, dac_model):
        return [_decode_one(f, dac_model) for f in lat_list]

    has_ref = fd_dac_ref_stats is not None

    # ===== CONDITIONAL generation =====
    # Distributional metrics (FD-DAC + KL both directions) are latent-only and
    # share the SAME real reference (fd_dac_ref_stats). The DAC decode below is
    # needed ONLY for conditioning fidelity (re-extract from audio) and for the
    # rich audio/spectrogram logging -- NOT for the distributional metrics.
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
    paired = (spf >= 1 and compute_uncond and (guidance > 1.0) and any_cond_active)
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
    frame_active = (fidelity_evaluator is not None and fidelity_evaluator.active)
    text_active  = ("text" in (global_configs or {})) and (clap_audio_embedder is not None)
    influence_active = frame_active or text_active

    fd_dac_uncond = None
    kl_uncond = {"kl_real_gen": None, "kl_gen_real": None}
    unc_lat = []
    if compute_uncond:
        # reuse the paired uncond latents if already generated, else generate now
        unc_lat = _unc_pre if _unc_pre is not None else _generate(conditioned=False)[0]
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
    # (DAC decode + re-extraction via basic-pitch/librosa + CLAP) is the
    # RAM-hungry step. Instead of decoding ALL generations into a big list and
    # re-extracting from all of them at once (which OOM-kills the process at
    # large n_metrics_samples), we STREAM it: decode ONE generation -> re-extract
    # its descriptors -> ACCUMULATE the metric -> DISCARD the waveform. Peak RAM
    # is therefore independent of how many samples we score, so n_influence_samples
    # can be raised freely -- the only cost of raising it is TIME (basic-pitch
    # runs once per sample). We still HOLD the first n_keep decoded waveforms,
    # which are needed for the disk dump (n_val_save) and the TB previews (n_log).
    n_val_save  = int(getattr(sampling_cfg, "n_val_save", 8) or 8)
    n_influence = int(getattr(sampling_cfg, "n_influence_samples", 64) or 64)
    n_keep = max(n_val_save, n_log)
    n_fid  = min(n_influence, n_samples) if influence_active else 0

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
        fid_pos = []
        if n_fid > 0 and n_lat > 0:
            fid_pos = sorted(set(
                torch.linspace(0, n_lat - 1, min(n_fid, n_lat))
                .round().long().tolist()))
        fid_set = set(fid_pos)
        keep_set = set(range(min(n_keep, n_lat)))
        if frame_active:
            fidelity_evaluator.reset()
        clap_sims, kept = [], []
        for i in sorted(fid_set | keep_set):           # decode each index ONCE
            wav = _decode_one(lat_list[i], dac_model)  # decode ONE
            if i in fid_set:
                wn = wav.numpy()
                if frame_active:
                    fidelity_evaluator.add_sample(
                        wn, DAC_SAMPLE_RATE, n_frames, cond_targets[i])
                if text_active:
                    t = cond_globals[i].get("text") if i < len(cond_globals) else None
                    if t is not None:
                        emb = clap_audio_embedder.embed(wn, DAC_SAMPLE_RATE)
                        clap_sims.append(float(np.dot(emb, t)))
            if i in keep_set:
                kept.append(wav)                       # sorted -> order preserved
            # else: wav is dropped here -> RAM stays flat regardless of n_fid
        fid  = fidelity_evaluator.results() if frame_active else {}
        # coverage travels WITH the means: reporting a score without saying how
        # many samples reached it is how failures disappear from the numbers.
        cov  = fidelity_evaluator.coverage() if frame_active else {}
        clap = float(np.mean(clap_sims)) if clap_sims else None
        return fid, clap, kept, len(fid_pos), cov

    if influence_active:
        print(f"    measuring condition influence on "
              f"{min(n_fid, len(cond_lat))} generations "
              f"(uniformly spread, streamed, memory-flat)...")
    fid_cond, sim_cond, cond_wavs, n_fid_used, cov_cond = _stream_audio(cond_lat)
    fid_null, sim_null, unc_wavs, cov_cond_null = {}, None, [], {}
    if compute_uncond and unc_lat:
        if not any_cond_active:
            # Pure-unconditional run: unc_lat IS cond_lat (reused above), so the
            # decoded waveforms are identical -- reuse them instead of running the
            # DAC decoder (CPU-bound) a second time over the same latents.
            unc_wavs = cond_wavs
        else:
            fid_null, sim_null, unc_wavs, _, cov_cond_null = _stream_audio(unc_lat)

    del cond_lat, unc_lat    # latents no longer needed
    if device == "cuda":
        torch.cuda.empty_cache()

    # ===== CONDITION INFLUENCE (delta: with-cond vs null, paired) =====
    # influence[cond_name][metric] = {"cond":.., "null":.., "delta":..}.
    # delta>0 means the condition pulled the generation toward its target. Built
    # only from the conditions ACTIVE in this run (registry-driven).
    influence = {}
    if frame_active:
        for key, cval in fid_cond.items():
            name, _, metric = key.partition("/")
            nval = fid_null.get(key)
            influence.setdefault(name, {})[metric] = {
                "cond": cval,
                "null": nval,
                "delta": (cval - nval) if (nval is not None) else None,
            }
    if text_active:
        influence["text"] = {"clap_sim": {
            "cond": sim_cond, "null": sim_null,
            "delta": (sim_cond - sim_null) if (sim_cond is not None and sim_null is not None) else None,
        }}

    # ---- image (CLIP): no direct audio<->CLIP metric available ----
    # Measuring image-condition influence on AUDIO needs an audio-visual model
    # in a shared space (e.g. Wav2CLIP / ImageBind). Until one is wired in, the
    # row is reported as not-available so the panel layout stays complete.
    if "image" in (global_configs or {}):
        influence["image"] = {"clip_sim": {
            "cond": None, "null": None, "delta": None,
            "note": "needs audio-visual model (e.g. Wav2CLIP)",
        }}

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
    else:
        # unconditional run: fd_dac_cond/kl_cond ARE the unconditional numbers
        if fd_dac_cond is not None:
            writer.add_scalar("Validation/Metrics/Fd_dac", fd_dac_cond, step)
        if kl_cond["kl_real_gen"] is not None:
            writer.add_scalar("Validation/Metrics/Kl_real_gen",
                              kl_cond["kl_real_gen"], step)
            writer.add_scalar("Validation/Metrics/Kl_gen_real",
                              kl_cond["kl_gen_real"], step)

    # ===== CONDITION-INFLUENCE PANEL (consolidated text table) =====
    # All per-condition adherence/influence lives HERE now, as a single table,
    # NOT as separate scalar curves. TensorBoard keeps a per-step history of the
    # text, so the step slider walks the panel across training.
    if influence:
        from condition_metrics import format_influence_panel, format_influence_legend
        panel_md = format_influence_panel(
            influence, step=step, prefix=prefix,
            guidance=guidance,
            # the influence is measured on n_fid_used generations, NOT on the
            # n_metrics_samples used for FD/KL: reporting the latter would claim
            # a sample size that was never used for these numbers.
            n_samples=n_fid_used,
            coverage=cov_cond,
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
          f"KL(gen||real): {_f(kl_cond['kl_gen_real'])}")
    if compute_uncond:
        print(f"  [uncond] FD-DAC: {_f(fd_dac_uncond)} | "
              f"KL(real||gen): {_f(kl_uncond['kl_real_gen'])} | "
              f"KL(gen||real): {_f(kl_uncond['kl_gen_real'])}")
    if influence:
        parts = []
        for cname, metrics in influence.items():
            for m, vals in metrics.items():
                d = vals.get("delta")
                parts.append(f"{cname}/{m} Δ={_f(d)}")
        if parts:
            print("  [influence] " + " | ".join(parts))

    # ===== SAVE VALIDATION ARTIFACTS TO DISK (one dir per step, one sub-dir per generation) =====
    # output_dir/step_{step}/generation_{i}/ contains, for generation i:
    #   conditions.npz   - the EXACT input conditions used (melody, energy, ...)
    #   cond_{name}.wav  - AUDIBLE rendering of each condition (melody as sine
    #                      tones, energy as amplitude-modulated tone) so one can
    #                      hear how it maps into the conditioned audio
    #   cond.wav         - the conditioned generation
    #   uncond.wav       - the unconditioned (null) generation, same index
    #   real.wav         - the reference audio (first n_log generations only)
    # How many generations are dumped is sampling.n_val_save (default: all).
    def _to_np(wav):
        return wav.numpy() if wav.dim() == 1 else wav.squeeze().numpy()

    from condition_metrics import sonify_condition

    # n_val_save was already resolved above; the streamed cond_wavs list holds
    # the first n_keep = max(n_val_save, n_log) decoded waveforms.
    n_save = min(n_val_save, len(cond_wavs))
    step_dir = os.path.join(output_dir, f"step_{step:07d}")

    def _gen_dir(i):
        d = os.path.join(step_dir, f"generation_{i:03d}")
        os.makedirs(d, exist_ok=True)
        return d

    for i in range(n_save):
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
        if compute_uncond and i < len(unc_wavs):
            sf.write(os.path.join(gdir, "uncond.wav"),
                     _to_np(unc_wavs[i]), DAC_SAMPLE_RATE)
    print(f"    saved {n_save} validation generations "
          + ("(per-generation dirs: cond+uncond+conditions+sonified) to "
             if any_cond_active else "(per-generation dirs: generated+real) to ")
          + step_dir)

    # ===== COMPARISON LOGGING on TensorBoard =====
    # Tagged by the generating weights `w` ("EMA"/"Model"). The scheme follows the
    # RUN MODE:
    #   * conditioned run -> real / with-cond / without-cond for a direct A/B
    #     (per-condition influence lives in the text panel);
    #   * unconditional run -> real / generated only, under the unconditional
    #     project's tag names (there is no with/without-cond A/B to make).
    w = prefix  # "EMA" or "Model"

    def _log_audio(wav, tag):
        wav_u = wav.unsqueeze(0) if wav.dim() == 1 else wav
        wn = wav_u / (wav_u.abs().max() + 1e-8)
        writer.add_audio(f"Validation/{tag}", wn, global_step=step,
                         sample_rate=DAC_SAMPLE_RATE)

    def _log_spec(wav, tag, title):
        wav_u = wav.unsqueeze(0) if wav.dim() == 1 else wav
        writer.add_image(f"Validation/{tag}",
                         make_spectrogram(wav_u, DAC_SAMPLE_RATE, title),
                         global_step=step)

    for i in range(n_log):
        # ---------- REAL ----------
        real_wav = _decode_one(real_frames[i], dac_model)
        _log_audio(real_wav, f"Audio_real_{i:02d}")
        _log_spec(real_wav, f"Spectrogram_real_{i:02d}",
                  f"real {i} - step {step}" if any_cond_active
                  else f"Real sample {i}")
        sf.write(os.path.join(_gen_dir(i), "real.wav"),
                 real_wav.numpy(), DAC_SAMPLE_RATE)

        if any_cond_active:
            # ---------- GENERATED WITH COND ----------
            cwav = cond_wavs[i]
            _log_audio(cwav, f"Audio_generated_with_cond_{w}_{i:02d}")
            _log_spec(cwav, f"Spectrogram_generated_with_cond_{w}_{i:02d}",
                      f"generated with cond ({w}) {i} - step {step} - guidance={guidance}")

            # ---------- GENERATED WITHOUT COND ----------
            if compute_uncond:
                uwav = unc_wavs[i]
                _log_audio(uwav, f"Audio_generated_without_cond_{w}_{i:02d}")
                _log_spec(uwav, f"Spectrogram_generated_without_cond_{w}_{i:02d}",
                          f"generated without cond ({w}) {i} - step {step}")
        else:
            # Unconditional run: there is no with/without-cond A/B to make (the
            # two are the same samples), so log ONE generation under the EXACT
            # tags/titles of the unconditional project.
            gwav = cond_wavs[i]
            _log_audio(gwav, f"Audio_generated_{w}_{i:02d}")
            _log_spec(gwav, f"Spectrogram_generated_{w}_{i:02d}",
                      f"{w} sample {i} - step {step}")

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
# RNG STATE (exact resume: continue the stream, not restart from the seed)
# ======================
def _capture_rng_state(data_generator=None):
    """Snapshot of every RNG stream so a --resume can continue EXACTLY where it
    left off (not restart from the seed): python, numpy, torch CPU, torch CUDA,
    and the DataLoader shuffle generator. Mirrors training.py."""
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
        print("[SEED] RNG state restored from checkpoint (exact resume).")
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
    stores every RNG stream so --resume continues EXACTLY (x0, t, CFG dropout,
    shuffle) rather than restarting from the seed. Mirrors training.py.
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

    # Cache safety (report #3): tie the shared normalizer / FD-DAC reference to
    # the dataset + duration + split they were computed on, so a stale cache from
    # a different preprocessing / split can never be silently reused.
    _n_frames_fp = frames_per_chunk(cfg.paths.dataset_root, cfg.model.duration_s)
    _validate_cache(cache_dir, _cache_fingerprint(cfg, _n_frames_fp),
                    guarded_files=[normalizer_path, fd_dac_cache_path])

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

    # Which distributional metrics to compute, mirroring the unconditional
    # project's `metrics.enabled` registry. A metric listed here is an EXPLICIT
    # request: asking for one this pipeline cannot produce is a HARD ERROR at
    # startup (not a silent skip), so you never train for hours believing a
    # metric is on when it is not.
    metrics_enabled = list(
        metrics_cfg.get("enabled", list(DAC_METRICS))
        if metrics_cfg is not None else list(DAC_METRICS))
    _unknown = [m for m in metrics_enabled if m not in DAC_METRICS]
    if _unknown:
        raise SystemExit(
            f"[metrics] metrics.enabled contains {_unknown}, which the CONDITIONED "
            f"pipeline does not provide. Available: {list(DAC_METRICS)}. "
            f"(The audio-FAD metrics fad_encodec/fad_vggish belong to the "
            f"unconditional evaluate_generation path and are not wired here.)")
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
                f"--conditions melody --device cuda")
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
            # split config (data.split in the YAML; leakage-safe defaults)
            split_ratios=tuple(_split_param(cfg, "ratios", [0.8, 0.1, 0.1])),
            split_seed=int(_split_param(cfg, "seed", 42)),
            group_by_source=bool(_split_param(cfg, "group_by_source", True)),
            stratify_by_class=bool(_split_param(cfg, "stratify_by_class", True)),
            save_test_manifest=bool(_split_param(cfg, "save_test_manifest", True)),
        )

    n_classes = len(label_map)
    print(f"\nDetected {n_classes} classes: {list(label_map.keys())}")
    print(f"[split] files  -> train {split_info['file_counts']['train']} | "
          f"val {split_info['file_counts']['val']} | "
          f"test {split_info['file_counts']['test']}")
    if split_info["manifest_path"]:
        print(f"[split] test manifest: {split_info['manifest_path']}")

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
        val_dataset, batch_size=cfg.data.val_batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=(device == "cuda"),
        persistent_workers=(n_workers > 0),
        drop_last=False, collate_fn=collate_conditioned,
    )

    train_iter = infinite_loader(train_loader)
    val_iter   = infinite_loader(val_loader)

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

    # Conditioning-influence evaluators (validation only, never affect training):
    #   - frame conditions: re-extract the enabled frame conditions from the
    #     generations and compare, paired, to the input ones (melody RPA/RCA,
    #     chroma cosine, rhythm/energy correlation).
    #   - text (CLAP): the audio side of the same CLAP checkpoint, to score
    #     audio<->text adherence; loaded lazily ONLY if 'text' is active.
    # The per-condition influence (delta vs null) is consolidated in the
    # Validation/Condition_influence text panel.
    from condition_metrics import ConditionFidelityEvaluator
    fidelity_evaluator = ConditionFidelityEvaluator(
        enabled_frame=list(FRAME_COND_DIMS.keys()),
        device=device,
        registry=registry,   # #15: re-extract with the run's exact extractor config
    )
    clap_audio_embedder = None
    if "text" in GLOBAL_CONFIGS:
        from conditions import ClapAudioEmbedder
        # match the CLAP checkpoint used by the text condition
        clap_model_name = CONDITION_CONFIG["global"]["text"]["kwargs"].get(
            "model_name", "laion/larger_clap_music")
        clap_audio_embedder = ClapAudioEmbedder(model_name=clap_model_name, device=device)
        print(f"Text-influence (CLAP audio) enabled: {clap_model_name}")

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
    _cfg_log.data.split.composition = {
        "file_counts":  dict(split_info["file_counts"]),
        "n_classes":    int(split_info["n_classes"]),
        "test_manifest": split_info["manifest_path"],
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
        # Restore every RNG stream so the run continues EXACTLY (x0, t, CFG
        # dropout, shuffle) instead of restarting from the seed. Best-effort for
        # old checkpoints without rng_state.
        _restore_rng_state(ckpt.get("rng_state"), data_generator)
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
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
    print(f"CFG dropout: all={cfg.conditioning.p_drop_all} "
          f"frame={cfg.conditioning.p_drop_frame} "
          f"global={cfg.conditioning.p_drop_global}")
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
    # the first metrics step. Tags: Validation/Audio_real_* / Spectrogram_real_*.
    log_real_audio_samples(
        val_dataset=val_dataset,
        normalizer=normalizer,
        writer=writer,
        n_samples=cfg.sampling.n_audio_samples,
    )

    # ======================
    # Real audio / spectrogram / melody are logged inside evaluate_and_log_metrics
    # at every metrics step (tags Validation/*_real_*), alongside the with-cond
    # and without-cond generations, so the whole comparison sits in the same
    # windows and walks with the TensorBoard step slider.

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
                n_val = cfg.data.num_val_batches

                if device == "cuda":
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    val_losses = []
                    for _ in range(n_val):
                        vb = next(val_iter)
                        vl = compute_loss(
                            model, vb, device,
                            use_amp=cfg.training.use_amp,
                            t_min=cfg.sampling.t_min,
                            t_max=cfg.sampling.t_max,
                            global_configs=GLOBAL_CONFIGS,
                            p_drop_all=cfg.conditioning.p_drop_all,
                            p_drop_frame=cfg.conditioning.p_drop_frame,
                            p_drop_global=cfg.conditioning.p_drop_global,
                            training=False,
                        ).item()
                        val_losses.append(vl)
                        del vb
                    val_loss = sum(val_losses) / len(val_losses)

                    ema_val_loss = val_loss
                    if cfg.training.use_ema and step >= cfg.training.ema_start:
                        ema_vl = []
                        for _ in range(n_val):
                            vb = next(val_iter)
                            evl = compute_loss(
                                ema.model, vb, device,
                                use_amp=cfg.training.use_amp,
                                t_min=cfg.sampling.t_min,
                                t_max=cfg.sampling.t_max,
                                global_configs=GLOBAL_CONFIGS,
                                p_drop_all=cfg.conditioning.p_drop_all,
                                p_drop_frame=cfg.conditioning.p_drop_frame,
                                p_drop_global=cfg.conditioning.p_drop_global,
                                training=False,
                            ).item()
                            ema_vl.append(evl)
                            del vb
                        ema_val_loss = sum(ema_vl) / len(ema_vl)
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
            # Generated-audio preview every intervals.audio steps (tags
            # Validation/Audio_generated_*). The richer real / with-cond /
            # without-cond comparison is logged separately at every metrics step.
            # NB: no enable flag -- `intervals.audio` means what it says. (It used
            # to be gated by an `audio_preview` toggle, so the interval silently
            # did nothing; set intervals.audio very large to effectively disable.)
            # ======================
            if step > 0 and step % cfg.intervals.audio == 0:
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
