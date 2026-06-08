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
#   - FD-DAC + FAD computed every intervals.metrics step on conditioned
#     samples, with reference pre-computed on the full validation set
#     (reference is unchanged: real validation latents / WAVs)
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
    AudioLatentDataset, DAC_LATENT_DIM, DAC_SAMPLE_RATE,
)
from network_cond import ConditionedAudioDiT, TOKEN_DIM
from audio_dataset_cond import (
    build_conditioned_datasets, collate_conditioned,
)
from conditions import (
    ConditionRegistry,
    make_null_frame_conditions, make_null_global_conditions,
)
from metrics import (
    FADCalculator,
    precompute_fd_dac_reference,
    compute_fd_dac,
)


# ======================
# CONFIG LOADING
# ======================
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

    # CLI overrides win over everything (YAML + checkpoint config).
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, cli_cfg)

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
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

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


def make_pianoroll(melody_onehot, title=""):
    """Render an (T, 88) one-hot melody as a piano-roll image (pitch x time).
    Row 0 = MIDI 21 (A0), row 87 = MIDI 108 (C8). Uses plot_to_image."""
    import numpy as _np
    m = _np.asarray(melody_onehot, dtype=_np.float32)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(m.T, aspect='auto', origin='lower', interpolation='nearest',
              cmap='magma', vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pitch bin (0=A0 .. 87=C8)")
    img = plot_to_image(fig)
    plt.close(fig)
    return img
@torch.no_grad()
def euler_sample_cfg(model, n_frames, device, steps, t_min, t_max, use_amp,
                      frame_cond, global_cond, guidance,
                      frame_dims, global_configs):
    """
    Euler integrator with classifier-free guidance.
    Both `frame_cond` and `global_cond` are expected as batch=1 dicts on device.
    If `guidance` <= 1.0 or both conditioning sources are absent, a single
    forward pass per step is used.
    """
    model.eval()
    x = torch.randn(1, n_frames, TOKEN_DIM, device=device)
    dt = (t_max - t_min) / steps

    null_fc = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc = make_null_global_conditions(1, global_configs or {}, device)

    has_cond = (frame_cond is not None) or (global_cond is not None)
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

    # Decode with DAC on CPU
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu")
    dac_model.eval()

    for i, gen in enumerate(generated_frames):
        if not torch.isfinite(gen).all():
            continue
        z = gen.T
        z = normalizer.denormalize(z)
        z_in = z.unsqueeze(0).float()
        waveform = dac_model.decode(z_in).squeeze(0)
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

    del dac_model


# ======================
# LOG REAL AUDIO SAMPLES (once at startup, step=0)
# ======================
@torch.no_grad()
def log_real_audio_samples(val_dataset, normalizer, writer, n_samples):
    """Logs real audio from the val dataset for comparison on TensorBoard."""
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu")
    dac_model.eval()

    total = len(val_dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()

    for i, idx in enumerate(indices):
        # ConditionedAudioDataset returns a 5-tuple: take only frames + label
        frames, _frame_cond, label_idx, _text_emb, _image_emb = val_dataset[idx]
        class_name = val_dataset.idx_to_label.get(label_idx, str(label_idx))
        z = frames.T
        z = normalizer.denormalize(z)
        z_in = z.unsqueeze(0).float()
        waveform = dac_model.decode(z_in).squeeze(0)
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

    del dac_model
    print(f"  {n_samples} real audios logged on TensorBoard")


# ======================
# METRICS EVALUATION (conditioned generation, FD-DAC + FAD)
# ======================
@torch.no_grad()
def evaluate_and_log_metrics(
    model, normalizer, val_dataset, step, writer, device, output_dir,
    fad_calculator, fd_dac_ref_stats, n_samples,
    sampling_cfg, conditioning_cfg, use_amp,
    frame_dims, global_configs,
    fidelity_evaluator=None, compute_uncond=True,
    prefix="EMA",
):
    """
    Computes the full validation metric suite on a FIXED subset of the val set
    (deterministic linspace indices, so the curves are comparable across steps),
    in three independent axes:

      1. UNCONDITIONAL generation  -> Fd_dac_uncond / Fad_uncond
         Generated with NULL conditions (no CFG). This is the ONLY metric that
         is apples-to-apples comparable with the unconditional model: it answers
         "how well does this (conditioned-trained) model generate freely?".

      2. CONDITIONAL generation    -> Fd_dac_cond / Fad_cond
         Each sample generated from one specific validation condition (with CFG
         guidance). Distributional fidelity of the conditioned generations to
         the real data. NOT comparable with the unconditional model (the
         conditioning restricts the distribution).

      3. CONDITIONING FIDELITY     -> Cond_fidelity/<name>/<metric>
         Paired: re-extract the condition from each conditioned generation and
         compare it one-to-one with the input condition (melody RPA/RCA, chroma
         cosine, rhythm correlation). Answers "does it follow the condition?".

    FD-DAC / FAD references stay the real validation latents / WAVs in both the
    cond and uncond cases. `prefix` ("EMA" / "Model") tags the audio/spectrogram
    by the generating weights. Returns (fd_dac_cond, fad_cond) for the caller.
    """
    guidance = float(conditioning_cfg.guidance_scale)
    n_frames = val_dataset.n_frames
    total = len(val_dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()
    n_log = min(2, n_samples)   # how many samples to log richly (audio/specs/rolls)

    print(f"\n  Compute metrics @ step {step}: {n_samples} generations "
          f"(cond guidance={guidance}"
          f"{', + uncond' if compute_uncond else ''}) "
          f"vs reference ({fad_calculator.ref_n_samples} samples)...")

    # ---- generate n_samples latents, conditioned or unconditional ----
    # For the conditioned pass we also keep, for the first n_log samples, the
    # real latent (to decode the real audio) and the target condition.
    def _generate(conditioned):
        lat_list = []
        targets = []        # paired input conditions (cpu numpy); cond only
        real_frames = []    # real latents of the first n_log samples; cond only
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
            )
            lat_list.append(gen)
            if device == "cuda":
                torch.cuda.empty_cache()
        return lat_list, targets, real_frames

    # ---- DAC decode helpers (model loaded once, reused) ----
    def _decode_one(frames, dac_model):
        z = normalizer.denormalize(frames.T)
        wav = dac_model.decode(z.unsqueeze(0).float()).squeeze()
        return wav.cpu()

    def _decode(lat_list, dac_model):
        return [_decode_one(f, dac_model) for f in lat_list]

    # FAD only needs the reference STATISTICS to be loaded (mu_ref / sigma_ref).
    # `ref_dir` is the reference-WAV folder, used ONLY when (re)computing those
    # stats from disk; on a cache-hit it may be None even though the stats are
    # loaded, so gating FAD on ref_dir wrongly skips it. Gate on mu_ref instead.
    has_fad = (fad_calculator is not None and fad_calculator.mu_ref is not None)

    # ===== CONDITIONAL generation =====
    cond_lat, cond_targets, real_frames = _generate(conditioned=True)
    fd_dac_cond = (compute_fd_dac(torch.stack(cond_lat), fd_dac_ref_stats, device=device)
                   if fd_dac_ref_stats is not None else None)

    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu")
    dac_model.eval()

    cond_wavs = _decode(cond_lat, dac_model)
    fad_cond = (fad_calculator.compute_fad_against_reference(cond_wavs, sr=DAC_SAMPLE_RATE)
                if has_fad else None)

    # ===== UNCONDITIONAL generation (comparable to the unconditional model) =====
    fd_dac_uncond = None
    fad_uncond = None
    unc_wavs = []
    if compute_uncond:
        unc_lat, _, _ = _generate(conditioned=False)
        fd_dac_uncond = (compute_fd_dac(torch.stack(unc_lat), fd_dac_ref_stats, device=device)
                         if fd_dac_ref_stats is not None else None)
        unc_wavs = _decode(unc_lat, dac_model)
        fad_uncond = (fad_calculator.compute_fad_against_reference(unc_wavs, sr=DAC_SAMPLE_RATE)
                      if has_fad else None)

    # ===== CONDITIONING FIDELITY (paired, on the conditioned generations) =====
    fidelity = {}
    if fidelity_evaluator is not None and fidelity_evaluator.active:
        print(f"    re-extracting conditions from {len(cond_wavs)} generations "
              f"for fidelity ({list(fidelity_evaluator.extractors)})...")
        fidelity_evaluator.reset()
        for wav, tgt in zip(cond_wavs, cond_targets):
            fidelity_evaluator.add_sample(wav.numpy(), DAC_SAMPLE_RATE, n_frames, tgt)
        fidelity = fidelity_evaluator.results()

    # ===== LOG SCALARS =====
    if fd_dac_cond is not None:
        writer.add_scalar("Validation/Metrics/Fd_dac_cond", fd_dac_cond, step)
    if fad_cond is not None:
        writer.add_scalar("Validation/Metrics/Fad_cond", fad_cond, step)
    if fd_dac_uncond is not None:
        writer.add_scalar("Validation/Metrics/Fd_dac_uncond", fd_dac_uncond, step)
    if fad_uncond is not None:
        writer.add_scalar("Validation/Metrics/Fad_uncond", fad_uncond, step)
    for key, val in fidelity.items():
        writer.add_scalar(f"Validation/Cond_fidelity/{key}", val, step)

    # ===== console summary =====
    def _f(x):
        return f"{x:.4f}" if x is not None else "n/a"
    print(f"  [cond]   FD-DAC: {_f(fd_dac_cond)} | FAD: {_f(fad_cond)}")
    if compute_uncond:
        print(f"  [uncond] FD-DAC: {_f(fd_dac_uncond)} | FAD: {_f(fad_uncond)}")
    if fidelity:
        print("  [fidelity] " + " | ".join(f"{k}={v:.4f}" for k, v in fidelity.items()))

    # ===== RICH COMPARISON LOGGING (real / with-cond / without-cond) =====
    # For the first n_log samples (same indices as the metrics generations),
    # log, all tagged by the generating weights `w` (EMA or Model):
    #   IMAGES: spectrogram + melody piano-roll, for real / with-cond / without-cond
    #   AUDIO : the audio + the sonified melody, for real / with-cond / without-cond
    # Everything is logged with global_step=step so TensorBoard's slider walks
    # across the metric steps. The melody of the generations is RE-EXTRACTED from
    # the generated audio with the same extractor that produced the targets.
    from condition_metrics import sonify_melody
    import numpy as _np
    w = prefix  # "EMA" or "Model"
    mel_ext = (fidelity_evaluator.extractors.get("melody")
               if (fidelity_evaluator is not None and fidelity_evaluator.active)
               else None)

    def _log_audio(wav, tag):
        wav_u = wav.unsqueeze(0) if wav.dim() == 1 else wav
        wn = wav_u / (wav_u.abs().max() + 1e-8)
        writer.add_audio(f"Validation/{tag}", wn, global_step=step,
                         sample_rate=DAC_SAMPLE_RATE)

    def _log_mel_audio(mel_arr, tag):
        son = torch.from_numpy(sonify_melody(mel_arr, DAC_SAMPLE_RATE)).unsqueeze(0)
        writer.add_audio(f"Validation/{tag}", son, global_step=step,
                         sample_rate=DAC_SAMPLE_RATE)

    def _log_spec(wav, tag, title):
        wav_u = wav.unsqueeze(0) if wav.dim() == 1 else wav
        writer.add_image(f"Validation/{tag}",
                         make_spectrogram(wav_u, DAC_SAMPLE_RATE, title),
                         global_step=step)

    def _log_roll(mel_arr, tag, title):
        writer.add_image(f"Validation/{tag}",
                         make_pianoroll(mel_arr, title), global_step=step)

    def _reextract(wav):
        if mel_ext is None:
            return None
        a = wav.numpy() if wav.dim() == 1 else wav.squeeze().numpy()
        return mel_ext.extract(_np.ascontiguousarray(a, dtype=_np.float32),
                               DAC_SAMPLE_RATE, n_frames)

    for i in range(n_log):
        # ---------- REAL ----------
        real_wav = _decode_one(real_frames[i], dac_model)
        _log_audio(real_wav, f"Audio_real_{i:02d}")
        _log_spec(real_wav, f"Spectrogram_real_{i:02d}", f"real {i} - step {step}")
        real_mel = cond_targets[i].get("melody") if i < len(cond_targets) else None
        if real_mel is not None:
            _log_roll(real_mel, f"Melody_pianoroll_real_{i:02d}",
                      f"melody real {i} - step {step}")
            _log_mel_audio(real_mel, f"Melody_audio_real_{i:02d}")

        # ---------- GENERATED WITH COND ----------
        cwav = cond_wavs[i]
        _log_audio(cwav, f"Audio_generated_with_cond_{w}_{i:02d}")
        _log_spec(cwav, f"Spectrogram_generated_with_cond_{w}_{i:02d}",
                  f"generated with cond ({w}) {i} - step {step} - guidance={guidance}")
        cmel = _reextract(cwav)
        if cmel is not None:
            _log_roll(cmel, f"Melody_pianoroll_generated_with_cond_{w}_{i:02d}",
                      f"melody generated with cond ({w}) {i} - step {step}")
            _log_mel_audio(cmel, f"Melody_audio_generated_with_cond_{w}_{i:02d}")

        # ---------- GENERATED WITHOUT COND ----------
        if compute_uncond:
            uwav = unc_wavs[i]
            _log_audio(uwav, f"Audio_generated_without_cond_{w}_{i:02d}")
            _log_spec(uwav, f"Spectrogram_generated_without_cond_{w}_{i:02d}",
                      f"generated without cond ({w}) {i} - step {step}")
            umel = _reextract(uwav)
            if umel is not None:
                _log_roll(umel, f"Melody_pianoroll_generated_without_cond_{w}_{i:02d}",
                          f"melody generated without cond ({w}) {i} - step {step}")
                _log_mel_audio(umel, f"Melody_audio_generated_without_cond_{w}_{i:02d}")

        # ---------- save wavs to disk ----------
        sf.write(os.path.join(output_dir, f"metrics_step{step:07d}_real_{i:02d}.wav"),
                 real_wav.numpy(), DAC_SAMPLE_RATE)
        sf.write(os.path.join(output_dir, f"metrics_step{step:07d}_with_cond_{w}_{i:02d}.wav"),
                 cwav.numpy() if cwav.dim() == 1 else cwav.squeeze().numpy(),
                 DAC_SAMPLE_RATE)
        if compute_uncond:
            sf.write(os.path.join(output_dir, f"metrics_step{step:07d}_without_cond_{w}_{i:02d}.wav"),
                     uwav.numpy() if uwav.dim() == 1 else uwav.squeeze().numpy(),
                     DAC_SAMPLE_RATE)

    del dac_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Backward-compatible return: the conditioned FD-DAC / FAD.
    return fd_dac_cond, fad_cond


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
# CHECKPOINT HELPER
# ======================
def build_ckpt_data(model, ema, optimizer, scheduler, scaler, step,
                    val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                    frame_cond_dims, frame_cond_out_dims, global_configs):
    """
    Assemble the conditioned checkpoint dict. The full `config` is stored so a
    later --resume can rebuild the exact same model / conditioning / training
    setup without the user having to re-pass model.kind, the enabled conditions,
    batch sizes, etc. `model_kind` and the per-condition dims are also kept as
    top-level fields for backward compatibility with sampling_cond.py /
    test_cond.py (which read them directly from the checkpoint).
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
    }
    if cfg.training.use_ema and ema is not None:
        data["ema_state_dict"] = ema.state_dict()
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
    fad_cache_dir     = os.path.join(cache_dir, "fad_cache")

    # Config's dump (with CLI override already applied) in the run dir
    config_dump_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(cfg, config_dump_path)
    print(f"[CONFIG DUMP] {config_dump_path}")
    print(f"[RUN DIR]     {run_dir}")
    print(f"[CACHE DIR]   {cache_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")

    # ======================
    # CONDITION REGISTRY
    # ======================
    # The active set of conditions (which frame extractors, which global
    # encoders, their dims) comes from CONDITION_CONFIG in conditions.py.
    # The class count is scanned from the train latents folder, but is
    # only used to drive text-CLAP prompt building inside the dataset.
    tmp_train = AudioLatentDataset(
        root_dir=cfg.paths.dataset_root, split="train",
        duration_s=cfg.model.duration_s, normalizer=None, preload=False,
    )
    n_classes = len(tmp_train.label_to_idx)
    print(f"\nDetected {n_classes} classes: {list(tmp_train.label_to_idx.keys())}")
    del tmp_train

    # Per-run selection of which conditions to activate, driven by the YAML
    # config. Default null = use all conditions enabled=True in CONDITION_CONFIG.
    # Empty list [] = use no condition of that type. Explicit list = subset.
    enabled_f = cfg.conditioning.get("enabled_frame",  None)
    enabled_g = cfg.conditioning.get("enabled_global", None)
    # OmegaConf converts YAML null -> None, YAML list -> ListConfig.
    # Convert ListConfig to a plain list before passing to the registry.
    if enabled_f is not None:
        enabled_f = list(enabled_f)
    if enabled_g is not None:
        enabled_g = list(enabled_g)

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

    train_dataset, val_dataset, normalizer, label_map = build_conditioned_datasets(
        latent_root=cfg.paths.dataset_root,
        condition_root=cond_root,
        image_root=img_root,
        duration_s=cfg.model.duration_s,
        normalizer_path=(normalizer_path
                         if os.path.exists(normalizer_path) else None),
        registry=registry,
        preload=False,
    )

    # Save the normalizer in the cache_dir
    if not os.path.exists(normalizer_path):
        normalizer.save(normalizer_path)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.train_batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
        drop_last=True, collate_fn=collate_conditioned,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.val_batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
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

    fd_dac_ref_stats = precompute_fd_dac_reference(
        metrics_val_ds,
        cache_path=fd_dac_cache_path,
    )

    fad_calculator = FADCalculator(device="cuda")
    fad_calculator.precompute_reference_stats(
        val_dataset=metrics_val_ds,
        normalizer=normalizer,
        wav_root=cfg.paths.wav_root,
        latent_root=cfg.paths.dataset_root,
        sr=DAC_SAMPLE_RATE,
        cache_dir=fad_cache_dir,
    )
    print(f"Reference stats ready: "
          f"FD-DAC on {len(val_dataset)} latent samples, "
          f"FAD on {fad_calculator.ref_n_samples} audio samples\n")

    # Conditioning-fidelity evaluator: re-extracts the enabled frame conditions
    # from the conditioned generations and compares them, paired, to the input
    # conditions (melody RPA/RCA, chroma cosine, rhythm correlation). Built once
    # from the active frame conditions; evaluation-only, never affects training.
    from condition_metrics import ConditionFidelityEvaluator
    fidelity_evaluator = ConditionFidelityEvaluator(
        enabled_frame=list(FRAME_COND_DIMS.keys()),
        device=device,
    )
    metrics_uncond = bool(cfg.sampling.get("metrics_uncond", True))
    print(f"Conditioning fidelity: {list(FRAME_COND_DIMS.keys()) or 'none'} "
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

    # Logs the config in TB (tab "Text" -> visible on the dashboard)
    writer.add_text(
        "config",
        "```yaml\n" + OmegaConf.to_yaml(cfg) + "\n```",
        global_step=0,
    )

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

        # Free the CPU copy of the checkpoint and clear any cached GPU blocks
        # left over from the load before the training loop starts.
        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
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
    print(f"Metrics: {cfg.sampling.n_metrics_samples} generated vs "
          f"{fad_calculator.ref_n_samples} reference")
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
                        FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS)
                    torch.save(ckpt_data, save_path)
                    for old in Path(ckpt_dir).glob("best_model_step*.pt"):
                        if old.resolve() != Path(save_path).resolve():
                            old.unlink()
                    pbar.write(f"  -> Best model: {save_path}")

            # ======================
            # AUDIO PREVIEW (optional, conditioned-only, separate from metrics)
            # Off by default: the full real / with-cond / without-cond comparison
            # (audio + spectrogram + melody piano-roll + sonified melody) is
            # logged at every metrics step inside evaluate_and_log_metrics. Enable
            # intervals.audio_preview to ALSO get more frequent conditioned-only
            # previews between metric steps (tags Validation/Audio_generated_*).
            # ======================
            if (cfg.intervals.get("audio_preview", False)
                    and step > 0 and step % cfg.intervals.audio == 0):
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
            # METRICS (FD-DAC + FAD) on conditioned generations
            # ======================
            if step > 0 and step % cfg.intervals.metrics == 0:
                gen_model = (ema.model
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else model)
                fd_dac, fad = evaluate_and_log_metrics(
                    model=gen_model,
                    normalizer=normalizer,
                    val_dataset=val_dataset,
                    step=step,
                    writer=writer,
                    device=device,
                    output_dir=audio_dir,
                    fad_calculator=fad_calculator,
                    fd_dac_ref_stats=fd_dac_ref_stats,
                    n_samples=cfg.sampling.n_metrics_samples,
                    sampling_cfg=cfg.sampling,
                    conditioning_cfg=cfg.conditioning,
                    use_amp=cfg.training.use_amp,
                    frame_dims=FRAME_COND_DIMS,
                    global_configs=GLOBAL_CONFIGS,
                    fidelity_evaluator=fidelity_evaluator,
                    compute_uncond=metrics_uncond,
                    prefix=("EMA"
                            if cfg.training.use_ema and step >= cfg.training.ema_start
                            else "Model"),
                )
                model.train()

            # ======================
            # PERIODICAL CHECKPOINT
            # ======================
            if step % cfg.intervals.ckpt == 0 and step > 0:
                p = os.path.join(ckpt_dir, f"checkpoint_step{step}.pt")
                ckpt_data = build_ckpt_data(
                    model, ema, optimizer, scheduler, scaler, step,
                    val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                    FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS)
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
                FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS)
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
