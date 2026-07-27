# test_cond.py
#
# Generates conditioned audio with the EMA model and compares each
# generation to its real counterpart on TensorBoard.
# Aligned with the refactored training_cond.py (OmegaConf + run_name
# layout, ConditionedAudioDiT, classifier-free guidance).
#
# For every test-set sample, the test pulls its multi-modal conditions
# (pitch + chroma + CLAP-text + CLIP-image), generates a latent
# trajectory with CFG, decodes it through DAC, and logs both the
# generation and the real reference on TensorBoard.
#
# The test set is the SAME one training persisted: the split parameters are
# restored from the checkpoint's config, so compute_split reuses the exact
# splits/test_split_<hash>.json written at train time (no leakage vs training,
# comparable conditioned generations across checkpoints).
#
# Optional overrides:
#   --prompt   forces a single CLAP text embedding for ALL generations
#              (replaces the per-sample text embedding from the test set)
#   --image    forces a single CLIP image embedding for ALL generations
#              (replaces the per-sample image embedding from the test set)
#   --guidance overrides cfg.conditioning.guidance_scale
#   --seed     reproducible generation (seeds python/numpy/torch and the
#              per-generation noise via a dedicated Generator)
#
# Frame-level conditions (pitch, chroma) are always taken from the test
# sample they belong to (no global override makes physical sense for
# time-varying signals).
#
# Usage:
#   python test_cond.py --ckpt runs/<run_name>/checkpoints/best_model.pt
#   python test_cond.py --ckpt runs/<run_name>/checkpoints/best_model.pt \
#       --config configs/cond_default.yaml --guidance 5.0
#   python test_cond.py --ckpt path/to/ckpt.pt --n_samples 16 --steps 100 \
#       --prompt "slow piano in C minor"
#   python test_cond.py --ckpt path/to/ckpt.pt --image path/to/cover.jpg
#
# Outputs:
#   - WAV files in runs/<run_name>/test_outputs/
#   - TensorBoard logs in runs/<run_name>/test_logs/
#     (visible alongside the training logs of the same run)

import os
os.environ.setdefault("USE_TF", "0")   # transformers -> PyTorch backend (no TF)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Use a machine-local cache for HuggingFace / DAC weights (avoids NFS issues).
os.environ.setdefault("XDG_CACHE_HOME", "/data/anasynth_nonbp/baione/.cache")

import argparse
import random
from io import BytesIO
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from audio_dataset_npy import (
    LatentNormalizer,
    decode_latents,
    DAC_SAMPLE_RATE,
)
from audio_dataset_cond import ConditionedAudioDataset, compute_split
from network_cond import ConditionedAudioDiT, TOKEN_DIM
from conditions import (
    ConditionRegistry,
    ImageDatasetManager,
    CLAPTextCondition,
    ImageCondition,
    make_null_frame_conditions,
    make_null_global_conditions,
)


# ============================================================
# CLI / CONFIG LOADING
# ============================================================
def load_config():
    """
    Builds the test config the SAME way training_cond.py rebuilds it on --resume:
    the full training config stored inside the checkpoint is authoritative and is
    layered ON TOP of the (now optional) external YAML, then CLI overrides win.

    This guarantees the model is tested with the exact dataset_root /
    condition_root / cache_dir / duration_s / precision / conditioning selection
    it was trained with, instead of whatever the external YAML happens to hold.

    Precedence (low -> high):
        external YAML  <  checkpoint config  <  CLI dotlist  <  CLI scalar flags

    The external YAML is only REQUIRED for old checkpoints that stored no full
    config (only `model_kind`); otherwise it is an optional base.
    Returns (cfg, args).
    """
    parser = argparse.ArgumentParser(
        description="Generate conditioned audio with a trained Audio DiT "
                     "and compare with real test samples.",
        add_help=True,
    )
    parser.add_argument("--config", type=str,
                        default="configs/cond_default.yaml",
                        help="OPTIONAL base YAML config. The checkpoint's stored "
                              "training config takes priority and is layered on "
                              "top of this YAML; if the YAML is missing and the "
                              "checkpoint has a full config, the checkpoint "
                              "config is used as the sole base "
                              "(default: configs/cond_default.yaml)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pt) - typically "
                              "runs/<run_name>/checkpoints/best_model.pt")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override run_name. If not given, it is inferred "
                              "from the checkpoint path "
                              "(runs/<run_name>/checkpoints/...).")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of samples to generate (default: 8)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of Euler steps "
                              "(default: cfg.sampling.euler_steps)")
    parser.add_argument("--duration_s", type=float, default=None,
                        help="Audio duration in seconds "
                              "(default: cfg.model.duration_s)")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Classifier-free guidance scale "
                              "(default: cfg.conditioning.guidance_scale)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducible generation (default: free-running)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Free CLAP text prompt that overrides the "
                              "per-sample text embedding for ALL generations "
                              "(e.g. 'slow piano in C minor')")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path that overrides the per-sample CLIP "
                              "embedding for ALL generations.")
    parser.add_argument("--allow_invalid_conditions", action="store_true",
                        help="DEBUG ONLY: tolerate missing/corrupt conditions on the "
                             "test set (zero-fill them). Off by default so the "
                             "evaluation cannot silently score NULL-conditioned "
                             "generations as if they were conditioned (report #12).")
    args, unknown = parser.parse_known_args()

    # ---- Base config from the external YAML (now OPTIONAL) ----
    base_cfg = OmegaConf.load(args.config) if os.path.exists(args.config) else None

    # ---- Training config restored from the checkpoint (authoritative) ----
    # Only the lightweight metadata is read here (map_location='cpu'); the model
    # weights are (re)loaded later in main(). Mirrors training_cond.load_config().
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    _meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_config     = _meta.get("config")
    ckpt_model_kind = _meta.get("model_kind")
    del _meta

    if ckpt_config is not None:
        # Normal case: the checkpoint carries the full training config. Layer it
        # ON TOP of the YAML base (or use it alone if no YAML is present).
        ckpt_cfg = OmegaConf.create(ckpt_config)
        cfg = ckpt_cfg if base_cfg is None else OmegaConf.merge(base_cfg, ckpt_cfg)
        print("[test_cond] Config restored from checkpoint "
              f"(model.kind={cfg.model.kind}, duration_s={cfg.model.duration_s}, "
              f"dataset_root={cfg.paths.dataset_root}, "
              f"enabled_frame={cfg.conditioning.enabled_frame}, "
              f"enabled_global={cfg.conditioning.enabled_global}).")
        if base_cfg is None:
            print(f"[test_cond] (external YAML '{args.config}' not found; using "
                   f"the checkpoint config as the sole base.)")
    else:
        # Old checkpoint without a stored config: the external YAML is mandatory.
        if base_cfg is None:
            raise FileNotFoundError(
                f"Config not found ('{args.config}') AND the checkpoint stores "
                f"no 'config'. Pass --config with a valid YAML matching the "
                f"training setup (dataset_root, duration_s, cache_dir, ...)."
            )
        cfg = base_cfg
        if ckpt_model_kind is not None:
            # model.kind MUST match to load the weights at all.
            cfg.model.kind = ckpt_model_kind
            print(f"[test_cond] model.kind restored from checkpoint: "
                   f"{cfg.model.kind} (old checkpoint without full config; other "
                   f"params come from the YAML/CLI).")
        else:
            print("[test_cond] WARNING: checkpoint has neither 'config' nor "
                   "'model_kind'; all params come from the YAML/CLI.")

    # ---- CLI overrides win over everything (YAML + checkpoint config) ----
    # CLI dotlist overrides (e.g. sampling.euler_steps=80)
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # CLI scalars take priority when explicitly set
    if args.steps is not None:
        cfg.sampling.euler_steps = args.steps
    if args.duration_s is not None:
        cfg.model.duration_s = args.duration_s
    if args.guidance is not None:
        cfg.conditioning.guidance_scale = args.guidance

    # ---- run_name (test output location), set AFTER the merge so the checkpoint
    # config's own paths.run_name never leaks into the test output path ----
    # Expected layout: runs/<run_name>/checkpoints/<file>.pt
    if args.run_name is not None:
        run_name = args.run_name
    else:
        ckpt_path = Path(args.ckpt).resolve()
        # Walk up: <run_name>/checkpoints/<file>
        if ckpt_path.parent.name == "checkpoints":
            run_name = ckpt_path.parent.parent.name
        else:
            run_name = "test"   # fallback if checkpoint is not in the expected layout
    cfg.paths.run_name = run_name

    return cfg, args


# ============================================================
# UTILITIES
# ============================================================
def plot_to_image(fig):
    """Convert a matplotlib figure to a torch tensor (3, H, W) for TensorBoard."""
    import torchvision
    import PIL.Image as Image
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = torchvision.transforms.ToTensor()(Image.open(buf))
    buf.close()
    return img


def make_spectrogram_image(waveform, sample_rate, title=""):
    """Build a mel-spectrogram image (tensor) for TensorBoard."""
    import torchaudio   # lazy: only a pure-DSP transform, no torchcodec/FFmpeg
    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=128, n_fft=2048, hop_length=512,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()
    spec_db = amp_to_db(spec_transform(waveform.cpu().float()))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(spec_db[0].numpy(), aspect="auto", origin="lower",
              cmap="viridis", vmin=-80, vmax=0)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel Bin")
    plt.colorbar(ax.images[0], ax=ax, label="dB")
    img = plot_to_image(fig)
    plt.close(fig)
    return img


# ============================================================
# EULER SAMPLING WITH CFG (standalone copy - kept in sync with training_cond.py)
# ============================================================
@torch.no_grad()
def euler_sample_cfg(model, n_frames, device, steps, t_min, t_max, use_amp,
                      frame_cond, global_cond, guidance,
                      frame_dims, global_configs, gen_rng=None):
    """
    Euler integrator with classifier-free guidance.
    Both `frame_cond` and `global_cond` are expected as batch=1 dicts on device.
    If `guidance` <= 1.0 or both conditioning sources are absent, a single
    forward pass per step is used.
    `gen_rng` (optional torch.Generator) fixes the initial-noise x0 for
    reproducible generation; None = free-running. Signature kept identical to
    training_cond.euler_sample_cfg.
    """
    model.eval()
    x = torch.randn(1, n_frames, TOKEN_DIM, device=device, generator=gen_rng)
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


# ============================================================
# MAIN
# ============================================================
def main():
    cfg, args = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducible generation (A/B): seed the global RNG and build a dedicated
    # Generator for the per-generation noise (mirrors training_cond metrics seed).
    gen_rng = None
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(int(args.seed))
        print(f"[test_cond] Seed: {args.seed}")

    print(f"[test_cond] Device:     {device}")
    print(f"[test_cond] Config:     {args.config}")
    print(f"[test_cond] Checkpoint: {args.ckpt}")
    print(f"[test_cond] Run name:   {cfg.paths.run_name}")

    # Output paths
    run_dir    = Path(cfg.paths.runs_dir) / cfg.paths.run_name
    output_dir = run_dir / "test_outputs"
    log_dir    = run_dir / "test_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[test_cond] Outputs:    {output_dir}")
    print(f"[test_cond] TB logs:    {log_dir}")

    writer = SummaryWriter(str(log_dir))

    # ============================================================
    # LOAD CHECKPOINT + MODEL
    # ============================================================
    print(f"\n[test_cond] Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    model_kind          = ckpt.get("model_kind",          cfg.model.kind)
    frame_cond_dims     = ckpt.get("frame_cond_dims",     {})
    frame_cond_out_dims = ckpt.get("frame_cond_out_dims", {})
    global_configs      = ckpt.get("global_configs",      {})
    # Back-compat: if a checkpoint stored frame_cond_dims but not
    # frame_cond_out_dims, recover the per-condition out_dim from
    # CONDITION_CONFIG (conditions.py) so the model can still be rebuilt.
    if frame_cond_dims and not frame_cond_out_dims:
        _reg = ConditionRegistry(
            enabled_frame=list(frame_cond_dims.keys()), enabled_global=[],
        )
        frame_cond_out_dims = _reg.frame_cond_out_dims
    print(f"[test_cond] Model kind:           {model_kind}")
    print(f"[test_cond] Frame cond dims:      {frame_cond_dims}")
    print(f"[test_cond] Frame cond out dims:  {frame_cond_out_dims}")
    print(f"[test_cond] Global cond configs:  {global_configs}")

    model = ConditionedAudioDiT(
        kind=model_kind,
        frame_cond_dims=frame_cond_dims,
        frame_cond_out_dims=frame_cond_out_dims,
        global_cond_configs=global_configs,
    ).to(device)

    # Prefer the EMA weights, but ONLY if the shadow was being updated when the
    # checkpoint was written (before training.ema_start it is still the random
    # init). Old checkpoints have no 'ema_ready' key -> assume ready.
    if "ema_state_dict" in ckpt and ckpt.get("ema_ready", True):
        model.load_state_dict(ckpt["ema_state_dict"])
        print("[test_cond] Using EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            print("[test_cond] EMA present but NOT trained yet (checkpoint "
                  "predates training.ema_start) - using main model weights")
        else:
            print("[test_cond] EMA not available - using main model weights")
    model.eval()

    # ============================================================
    # NORMALIZER
    # ============================================================
    # Resolve normalizer path: prefer cache_dir from config, fallback to run_dir
    cache_dir = Path(cfg.paths.cache_dir)
    normalizer_candidates = [
        cache_dir / "normalizer.pt",
        run_dir / "checkpoints" / "normalizer.pt",
    ]
    normalizer_path = None
    for p in normalizer_candidates:
        if p.exists():
            normalizer_path = p
            break
    if normalizer_path is None:
        raise FileNotFoundError(
            f"normalizer.pt not found in any of: "
            f"{[str(p) for p in normalizer_candidates]}"
        )

    normalizer = LatentNormalizer()
    normalizer.load(str(normalizer_path))
    print(f"[test_cond] Normalizer: {normalizer_path}")

    # Label map (for naming)
    label_map     = ckpt.get("label_map", {})
    idx_to_label  = {v: k for k, v in label_map.items()}

    # ============================================================
    # TEST SET (conditioned)
    # ============================================================
    # The registry is built from CONDITION_CONFIG in conditions.py; the
    # checkpoint stores frame_cond_dims and global_configs to drive the
    # model architecture, but the dataset still needs the live extractors
    # to pre-compute text embeddings (CLAP) and to know which frame names
    # to load from the .npz files.
    # Build the registry restricted to the conditions actually present in
    # the checkpoint, so we instantiate only the encoders we really need
    # (e.g. for a pitch-only training, no CLAP / CLIP are loaded here).
    registry = ConditionRegistry(
        enabled_frame  = list(frame_cond_dims.keys()),
        enabled_global = list(global_configs.keys()),
    )

    cond_root = (cfg.paths.condition_root
                  if Path(cfg.paths.condition_root).exists() else None)
    img_root  = (cfg.paths.image_root
                  if Path(cfg.paths.image_root).exists() else None)

    # Reuse the SAME test set training persisted. The split parameters come from
    # the checkpoint's config (restored in load_config), so the manifest hash
    # matches and compute_split reuses the exact test_split_<hash>.json written
    # at train time. save_test_manifest=False: never create one from the test.
    def _sp(key, default):
        d = cfg.data.get("split", None) if hasattr(cfg, "data") else None
        return default if d is None else d.get(key, default)

    split = compute_split(
        cfg.paths.dataset_root,
        ratios=tuple(_sp("ratios", [0.8, 0.1, 0.1])),
        seed=int(_sp("seed", 42)),
        group_by_source=bool(_sp("group_by_source", True)),
        stratify_by_class=bool(_sp("stratify_by_class", True)),
        save_test_manifest=False,
    )
    test_files = split["splits"]["test"]
    if not test_files:
        raise RuntimeError(
            f"Empty test split for {cfg.paths.dataset_root}. Expected a persisted "
            f"test manifest under {Path(cfg.paths.dataset_root).parent/'splits'} "
            f"(written by training_cond.py). Check the split params match training.")

    # label_to_idx: the checkpoint's mapping is authoritative (it is what the
    # model's class conditioning was trained with); fall back to the split scan.
    if label_map:
        dataset_label_map = dict(label_map)
    else:
        dataset_label_map = {c: i for i, c in enumerate(split["classes"])}
    # keep the naming map consistent with the mapping the dataset actually uses
    idx_to_label = {v: k for k, v in dataset_label_map.items()}

    # image manager only if image conditioning is active (split-less: split=None)
    image_manager = None
    image_active = "image" in getattr(registry, "global_extractors", {})
    if image_active and img_root is not None:
        try:
            image_manager = ImageDatasetManager(img_root, split=None)
        except Exception:
            print("[test_cond] image_root present but unreadable -> per-sample "
                  "image embeddings will be zeros unless --image is provided.")
            image_manager = None

    test_dataset = ConditionedAudioDataset(
        files=test_files,
        label_to_idx=dataset_label_map,
        split="test",
        latent_root=cfg.paths.dataset_root,
        condition_root=cond_root,
        image_root=img_root,
        duration_s=cfg.model.duration_s,
        normalizer=normalizer,
        registry=registry,
        image_manager=image_manager,
        preload_latents=False,
        strict_conditions=not args.allow_invalid_conditions,   # #12: strict by default
    )

    total = len(test_dataset)
    if total == 0:
        raise RuntimeError(
            f"Test split has {len(test_files)} files but 0 usable chunks "
            f"(duration_s / n_frames mismatch?).")

    n_samples = min(args.n_samples, total)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()
    print(f"[test_cond] Test set: {total} samples | using {n_samples}")

    # ============================================================
    # OPTIONAL GLOBAL OVERRIDES (--prompt, --image)
    # ============================================================
    text_emb_override  = None
    image_emb_override = None

    if args.prompt is not None:
        if "text" not in global_configs:
            print(f"[test_cond] WARNING: --prompt was given but the model "
                   f"has no 'text' global condition. Ignored.")
        else:
            text_enc = CLAPTextCondition()
            emb = text_enc.encode_text(args.prompt)
            text_emb_override = torch.from_numpy(emb).to(device)
            text_enc.unload()
            print(f"[test_cond] CLAP-text OVERRIDE: {args.prompt!r}")

    if args.image is not None:
        if "image" not in global_configs:
            print(f"[test_cond] WARNING: --image was given but the model "
                   f"has no 'image' global condition. Ignored.")
        elif not os.path.exists(args.image):
            print(f"[test_cond] WARNING: --image path does not exist "
                   f"({args.image}). Ignored.")
        else:
            img_enc = ImageCondition()
            emb = img_enc.encode_image(args.image)
            image_emb_override = torch.from_numpy(emb).to(device)
            img_enc.unload()
            print(f"[test_cond] CLIP-image OVERRIDE: {args.image}")

    # ============================================================
    # GENERATION + LOGGING
    # ============================================================
    n_frames    = test_dataset.n_frames
    euler_steps = int(cfg.sampling.euler_steps)
    guidance    = float(cfg.conditioning.guidance_scale)
    use_amp     = bool(cfg.training.use_amp)

    print(f"\n[test_cond] --- Generating {n_samples} samples "
          f"({n_frames} frames each, {euler_steps} Euler steps, "
          f"guidance={guidance}) ---")

    for i, idx in enumerate(indices):
        frames_real, frame_cond_real, label_idx, text_emb, image_emb \
            = test_dataset[idx]
        label_name = idx_to_label.get(label_idx, str(label_idx))
        print(f"\n[test_cond] Sample {i+1}/{n_samples} | "
              f"idx={idx} | label={label_name}")

        # Build batch=1 conditions on device, applying global overrides if any
        fc = {k: v.unsqueeze(0).to(device).float()
              for k, v in frame_cond_real.items()}
        gc = {}
        if "text" in global_configs:
            if text_emb_override is not None:
                gc["text"] = text_emb_override.unsqueeze(0)
            else:
                gc["text"] = text_emb.unsqueeze(0).to(device)
        if "image" in global_configs:
            if image_emb_override is not None:
                gc["image"] = image_emb_override.unsqueeze(0)
            else:
                gc["image"] = image_emb.unsqueeze(0).to(device)

        # --- Generate latent with CFG ---
        with torch.no_grad():
            frames_gen = euler_sample_cfg(
                model=model, n_frames=n_frames, device=device,
                steps=euler_steps,
                t_min=cfg.sampling.t_min, t_max=cfg.sampling.t_max,
                use_amp=use_amp,
                frame_cond=fc, global_cond=gc, guidance=guidance,
                frame_dims=frame_cond_dims, global_configs=global_configs,
                gen_rng=gen_rng,
            )

        # --- Latent -> audio (denormalize + DAC decode via from_latents) ---
        z_gen = frames_gen.T                       # (72, n_frames)
        z_gen = normalizer.denormalize(z_gen)
        waveform_gen = decode_latents(z_gen, device=device)

        # Save WAV
        out_path = output_dir / f"generated_{i:04d}_{label_name}.wav"
        sf.write(str(out_path), waveform_gen.cpu().numpy().T, DAC_SAMPLE_RATE)
        print(f"[test_cond] Saved: {out_path}")

        # --- Log generated audio ---
        wn = waveform_gen / (waveform_gen.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/generated/{label_name}", wn.cpu(),
            global_step=i, sample_rate=DAC_SAMPLE_RATE,
        )

        # --- Generated spectrogram ---
        spec_img_gen = make_spectrogram_image(
            waveform_gen, DAC_SAMPLE_RATE,
            f"Generated - {label_name}",
        )
        writer.add_image(
            f"Spectrogram/generated/{label_name}", spec_img_gen, global_step=i,
        )

        # --- Real audio reference ---
        z_real = frames_real.T                     # (72, n_frames)
        z_real = normalizer.denormalize(z_real)
        waveform_real = decode_latents(z_real, device=device)
        wrn = waveform_real / (waveform_real.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/real/{label_name}", wrn.cpu(),
            global_step=i, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img_real = make_spectrogram_image(
            waveform_real, DAC_SAMPLE_RATE,
            f"Real - {label_name}",
        )
        writer.add_image(
            f"Spectrogram/real/{label_name}", spec_img_real, global_step=i,
        )

    writer.close()
    print(f"\n[test_cond] Done!")
    print(f"[test_cond] WAV files:    {output_dir}")
    print(f"[test_cond] TensorBoard:  tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
