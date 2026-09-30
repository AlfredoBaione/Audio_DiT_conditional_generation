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
# The test set is the SAME one the training held out: both read it from the
# dataset's splits.json (written by preprocess_stream.py). Nothing is
# recomputed, so the test set cannot drift from the one the checkpoint was
# trained against (no leakage, comparable generations across checkpoints).
#
# Optional overrides:
#   --prompt   forces a single CLAP text embedding for ALL generations
#              (replaces the per-sample text embedding from the test set)
#   --image    forces a single CLIP image embedding for ALL generations
#              (replaces the per-sample image embedding from the test set)
#   --guidance overrides cfg.conditioning.guidance_scale
#   --seed     reproducible generation (seeds python/numpy/torch and the
#              per-generation noise via a dedicated Generator)
#   --metrics_samples N | all
#              ALSO compute the metrics on the test set -- FD-DAC, KL (both
#              directions), FAD-VGGish (those listed in metrics.enabled) and
#              the condition-influence table -- over N test samples spread
#              over the whole split, or over every one of them with 'all'.
#              Off by default. See run_test_metrics for what is computed and
#              how; --n_samples then only says how many of those generations
#              are also saved as WAV / logged as audio to listen to.
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
#   python test_cond.py --ckpt path/to/ckpt.pt --metrics_samples all
#   python test_cond.py --ckpt path/to/ckpt.pt --metrics_samples 256 \
#       metrics.fad_reference=decoded
# On an IRCAM GPU server, through the test's own GPU-lock wrapper (same
# arguments, passed through unchanged):
#   python launch_test_cond.py --ckpt path/to/ckpt.pt --metrics_samples all
#
# Outputs:
#   - WAV files in runs/<run_name>/test_outputs/
#   - TensorBoard logs in runs/<run_name>/test_logs/
#     (visible alongside the training logs of the same run)
#   - with --metrics_samples: the numbers also in
#     runs/<run_name>/test_outputs/metrics_<checkpoint>_<N|all>.json, and on
#     TensorBoard as Test/Metrics/* scalars + the Test/Condition_influence
#     table, at the checkpoint's training step (so several checkpoints of one
#     run line up as a curve)

import os
os.environ.setdefault("USE_TF", "0")   # transformers -> PyTorch backend (no TF)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# Use a machine-local cache for HuggingFace / DAC weights (avoids NFS issues).
os.environ.setdefault("XDG_CACHE_HOME", "/data/anasynth_nonbp/baione/.cache")
# TORCH_HOME is an ASSIGNMENT, and guarded to IRCAM: the nodes already export
# it into the SHARED conda env (read-only for us), and torch.hub prefers it
# over XDG_CACHE_HOME -- the first download on a machine with a cold cache
# (beat_this fetching its checkpoint) would die with PermissionError.
if os.path.isdir("/data/anasynth_nonbp/baione"):
    os.environ["TORCH_HOME"] = "/data/anasynth_nonbp/baione/.cache/torch"

import argparse
import random
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from audio_dataset_npy import (
    LatentNormalizer,
    decode_latents,
    DAC_SAMPLE_RATE,
)
from audio_dataset_cond import ConditionedAudioDataset, load_source_split
from network_cond import (ConditionedAudioDiT, TOKEN_DIM,
                          ckpt_frame_reinject_every,
                          ckpt_text_cross_every, ckpt_text_ctx_dim,
                          check_ckpt_reinject_gate)
from conditions import (
    ConditionRegistry,
    # CLAPTextCondition / ImageCondition are still needed HERE: --prompt and
    # --image encode a free prompt or a chosen picture at inference time. What
    # is gone is ImageDatasetManager -- the per-sample image now comes from the
    # dataset's own CLIP bank, not from scanning the image folder again.
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
    parser.add_argument("--metrics_samples", type=str, default=None,
                        metavar="N|all",
                        help="Also compute the metrics on the TEST set: FD-DAC, "
                             "KL, FAD-VGGish (those in metrics.enabled) and the "
                             "condition-influence table, over N test samples "
                             "spread over the whole split, or 'all' of them. "
                             "Off by default (generation for listening only). "
                             "--n_samples then says how many of the scored "
                             "generations are also saved / logged as audio.")
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
# EULER SAMPLING WITH CFG (standalone copy - kept in sync with training_cond.py)
# ============================================================
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
    `gen_rng` (optional torch.Generator) fixes the initial-noise x0 for
    reproducible generation; None = free-running. Signature kept identical to
    training_cond.euler_sample_cfg.
    """
    model.eval()
    x = torch.randn(1, n_frames, TOKEN_DIM, device=device, generator=gen_rng)
    dt = (t_max - t_min) / steps

    null_fc = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc = make_null_global_conditions(1, global_configs or {}, device)

    # NB: test the CONTENT, not `is not None`: with every condition disabled the
    # caller passes empty dicts ({}), which are "no conditioning" -- treating them
    # as present would engage CFG and burn two IDENTICAL forwards per step.
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


# ============================================================
# TEST-SET METRICS  (--metrics_samples)
# ============================================================
def parse_metrics_samples(value):
    """--metrics_samples -> None (off), 'all', or a positive int."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v == "all":
        return "all"
    try:
        n = int(v)
    except ValueError:
        n = 0
    if n <= 0:
        raise SystemExit(f"[test_cond] --metrics_samples must be a positive "
                         f"number or 'all', got {value!r}.")
    return n


@torch.no_grad()
def run_test_metrics(tc, model, normalizer, test_dataset, cfg, args, device,
                     frame_dims, global_configs, registry, writer, output_dir,
                     idx_to_label, n_metrics, ckpt_step, weights):
    """
    Generate the scored test samples ONCE and compute on them every metric the
    training computes on the validation set -- the same definitions, on the
    held-out split:

      FD-DAC, KL (both directions)  latent-only, against the REAL TEST latents,
                                    all of them (the validation reference is
                                    the whole val split in the same way);
      FAD-VGGish                    decode + VGGish, against the real TEST audio:
                                    metrics.fad_reference 'wav' -> the test wavs
                                    on disk, 'decoded' -> the test latents
                                    through DAC;
      condition influence           each sample is generated WITH its conditions
                                    and WITHOUT, from the same noise (the fused
                                    CFG sampler hands the second one over nearly
                                    for free); each condition is re-extracted
                                    from both and Δ = with-cond - without, paired
                                    sample by sample, scored with
                                    metrics.influence_family.

    Everything that decides HOW -- which metrics (metrics.enabled), whether the
    no-condition generations are also scored distributionally
    (sampling.metrics_uncond), the noise seed (metrics.seed, or --seed), the
    influence family and threshold, guidance, Euler steps, how the text slot is
    filled (conditioning.text_source + sampling.validation_text_from_caption)
    -- comes from the checkpoint's config, and CLI dotlist overrides still win.
    A test number is therefore the twin of the validation curve of the run.

    The references are computed here, on the test split, every time: they are
    cheap next to the generation, and nothing is written into the shared
    cache_dir, whose files are tied to the training's own splits.

    STREAMED: each generation is decoded ONCE and serves all the metrics that
    need audio (re-extraction, CLAP/Wav2CLIP, VGGish); the waveform is then
    dropped. Only the latents (for FD-DAC/KL, ~0.12 MB each) and the listening
    clips stay. Memory is flat in the number of samples; the cost is time, and
    the time is the generation.
    """
    import json
    import time
    from tqdm import tqdm
    from condition_metrics import (ConditionFidelityEvaluator, pair_influence,
                                   pair_scalar, format_influence_panel,
                                   format_influence_legend)
    from metrics import (precompute_latent_reference, compute_dac_metrics,
                         precompute_audio_reference, compute_audio_mu_sigma,
                         compute_mu_sigma, compute_fad, COND_METRICS)

    frame_dims = frame_dims or {}
    global_configs = global_configs or {}
    mcfg = cfg.get("metrics", None) or {}
    enabled = list(mcfg.get("enabled", list(COND_METRICS)) or [])
    _unknown = [m for m in enabled if m not in COND_METRICS]
    if _unknown:
        raise SystemExit(f"[test_cond] metrics.enabled contains {_unknown}; "
                         f"available here: {list(COND_METRICS)}.")

    total = len(test_dataset)
    n_scored = total if n_metrics == "all" else min(int(n_metrics), total)
    # Spread over the WHOLE split (same rule as the listening path): a prefix
    # would describe the first classes/sources of the list, not the test set.
    indices = (list(range(total)) if n_scored == total else
               torch.linspace(0, total - 1, n_scored).long().tolist())
    n_listen = min(int(args.n_samples), n_scored)
    listen_pos = sorted(set(
        torch.linspace(0, n_scored - 1, n_listen).long().tolist())) if n_listen else []

    n_frames = test_dataset.n_frames
    steps = int(cfg.sampling.euler_steps)
    t_min, t_max = float(cfg.sampling.t_min), float(cfg.sampling.t_max)
    guidance = float(cfg.conditioning.guidance_scale)
    use_amp = bool(cfg.training.use_amp)
    spf = int(cfg.sampling.get("metrics_samples_per_forward", 1))
    compute_uncond = bool(cfg.sampling.get("metrics_uncond", False))
    seed = args.seed if args.seed is not None else mcfg.get("seed", 0)
    family = str(mcfg.get("influence_family", "influence_metrics"))
    mir_threshold = float(mcfg.get("mir_threshold", 0.5))
    any_cond = bool(frame_dims) or bool(global_configs)
    # Fused: B samples per forward, batch 3B, with-cond AND without from one x0.
    # The serial path (spf=0, or no CFG to fuse) makes the "without" with a
    # second generator on the same seed, as the training does.
    paired = spf >= 1 and guidance > 1.0 and any_cond
    group = max(1, spf) if paired else 1

    def _dev(key, fallback="cuda"):
        d = str(mcfg.get(key, fallback))
        return "cpu" if d.startswith("cuda") and not torch.cuda.is_available() else d
    fid_device, fad_device = _dev("fidelity_device"), _dev("fad_device")

    def decode(frames):
        """(n_frames, 72) normalized latent -> 1-D waveform on CPU. The test's
        own decoder, for the generations AND a 'decoded' FAD reference alike."""
        z = normalizer.denormalize(frames.T)
        return decode_latents(z, device=device).reshape(-1).float().cpu()

    print(f"\n[test_cond] === TEST-SET METRICS on {n_scored}/{total} samples "
          f"({weights} weights, step {ckpt_step}) ===")
    print(f"[test_cond] metrics: {enabled or '(no distributional metric)'} | "
          f"influence: {family if frame_dims else '-'}"
          f"{' + ' + '/'.join(sorted(global_configs)) if global_configs else ''}"
          f" | guidance={guidance} | {steps} Euler steps | seed={seed} | "
          f"{'fused, ' + str(group) + ' sample(s) per forward' if paired else 'serial sampler'}")

    # ---------------- references: the TEST split ----------------
    lat_root = Path(cfg.paths.dataset_root)
    dac_ref = None
    if "fd_dac" in enabled or "kl_dac" in enabled:
        dac_ref = precompute_latent_reference(
            tc.MetricsAdapter(test_dataset), cache_path=None, device=device)

    fad_emb = fad_ref = None
    fad_mode = str(mcfg.get("fad_reference", "wav"))
    if "fad_vggish" in enabled:
        from metrics import VGGishEmbedder
        fad_emb = VGGishEmbedder(device=fad_device)
        # One reference clip per test CHUNK, from the split's own file list
        # (never a glob of wav/, which holds every split together).
        _npy = sorted({str(s[0]) for s in test_dataset.samples})
        if fad_mode == "wav":
            _wav_root = Path(cfg.paths.wav_root)
            _wavs = [_wav_root / Path(f).relative_to(lat_root).with_suffix(".wav")
                     for f in _npy]
            _missing = [w for w in _wavs if not w.exists()]
            if _missing:
                raise SystemExit(
                    f"[test_cond] FAD-VGGish with metrics.fad_reference='wav' "
                    f"needs the real TEST wavs, but {len(_missing)}/{len(_wavs)} "
                    f"are missing under {_wav_root} (first: {_missing[0]}).\n"
                    f"  Either add them -- re-run preprocess_stream.py on the "
                    f"same output dir with --save_wav test (it does NOT "
                    f"re-encode the latents) -- or run this test with "
                    f"metrics.fad_reference=decoded, which decodes the real test "
                    f"latents through DAC instead (not comparable with published "
                    f"FAD values).")
            fad_ref = precompute_audio_reference(_wavs, fad_emb, cache_path=None,
                                                 device=fad_device)
        elif fad_mode == "decoded":
            def _real_clips():
                for i in range(total):
                    yield decode(test_dataset[i][0]).view(1, 1, -1), DAC_SAMPLE_RATE
            _mu, _sig, _n = compute_audio_mu_sigma(
                _real_clips(), total, fad_emb, device=fad_device,
                desc="FAD ref (test, decoded)")
            fad_ref = {"mu": _mu.cpu(), "sigma": _sig.cpu(), "n_total": _n}
        else:
            raise SystemExit(
                f"[test_cond] metrics.fad_reference='{fad_mode}' is not valid: "
                f"'wav' (real test wavs) or 'decoded' (test latents via DAC).")

    # ---------------- influence: scorers ----------------
    ev_c = ev_n = None
    if frame_dims:
        def _evaluator():
            return ConditionFidelityEvaluator(
                enabled_frame=list(frame_dims), device=fid_device,
                registry=registry, family=family, mir_threshold=mir_threshold)
        # Two, one per pass: with-cond and without are scored in the same loop,
        # and each evaluator keeps its own per-sample values to pair afterwards.
        ev_c, ev_n = _evaluator(), _evaluator()
        for _name in ev_c.extractors:
            print(f"[test_cond] influence {_name}: {ev_c.families[_name]}")
    gemb = {}
    if "text" in global_configs:
        from conditions import ClapAudioEmbedder, CONDITION_CONFIG
        gemb["text"] = ClapAudioEmbedder(
            model_name=CONDITION_CONFIG["global"]["text"]["kwargs"].get(
                "model_name", "laion/clap-htsat-unfused"),
            device=fid_device)
    if "image" in global_configs:
        try:
            from conditions import Wav2ClipAudioEmbedder
            _w2c = Wav2ClipAudioEmbedder(device=fid_device)
            _w2c._load()
            gemb["image"] = _w2c
        except Exception as _e:
            print(f"[test_cond] image influence unavailable: "
                  f"{type(_e).__name__}: {_e}")
    gsim_names = sorted(gemb)

    # The text slot of a scored generation: the caption's CLAP TEXT vector when
    # sampling.validation_text_from_caption is on (what the validation metrics
    # use), else the dataset's own vector (conditioning.text_source).
    cap_ids, cap_emb = {}, None
    if (bool(cfg.sampling.get("validation_text_from_caption", False))
            and "text" in global_configs):
        cap_ids, cap_emb = tc.load_caption_conditions(test_dataset.latent_root)
        print("[test_cond] text slot: "
              + ("the caption's CLAP text vector (validation_text_from_caption)"
                 if cap_emb is not None else
                 "no caption embeddings in this dataset -> the chunk's own vector"))

    def text_vec(idx, text_emb):
        if cap_emb is None:
            return text_emb
        cid = cap_ids.get(tc._chunk_key_of(test_dataset,
                                           test_dataset.samples[idx][1]))
        return text_emb if cid is None else torch.from_numpy(cap_emb[cid].copy())

    wants_ctx = bool(getattr(model, "text_cross_layers", None))

    # ---------------- generate + score, streamed ----------------
    def _new_stats():
        return {"sx": None, "sxx": None, "n": 0}

    def _add_stats(st, emb):
        e = emb.to(dtype=torch.float64)
        if st["sx"] is None:
            st["sx"] = torch.zeros(e.shape[-1], dtype=torch.float64, device=e.device)
            st["sxx"] = torch.zeros(e.shape[-1], e.shape[-1], dtype=torch.float64,
                                    device=e.device)
        st["sx"] += e.sum(dim=0)
        st["sxx"] += e.T @ e
        st["n"] += e.shape[0]

    fad_c, fad_u = _new_stats(), _new_stats()
    gs_c = {c: {} for c in gsim_names}
    gs_n = {c: {} for c in gsim_names}
    cond_lat, null_lat = [], []
    need_null = any_cond          # the influence baseline
    gen_rng = null_rng = None
    if seed is not None:
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(int(seed))
        if not paired and need_null:
            null_rng = torch.Generator(device=device)
            null_rng.manual_seed(int(seed))
    n_listened = 0
    t0 = time.time()

    def _score_one(p, wav, ev, gs, targets, gtargets, fad_stats):
        wn = wav.numpy()
        if ev is not None:
            ev.add_sample(wn, DAC_SAMPLE_RATE, n_frames, targets, sample_id=p)
        for c in gsim_names:
            tgt = gtargets.get(c)
            if tgt is None:
                continue
            try:
                gs[c][p] = float(np.dot(gemb[c].embed(wn, DAC_SAMPLE_RATE),
                                        np.asarray(tgt).reshape(-1)))
            except Exception as _e:
                if not gs[c]:
                    print(f"[test_cond] {c} similarity unavailable: "
                          f"{type(_e).__name__}: {_e}")
        if fad_stats is not None:
            _add_stats(fad_stats, fad_emb.embed(wav.view(1, 1, -1),
                                                DAC_SAMPLE_RATE))

    bar = tqdm(total=n_scored, desc="Test metrics")
    for start in range(0, n_scored, group):
        grp = list(range(start, min(start + group, n_scored)))
        items = []
        for p in grp:
            (frames_real, frame_cond, label_idx, text_emb, image_emb,
             text_ctx) = test_dataset[indices[p]]
            items.append((frames_real, frame_cond, label_idx,
                          text_vec(indices[p], text_emb), image_emb, text_ctx))
        fc = {k: torch.stack([it[1][k] for it in items]).to(device).float()
              for k in frame_dims}
        gc = {}
        if "text" in global_configs:
            gc["text"] = torch.stack([it[3] for it in items]).to(device)
        if "image" in global_configs:
            gc["image"] = torch.stack([it[4] for it in items]).to(device)
        ctx = ({"tokens": torch.stack([it[5]["tokens"] for it in items]),
                "mask": torch.stack([it[5]["mask"] for it in items])}
               if wants_ctx else None)

        if paired:
            lats_c, lats_u = tc.euler_sample_cfg_paired(
                model, n_frames, device, steps=steps, t_min=t_min, t_max=t_max,
                use_amp=use_amp, frame_cond=fc, global_cond=gc,
                guidance=guidance, frame_dims=frame_dims,
                global_configs=global_configs, gen_rng=gen_rng, text_ctx=ctx)
        else:
            lats_c = [tc.euler_sample_cfg(
                model, n_frames, device, steps=steps, t_min=t_min, t_max=t_max,
                use_amp=use_amp, frame_cond=fc or None, global_cond=gc or None,
                guidance=guidance, frame_dims=frame_dims,
                global_configs=global_configs, gen_rng=gen_rng, text_ctx=ctx)]
            lats_u = [tc.euler_sample_cfg(
                model, n_frames, device, steps=steps, t_min=t_min, t_max=t_max,
                use_amp=use_amp, frame_cond=None, global_cond=None,
                guidance=1.0, frame_dims=frame_dims,
                global_configs=global_configs, gen_rng=null_rng,
                text_ctx=None)] if need_null else [None]

        for k, p in enumerate(grp):
            frames_real, frame_cond, label_idx, t_vec, i_vec, _ = items[k]
            targets = {c: v.cpu().numpy() for c, v in frame_cond.items()}
            gtargets = {"text": t_vec.cpu().numpy() if "text" in global_configs else None,
                        "image": i_vec.cpu().numpy() if "image" in global_configs else None}
            cond_lat.append(lats_c[k])
            wav_c = decode(lats_c[k])
            _score_one(p, wav_c, ev_c, gs_c, targets, gtargets,
                       fad_c if fad_emb is not None else None)
            if lats_u[k] is not None:
                null_lat.append(lats_u[k])
                wav_u = decode(lats_u[k])
                _score_one(p, wav_u, ev_n, gs_n, targets, gtargets,
                           fad_u if (fad_emb is not None and compute_uncond) else None)
            elif not any_cond:
                # No condition at all: the "conditioned" generation IS the
                # unconditional one (null inputs, no CFG) -- nothing to pair.
                null_lat.append(lats_c[k])

            if p in listen_pos:
                # Same files and tags as a plain listening run.
                label_name = idx_to_label.get(label_idx, str(label_idx))
                out_path = output_dir / f"generated_{n_listened:04d}_{label_name}.wav"
                sf.write(str(out_path), wav_c.numpy(), DAC_SAMPLE_RATE)
                writer.add_audio(f"Audio/generated/{label_name}",
                                 (wav_c / (wav_c.abs().max() + 1e-8)).view(1, -1),
                                 global_step=n_listened, sample_rate=DAC_SAMPLE_RATE)
                wav_r = decode(frames_real)
                writer.add_audio(f"Audio/real/{label_name}",
                                 (wav_r / (wav_r.abs().max() + 1e-8)).view(1, -1),
                                 global_step=n_listened, sample_rate=DAC_SAMPLE_RATE)
                n_listened += 1
            bar.update(1)
    bar.close()
    elapsed = time.time() - t0

    # ---------------- distributional metrics ----------------
    dist = {}
    if dac_ref is not None:
        for tag, lats in (("cond", cond_lat),
                          ("uncond", null_lat if compute_uncond else [])):
            if not lats:
                continue
            m = compute_dac_metrics(torch.stack(lats), dac_ref, enabled=enabled,
                                    device=device)
            dist[f"fd_dac_{tag}"] = m["fd_dac"]
            dist[f"kl_{tag}_real_gen"] = m["kl_real_gen"]
            dist[f"kl_{tag}_gen_real"] = m["kl_gen_real"]
    if fad_emb is not None:
        for tag, st in (("cond", fad_c), ("uncond", fad_u)):
            if st["n"] > 1:
                mu, sig, _ = compute_mu_sigma(st["sx"], st["sxx"], st["n"])
                dist[f"fad_vggish_{tag}"] = compute_fad(mu, sig, fad_ref,
                                                        device=fad_device)

    # ---------------- condition influence (paired) ----------------
    have_null = bool(null_lat) and any_cond
    influence, cov = {}, {}
    if ev_c is not None:
        influence, cov = pair_influence(ev_c.per_sample(), ev_n.per_sample(),
                                        coverage_cond=ev_c.coverage(),
                                        have_null=have_null)
    _gmetric = {"text": "clap_sim", "image": "clip_sim"}
    for c in gsim_names:
        if not gs_c[c]:
            continue
        _cm, _nm, _dm, _npair = pair_scalar(gs_c[c], gs_n[c], have_null=have_null)
        key = _gmetric.get(c, "sim")
        influence[c] = {key: {"cond": _cm, "null": _nm, "delta": _dm}}
        cov[f"{c}/{key}"] = {
            "valid": _npair, "attempted": len(gs_c[c]),
            "unpaired": len(set(gs_c[c]) ^ set(gs_n[c])) if have_null else 0}
    for c in global_configs:
        if c not in influence:
            influence[c] = {_gmetric.get(c, "sim"): {
                "cond": None, "null": None, "delta": None,
                "note": ("no audio->CLIP embedder: pip install wav2clip"
                         if c == "image" else "no embedder for this condition")}}
    # Rows read "<cond>_test", as the training's read "<cond>_validation".
    inf_named = {f"{c}_test": v for c, v in influence.items()}
    cov_named = {}
    for k, v in cov.items():
        c, _, m = k.partition("/")
        cov_named[f"{c}_test/{m}"] = v
    panel = (format_influence_panel(inf_named, step=ckpt_step,
                                    prefix=f"{weights} · test set",
                                    guidance=guidance, n_samples=n_scored,
                                    coverage=cov_named)
             if influence else "")

    # ---------------- report: console, TensorBoard, JSON ----------------
    _tb = {"fd_dac_cond": "Fd_dac_cond", "kl_cond_real_gen": "Kl_cond/real_gen",
           "kl_cond_gen_real": "Kl_cond/gen_real",
           "fd_dac_uncond": "Fd_dac_uncond",
           "kl_uncond_real_gen": "Kl_uncond/real_gen",
           "kl_uncond_gen_real": "Kl_uncond/gen_real",
           "fad_vggish_cond": "Fad_vggish_cond",
           "fad_vggish_uncond": "Fad_vggish_uncond"}
    print(f"\n[test_cond] {n_scored} test samples in {elapsed / 60:.1f} min "
          f"({elapsed / max(1, n_scored):.1f} s per sample)")
    for k, v in dist.items():
        if v is not None:
            print(f"[test_cond]   {_tb[k]:22s} {v:.4f}")
            writer.add_scalar(f"Test/Metrics/{_tb[k]}", v, ckpt_step)
    if panel:
        print("\n" + panel + "\n")
        writer.add_text("Test/Condition_influence", panel, ckpt_step)
        writer.add_text("Test/Condition_influence_legend",
                        format_influence_legend(), 0)

    tag = "all" if n_scored == total else str(n_scored)
    out_json = output_dir / f"metrics_{Path(args.ckpt).stem}_{tag}.json"
    out_json.write_text(json.dumps({
        "checkpoint": str(args.ckpt), "step": ckpt_step, "weights": weights,
        "n_scored": n_scored, "n_test": total, "seed": seed,
        "guidance": guidance, "euler_steps": steps,
        "metrics_enabled": enabled, "fad_reference": fad_mode,
        "metrics_uncond": compute_uncond, "influence_family": family,
        "mir_threshold": mir_threshold,
        "text_from_caption": cap_emb is not None,
        "minutes": round(elapsed / 60, 2),
        "distributional": dist, "influence": influence, "coverage": cov,
        "influence_table_markdown": panel,
    }, indent=2, default=float), encoding="utf-8")
    print(f"[test_cond] metrics saved: {out_json}")
    print(f"[test_cond] {n_listened} of them also saved to listen to: {output_dir}")


# ============================================================
# MAIN
# ============================================================
def main():
    cfg, args = load_config()

    # ---- test-set metrics (--metrics_samples) ----
    n_metrics = parse_metrics_samples(args.metrics_samples)
    tc = None
    if n_metrics is not None:
        if args.prompt is not None or args.image is not None:
            raise SystemExit(
                "[test_cond] --metrics_samples cannot be combined with --prompt "
                "/ --image: the metrics score the test set's OWN conditions, and "
                "an override replaces them for every generation.")
        # The metrics reuse the training's own sampler, reference adapter and
        # caption lookup, so a test number is computed exactly like its
        # validation twin. Importing the module also applies its process
        # settings -- TF32 matmuls, the CUDA allocator config, the IRCAM cache
        # redirection -- i.e. the ones the validation metrics run under. Done
        # HERE, before anything touches CUDA, and only for a metrics run: a
        # plain listening test runs exactly as before.
        import training_cond as tc

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

    # Architecture parameter (it adds one tensor per selected block to the
    # state_dict), so it has to be rebuilt exactly as trained. 0 for any
    # checkpoint written before the option existed.
    frame_reinject_every = ckpt_frame_reinject_every(ckpt)
    check_ckpt_reinject_gate(ckpt, args.ckpt)
    print(f"[test_cond] Frame re-injection:    "
          f"{'every ' + str(frame_reinject_every) + ' block(s)' if frame_reinject_every > 0 else 'OFF'}")

    # The cross-attention, same contract: read the stride off the checkpoint
    # and the CONTEXT WIDTH off the weights themselves, which is the only
    # source that cannot disagree with the tensors being loaded.
    text_cross_every = ckpt_text_cross_every(ckpt)
    text_ctx_dim = ckpt_text_ctx_dim(ckpt)

    model = ConditionedAudioDiT(
        kind=model_kind,
        frame_cond_dims=frame_cond_dims,
        frame_cond_out_dims=frame_cond_out_dims,
        global_cond_configs=global_configs,
        frame_reinject_every=frame_reinject_every,
        text_cross_every=text_cross_every,
        text_ctx_dim=text_ctx_dim,
    ).to(device)

    # Prefer the EMA weights, but ONLY if the shadow was being updated when the
    # checkpoint was written (before training.ema_start it is still the random
    # init). Old checkpoints have no 'ema_ready' key -> assume ready.
    weights = "EMA"
    if "ema_state_dict" in ckpt and ckpt.get("ema_ready", True):
        model.load_state_dict(ckpt["ema_state_dict"])
        print("[test_cond] Using EMA weights")
    else:
        weights = "Model"
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

    # The test set is READ from the dataset's splits.json -- the same file the
    # training read. Nothing is recomputed here, so the test set cannot differ
    # from the one the checkpoint was held out from: that used to depend on the
    # split parameters restored from the checkpoint config still matching.
    split = load_source_split(
        cfg.paths.dataset_root,
        splits_path=cfg.paths.get("splits_path", None)
        if hasattr(cfg, "paths") else None,
    )
    test_files = split["splits"]["test"]
    if not test_files:
        raise RuntimeError(
            f"Empty test split for {cfg.paths.dataset_root}: "
            f"{split['manifest_path']} assigns no source to 'test'. The dataset "
            f"was preprocessed with a test ratio of 0, or every class has too few "
            f"sources to hold one out.")

    # label_to_idx: the checkpoint's mapping is authoritative (it is what the
    # model's class conditioning was trained with); fall back to the split scan.
    if label_map:
        dataset_label_map = dict(label_map)
    else:
        dataset_label_map = {c: i for i, c in enumerate(split["classes"])}
    # keep the naming map consistent with the mapping the dataset actually uses
    idx_to_label = {v: k for k, v in dataset_label_map.items()}

    # No image manager: the per-sample image embedding is read from the
    # dataset's own global_conditions/image/ bank, written once by
    # preprocess_stream.py --global image. The raw image folder is not opened
    # here and CLIP is never loaded.
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
        preload_latents=False,
        strict_conditions=not args.allow_invalid_conditions,   # #12: strict by default
        # The text slot is filled as the training fills it (the chunk's CLAP
        # audio vector, its caption's CLAP text vector, or a mix -- deterministic
        # outside train): the same arguments build_conditioned_datasets passes
        # for the training's own test split. Without them the dataset defaulted
        # to 'audio' whatever the model was trained on.
        text_source=cfg.conditioning.get("text_source", "audio"),
        text_mix_p=cfg.conditioning.get("text_mix_p", 0.5),
    )

    total = len(test_dataset)
    if total == 0:
        raise RuntimeError(
            f"Test split has {len(test_files)} files but 0 usable chunks "
            f"(duration_s / n_frames mismatch?).")

    n_samples = min(args.n_samples, total)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()
    if n_metrics is None:
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
    if n_metrics is not None:
        run_test_metrics(
            tc, model, normalizer, test_dataset, cfg, args, device,
            frame_dims=frame_cond_dims, global_configs=global_configs,
            registry=registry, writer=writer, output_dir=output_dir,
            idx_to_label=idx_to_label, n_metrics=n_metrics,
            ckpt_step=int(ckpt.get("step", 0) or 0), weights=weights)
        writer.close()
        print(f"\n[test_cond] Done!")
        print(f"[test_cond] TensorBoard:  tensorboard --logdir {log_dir}")
        return

    n_frames    = test_dataset.n_frames
    euler_steps = int(cfg.sampling.euler_steps)
    guidance    = float(cfg.conditioning.guidance_scale)
    use_amp     = bool(cfg.training.use_amp)

    print(f"\n[test_cond] --- Generating {n_samples} samples "
          f"({n_frames} frames each, {euler_steps} Euler steps, "
          f"guidance={guidance}) ---")

    for i, idx in enumerate(indices):
        (frames_real, frame_cond_real, label_idx, text_emb, image_emb,
         text_ctx) \
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
        # The caption of THIS test chunk as a sequence, for a checkpoint whose
        # blocks cross-attend. It is the dataset's own, so it describes the
        # same chunk the pooled vector above does.
        tctx = {"tokens": text_ctx["tokens"].unsqueeze(0),
                "mask": text_ctx["mask"].unsqueeze(0)}

        # --- Generate latent with CFG ---
        with torch.no_grad():
            frames_gen = euler_sample_cfg(
                model=model, n_frames=n_frames, device=device,
                steps=euler_steps,
                t_min=cfg.sampling.t_min, t_max=cfg.sampling.t_max,
                use_amp=use_amp,
                frame_cond=fc, global_cond=gc, guidance=guidance,
                frame_dims=frame_cond_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=tctx,
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

        # --- Real audio reference ---
        z_real = frames_real.T                     # (72, n_frames)
        z_real = normalizer.denormalize(z_real)
        waveform_real = decode_latents(z_real, device=device)
        wrn = waveform_real / (waveform_real.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/real/{label_name}", wrn.cpu(),
            global_step=i, sample_rate=DAC_SAMPLE_RATE,
        )

    writer.close()
    print(f"\n[test_cond] Done!")
    print(f"[test_cond] WAV files:    {output_dir}")
    print(f"[test_cond] TensorBoard:  tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
