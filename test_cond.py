# test_cond.py
#
# The TEST of a conditioned checkpoint: everything the training's metrics step
# does on the validation set, done on the TEST set with the SAME code
# (training_cond's own functions, not a copy), and written to TensorBoard the
# same way -- every window of the training but the loss curves:
#
#   Time series  Test/Metrics/*: FD-DAC, KL (both directions), FAD-VGGish
#                (those in metrics.enabled), on sampling.n_metrics_samples test
#                samples -- or the whole test split -- against the REAL test
#                data;
#   Text         Test/Condition_influence: the training's panel, a row per
#                metric of every active condition (text / image included) and
#                columns with-cond / null / Δ / valid/used, on those same
#                samples; its
#                legend; and the config of the test. An unconditioned
#                checkpoint has no table, only the config;
#   Images/Audio the test panels, the probe panels and the uncond cards -- the
#                very ones the training shows at the same step.
#
# The test set is the SAME one the training held out: both read it from the
# dataset's splits.json (written by preprocess_stream.py), so it cannot drift
# from the one the checkpoint was trained against.
#
# CONFIG (low -> high priority): the checkpoint's own training config (model,
# dataset, conditions, guidance, Euler steps, metrics.*, seed) < the test
# config, configs/test_cond.yaml by default (how many test samples are
# measured, how many panels / cards are shown) < CLI dotlist overrides, checked
# like the training's (a key that does not exist stops the test) < the CLI
# flags below.
#
# Usage:
#   python test_cond.py --ckpt runs/<run_name>/checkpoints/checkpoint_step<N>.pt
#   python test_cond.py --ckpt <ckpt> --metrics_samples all
#   python test_cond.py --ckpt <ckpt> sampling.n_metrics_samples=0   # panels only
#   python test_cond.py --ckpt <ckpt> --config configs/my_test.yaml \
#       metrics.dac_device=cuda
# On an IRCAM GPU server, through the test's own GPU-lock wrapper (same
# arguments, passed through unchanged):
#   python launch_test_cond.py --ckpt <ckpt> [--config ...] [overrides]
#
# Outputs:
#   - TensorBoard logs in runs/<run_name>/test_logs/ (beside the training's
#     own, so the two are seen together; several checkpoints of one run line
#     up along the step axis)
#   - the generations as .wav in runs/<run_name>/test_outputs/, and with the
#     metrics the numbers in test_outputs/metrics_<checkpoint>_<N|all>.json

import os
import json
import time
import argparse
from pathlib import Path

# The training module FIRST: importing it applies the process settings the
# validation metrics run under -- the IRCAM cache redirection, the CUDA
# allocator config, TF32 matmuls -- before anything touches CUDA, and it is
# where every function this test runs lives.
import training_cond as tc

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from audio_dataset_npy import LatentNormalizer, DAC_SAMPLE_RATE
from audio_dataset_cond import ConditionedAudioDataset, load_source_split
from network_cond import (ConditionedAudioDiT,
                          ckpt_frame_reinject_every,
                          ckpt_text_cross_every, ckpt_text_ctx_dim,
                          check_ckpt_reinject_gate)
from conditions import ConditionRegistry
from metrics import (precompute_latent_reference, precompute_audio_reference,
                     compute_audio_mu_sigma, COND_METRICS)


# ============================================================
# CLI / CONFIG LOADING
# ============================================================
def parse_metrics_samples(value):
    """sampling.n_metrics_samples (or --metrics_samples) -> 0 (no metrics),
    'all', or a positive int."""
    if value is None:
        return 0
    v = str(value).strip().lower()
    if v == "all":
        return "all"
    try:
        n = int(v)
    except ValueError:
        n = -1
    if n < 0:
        raise SystemExit(f"[test_cond] sampling.n_metrics_samples must be a "
                         f"number >= 0 or 'all', got {value!r}.")
    return n


def load_config():
    """
    Builds the test config in layers (low -> high):
        checkpoint's training config  <  test config (--config)
                                      <  CLI dotlist  <  CLI flags

    The checkpoint's own config is the base: it is the model, the dataset and
    HOW the metrics are computed, so a test number is computed exactly like
    its validation twin. The test config holds what the TEST decides (how many
    test samples are measured, how many panels and cards are shown), with the
    training config's key names. The keys that no longer exist are dropped and
    a CLI typo stops the test, by the training's own functions
    (drop_obsolete_sampling_keys, merge_cli_overrides).

    A checkpoint too old to carry its config falls back to the training YAML,
    configs/cond_default.yaml, with the model kind it does carry.
    Returns (cfg, args).
    """
    parser = argparse.ArgumentParser(
        description="Test a conditioned Audio DiT checkpoint on the test set: "
                    "the training's metrics step, on the test split.",
        add_help=True,
    )
    parser.add_argument("--config", type=str,
                        default="configs/test_cond.yaml",
                        help="Test config, layered ON TOP of the checkpoint's "
                             "own training config (default: "
                             "configs/test_cond.yaml)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pt) - typically "
                             "runs/<run_name>/checkpoints/checkpoint_step<N>.pt")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override run_name. If not given, it is inferred "
                             "from the checkpoint path "
                             "(runs/<run_name>/checkpoints/...).")
    parser.add_argument("--metrics_samples", type=str, default=None,
                        metavar="N|all",
                        help="Shortcut for sampling.n_metrics_samples: how "
                             "many test samples FD-DAC, KL, FAD and the table "
                             "are computed on; 'all' = the whole test split, "
                             "0 = no metrics (panels only).")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Shortcut for sampling.n_test_panels: how many "
                             "test samples are shown as panels.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of Euler steps "
                             "(default: the checkpoint's sampling.euler_steps)")
    parser.add_argument("--duration_s", type=float, default=None,
                        help="Audio duration in seconds "
                             "(default: the checkpoint's model.duration_s)")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Classifier-free guidance scale (default: the "
                             "checkpoint's conditioning.guidance_scale)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed of the generation noise (default: the "
                             "checkpoint's metrics.seed)")
    parser.add_argument("--allow_invalid_conditions", action="store_true",
                        help="DEBUG ONLY: tolerate missing/corrupt conditions on the "
                             "test set (zero-fill them). Off by default so the "
                             "evaluation cannot silently score NULL-conditioned "
                             "generations as if they were conditioned (report #12).")
    args, unknown = parser.parse_known_args()

    # ---- the base: the training config stored in the checkpoint ----
    # Only the lightweight metadata is read here (map_location='cpu'); the
    # weights are (re)loaded later in main(). Mirrors training_cond.load_config.
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    _meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_config = _meta.get("config")
    ckpt_model_kind = _meta.get("model_kind")
    del _meta

    if ckpt_config is not None:
        cfg = OmegaConf.create(ckpt_config)
        print("[test_cond] Config restored from checkpoint "
              f"(model.kind={cfg.model.kind}, duration_s={cfg.model.duration_s}, "
              f"dataset_root={cfg.paths.dataset_root}, "
              f"enabled_frame={cfg.conditioning.enabled_frame}, "
              f"enabled_global={cfg.conditioning.enabled_global}).")
    else:
        # Old checkpoint without a stored config: the training YAML is the base.
        train_yaml = "configs/cond_default.yaml"
        if not os.path.exists(train_yaml):
            raise FileNotFoundError(
                f"The checkpoint stores no training config and {train_yaml} "
                f"is not here to stand in for it.")
        cfg = OmegaConf.load(train_yaml)
        if ckpt_model_kind is not None:
            # model.kind MUST match to load the weights at all.
            cfg.model.kind = ckpt_model_kind
        print(f"[test_cond] the checkpoint stores no config: base = "
              f"{train_yaml} (model.kind={cfg.model.kind}).")
    tc.drop_obsolete_sampling_keys(cfg)

    # ---- the test config on top ----
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Test config not found: {args.config}")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    print(f"[test_cond] Test config: {args.config}")

    # ---- CLI dotlist, then the flags ----
    cfg = tc.merge_cli_overrides(
        cfg, unknown, f"the checkpoint's config + {args.config}")
    if args.metrics_samples is not None:
        cfg.sampling.n_metrics_samples = args.metrics_samples
    if args.n_samples is not None:
        cfg.sampling.n_test_panels = int(args.n_samples)
    if args.steps is not None:
        cfg.sampling.euler_steps = args.steps
    if args.duration_s is not None:
        cfg.model.duration_s = args.duration_s
    if args.guidance is not None:
        cfg.conditioning.guidance_scale = args.guidance
    if args.seed is not None:
        cfg.metrics.seed = int(args.seed)

    # ---- run_name (test output location), set AFTER the merge so the checkpoint
    # config's own paths.run_name never leaks into the test output path ----
    # Expected layout: runs/<run_name>/checkpoints/<file>.pt
    if args.run_name is not None:
        run_name = args.run_name
    else:
        ckpt_path = Path(args.ckpt).resolve()
        if ckpt_path.parent.name == "checkpoints":
            run_name = ckpt_path.parent.parent.name
        else:
            run_name = "test"   # fallback if checkpoint is not in the expected layout
    cfg.paths.run_name = run_name

    return cfg, args


# ============================================================
# MAIN
# ============================================================
def main():
    cfg, args = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_metrics = parse_metrics_samples(cfg.sampling.get("n_metrics_samples", 0))

    print(f"[test_cond] Device:     {device}")
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

    # Where the shared DAC decoder lives: metrics.dac_device, as in the
    # training (the checkpoint's value unless overridden). Decided before
    # anything decodes, because the decoder is a load-once singleton.
    tc.set_dac_device(cfg.metrics.get("dac_device", "cpu"))

    # ============================================================
    # LOAD CHECKPOINT + MODEL
    # ============================================================
    print("\n[test_cond] Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_step = int(ckpt.get("step", 0) or 0)

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
    # init). Old checkpoints have no 'ema_ready' key -> assume ready. The same
    # weights the training's metrics step used at this step.
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

    label_map = ckpt.get("label_map", {})
    del ckpt

    # ============================================================
    # TEST SET (conditioned)
    # ============================================================
    # The registry is built from CONDITION_CONFIG in conditions.py, restricted
    # to the conditions actually present in the checkpoint, so only the
    # encoders really needed are instantiated (e.g. for a pitch-only training,
    # no CLAP / CLIP are loaded here).
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
    # from the one the checkpoint was held out from.
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

    # The per-sample image embedding is read from the dataset's own
    # global_conditions/image/ bank, written once by preprocess_stream.py; the
    # text slot is filled as the training fills it (conditioning.text_source).
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
        text_source=cfg.conditioning.get("text_source", "audio"),
        text_mix_p=cfg.conditioning.get("text_mix_p", 0.5),
    )
    total = len(test_dataset)
    if total == 0:
        raise RuntimeError(
            f"Test split has {len(test_files)} files but 0 usable chunks "
            f"(duration_s / n_frames mismatch?).")
    n_scored = (0 if n_metrics == 0 else
                total if n_metrics == "all" else min(int(n_metrics), total))
    print(f"[test_cond] Test set: {total} samples | measured: "
          f"{n_scored if n_scored else 'none (panels only)'}"
          f"{' (all)' if n_scored == total else ''}")

    # ============================================================
    # THE CONFIG, in the Text window -- as the training writes its own
    # ============================================================
    writer.add_text(
        "config",
        f"checkpoint `{args.ckpt}` · step {ckpt_step} · {weights} weights · "
        f"test config `{args.config}`\n\n"
        "```yaml\n" + OmegaConf.to_yaml(cfg) + "\n```",
        global_step=ckpt_step)

    # ============================================================
    # WHAT MEASURES, AND THE PROBE BANKS -- the training's own builders
    # ============================================================
    metrics_cfg = cfg.get("metrics", None)
    influence_family, mir_threshold = tc.read_influence_settings(metrics_cfg)
    fidelity_evaluator, global_embedders = tc.build_condition_scorers(
        cfg, frame_cond_dims, global_configs, registry,
        influence_family, mir_threshold)
    probe_sets = tc.build_probe_sets(cfg, registry, fidelity_evaluator,
                                     test_dataset, frame_cond_dims,
                                     global_configs)

    metrics_seed = (metrics_cfg.get("seed", None)
                    if metrics_cfg is not None else None)
    use_amp = bool(cfg.training.get("use_amp", False))
    image_root = cfg.paths.get("image_root", None)
    n_frames = test_dataset.n_frames
    conditioned = bool(frame_cond_dims) or bool(global_configs)

    # The recordings of an UNCONDITIONED checkpoint, as the training logs its
    # own ('ground truth'): a conditioned one has them at the top of the test
    # panels instead.
    if not conditioned:
        tc.log_real_audio_samples(
            test_dataset=test_dataset,
            normalizer=normalizer, writer=writer,
            n_samples=cfg.sampling.get("n_audio_samples", 4),
            sampling_cfg=cfg.sampling, frame_dims=frame_cond_dims,
            global_configs=global_configs)

    if n_scored == 0:
        # ============================================================
        # NO METRICS: what there is to listen to and look at, only
        # ============================================================
        print(f"\n[test_cond] --- panels only (sampling.n_metrics_samples = 0), "
              f"step {ckpt_step} ---")
        tc.log_listening_panels(
            model=model, normalizer=normalizer, step=ckpt_step, writer=writer,
            device=device, output_dir=str(output_dir), sampling_cfg=cfg.sampling,
            conditioning_cfg=cfg.conditioning, use_amp=use_amp,
            frame_dims=frame_cond_dims, global_configs=global_configs,
            fidelity_evaluator=fidelity_evaluator, test_dataset=test_dataset,
            probe_sets=probe_sets, n_frames=n_frames, prefix=weights,
            metrics_seed=metrics_seed, image_root=image_root)
        writer.close()
        print("\n[test_cond] Done!")
        print(f"[test_cond] TensorBoard:  tensorboard --logdir {log_dir} "
              f"--samples_per_plugin text=0")
        return

    # ============================================================
    # THE METRICS STEP, ON THE TEST SET
    # ============================================================
    # Which distributional metrics: metrics.enabled, as in the training.
    metrics_enabled = list(metrics_cfg.get("enabled", list(COND_METRICS))
                           if metrics_cfg is not None else list(COND_METRICS))
    _unknown = [m for m in metrics_enabled if m not in COND_METRICS]
    if _unknown:
        raise SystemExit(f"[test_cond] metrics.enabled contains {_unknown}; "
                         f"available here: {list(COND_METRICS)}.")
    compute_uncond = bool(cfg.sampling.get("metrics_uncond", False))

    # ---- the references: the REAL test data, computed here every time ----
    # They are cheap next to the generation, and nothing is written into the
    # shared cache_dir, whose files are tied to the training's validation set.
    fd_dac_ref_stats = None
    if "fd_dac" in metrics_enabled or "kl_dac" in metrics_enabled:
        fd_dac_ref_stats = precompute_latent_reference(
            tc.MetricsAdapter(test_dataset), cache_path=None, device=device)

    fad_embedder = fad_ref_stats = None
    fad_device = "cpu"
    fad_mode = str(metrics_cfg.get("fad_reference", "wav")
                   if metrics_cfg is not None else "wav")
    if "fad_vggish" in metrics_enabled:
        from metrics import VGGishEmbedder
        fad_device = str(metrics_cfg.get("fad_device", "cuda"))
        if fad_device.startswith("cuda") and not torch.cuda.is_available():
            print("[test_cond] fad_device='cuda' but no GPU is available "
                  "-> FAD falls back to CPU.")
            fad_device = "cpu"
        fad_embedder = VGGishEmbedder(device=fad_device)
        # One reference clip per test CHUNK, from the split's own file list
        # (never a glob of wav/, which holds every split together).
        lat_root = Path(cfg.paths.dataset_root)
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
            fad_ref_stats = precompute_audio_reference(
                _wavs, fad_embedder, cache_path=None, device=fad_device)
        elif fad_mode == "decoded":
            _dac = tc.get_dac()

            def _real_clips():
                for i in range(total):
                    yield (tc.decode_frames_to_wav(test_dataset[i][0], normalizer,
                                                   _dac).view(1, 1, -1),
                           DAC_SAMPLE_RATE)
            _mu, _sig, _n = compute_audio_mu_sigma(
                _real_clips(), total, fad_embedder, device=fad_device,
                desc="FAD ref (test, decoded)")
            fad_ref_stats = {"mu": _mu.cpu(), "sigma": _sig.cpu(), "n_total": _n}
        else:
            raise SystemExit(
                f"[test_cond] metrics.fad_reference='{fad_mode}' is not valid: "
                f"'wav' (real test wavs) or 'decoded' (test latents via DAC).")

    # ---- the training's metrics step, on the test set ----
    # FD-DAC, KL and FAD on the SAME n_scored generations (the FAD on all of
    # them), the table on those same ones, the dump, then the panels and the
    # uncond cards. Tagged Test/..., at the checkpoint's step.
    print(f"\n[test_cond] === METRICS on {n_scored}/{total} test samples "
          f"({weights} weights, step {ckpt_step}) ===")
    report = {}
    t0 = time.time()
    tc.evaluate_and_log_metrics(
        model=model, normalizer=normalizer, val_dataset=test_dataset,
        step=ckpt_step, writer=writer, device=device,
        output_dir=str(output_dir), fd_dac_ref_stats=fd_dac_ref_stats,
        n_samples=n_scored, sampling_cfg=cfg.sampling,
        conditioning_cfg=cfg.conditioning, use_amp=use_amp,
        frame_dims=frame_cond_dims, global_configs=global_configs,
        fidelity_evaluator=fidelity_evaluator,
        global_embedders=global_embedders, compute_uncond=compute_uncond,
        prefix=weights, metrics_seed=metrics_seed,
        metrics_enabled=metrics_enabled, fad_embedder=fad_embedder,
        fad_ref_stats=fad_ref_stats, n_fad=n_scored, fad_device=fad_device,
        probe_sets=probe_sets, image_root=image_root,
        test_dataset=test_dataset, tag_prefix="Test", split_name="test",
        report=report)
    elapsed = time.time() - t0
    writer.close()

    # ---- the numbers, also as a file ----
    tag = "all" if n_scored == total else str(n_scored)
    out_json = output_dir / f"metrics_{Path(args.ckpt).stem}_{tag}.json"
    out_json.write_text(json.dumps({
        "checkpoint": str(args.ckpt), "step": ckpt_step, "weights": weights,
        "test_config": str(args.config),
        "n_scored": n_scored, "n_test": total,
        "seed": metrics_seed,
        "guidance": float(cfg.conditioning.guidance_scale),
        "euler_steps": int(cfg.sampling.euler_steps),
        "metrics_enabled": metrics_enabled, "fad_reference": fad_mode,
        "metrics_uncond": compute_uncond, "influence_family": influence_family,
        "mir_threshold": mir_threshold,
        "text_from_caption": bool(cfg.sampling.get(
            "validation_text_from_caption", False)),
        "dac_device": str(cfg.metrics.get("dac_device", "cpu")),
        "minutes": round(elapsed / 60, 2),
        **report,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\n[test_cond] {n_scored} test samples in {elapsed / 60:.1f} min")
    print(f"[test_cond] metrics saved: {out_json}")
    print("\n[test_cond] Done!")
    print(f"[test_cond] TensorBoard:  tensorboard --logdir {log_dir} "
          f"--samples_per_plugin text=0")


if __name__ == "__main__":
    main()
