# Test of a checkpoint on the test split: the training's metrics step.

import os
import sys
import json
import time
import argparse
from pathlib import Path

import training_cond as tc

import torch
import torch.distributed as dist
import soundfile as sf
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audio_dataset_npy import LatentNormalizer, DAC_SAMPLE_RATE
from audio_dataset_cond import ConditionedAudioDataset, load_source_split
from network_cond import (ConditionedAudioDiT, TOKEN_DIM,
                          ckpt_frame_reinject_every,
                          ckpt_text_cross_every, ckpt_text_ctx_dim,
                          check_ckpt_reinject_gate, ckpt_attention)
from conditions import ConditionRegistry
import latent_codec as lc
from metrics import (precompute_latent_reference, precompute_audio_reference,
                     compute_audio_mu_sigma, compute_mu_sigma, COND_METRICS)


def parse_metrics_samples(value):
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

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    _meta = torch.load(args.ckpt, map_location="cpu", weights_only=False,
                       mmap=True)
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
        train_yaml = "configs/cond_default.yaml"
        if not os.path.exists(train_yaml):
            raise FileNotFoundError(
                f"The checkpoint stores no training config and {train_yaml} "
                f"is not here to stand in for it.")
        cfg = OmegaConf.load(train_yaml)
        if ckpt_model_kind is not None:
            cfg.model.kind = ckpt_model_kind
        print(f"[test_cond] the checkpoint stores no config: base = "
              f"{train_yaml} (model.kind={cfg.model.kind}).")
    tc.drop_obsolete_sampling_keys(cfg)

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Test config not found: {args.config}")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config))
    print(f"[test_cond] Test config: {args.config}")

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
    tc.t_schedule_of(cfg.sampling)

    if args.run_name is not None:
        run_name = args.run_name
    else:
        ckpt_path = Path(args.ckpt).resolve()
        if ckpt_path.parent.name == "checkpoints":
            run_name = ckpt_path.parent.parent.name
        else:
            run_name = "test"
    cfg.paths.run_name = run_name

    return cfg, args


def _stats_over_ranks(sum_x, sum_xx, count, di):
    sum_x, sum_xx, count = tc.dist_sum_tensors([sum_x, sum_xx, count], di)
    mu, sigma, n = compute_mu_sigma(sum_x, sum_xx, count)
    return {"mu": mu.cpu(), "sigma": sigma.cpu(), "n_total": int(n)}


@torch.no_grad()
def split_latent_reference(dataset, di, device):
    _d = lc.active().latent_dim
    sum_x = torch.zeros(_d, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(_d, _d, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for i in tqdm(range(di.rank, len(dataset), di.world),
                  desc="Latent reference", disable=not di.is_main):
        x = dataset[i][0].to(device=device, dtype=torch.float64)
        sum_x = sum_x + x.sum(dim=0)
        sum_xx = sum_xx + x.T @ x
        count = count + x.shape[0]
    stats = _stats_over_ranks(sum_x, sum_xx, count, di)
    print(f"[Latent Reference] {stats['n_total']} frames of {len(dataset)} "
          f"test samples, over {di.world} GPUs")
    return stats


@torch.no_grad()
def split_audio_reference(clips, n_clips, embedder, device, di, desc):
    d = int(embedder.embedding_dim)
    sum_x = torch.zeros(d, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(d, d, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for wav, sr in tqdm(clips, total=n_clips, desc=desc,
                        disable=not di.is_main):
        e = embedder.embed(wav, audio_sr=sr).to(device=device,
                                                dtype=torch.float64)
        sum_x = sum_x + e.sum(dim=0)
        sum_xx = sum_xx + e.T @ e
        count = count + e.shape[0]
        del e
    return _stats_over_ranks(sum_x, sum_xx, count, di)


def wav_clips(paths):
    for p in paths:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        yield torch.from_numpy(data.T.copy()).unsqueeze(0), sr


def main():
    DIST = tc.init_distributed()
    if DIST.enabled and not DIST.is_main:
        sys.stdout = open(os.devnull, "w")
    cfg, args = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_metrics = parse_metrics_samples(cfg.sampling.get("n_metrics_samples", 0))

    print(f"[test_cond] Device:     {device}")
    print(f"[test_cond] Checkpoint: {args.ckpt}")
    print(f"[test_cond] Run name:   {cfg.paths.run_name}")

    run_dir    = Path(cfg.paths.runs_dir) / cfg.paths.run_name
    output_dir = run_dir / "test_outputs"
    log_dir    = run_dir / "test_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[test_cond] Outputs:    {output_dir}")
    print(f"[test_cond] TB logs:    {log_dir}")

    if DIST.enabled:
        rank_log = f"rank<R>_{Path(args.ckpt).stem}.log"
        if not DIST.is_main:
            sys.stdout = open(log_dir / rank_log.replace("<R>", str(DIST.rank)),
                              "w", buffering=1, encoding="utf-8",
                              errors="backslashreplace")
        print(f"[multi-GPU] rank {DIST.rank} of {DIST.world} on "
              f"{f'cuda:{DIST.local_rank}' if device == 'cuda' else 'cpu'} | "
              f"backend {DIST.backend} | test sample j on GPU j mod "
              f"{DIST.world} | the other ranks' console: {log_dir / rank_log}")
        if n_metrics == 0:
            if not DIST.is_main:
                print("[test_cond] panels only (sampling.n_metrics_samples = "
                      "0): GPU 0 makes them, nothing to do here.")
                dist.destroy_process_group()
                return
            print(f"[test_cond] panels only (sampling.n_metrics_samples = 0): "
                  f"nothing to split, GPU 0 makes them and the other "
                  f"{DIST.world - 1} GPU(s) stay idle.")

    writer = SummaryWriter(str(log_dir)) if DIST.is_main else tc.NullWriter()

    tc.set_dac_device(cfg.metrics.get("dac_device", "cpu"))

    print("\n[test_cond] Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False,
                      mmap=True)
    ckpt_step = int(ckpt.get("step", 0) or 0)

    CODEC = lc.activate(lc.codec_from_ckpt(ckpt))
    _ds_codec = lc.dataset_codec(cfg.paths.dataset_root)
    if _ds_codec != CODEC.name:
        raise SystemExit(
            f"[test_cond] the checkpoint was trained on {CODEC.name} latents, "
            f"the dataset at {cfg.paths.dataset_root} is {_ds_codec}. Test it "
            f"on a dataset preprocessed with --codec {CODEC.name}.")
    print(f"[test_cond] Codec:                {CODEC.name} ({CODEC.sample_rate} Hz, "
          f"{CODEC.frames_per_s:.2f} frames/s, {CODEC.latent_dim}-d)")

    model_kind          = ckpt.get("model_kind",          cfg.model.kind)
    frame_cond_dims     = ckpt.get("frame_cond_dims",     {})
    frame_cond_out_dims = ckpt.get("frame_cond_out_dims", {})
    global_configs      = ckpt.get("global_configs",      {})
    if frame_cond_dims and not frame_cond_out_dims:
        _reg = ConditionRegistry(
            enabled_frame=list(frame_cond_dims.keys()), enabled_global=[],
        )
        frame_cond_out_dims = _reg.frame_cond_out_dims
    print(f"[test_cond] Model kind:           {model_kind}")
    print(f"[test_cond] Frame cond dims:      {frame_cond_dims}")
    print(f"[test_cond] Frame cond out dims:  {frame_cond_out_dims}")
    print(f"[test_cond] Global cond configs:  {global_configs}")

    frame_reinject_every = ckpt_frame_reinject_every(ckpt)
    check_ckpt_reinject_gate(ckpt, args.ckpt)
    print(f"[test_cond] Frame re-injection:    "
          f"{'every ' + str(frame_reinject_every) + ' block(s)' if frame_reinject_every > 0 else 'OFF'}")

    text_cross_every = ckpt_text_cross_every(ckpt)
    text_ctx_dim = ckpt_text_ctx_dim(ckpt)
    attention = ckpt_attention(ckpt)
    print(f"[test_cond] Self-attention:        {attention}")
    print(f"[test_cond] Euler steps:           {cfg.sampling.euler_steps} | "
          f"t schedule: {tc.t_schedule_of(cfg.sampling)}")

    model = ConditionedAudioDiT(
        kind=model_kind,
        frame_cond_dims=frame_cond_dims,
        frame_cond_out_dims=frame_cond_out_dims,
        global_cond_configs=global_configs,
        frame_reinject_every=frame_reinject_every,
        text_cross_every=text_cross_every,
        text_ctx_dim=text_ctx_dim,
        attention=attention,
        token_dim=CODEC.latent_dim,
    ).to(device)

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

    registry = ConditionRegistry(
        enabled_frame  = list(frame_cond_dims.keys()),
        enabled_global = list(global_configs.keys()),
    )

    cond_root = (cfg.paths.condition_root
                 if Path(cfg.paths.condition_root).exists() else None)
    img_root  = (cfg.paths.image_root
                 if Path(cfg.paths.image_root).exists() else None)

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

    if label_map:
        dataset_label_map = dict(label_map)
    else:
        dataset_label_map = {c: i for i, c in enumerate(split["classes"])}

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
        strict_conditions=not args.allow_invalid_conditions,
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

    writer.add_text(
        "config",
        f"checkpoint `{args.ckpt}` · step {ckpt_step} · {weights} weights · "
        f"test config `{args.config}`\n\n"
        "```yaml\n" + OmegaConf.to_yaml(cfg) + "\n```",
        global_step=ckpt_step)

    metrics_cfg = cfg.get("metrics", None)
    influence_family, mir_threshold = tc.read_influence_settings(metrics_cfg)
    fidelity_evaluator, global_embedders = tc.build_condition_scorers(
        cfg, frame_cond_dims, global_configs, registry,
        influence_family, mir_threshold)
    probe_sets = (tc.build_probe_sets(cfg, registry, fidelity_evaluator,
                                      test_dataset, frame_cond_dims,
                                      global_configs)
                  if DIST.is_main else None)

    metrics_seed = (metrics_cfg.get("seed", None)
                    if metrics_cfg is not None else None)
    use_amp = bool(cfg.training.get("use_amp", False))
    image_root = cfg.paths.get("image_root", None)
    n_frames = test_dataset.n_frames
    conditioned = bool(frame_cond_dims) or bool(global_configs)

    if not conditioned and DIST.is_main:
        tc.log_real_audio_samples(
            test_dataset=test_dataset,
            normalizer=normalizer, writer=writer,
            n_samples=cfg.sampling.get("n_audio_samples", 4),
            sampling_cfg=cfg.sampling, frame_dims=frame_cond_dims,
            global_configs=global_configs, output_dir=str(output_dir))

    if n_scored == 0:
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
        if DIST.enabled:
            dist.destroy_process_group()
        print("\n[test_cond] Done!")
        print(f"[test_cond] TensorBoard:  tensorboard --logdir {log_dir} "
              f"--samples_per_plugin text=0")
        return

    metrics_enabled = list(metrics_cfg.get("enabled", list(COND_METRICS))
                           if metrics_cfg is not None else list(COND_METRICS))
    _unknown = [m for m in metrics_enabled if m not in COND_METRICS]
    if _unknown:
        raise SystemExit(f"[test_cond] metrics.enabled contains {_unknown}; "
                         f"available here: {list(COND_METRICS)}.")
    compute_uncond = bool(cfg.sampling.get("metrics_uncond", False))

    fd_dac_ref_stats = None
    if "fd_dac" in metrics_enabled or "kl_dac" in metrics_enabled:
        if DIST.enabled:
            fd_dac_ref_stats = split_latent_reference(test_dataset, DIST, device)
        else:
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
            if DIST.enabled:
                _own = _wavs[DIST.rank::DIST.world]
                print(f"[Audio ref] embedding {len(_wavs)} real test wavs "
                      f"(NO DAC), over {DIST.world} GPUs...")
                fad_ref_stats = split_audio_reference(
                    wav_clips(_own), len(_own), fad_embedder, fad_device, DIST,
                    "Audio ref")
            else:
                fad_ref_stats = precompute_audio_reference(
                    _wavs, fad_embedder, cache_path=None, device=fad_device)
        elif fad_mode == "decoded":
            _dac = tc.get_dac()

            def _real_clips(idxs):
                for i in idxs:
                    yield (tc.decode_frames_to_wav(test_dataset[i][0], normalizer,
                                                   _dac).view(1, 1, -1),
                           lc.active().sample_rate)
            if DIST.enabled:
                _own = range(DIST.rank, total, DIST.world)
                fad_ref_stats = split_audio_reference(
                    _real_clips(_own), len(_own), fad_embedder, fad_device,
                    DIST, "FAD ref (test, decoded)")
            else:
                _mu, _sig, _n = compute_audio_mu_sigma(
                    _real_clips(range(total)), total, fad_embedder,
                    device=fad_device, desc="FAD ref (test, decoded)")
                fad_ref_stats = {"mu": _mu.cpu(), "sigma": _sig.cpu(),
                                 "n_total": _n}
        else:
            raise SystemExit(
                f"[test_cond] metrics.fad_reference='{fad_mode}' is not valid: "
                f"'wav' (real test wavs) or 'decoded' (test latents via DAC).")

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
        report=report, dist_info=DIST)
    elapsed = time.time() - t0
    writer.close()
    if DIST.enabled:
        tc.dist_barrier(DIST)
        dist.destroy_process_group()
        if not DIST.is_main:
            print(f"\n[test_cond] rank {DIST.rank}: done.")
            return

    tag = "all" if n_scored == total else str(n_scored)
    out_json = output_dir / f"metrics_{Path(args.ckpt).stem}_{tag}.json"
    out_json.write_text(json.dumps({
        "checkpoint": str(args.ckpt), "step": ckpt_step, "weights": weights,
        "test_config": str(args.config),
        "n_scored": n_scored, "n_test": total,
        "seed": metrics_seed,
        "guidance": float(cfg.conditioning.guidance_scale),
        "euler_steps": int(cfg.sampling.euler_steps),
        "t_schedule": tc.t_schedule_of(cfg.sampling),
        "metrics_enabled": metrics_enabled, "fad_reference": fad_mode,
        "metrics_uncond": compute_uncond, "influence_family": influence_family,
        "mir_threshold": mir_threshold,
        "text_from_caption": bool(cfg.sampling.get(
            "validation_text_from_caption", False)),
        "dac_device": str(cfg.metrics.get("dac_device", "cpu")),
        "gpus": DIST.world,
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
