# Generation and editing from a checkpoint; the Euler time grid (sampling.t_schedule).

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ["HF_HOME"] = os.path.join(_cache, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(_cache, "torch")

import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
import soundfile as sf

from dataset_utils import LatentNormalizer
from network_cond import ConditionedAudioDiT, ckpt_text_ctx_dim
import latent_codec as lc
import lora
from conditions import (
    ConditionRegistry,
    CLAPTextCondition, ImageCondition,
    make_null_frame_conditions, make_null_global_conditions,
)


T_MIN = 0.001
T_MAX = 0.999


T_SCHEDULES = ("uniform", "logsnr_uniform")

LOGSNR_MIN, LOGSNR_MAX = -6.2, 2.0


def logsnr_uniform_grid(steps, t_start=None):
    n = int(steps)
    if n < 1:
        raise ValueError(f"logsnr_uniform_grid: steps must be >= 1, got {steps}")
    first = 1.0 / (1.0 + math.exp(-LOGSNR_MIN))
    last = 1.0 / (1.0 + math.exp(-LOGSNR_MAX))
    if t_start is None or t_start <= first:
        lo, start = LOGSNR_MIN, 0.0
    elif t_start < last:
        lo, start = math.log(t_start / (1.0 - t_start)), float(t_start)
    else:
        dt = (1.0 - t_start) / n
        return [t_start + i * dt for i in range(n)] + [1.0]
    hi = LOGSNR_MAX
    inner = [1.0 / (1.0 + math.exp(-(lo + k * (hi - lo) / n)))
             for k in range(1, n)]
    return [start] + inner + [1.0]


def euler_grid(steps, t_lo, t_hi, schedule="uniform", t_start=None):
    if schedule == "uniform":
        lo = t_lo if t_start is None else t_start
        dt = (t_hi - lo) / steps
        return [(lo + i * dt, dt) for i in range(steps)]
    if schedule == "logsnr_uniform":
        g = logsnr_uniform_grid(steps, t_start)
        return [(g[i], g[i + 1] - g[i]) for i in range(len(g) - 1)]
    raise ValueError(f"t_schedule {schedule!r}: expected one of "
                     f"{list(T_SCHEDULES)}")


@torch.no_grad()
def euler_sampling_cfg(
    model, n_frames, device,
    frame_cond=None, global_cond=None,
    guidance=3.0, steps=50,
    frame_dims=None, global_configs=None,
    x_start=None, t_start=0.0,
    use_amp=True, text_ctx=None, t_schedule="uniform", inpaint=None,
):
    model.eval()
    from_noise = x_start is None or t_start <= 0.0

    if x_start is not None:
        x = x_start.to(device)
    else:
        x = torch.randn(1, n_frames, lc.active().latent_dim, device=device)
        t_start = T_MIN

    null_fc = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc = make_null_global_conditions(1, global_configs or {}, device)

    ctx = ctx_mask = None
    if getattr(model, "text_cross_layers", None) and text_ctx is not None:
        ctx = text_ctx["tokens"].to(device)
        ctx_mask = text_ctx["mask"].to(device)

    ip_m = ip_lat = ip_ctx = None
    if inpaint is not None:
        ip_m = inpaint[0].to(device=device, dtype=torch.float32).view(1, n_frames, 1)
        ip_lat = inpaint[1].to(device=device, dtype=torch.float32).view(1, n_frames, -1)
        ip_ctx = torch.cat([ip_m, ip_lat * ip_m], dim=-1)

    actual_start = min(max(t_start, T_MIN), T_MAX)
    grid = euler_grid(steps, T_MIN, T_MAX, t_schedule,
                      t_start=None if from_noise else actual_start)

    for tv, dt in grid:
        t = torch.ones(1, device=device) * tv

        with torch.amp.autocast('cuda', enabled=use_amp):
            has_cond = ((frame_cond is not None) or (global_cond is not None)
                        or (ctx is not None))
            if guidance > 1.0 and has_cond:
                fc = frame_cond if frame_cond else null_fc
                gc = global_cond if global_cond else null_gc
                v_c = model(x, t, frame_conditions=fc, global_conditions=gc,
                            text_context=ctx, text_context_mask=ctx_mask,
                            inpaint_context=ip_ctx)
                v_u = model(x, t, frame_conditions=null_fc, global_conditions=null_gc,
                            text_context=None, inpaint_context=ip_ctx)
                v = v_u + guidance * (v_c - v_u)
            else:
                v = model(x, t,
                          frame_conditions=frame_cond or null_fc,
                          global_conditions=global_cond or null_gc,
                          text_context=ctx, text_context_mask=ctx_mask,
                          inpaint_context=ip_ctx)
        x = x + v.float() * dt

    if ip_m is not None:
        x = ip_m * ip_lat + (1 - ip_m) * x
    return x[0].cpu()


def encode_source(normalizer, source_path, device):
    spec = lc.active()
    codec_model = lc.load_model(spec.name, device, cache=False)

    audio, sr = sf.read(source_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != spec.sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=spec.sample_rate)

    audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    latents = lc.encode(codec_model, audio_t, spec.sample_rate)
    z = latents.squeeze(0).cpu()
    del codec_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return normalizer.normalize(z).T.unsqueeze(0)


@torch.no_grad()
def edit_audio(
    model, normalizer, source_path, device,
    frame_cond_builder=None, global_cond=None,
    edit_strength=0.3, guidance=3.0, steps=50,
    frame_dims=None, global_configs=None,
    use_amp=True, text_ctx=None, t_schedule="uniform",
):
    spec = lc.active()
    x1 = encode_source(normalizer, source_path, device)
    n_frames = x1.shape[1]

    frame_cond = frame_cond_builder(n_frames) if frame_cond_builder is not None else None
    if frame_dims:
        print(f"  [edit] source n_frames={n_frames} -> "
              f"frame_cond={'ON' if frame_cond else 'NULL'}")

    if not (0.0 <= edit_strength <= 1.0):
        raise ValueError(f"--strength must be in [0, 1], got {edit_strength}")

    if edit_strength == 0.0:
        print("  [edit] strength=0 -> identity: returning the source through the "
              "codec round-trip, no integration.")
        frames = x1.squeeze(0)
    else:
        t_start = 1.0 - edit_strength
        noise = torch.randn_like(x1)
        x_corrupted = (1 - t_start) * noise + t_start * x1

        frames = euler_sampling_cfg(
            model, n_frames, device,
            frame_cond=frame_cond, global_cond=global_cond,
            guidance=guidance, steps=steps,
            frame_dims=frame_dims, global_configs=global_configs,
            x_start=x_corrupted, t_start=t_start,
            use_amp=use_amp, text_ctx=text_ctx, t_schedule=t_schedule,
        )

    z_out = normalizer.denormalize(frames.T)
    codec_model = lc.load_model(spec.name, "cpu", cache=False)
    wav = lc.decode(codec_model, z_out.unsqueeze(0).float()).squeeze()
    del codec_model

    return wav


@torch.no_grad()
def inpaint_audio(
    model, normalizer, source_path, device, start_s, end_s,
    frame_cond_builder=None, global_cond=None,
    guidance=3.0, steps=50, n_samples=1,
    frame_dims=None, global_configs=None,
    use_amp=True, text_ctx=None, t_schedule="uniform",
):
    spec = lc.active()
    x1 = encode_source(normalizer, source_path, device)
    n_frames = x1.shape[1]
    fps = spec.frames_per_s
    lo = max(0, int(math.floor(start_s * fps)))
    hi = min(n_frames, int(math.ceil(end_s * fps)))
    if not 0.0 <= start_s < end_s or lo >= hi:
        raise ValueError(
            f"--inpaint_start {start_s} / --inpaint_end {end_s}: need "
            f"0 <= start < end inside the {n_frames / fps:.2f} s of the source")
    keep = torch.ones(n_frames)
    keep[lo:hi] = 0.0

    frame_cond = frame_cond_builder(n_frames) if frame_cond_builder is not None else None
    print(f"  [inpaint] source {n_frames} frames ({n_frames / fps:.2f} s) | "
          f"regenerating frames {lo}-{hi} ({lo / fps:.2f}-{hi / fps:.2f} s), "
          f"the rest is kept as it is | "
          f"frame_cond={'ON' if frame_cond else 'NULL'}")

    codec_model = lc.load_model(spec.name, "cpu", cache=False)
    wavs = []
    for _ in range(int(n_samples)):
        frames = euler_sampling_cfg(
            model, n_frames, device,
            frame_cond=frame_cond, global_cond=global_cond,
            guidance=guidance, steps=steps,
            frame_dims=frame_dims, global_configs=global_configs,
            use_amp=use_amp, text_ctx=text_ctx, t_schedule=t_schedule,
            inpaint=(keep, x1[0]),
        )
        z_out = normalizer.denormalize(frames.T)
        wavs.append(lc.decode(codec_model, z_out.unsqueeze(0).float()).squeeze())
    del codec_model
    return wavs


def build_global_cond(
    label_name, ckpt, device, image_path=None, prompt=None, ctx_out=None,
) -> Dict[str, torch.Tensor]:
    gc = {}
    ctx_out = {} if ctx_out is None else ctx_out
    global_configs = ckpt.get("global_configs", {})

    if "text" in global_configs:
        if prompt is not None:
            text_input = prompt
        elif label_name is not None:
            text_input = label_name.replace("_", " ")
        else:
            text_input = None

        if text_input is not None:
            text_enc = CLAPTextCondition()
            emb = text_enc.encode_text(text_input)
            gc["text"] = torch.from_numpy(emb).unsqueeze(0).to(device)
            tok, tlen = text_enc.encode_tokens([text_input])
            m = torch.zeros(1, tok.shape[1], dtype=torch.bool)
            m[0, :int(tlen[0])] = True
            ctx_out["text"] = {"tokens": torch.from_numpy(tok).to(device),
                               "mask": m.to(device)}
            text_enc.unload()
            print(f"  [CLAP-text] prompt: {text_input!r} "
                  f"({int(tlen[0])} token(s))")

    if "image" in global_configs and image_path:
        img_enc = ImageCondition()
        emb = img_enc.encode_image(image_path)
        gc["image"] = torch.from_numpy(emb).unsqueeze(0).to(device)
        img_enc.unload()
        print(f"  [CLIP-image] {image_path}")

    return gc


def _align_to_frames(arr: np.ndarray, n_frames: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T = arr.shape[0]
    if T == n_frames:
        return arr
    if T > n_frames:
        return arr[:n_frames]
    pad = np.zeros((n_frames - T, arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


def build_frame_cond(args, ckpt, n_frames, device) -> Optional[Dict[str, torch.Tensor]]:
    frame_cond_dims = ckpt.get("frame_cond_dims", {}) or {}
    if not frame_cond_dims:
        return None

    raw: Dict[str, np.ndarray] = {}

    if getattr(args, "condition_npz", None):
        if not os.path.exists(args.condition_npz):
            raise FileNotFoundError(f"--condition_npz not found: {args.condition_npz}")
        data = np.load(args.condition_npz)
        for name in frame_cond_dims:
            if name in data:
                raw[name] = np.asarray(data[name], dtype=np.float32)
        print(f"  [frame-cond] loaded from npz: {sorted(raw.keys())}")

    elif getattr(args, "condition_wav", None):
        if not os.path.exists(args.condition_wav):
            raise FileNotFoundError(f"--condition_wav not found: {args.condition_wav}")
        audio, sr = sf.read(args.condition_wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        reg = ConditionRegistry(
            enabled_frame=list(frame_cond_dims.keys()), enabled_global=[],
        )
        raw = reg.extract_frame_conditions(audio, sr, n_frames)
        print(f"  [frame-cond] re-extracted from wav: {sorted(raw.keys())}")

    else:
        return None

    fc: Dict[str, torch.Tensor] = {}
    for name, rdim in frame_cond_dims.items():
        arr = raw.get(name)
        if arr is None:
            print(f"  [frame-cond] WARNING: '{name}' not in source -> null (zeros)")
            arr = np.zeros((n_frames, rdim), dtype=np.float32)
        else:
            arr = _align_to_frames(arr, n_frames)
            if arr.shape[1] != rdim:
                raise ValueError(
                    f"Condition '{name}' has raw_dim {arr.shape[1]} but the "
                    f"checkpoint expects {rdim}.")
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        fc[name] = t.unsqueeze(0).to(device)
    return fc


def resolve_normalizer_path(ckpt: dict, ckpt_path: str, cli_path: str = None) -> str:
    candidates = []

    if cli_path:
        candidates.append(Path(cli_path))

    cfg = ckpt.get("config", None)
    if isinstance(cfg, dict):
        cache_dir = cfg.get("paths", {}).get("cache_dir", None)
        if cache_dir:
            candidates.append(Path(cache_dir) / "normalizer.pt")

    ckpt_p = Path(ckpt_path).resolve()
    if ckpt_p.parent.name == "checkpoints":
        run_dir = ckpt_p.parent.parent
        candidates.append(run_dir / "checkpoints" / "normalizer.pt")
        candidates.append(run_dir.parent.parent / "cache" / "normalizer.pt")
    candidates.append(Path(ckpt_path).parent / "normalizer.pt")

    candidates.append(Path("cache") / "normalizer.pt")
    candidates.append(Path("/data/anasynth_nonbp/baione/cache/normalizer.pt"))

    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "normalizer.pt not found. Tried:\n  " +
        "\n  ".join(str(c) for c in candidates) +
        "\nPass the normalizer path explicitly with --normalizer."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("mode", choices=["generate", "edit", "inpaint"])
    parser.add_argument("--source", type=str, default=None,
                        help="Source audio for edit and inpaint modes")
    parser.add_argument("--inpaint_start", type=float, default=None,
                        help="Inpaint mode: start, in seconds, of the span of "
                             "--source to regenerate; the rest is kept")
    parser.add_argument("--inpaint_end", type=float, default=None,
                        help="Inpaint mode: end of that span, in seconds")
    parser.add_argument("--label", type=str, default=None,
                        help="Class name (will be used as the CLAP prompt "
                             "if --prompt is not passed)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Free-form text prompt for CLAP "
                             "(e.g. 'slow piano in C minor'). If passed, it "
                             "OVERRIDES --label as the source of the text embedding.")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path for visual conditioning")
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--strength", type=float, default=0.3,
                        help="Edit strength 0-1")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--t_schedule", choices=list(T_SCHEDULES), default=None,
                        help="Where the Euler steps go along t: 'uniform' "
                             "(equal steps) or 'logsnr_uniform' (Stable Audio 3's grid, "
                             "equally spaced in log-SNR, crowded at the noise "
                             "end). Default: the one the checkpoint's run "
                             "generated with (its sampling.t_schedule).")
    parser.add_argument("--lora_strength", type=float, default=1.0,
                        help="LoRA checkpoint: 0 = the base model, 1 = as trained.")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--duration", type=float, default=None,
                        help="Generation length in seconds (generate mode). If "
                             "omitted, uses the exact n_frames the checkpoint was "
                             "trained with (recommended). A value here is converted "
                             "to codec frames via round(), matching preprocessing.")
    parser.add_argument("--output", type=str, default="./generated_cond")
    parser.add_argument("--normalizer", type=str, default=None,
                        help="Explicit path to normalizer.pt (otherwise resolved "
                             "from the checkpoint config / standard locations).")
    parser.add_argument("--condition_npz", type=str, default=None,
                        help="Path to an .npz of frame conditions (keys = condition "
                             "names, e.g. f0/energy), as produced by "
                             "extract_conditions.py. Aligned to n_frames "
                             "(truncate/zero-pad) like the dataset.")
    parser.add_argument("--condition_wav", type=str, default=None,
                        help="Path to a WAV from which the frame conditions required "
                             "by the checkpoint are RE-EXTRACTED with the same "
                             "ConditionRegistry used at training time.")
    parser.add_argument("--allow_null_frame_conditions", action="store_true",
                        help="Permit generation with NULL (zero) frame conditions "
                             "when the checkpoint requires frame conditioning and no "
                             "--condition_npz/--condition_wav is given. Off by default "
                             "so you do not silently generate uncontrolled audio.")
    parser.add_argument("--allow_null_global_conditions", action="store_true",
                        help="Permit generation with NULL global conditions when the "
                             "checkpoint requires text/image conditioning and none is "
                             "given (no --prompt/--label for text, no --image for "
                             "image). Off by default so you do not silently generate "
                             "with uncontrolled global conditioning.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Carico checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if lora.adapter_of(ckpt["config"]):
        ckpt = lora.merge_checkpoint(ckpt, ckpt["config"]["paths"]["init_from"],
                                     args.lora_strength)

    CODEC = lc.activate(ckpt["codec"])
    print(f"Codec: {CODEC.name} ({CODEC.sample_rate} Hz, "
          f"{CODEC.frames_per_s:.2f} frames/s, {CODEC.latent_dim}-d)")

    frame_cond_dims     = ckpt["frame_cond_dims"]
    frame_cond_out_dims = ckpt["frame_cond_out_dims"]
    global_configs_ckpt = ckpt["global_configs"]

    frame_reinject_every = ckpt["frame_reinject_every"]
    text_cross_every = ckpt["text_cross_every"]
    text_ctx_dim = ckpt_text_ctx_dim(ckpt)
    attention = ckpt["attention"]
    qk_norm = ckpt["qk_norm"]
    inpainting = ckpt["inpainting"]

    model = ConditionedAudioDiT(
        kind=ckpt.get("model_kind", "L"),
        frame_cond_dims=frame_cond_dims,
        frame_cond_out_dims=frame_cond_out_dims,
        global_cond_configs=global_configs_ckpt,
        frame_reinject_every=frame_reinject_every,
        text_cross_every=text_cross_every,
        text_ctx_dim=text_ctx_dim,
        attention=attention,
        qk_norm=qk_norm,
        inpainting=inpainting,
        token_dim=CODEC.latent_dim,
    ).to(device)

    if "ema_state_dict" in ckpt and ckpt.get("ema_ready", True):
        model.load_state_dict(ckpt["ema_state_dict"])
        print("  -> Usando EMA model")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            print("  -> EMA present but NOT trained yet (checkpoint predates "
                  "training.ema_start): using the live model weights")

    normalizer = LatentNormalizer()
    norm_path = resolve_normalizer_path(ckpt, args.checkpoint, cli_path=args.normalizer)
    print(f"Normalizer: {norm_path}")
    normalizer.load(norm_path)

    t_schedule = args.t_schedule or ckpt["config"]["sampling"]["t_schedule"]
    print(f"Euler steps: {args.steps} | t schedule: {t_schedule}"
          + (" (the checkpoint's)" if args.t_schedule is None else ""))

    text_ctx_out = {}
    gc = build_global_cond(
        args.label, ckpt, device,
        image_path=args.image, prompt=args.prompt, ctx_out=text_ctx_out,
    )
    prompt_ctx = text_ctx_out.get("text")

    frame_dims = frame_cond_dims
    global_configs = global_configs_ckpt

    missing_global = [g for g in global_configs if g not in gc]
    if missing_global:
        how = {"text": "--prompt or --label", "image": "--image"}
        need = ", ".join(f"{g} (needs {how.get(g, 'its input')})"
                         for g in missing_global)
        if args.allow_null_global_conditions:
            print(f"[WARN] Checkpoint requires global conditions {missing_global} "
                  f"but none were given -> generating with NULL global conditions "
                  f"(--allow_null_global_conditions).")
        else:
            print(f"[ERROR] This checkpoint was trained with global conditions "
                  f"{list(global_configs.keys())}, but these are missing: {need}. "
                  f"Provide them, or pass --allow_null_global_conditions to "
                  f"generate with null global conditioning on purpose.")
            sys.exit(1)

    has_frame_source = bool(getattr(args, "condition_npz", None)
                            or getattr(args, "condition_wav", None))
    if frame_dims and not has_frame_source:
        if args.allow_null_frame_conditions:
            print(f"[WARN] Checkpoint requires frame conditions {list(frame_dims)} "
                  f"but none were given -> generating with NULL frame conditions "
                  f"(--allow_null_frame_conditions).")
        else:
            print(f"[ERROR] This checkpoint was trained with frame conditions "
                  f"{list(frame_dims)}, but no --condition_npz/--condition_wav was "
                  f"given. Provide one, or pass --allow_null_frame_conditions to "
                  f"generate with null (zero) frame conditions on purpose.")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    def _out_tag(fallback):
        if args.label:
            return args.label
        if args.prompt:
            slug = "".join(c if c.isalnum() else "_" for c in args.prompt)
            slug = "_".join(p for p in slug.split("_") if p)[:48]
            if slug:
                return slug
        return fallback

    def _active_cond_tag(frame_cond, global_cond):
        active = sorted(frame_cond or {}) + sorted(global_cond or {})
        return "_".join(active) if active else "uncond"

    if args.mode == "generate":
        if args.duration is None:
            n_frames = int(ckpt.get("n_frames", round(5.0 * CODEC.frames_per_s)))
        else:
            n_frames = int(round(args.duration * CODEC.frames_per_s))
        frame_cond = build_frame_cond(args, ckpt, n_frames, device)

        print(f"Generazione {args.n_samples} audio | "
              f"{n_frames} frame | guidance={args.guidance} | "
              f"frame_cond={'ON' if frame_cond else 'NULL'}")

        codec_m = lc.load_model(CODEC.name, "cpu", cache=False)

        for i in range(args.n_samples):
            gen = euler_sampling_cfg(
                model, n_frames, device,
                frame_cond=frame_cond,
                global_cond=gc if gc else None,
                guidance=args.guidance, steps=args.steps,
                frame_dims=frame_dims, global_configs=global_configs,
                text_ctx=prompt_ctx, t_schedule=t_schedule,
            )
            with torch.no_grad():
                z = normalizer.denormalize(gen.T)
                wav = lc.decode(codec_m, z.unsqueeze(0).float()).squeeze()
            tag = _out_tag(_active_cond_tag(frame_cond, gc))
            p = os.path.join(args.output, f"gen_{tag}_{i:02d}.wav")
            sf.write(p, wav.numpy(), CODEC.sample_rate)
            print(f"  {p}")

    elif args.mode == "edit":
        if not args.source:
            print("[ERROR] --source richiesto per mode=edit")
            sys.exit(1)

        def _frame_cond_builder(nf):
            return build_frame_cond(args, ckpt, nf, device)

        print(f"Editing {args.source} | strength={args.strength} | "
              f"guidance={args.guidance} | "
              f"frame_source={'ON' if has_frame_source else 'NULL'}")
        wav = edit_audio(
            model, normalizer, args.source, device,
            frame_cond_builder=_frame_cond_builder,
            global_cond=gc if gc else None,
            edit_strength=args.strength, guidance=args.guidance,
            steps=args.steps,
            frame_dims=frame_dims, global_configs=global_configs,
            text_ctx=prompt_ctx, t_schedule=t_schedule,
        )
        tag = _out_tag("edited")
        p = os.path.join(args.output, f"edit_{tag}.wav")
        sf.write(p, wav.numpy(), CODEC.sample_rate)
        print(f"  {p}")

    elif args.mode == "inpaint":
        if not inpainting:
            print("[ERROR] this checkpoint was trained without "
                  "training.inpainting: it has no layers to read a context. "
                  "Use a checkpoint trained with training.inpainting: true.")
            sys.exit(1)
        if not args.source or args.inpaint_start is None or args.inpaint_end is None:
            print("[ERROR] mode=inpaint needs --source, --inpaint_start and "
                  "--inpaint_end")
            sys.exit(1)

        def _frame_cond_builder(nf):
            return build_frame_cond(args, ckpt, nf, device)

        print(f"Inpainting {args.source} | {args.inpaint_start}-"
              f"{args.inpaint_end} s | {args.n_samples} version(s) | "
              f"guidance={args.guidance} | "
              f"frame_source={'ON' if has_frame_source else 'NULL'}")
        wavs = inpaint_audio(
            model, normalizer, args.source, device,
            args.inpaint_start, args.inpaint_end,
            frame_cond_builder=_frame_cond_builder,
            global_cond=gc if gc else None,
            guidance=args.guidance, steps=args.steps,
            n_samples=args.n_samples,
            frame_dims=frame_dims, global_configs=global_configs,
            text_ctx=prompt_ctx, t_schedule=t_schedule,
        )
        tag = _out_tag("inpainted")
        for i, wav in enumerate(wavs):
            p = os.path.join(args.output, f"inpaint_{tag}_{i:02d}.wav")
            sf.write(p, wav.numpy(), CODEC.sample_rate)
            print(f"  {p}")


if __name__ == "__main__":
    main()
