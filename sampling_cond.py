# sampling_cond.py
#
# Standalone conditioned sampling with CFG and editing.
#
# Modes:
#   1. generate: from noise -> audio with conditions
#   2. edit:     existing audio -> partial corruption -> reconstruction with new conditions
#
# Usage:
#   python sampling_cond.py checkpoint.pt generate \
#       --label "Baroque_sacred" --guidance 3.0 --n_samples 4
#
#   python sampling_cond.py checkpoint.pt edit input.wav \
#       --label "Romanticism_chamber" --strength 0.3 --guidance 3.0

import os
os.environ.setdefault("USE_TF", "0")   # transformers -> PyTorch backend (no TF)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
import soundfile as sf

from audio_dataset_npy import (
    LatentNormalizer, DAC_SAMPLE_RATE, DAC_FRAMES_PER_S,
)
from network_cond import ConditionedAudioDiT, TOKEN_DIM
from conditions import (
    ConditionRegistry,
    CLAPTextCondition, ImageCondition,
    make_null_frame_conditions, make_null_global_conditions,
)


T_MIN = 0.001
T_MAX = 0.999


@torch.no_grad()
def euler_sampling_cfg(
    model, n_frames, device,
    frame_cond=None, global_cond=None,
    guidance=3.0, steps=50,
    frame_dims=None, global_configs=None,
    x_start=None, t_start=0.0,
    use_amp=True,
):
    """
    Euler sampling con CFG.
    x_start + t_start: for editing (partial corruption).
    """
    model.eval()

    if x_start is not None:
        x = x_start.to(device)
    else:
        x = torch.randn(1, n_frames, TOKEN_DIM, device=device)
        t_start = T_MIN

    null_fc = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc = make_null_global_conditions(1, global_configs or {}, device)

    # Clamp into [T_MIN, T_MAX]: a t_start above T_MAX would make dt negative and
    # walk the integration backwards. The edit path already short-circuits
    # strength=0, so this is a guard for any other caller.
    actual_start = min(max(t_start, T_MIN), T_MAX)
    dt = (T_MAX - actual_start) / steps

    for i in range(steps):
        tv = actual_start + i * dt
        t = torch.ones(1, device=device) * tv

        with torch.amp.autocast('cuda', enabled=use_amp):
            has_cond = (frame_cond is not None) or (global_cond is not None)
            if guidance > 1.0 and has_cond:
                fc = frame_cond if frame_cond else null_fc
                gc = global_cond if global_cond else null_gc
                v_c = model(x, t, frame_conditions=fc, global_conditions=gc)
                v_u = model(x, t, frame_conditions=null_fc, global_conditions=null_gc)
                v = v_u + guidance * (v_c - v_u)
            else:
                v = model(x, t,
                          frame_conditions=frame_cond or null_fc,
                          global_conditions=global_cond or null_gc)
        x = x + v.float() * dt

    return x[0].cpu()


@torch.no_grad()
def edit_audio(
    model, normalizer, source_path, device,
    frame_cond_builder=None, global_cond=None,
    edit_strength=0.3, guidance=3.0, steps=50,
    frame_dims=None, global_configs=None,
    use_amp=True,
):
    """
    Editing: real audio -> partial corruption -> reconstruction.
    edit_strength: 0.0 = no change, 1.0 = full regeneration.

    BUG #4 FIX: the source audio may not last exactly --duration, so its true
    DAC length is only known AFTER encoding. Frame conditions therefore CANNOT be
    built ahead of time from a duration-derived n_frames (that produced an x vs
    frame_cond length mismatch). Instead the caller passes `frame_cond_builder`,
    a callable n_frames -> frame_cond dict (or None), which is invoked here once
    the source's actual n_frames is known, guaranteeing x and conditions align.
    """
    import dac

    # Encode source
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to(device); dac_model.eval()

    audio, sr = sf.read(source_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != DAC_SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=DAC_SAMPLE_RATE)

    audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    x = dac_model.preprocess(audio_t, DAC_SAMPLE_RATE)
    _z, _codes, latents, _, _ = dac_model.encode(x)
    z = latents.squeeze(0).cpu()  # (72, T) DAC pre-quantizer latents
    del dac_model
    if device == "cuda":
        torch.cuda.empty_cache()

    n_frames = z.shape[1]

    # Now that the source's true length is known, build the frame conditions
    # aligned to THIS n_frames (bug #4). The builder handles npz/wav loading,
    # per-condition alignment (truncate/zero-pad) and null fallback.
    frame_cond = frame_cond_builder(n_frames) if frame_cond_builder is not None else None
    if frame_dims:
        print(f"  [edit] source n_frames={n_frames} -> "
              f"frame_cond={'ON' if frame_cond else 'NULL'}")

    z_norm = normalizer.normalize(z)
    x1 = z_norm.T.unsqueeze(0)

    # strength is a fraction of the trajectory: validate it instead of letting a
    # bad value produce silent nonsense.
    if not (0.0 <= edit_strength <= 1.0):
        raise ValueError(f"--strength must be in [0, 1], got {edit_strength}")

    # strength=0 means "do not edit". Integrating anyway would set t_start=1.0 >
    # T_MAX, making dt = (T_MAX - t_start)/steps NEGATIVE: the sampler would walk
    # BACKWARDS and return something that is not the source at all. Short-circuit
    # to the identity: decode the untouched latent. (That is the codec
    # round-trip, i.e. the best this pipeline can reproduce -- not the raw input
    # bytes, which nothing in latent space can return.)
    if edit_strength == 0.0:
        print("  [edit] strength=0 -> identity: returning the source through the "
              "DAC round-trip, no integration.")
        frames = x1.squeeze(0)
    else:
        # Partial corruption
        t_start = 1.0 - edit_strength
        noise = torch.randn_like(x1)
        x_corrupted = (1 - t_start) * noise + t_start * x1

        # Re-integration with new conditions
        frames = euler_sampling_cfg(
            model, n_frames, device,
            frame_cond=frame_cond, global_cond=global_cond,
            guidance=guidance, steps=steps,
            frame_dims=frame_dims, global_configs=global_configs,
            x_start=x_corrupted, t_start=t_start,
            use_amp=use_amp,
        )

    # Decode (72-dim latents -> quantizer.from_latents -> 1024-dim z -> waveform)
    z_out = normalizer.denormalize(frames.T)
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu"); dac_model.eval()
    z_q, _, _ = dac_model.quantizer.from_latents(z_out.unsqueeze(0).float())
    wav = dac_model.decode(z_q).squeeze()
    del dac_model

    return wav


def build_global_cond(
    label_name, ckpt, device, image_path=None, prompt=None,
) -> Dict[str, torch.Tensor]:
    """
    Builds the global_cond dictionary for a class or for a free-form prompt.

    Args:
        label_name: class name (e.g. "Baroque_sacred"). If prompt is not passed,
                    it is used to automatically build the CLAP text
                    ("baroque sacred"). Also used for file naming.
        ckpt:       checkpoint dict (to read global_configs)
        image_path: optional path to a conditioning image
        prompt:     free-form text for CLAP. If passed, it OVERRIDES label_name
                    as the source of the text embedding (e.g. "slow piano in C minor").

    For the model the "class" is no longer a direct input: all the semantic
    signal flows through CLAP-text and/or CLIP-image.
    """
    gc = {}
    global_configs = ckpt.get("global_configs", {})

    # Text (CLAP): free-form prompt if passed, otherwise derived from the label
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
            text_enc.unload()
            print(f"  [CLAP-text] prompt: {text_input!r}")

    # Image (CLIP): se passata
    if "image" in global_configs and image_path:
        img_enc = ImageCondition()
        emb = img_enc.encode_image(image_path)
        gc["image"] = torch.from_numpy(emb).unsqueeze(0).to(device)
        img_enc.unload()
        print(f"  [CLIP-image] {image_path}")

    return gc


def _align_to_frames(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Align a per-frame condition (T, dim) to exactly n_frames on the time axis,
    exactly as the dataset does in __getitem__: truncate if too long, zero-pad if
    too short. Never interpolates: a condition is truncated or zero-padded,
    never resampled."""
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
    """
    Build the frame_conditions dict {name: (1, n_frames, raw_dim)} on `device`,
    matching the checkpoint's frame_cond_dims, from one of:
      --condition_npz : an .npz with keys = condition names (e.g. f0, energy),
                        each (T, raw_dim); typically produced by extract_conditions.py.
      --condition_wav : a WAV from which the required frame conditions are
                        RE-EXTRACTED with the same ConditionRegistry used at
                        extraction time (identical extractor configuration).

    Alignment to n_frames mirrors the dataset (truncate/zero-pad). Any required
    condition not found in the source is zero-filled (its null value) with a
    warning. Returns None if the model has no frame conditioning, or if no source
    was given (the caller enforces the safety policy in that case).
    """
    frame_cond_dims = ckpt.get("frame_cond_dims", {}) or {}
    if not frame_cond_dims:
        return None  # model has no frame conditioning at all

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
        raw = reg.extract_frame_conditions(audio, sr, n_frames)  # {name: (n_frames, dim)}
        print(f"  [frame-cond] re-extracted from wav: {sorted(raw.keys())}")

    else:
        return None  # no frame source given; caller decides what to do

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
        fc[name] = t.unsqueeze(0).to(device)          # (1, n_frames, rdim)
    return fc


def resolve_normalizer_path(ckpt: dict, ckpt_path: str, cli_path: str = None) -> str:
    """
    Find the normalizer.pt to use, in priority order:
      1. cli_path, if explicitly provided
      2. cache_dir/normalizer.pt, from the config stored in the checkpoint
      3. <run_dir>/checkpoints/normalizer.pt relative to the checkpoint location
      4. a few common fallbacks
    Returns the first existing path, or raises FileNotFoundError listing all
    the candidates that were tried. Mirrors sampling.py.
    """
    candidates = []

    if cli_path:
        candidates.append(Path(cli_path))

    # From the config saved inside the checkpoint (new-style checkpoints)
    cfg = ckpt.get("config", None)
    if isinstance(cfg, dict):
        cache_dir = cfg.get("paths", {}).get("cache_dir", None)
        if cache_dir:
            candidates.append(Path(cache_dir) / "normalizer.pt")

    # Relative to the checkpoint: runs/<run>/checkpoints/<file>
    ckpt_p = Path(ckpt_path).resolve()
    if ckpt_p.parent.name == "checkpoints":
        run_dir = ckpt_p.parent.parent
        candidates.append(run_dir / "checkpoints" / "normalizer.pt")
        candidates.append(run_dir.parent.parent / "cache" / "normalizer.pt")
    # Same directory as the checkpoint (legacy layout)
    candidates.append(Path(ckpt_path).parent / "normalizer.pt")

    # Common fallbacks
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
    parser.add_argument("mode", choices=["generate", "edit"])
    parser.add_argument("--source", type=str, default=None,
                        help="Source audio for edit mode")
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
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--duration", type=float, default=None,
                        help="Generation length in seconds (generate mode). If "
                             "omitted, uses the exact n_frames the checkpoint was "
                             "trained with (recommended). A value here is converted "
                             "to DAC frames via round(), matching preprocessing.")
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

    # Load checkpoint
    print(f"Carico checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    frame_cond_dims     = ckpt.get("frame_cond_dims",     {})
    frame_cond_out_dims = ckpt.get("frame_cond_out_dims", {})
    global_configs_ckpt = ckpt.get("global_configs",      {})
    # Back-compat: if a checkpoint stored frame_cond_dims but not
    # frame_cond_out_dims, recover the per-condition out_dim from
    # CONDITION_CONFIG (conditions.py) so the model can still be rebuilt.
    if frame_cond_dims and not frame_cond_out_dims:
        from conditions import ConditionRegistry
        _reg = ConditionRegistry(
            enabled_frame=list(frame_cond_dims.keys()), enabled_global=[],
        )
        frame_cond_out_dims = _reg.frame_cond_out_dims

    model = ConditionedAudioDiT(
        kind=ckpt.get("model_kind", "L"),
        frame_cond_dims=frame_cond_dims,
        frame_cond_out_dims=frame_cond_out_dims,
        global_cond_configs=global_configs_ckpt,
    ).to(device)

    # Prefer the EMA weights, but ONLY if the shadow was actually being updated
    # when the checkpoint was written. Before training.ema_start the shadow is
    # still the random initialisation, so loading it would generate noise while
    # reporting "using EMA". Checkpoints written before this flag existed have no
    # 'ema_ready' key -> assume ready (previous behaviour).
    if "ema_state_dict" in ckpt and ckpt.get("ema_ready", True):
        model.load_state_dict(ckpt["ema_state_dict"])
        print("  → Usando EMA model")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            print("  → EMA present but NOT trained yet (checkpoint predates "
                  "training.ema_start): using the live model weights")

    # Normalizer (robust resolution: CLI > checkpoint config > standard paths)
    normalizer = LatentNormalizer()
    norm_path = resolve_normalizer_path(ckpt, args.checkpoint, cli_path=args.normalizer)
    print(f"Normalizer: {norm_path}")
    normalizer.load(norm_path)

    # Global conditions
    gc = build_global_cond(
        args.label, ckpt, device,
        image_path=args.image, prompt=args.prompt,
    )

    frame_dims = frame_cond_dims
    global_configs = global_configs_ckpt

    # ---- SAFETY: global conditions (bug #5) ----
    # Symmetric to the frame-condition policy: if the checkpoint REQUIRES a global
    # condition (text/image) and the user provided nothing to build it, refuse to
    # silently generate with a NULL global embedding unless explicitly opted in.
    # `gc` only contains a key when its input was actually supplied
    # (text <- --prompt/--label, image <- --image), so a missing key == null.
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

    # ---- SAFETY: frame conditions (bug #5 sibling) ----
    # Based on whether a frame-condition SOURCE was given, not on n_frames (which
    # in edit mode is only known after encoding the source). If the checkpoint
    # requires frame conditioning and no source was given, refuse unless opted in.
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

    # --- GENERATE ---
    if args.mode == "generate":
        # Generation length: default to the exact n_frames the model trained on
        # (stored in the checkpoint), so a 5 s run generates 431 frames, not 430.
        # A user-provided --duration is converted with round() (codec-consistent).
        if args.duration is None:
            n_frames = int(ckpt.get("n_frames", round(5.0 * DAC_FRAMES_PER_S)))
        else:
            n_frames = int(round(args.duration * DAC_FRAMES_PER_S))
        frame_cond = build_frame_cond(args, ckpt, n_frames, device)

        print(f"Generazione {args.n_samples} audio | "
              f"{n_frames} frame | guidance={args.guidance} | "
              f"frame_cond={'ON' if frame_cond else 'NULL'}")

        import dac
        dac_m = dac.DAC.load(dac.utils.download(model_type="44khz"))
        dac_m.to("cpu"); dac_m.eval()

        for i in range(args.n_samples):
            gen = euler_sampling_cfg(
                model, n_frames, device,
                frame_cond=frame_cond,
                global_cond=gc if gc else None,
                guidance=args.guidance, steps=args.steps,
                frame_dims=frame_dims, global_configs=global_configs,
            )
            z = normalizer.denormalize(gen.T)
            z_q, _, _ = dac_m.quantizer.from_latents(z.unsqueeze(0).float())
            wav = dac_m.decode(z_q).squeeze()
            tag = args.label or "uncond"
            p = os.path.join(args.output, f"gen_{tag}_{i:02d}.wav")
            sf.write(p, wav.numpy(), DAC_SAMPLE_RATE)
            print(f"  {p}")

    # --- EDIT ---
    elif args.mode == "edit":
        if not args.source:
            print("[ERROR] --source richiesto per mode=edit")
            sys.exit(1)

        # Frame conditions are aligned to the SOURCE length (bug #4): pass a
        # builder that edit_audio calls once the source's n_frames is known.
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
        )
        tag = args.label or "edited"
        p = os.path.join(args.output, f"edit_{tag}.wav")
        sf.write(p, wav.numpy(), DAC_SAMPLE_RATE)
        print(f"  {p}")


if __name__ == "__main__":
    main()
