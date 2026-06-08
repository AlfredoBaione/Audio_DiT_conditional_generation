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
    CLAPTextCondition, ImageCondition, ImageDatasetManager,
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

    actual_start = max(t_start, T_MIN)
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
    frame_cond=None, global_cond=None,
    edit_strength=0.3, guidance=3.0, steps=50,
    frame_dims=None, global_configs=None,
    use_amp=True,
):
    """
    Editing: real audio -> partial corruption -> reconstruction.
    edit_strength: 0.0 = no change, 1.0 = full regeneration.
    """
    import dac

    # Encode source
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to(device); dac_model.eval()

    audio, sr = sf.read(source_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != DAC_SAMPLE_RATE:
        import torchaudio
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).float().unsqueeze(0), sr, DAC_SAMPLE_RATE
        ).squeeze(0).numpy()

    audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    x = dac_model.preprocess(audio_t, DAC_SAMPLE_RATE)
    z, _, _, _, _ = dac_model.encode(x)
    z = z.squeeze(0).cpu()  # (1024, T)
    del dac_model
    if device == "cuda":
        torch.cuda.empty_cache()

    n_frames = z.shape[1]
    z_norm = normalizer.normalize(z)
    x1 = z_norm.T.unsqueeze(0)

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

    # Decode
    z_out = normalizer.denormalize(frames.T)
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu"); dac_model.eval()
    wav = dac_model.decode(z_out.unsqueeze(0).float()).squeeze()
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
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="./generated_cond")
    parser.add_argument("--normalizer", type=str, default=None,
                        help="Explicit path to normalizer.pt (otherwise resolved "
                             "from the checkpoint config / standard locations).")
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

    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("  → Usando EMA model")
    else:
        model.load_state_dict(ckpt["model_state_dict"])

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

    os.makedirs(args.output, exist_ok=True)

    # --- GENERATE ---
    if args.mode == "generate":
        n_frames = int(args.duration * DAC_FRAMES_PER_S)
        print(f"Generazione {args.n_samples} audio | "
              f"{n_frames} frame | guidance={args.guidance}")

        import dac
        dac_m = dac.DAC.load(dac.utils.download(model_type="44khz"))
        dac_m.to("cpu"); dac_m.eval()

        for i in range(args.n_samples):
            gen = euler_sampling_cfg(
                model, n_frames, device,
                global_cond=gc if gc else None,
                guidance=args.guidance, steps=args.steps,
                frame_dims=frame_dims, global_configs=global_configs,
            )
            z = normalizer.denormalize(gen.T)
            wav = dac_m.decode(z.unsqueeze(0).float()).squeeze()
            tag = args.label or "uncond"
            p = os.path.join(args.output, f"gen_{tag}_{i:02d}.wav")
            sf.write(p, wav.numpy(), DAC_SAMPLE_RATE)
            print(f"  {p}")

    # --- EDIT ---
    elif args.mode == "edit":
        if not args.source:
            print("[ERROR] --source richiesto per mode=edit")
            sys.exit(1)

        print(f"Editing {args.source} | strength={args.strength} | "
              f"guidance={args.guidance}")
        wav = edit_audio(
            model, normalizer, args.source, device,
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
