# Latent codecs: DAC 44 kHz and EnCodec 32 kHz.

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CodecSpec:
    name: str
    label: str
    sample_rate: int
    hop_length: int
    latent_dim: int

    @property
    def frames_per_s(self) -> float:
        return self.sample_rate / self.hop_length


CODECS = {
    "dac_44khz": CodecSpec("dac_44khz", "DAC", 44100, 512, 72),
    "encodec_32khz": CodecSpec("encodec_32khz", "EnCodec", 32000, 640, 128),
}
CODEC_NAMES = tuple(CODECS)
DEFAULT_CODEC = "dac_44khz"

_ENCODEC_HF_ID = {"encodec_32khz": "facebook/encodec_32khz"}


def get_spec(name) -> CodecSpec:
    if isinstance(name, CodecSpec):
        return name
    if name not in CODECS:
        raise ValueError(f"unknown codec {name!r}: must be one of {list(CODECS)}")
    return CODECS[name]


def codec_from_meta(meta: Optional[dict]) -> str:
    name = (meta or {}).get("codec", None)
    if name is None:
        return DEFAULT_CODEC
    return get_spec(name).name


def dataset_meta_path(latent_root) -> Path:
    return Path(latent_root).parent / "dataset_meta.json"


def dataset_codec(latent_root) -> str:
    p = dataset_meta_path(latent_root)
    if not p.exists():
        return DEFAULT_CODEC
    try:
        meta = json.loads(p.read_text())
    except Exception:
        return DEFAULT_CODEC
    return codec_from_meta(meta)


def codec_from_ckpt(ckpt: dict) -> str:
    name = codec_from_meta({"codec": ckpt.get("codec", None)})
    width = ckpt_token_dim(ckpt)
    if width is not None and width != get_spec(name).latent_dim:
        raise RuntimeError(
            f"this checkpoint says codec={name!r} ({get_spec(name).latent_dim}-d "
            f"latents) but its weights produce {width}-d tokens. The weights are "
            f"what gets loaded; the field is wrong -- the checkpoint was edited "
            f"or assembled by hand.")
    return name


def ckpt_token_dim(ckpt: dict) -> Optional[int]:
    for key in ("model_state_dict", "ema_state_dict"):
        sd = ckpt.get(key, None)
        if isinstance(sd, dict):
            w = sd.get("final_layer.linear.weight", None)
            if w is not None and hasattr(w, "shape"):
                return int(w.shape[0])
    return None


_ACTIVE = [CODECS[DEFAULT_CODEC]]


def activate(name) -> CodecSpec:
    spec = get_spec(name)
    _ACTIVE[0] = spec
    return spec


def active() -> CodecSpec:
    return _ACTIVE[0]


def active_fps(fps: Optional[float] = None) -> float:
    return active().frames_per_s if fps is None else fps


def active_sr(sr: Optional[int] = None) -> int:
    return active().sample_rate if sr is None else sr


_MODELS = {}


def load_model(name, device: str = "cpu", cache: bool = True):
    spec = get_spec(name)
    key = (spec.name, str(device))
    if not cache or key not in _MODELS:
        if spec.name == "dac_44khz":
            import dac
            model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        else:
            from transformers import EncodecModel
            model = EncodecModel.from_pretrained(_ENCODEC_HF_ID[spec.name])
        model.to(device)
        model.eval()
        if not cache:
            return model
        _MODELS[key] = model
    return _MODELS[key]


def is_dac(model) -> bool:
    return hasattr(getattr(model, "quantizer", None), "from_latents")


def encode(model, wav, sr: int):
    import torch
    with torch.no_grad():
        if is_dac(model):
            x = model.preprocess(wav, sr)
            _z, _codes, latents, _, _ = model.encode(x)
            return latents
        want = int(model.config.sampling_rate)
        if int(sr) != want:
            raise ValueError(f"EnCodec expects {want} Hz audio, got {sr} Hz")
        return model.encoder(wav)


def decode(model, z):
    import torch
    with torch.no_grad():
        if is_dac(model):
            z_q, _, _ = model.quantizer.from_latents(z)
            return model.decode(z_q)
        return model.decoder(z)


def n_frames_for(model, n_samples: int, sr: int) -> int:
    import torch
    dev = next(model.parameters()).device
    z = encode(model, torch.zeros(1, 1, int(n_samples), device=dev), sr)
    return int(z.shape[-1])
