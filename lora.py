# LoRA and DoRA adapters on a frozen trained model.

import math
import os
import re
import shutil

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

ADAPTERS = ("lora", "dora-rows")
_ATTENTION = r"\.attn\.(qkv|proj)$|\.cross_attn\.(q|kv|proj)$"
_BLOCKS = _ATTENTION + r"|\.mlp\.w[123]$"
TARGETS = {
    "attention": _ATTENTION,
    "blocks": _BLOCKS,
    "sa3": _BLOCKS + r"|^t_embedder\.mlp\.[02]$|^global_encoder\.|^input_proj$"
                     r"|^final_layer\.linear$",
}
_ADAPTER_KEY = ".parametrizations.weight.0."


def adapter_of(cfg):
    return (cfg.get("lora", None) or {}).get("adapter", None)


def check_config(cfg):
    if cfg.lora.adapter not in ADAPTERS:
        raise SystemExit(f"[config] lora.adapter = {cfg.lora.adapter!r}, must be "
                         f"null or one of {list(ADAPTERS)}. Nothing was run.")
    if cfg.lora.targets not in TARGETS:
        raise SystemExit(f"[config] lora.targets = {cfg.lora.targets!r}, must be "
                         f"one of {list(TARGETS)}. Nothing was run.")
    if int(cfg.lora.rank) < 1:
        raise SystemExit(f"[config] lora.rank = {cfg.lora.rank}, must be >= 1. "
                         f"Nothing was run.")
    for key in ("init_from", "normalizer_from"):
        path = cfg.paths.get(key, None)
        if not path or not os.path.exists(path):
            raise SystemExit(f"[config] lora.adapter needs paths.{key}, got "
                             f"{path!r}. Nothing was run.")


def adapted_weight(weight, lora_A, lora_B, magnitude=None, strength=1.0):
    if magnitude is None:
        return weight + strength * (lora_B @ lora_A)
    if strength == 0:
        return weight
    v = weight + strength * (lora_B @ lora_A)
    if strength != 1:
        base_norm = weight.norm(dim=1)
        magnitude = base_norm + strength * (magnitude - base_norm)
    return v / (v.norm(dim=1, keepdim=True) + 1e-12) * magnitude.unsqueeze(1)


class Adapter(nn.Module):
    def __init__(self, weight, rank, adapter):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, weight.shape[1], device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.magnitude = (nn.Parameter(weight.detach().norm(dim=1))
                          if adapter == "dora-rows" else None)

    def forward(self, weight):
        return adapted_weight(weight, self.lora_A, self.lora_B, self.magnitude)


def base_weights(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if "ema_state_dict" in ckpt and ckpt.get("ema_ready", True):
        return ckpt["ema_state_dict"]
    return ckpt["model_state_dict"]


def attach(model, cfg):
    path = cfg.paths.init_from
    try:
        model.load_state_dict(base_weights(path))
    except RuntimeError as e:
        raise SystemExit(f"[lora] {path} does not fit this model: model.* and "
                         f"conditioning.enabled_* must be the ones it was "
                         f"trained with.\n{e}")
    model.requires_grad_(False)
    pattern = re.compile(TARGETS[cfg.lora.targets])
    layers = [m for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and pattern.search(n)]
    for m in layers:
        parametrize.register_parametrization(
            m, "weight", Adapter(m.weight, int(cfg.lora.rank), cfg.lora.adapter),
            unsafe=True)
    print(f"[lora] {cfg.lora.adapter}, rank {cfg.lora.rank}, targets "
          f"{cfg.lora.targets}: {len(layers)} layers | base model frozen: {path}")


def adapter_state(model):
    return {k: v for k, v in model.state_dict().items() if _ADAPTER_KEY in k}


def load_adapters(model, state):
    if set(state) != set(adapter_state(model)):
        raise RuntimeError("the checkpoint's adapters are not the ones of this "
                           "model: lora.targets or the model differ.")
    model.load_state_dict(state, strict=False)


def merge_checkpoint(ckpt, base_path, strength=1.0):
    merged = dict(base_weights(base_path))
    layers = {}
    for key, value in ckpt["model_state_dict"].items():
        layer, name = key.split(_ADAPTER_KEY)
        layers.setdefault(layer, {})[name] = value.cpu()
    for layer, p in layers.items():
        merged[f"{layer}.weight"] = adapted_weight(
            merged[f"{layer}.weight"], p["lora_A"], p["lora_B"],
            p.get("magnitude", None), strength)
    print(f"[lora] {len(layers)} layers merged into {base_path}, strength {strength}")
    return {**ckpt, "model_state_dict": merged}


def install_normalizer(source, dest):
    if not os.path.exists(dest):
        shutil.copyfile(source, dest)
        return
    a = torch.load(source, map_location="cpu", weights_only=False)
    b = torch.load(dest, map_location="cpu", weights_only=False)
    if not (torch.equal(a["mean"], b["mean"]) and torch.equal(a["std"], b["std"])):
        raise SystemExit(f"[lora] {dest} is not the normalizer of the base model "
                         f"({source}), and the references in that cache were made "
                         f"with it. Use a new paths.cache_dir.")
