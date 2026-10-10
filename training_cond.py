# Training of the conditioned audio DiT (rectified flow, classifier-free guidance).

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ["TORCH_HOME"] = os.path.join(_cache, "torch")

import copy
import sys
import math
import json
import random
import argparse
from datetime import datetime, timedelta
from contextlib import nullcontext
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="backslashreplace")
    except Exception:
        pass

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audio_dataset_npy import frames_per_chunk

import latent_codec as lc
from network_cond import (ConditionedAudioDiT, check_ckpt_reinject_gate,
                          ckpt_attention, ckpt_qk_norm, ATTENTION_KINDS)
from audio_dataset_cond import (
    build_conditioned_datasets, collate_conditioned, load_caption_table,
)
from conditions import (
    ConditionRegistry,
    CONDITION_CONFIG,
    make_null_frame_conditions, make_null_global_conditions,
)
from sampling_cond import T_SCHEDULES, T_SCHEDULE_RENAMED, euler_grid
from metrics import (
    precompute_latent_reference,
    compute_dac_metrics,
    precompute_audio_reference,
    compute_audio_mu_sigma,
    compute_mu_sigma,
    compute_fad,
    compute_frechet_distance,
    gaussian_kl_fullcov,
    COND_METRICS,
)


_DAC_MODEL  = None
_DAC_DEVICE = "cpu"


def set_dac_device(device: str):
    global _DAC_DEVICE
    dev = str(device).strip().lower()
    if dev not in ("cpu", "cuda") and not dev.startswith("cuda:"):
        raise ValueError(
            f"metrics.dac_device must be 'cpu', 'cuda' or 'cuda:<n>', got "
            f"{device!r}."
        )
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"metrics.dac_device={device!r} but CUDA is not available. Set it to "
            f"'cpu' (the default) to decode on the CPU."
        )
    if _DAC_MODEL is not None and dev != _DAC_DEVICE:
        raise RuntimeError(
            f"set_dac_device({device!r}) called after the DAC was already loaded "
            f"on {_DAC_DEVICE!r}. The device must be chosen before the first "
            f"get_dac()."
        )
    _DAC_DEVICE = dev


def get_dac():
    global _DAC_MODEL
    if _DAC_MODEL is None:
        if lc.active().name == "dac_44khz":
            import dac
            _DAC_MODEL = dac.DAC.load(dac.utils.download(model_type="44khz"))
            _DAC_MODEL.to(_DAC_DEVICE)
            _DAC_MODEL.eval()
        else:
            _DAC_MODEL = lc.load_model(lc.active().name, _DAC_DEVICE)
        dev = next(_DAC_MODEL.parameters()).device
        print(f"[{lc.active().label}] Model loaded once ({str(dev).upper()}) "
              f"and cached for the whole run.")
    return _DAC_MODEL


def decode_frames_to_wav(frames, normalizer, dac_model):
    z = normalizer.denormalize(frames.T)
    z = z.unsqueeze(0).float().to(next(dac_model.parameters()).device)
    if not lc.is_dac(dac_model):
        return lc.decode(dac_model, z).squeeze().detach().cpu()
    z_q, _, _ = dac_model.quantizer.from_latents(z)
    return dac_model.decode(z_q).squeeze().detach().cpu()


def _load_dataset_meta(latent_root):
    p = Path(latent_root).parent / "dataset_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _latent_file_list_hash(latent_root):
    import hashlib
    root = Path(latent_root)
    hasher = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*.npy")):
        st = p.stat()
        entry = f"{p.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}"
        if n:
            hasher.update(b"\n")
        hasher.update(entry.encode("utf-8"))
        n += 1
    return hasher.hexdigest(), n


def _splits_fingerprint(cfg):
    import hashlib
    p = Path(cfg.paths.dataset_root).parent / "splits.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return {"unreadable": True}
    groups = payload.get("groups", {})
    digest = hashlib.sha1(
        json.dumps(groups, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "params": payload.get("params", {}),
        "source": payload.get("source"),
        "n_sources": len(groups),
        "assignment_sha1": digest,
    }


def _cache_fingerprint(cfg, n_frames):
    flist_hash, flist_count = _latent_file_list_hash(cfg.paths.dataset_root)
    return {
        "latent_root": os.path.abspath(cfg.paths.dataset_root),
        "dataset_meta": _load_dataset_meta(cfg.paths.dataset_root),
        "duration_s": float(cfg.model.duration_s),
        "n_frames": int(n_frames),
        "latent_dim": int(lc.active().latent_dim),
        "latent_file_list_hash": flist_hash,
        "latent_file_count": flist_count,
        "split": _splits_fingerprint(cfg),
    }


def _validate_cache(cache_dir, fingerprint, guarded_files):
    meta_path = os.path.join(cache_dir, "cache_meta.json")
    present = [f for f in guarded_files if os.path.exists(f)]

    if os.path.exists(meta_path):
        try:
            old = json.loads(Path(meta_path).read_text())
        except Exception:
            old = None
        if old == fingerprint:
            print(f"[cache] verified against {meta_path} -> reusing cache.")
            return
        raise SystemExit(
            "[cache] STALE CACHE: the cached normalizer / FD-DAC reference in\n"
            f"  {cache_dir}\n"
            "were computed on a DIFFERENT dataset/duration/split than the current "
            "run, so reusing them would silently corrupt normalization and FD/KL.\n"
            f"  cached : {json.dumps(old,  sort_keys=True)}\n"
            f"  current: {json.dumps(fingerprint, sort_keys=True)}\n"
            "Point paths.cache_dir to a fresh directory, or delete "
            "normalizer.pt / fd_dac_ref_stats.pt / cache_meta.json there.")

    if present:
        raise SystemExit(
            "[cache] UNVERIFIABLE CACHE: found "
            f"{[os.path.basename(f) for f in present]} in\n  {cache_dir}\n"
            "but no cache_meta.json to tie them to a dataset. After the "
            "preprocessing refactor the latent statistics changed, so an old "
            "cache is almost certainly stale. Delete those files (and any "
            "cache_meta.json) or use a fresh paths.cache_dir.")

    os.makedirs(cache_dir, exist_ok=True)
    Path(meta_path).write_text(json.dumps(fingerprint, indent=2))
    print(f"[cache] fresh cache -> wrote {meta_path}")


def _npz_keys(npz):
    try:
        with np.load(str(npz)) as d:
            return set(d.files)
    except Exception:
        return None


def _scan_frame_conditions(condition_root, io_workers=16, block=4096):
    from collections import Counter as _Counter
    from concurrent.futures import ThreadPoolExecutor
    from itertools import islice
    root = Path(condition_root) if condition_root else None
    if root is None or not root.exists():
        return {"total": 0, "present": {}}
    total = 0
    present = _Counter()
    paths = root.rglob("*.npz")
    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as ex:
        with tqdm(desc="Condition scan", unit="file") as pbar:
            while True:
                chunk = list(islice(paths, block))
                if not chunk:
                    break
                for keys in ex.map(_npz_keys, chunk):
                    if keys is None:
                        continue
                    total += 1
                    for k in keys:
                        present[k] += 1
                pbar.update(len(chunk))
    return {"total": total, "present": dict(present)}


def _flatten_keys(d, prefix=""):
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.append(kk)
            out.extend(_flatten_keys(v, kk))
    return out


def merge_cli_overrides(cfg, dotlist, where):
    if not dotlist:
        return cfg
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    OmegaConf.set_struct(cfg, True)
    try:
        cfg = OmegaConf.merge(cfg, cli_cfg)
    except Exception as e:
        base_keys = set(_flatten_keys(OmegaConf.to_container(cfg, resolve=False)))
        bad = [k for k in _flatten_keys(OmegaConf.to_container(cli_cfg))
               if k not in base_keys]
        notes = dict.fromkeys(
            OBSOLETE_SAMPLING_KEYS[k.partition(".")[2]] for k in bad
            if k.startswith("sampling.")
            and k.partition(".")[2] in OBSOLETE_SAMPLING_KEYS)
        _hint = "".join(f"\n          {n}" for n in notes)
        raise SystemExit(
            f"[config] unknown CLI override key(s): {bad or [str(e)]}\n"
            f"          These do not exist in {where}. Check the "
            f"spelling and the dotted path (e.g. 'data.num_val_batches', not "
            f"'data_num_val_batches'). Nothing was run.{_hint}")
    OmegaConf.set_struct(cfg, False)
    return cfg


def load_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str,
                        default="configs/training_cond_default.yaml")
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

    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        _meta = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "config" in _meta and _meta["config"] is not None:
            ckpt_cfg = OmegaConf.create(_meta["config"])
            cfg = OmegaConf.merge(cfg, ckpt_cfg)
            print("[RESUME] Config restored from checkpoint "
                  f"(model.kind={cfg.model.kind}, "
                  f"frame_reinject_every={cfg.model.get('frame_reinject_every', 0)}, "
                  f"attention={cfg.model.get('attention', 'standard')}, "
                  f"t_sampler={cfg.training.get('t_sampler', 'logit_normal')}, "
                  f"t_schedule={cfg.sampling.get('t_schedule', None) or 'uniform'}, "
                  f"enabled_frame={cfg.conditioning.enabled_frame}, "
                  f"enabled_global={cfg.conditioning.enabled_global}, "
                  f"train_batch_size={cfg.data.train_batch_size}).")
        elif "model_kind" in _meta:
            cfg.model.kind = _meta["model_kind"]
            print(f"[RESUME] model.kind restored from checkpoint: {cfg.model.kind} "
                  "(older checkpoint without full config; other params come "
                  "from the YAML/CLI).")
        cfg.model.qk_norm = ckpt_qk_norm(_meta)
        _ck_trn = (_meta.get("config") or {}).get("training") or {}
        _legacy = {k: v for k, v in LEGACY_TRAINING_KEYS.items()
                   if k not in _ck_trn}
        for _k, _v in _legacy.items():
            cfg.training[_k] = _v
        print(f"[RESUME] model.qk_norm={cfg.model.qk_norm} (read off the weights)"
              + (f" | absent from this checkpoint, set to what it was trained "
                 f"with: {_legacy}" if _legacy else ""))
        del _meta

    drop_obsolete_sampling_keys(cfg)

    cfg = merge_cli_overrides(cfg, unknown, args.config)

    _smp = cfg.get("sampling", None)
    if _smp is not None:
        for _k in PANEL_KEYS.values():
            _v = _smp.get(_k, None)
            if _v is not None and int(_v) < 0:
                raise SystemExit(
                    f"[config] sampling.{_k} = {_v}, must be >= 0 (0 = no "
                    f"panels of that family). Nothing was run.")

    _p_all = float(cfg.conditioning.p_drop_all)
    _p_frm = float(cfg.conditioning.p_drop_frame)
    _p_gbl = float(cfg.conditioning.p_drop_global)
    _p_each = float(cfg.conditioning.get("p_drop_each_frame", 0.0))
    for _name, _val in (("p_drop_all", _p_all), ("p_drop_frame", _p_frm),
                        ("p_drop_global", _p_gbl),
                        ("p_drop_each_frame", _p_each)):
        if not 0.0 <= _val <= 1.0:
            raise SystemExit(
                f"[config] conditioning.{_name} = {_val} is outside [0, 1]. "
                f"These are probabilities. Nothing was run.")
    _p_sum = _p_all + _p_frm + _p_gbl
    if _p_sum >= 1.0:
        raise SystemExit(
            f"[config] conditioning.p_drop_all + p_drop_frame + p_drop_global "
            f"= {_p_all} + {_p_frm} + {_p_gbl} = {_p_sum:.3f}, which leaves "
            f"NOTHING for the 'keep everything' case: every training sample "
            f"would have some condition dropped and the model would never see "
            f"full conditioning. Their sum must stay below 1.0 (the default "
            f"0.10 + 0.05 + 0.05 = 0.20 leaves 80%). Nothing was run.")
    if _p_all <= 0.0 and (_p_frm <= 0.0 or _p_gbl <= 0.0):
        print(f"[config] WARNING: p_drop_all={_p_all}, p_drop_frame={_p_frm}, "
              f"p_drop_global={_p_gbl}. Classifier-free guidance needs a NULL "
              f"branch to extrapolate from: with no dropout on a branch, "
              f"guidance_scale on that branch is meaningless.")
    del _p_all, _p_frm, _p_gbl, _p_each, _p_sum

    _att = cfg.model.get("attention", "standard")
    if _att not in ATTENTION_KINDS:
        raise SystemExit(
            f"[config] model.attention = {_att!r}, must be one of "
            f"{list(ATTENTION_KINDS)}. Nothing was run.")
    _tsm = cfg.training.get("t_sampler", "logit_normal")
    if _tsm not in T_SAMPLERS:
        raise SystemExit(
            f"[config] training.t_sampler = {_tsm!r}, must be one of "
            f"{list(T_SAMPLERS)}. Nothing was run.")
    t_schedule_of(cfg.get("sampling", None))
    del _att, _tsm

    _trn = cfg.training
    _sch = _trn.get("lr_schedule", "cosine")
    if _sch not in LR_SCHEDULES:
        raise SystemExit(
            f"[config] training.lr_schedule = {_sch!r}, must be one of "
            f"{list(LR_SCHEDULES)}. Nothing was run.")
    if _sch == "inverse_power" and not (float(_trn.get("lr_inv_gamma", 1.0e6)) > 0
                                        and float(_trn.get("lr_power", 0.5)) >= 0):
        raise SystemExit(
            f"[config] training.lr_inv_gamma = {_trn.get('lr_inv_gamma')} must be "
            f"> 0 and training.lr_power = {_trn.get('lr_power')} must be >= 0. "
            f"Nothing was run.")
    _betas = list(_trn.get("adam_betas", LEGACY_TRAINING_KEYS["adam_betas"]))
    if len(_betas) != 2 or not all(0.0 <= float(b) < 1.0 for b in _betas):
        raise SystemExit(
            f"[config] training.adam_betas = {_betas}, must be two values in "
            f"[0, 1), e.g. [0.9, 0.95]. Nothing was run.")
    if not float(_trn.get("adam_eps", 1e-8)) > 0:
        raise SystemExit(
            f"[config] training.adam_eps = {_trn.get('adam_eps')}, must be > 0. "
            f"Nothing was run.")
    if float(_trn.weight_decay) < 0:
        raise SystemExit(
            f"[config] training.weight_decay = {_trn.weight_decay}, must be >= 0. "
            f"Nothing was run.")
    del _trn, _sch, _betas

    if args.resume is not None:
        cfg.paths.resume_from = args.resume

    if args.run_name is not None:
        run_name = args.run_name
    elif cfg.paths.get("run_name") is not None:
        run_name = cfg.paths.run_name
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    cfg.paths.run_name = run_name

    cfg.data.effective_bs = cfg.data.train_batch_size * cfg.data.grad_accum

    return cfg, run_name


LR_SCHEDULES = ("inverse_power", "cosine")
LEGACY_TRAINING_KEYS = {
    "lr_schedule": "cosine",
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "weight_decay_matrices_only": False,
}


def make_lr_lambda(num_steps: int, warmup_steps: int, decay_start_frac: float,
                   schedule: str = "cosine", inv_gamma: float = 1.0e6,
                   power: float = 0.5):
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"lr schedule must be one of {LR_SCHEDULES}, got {schedule!r}")
    decay_start = int(num_steps * decay_start_frac)

    def lr_lambda(step: int) -> float:
        if schedule == "inverse_power":
            warm = step / warmup_steps if step < warmup_steps else 1.0
            return warm * (1.0 + step / inv_gamma) ** (-power)
        if step < warmup_steps:
            return step / warmup_steps
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / (num_steps - decay_start)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)).item())

    return lr_lambda


def lr_lambda_from_cfg(train_cfg):
    return make_lr_lambda(
        num_steps=train_cfg.num_steps,
        warmup_steps=train_cfg.warmup_steps,
        decay_start_frac=train_cfg.decay_start_frac,
        schedule=str(train_cfg.get("lr_schedule", "cosine")),
        inv_gamma=float(train_cfg.get("lr_inv_gamma", 1.0e6)),
        power=float(train_cfg.get("lr_power", 0.5)),
    )


def is_decayed_param(name: str, p: torch.Tensor) -> bool:
    return p.ndim >= 2 and not name.endswith("text_null")


def optimizer_layout(train_cfg) -> str:
    return ("matrices"
            if bool(train_cfg.get("weight_decay_matrices_only", False))
            and float(train_cfg.weight_decay) > 0 else "single")


def build_optimizer(model, train_cfg, layout: str):
    wd = float(train_cfg.weight_decay)
    named = list(model.named_parameters())
    if layout == "matrices":
        groups = [
            {"params": [p for n, p in named if is_decayed_param(n, p)],
             "weight_decay": wd},
            {"params": [p for n, p in named if not is_decayed_param(n, p)],
             "weight_decay": 0.0},
        ]
    elif layout == "single":
        groups = [{"params": [p for _, p in named], "weight_decay": wd}]
    else:
        raise ValueError(f"optimizer layout must be 'single' or 'matrices', got {layout!r}")
    betas = train_cfg.get("adam_betas", LEGACY_TRAINING_KEYS["adam_betas"])
    return torch.optim.AdamW(
        groups,
        lr=float(train_cfg.lr),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(train_cfg.get("adam_eps", LEGACY_TRAINING_KEYS["adam_eps"])),
    )


def optimizer_overrides_ignored(optimizer, scheduler, train_cfg) -> list:
    g = optimizer.param_groups[0]
    betas = tuple(float(b) for b in
                  train_cfg.get("adam_betas", LEGACY_TRAINING_KEYS["adam_betas"]))
    eps = float(train_cfg.get("adam_eps", LEGACY_TRAINING_KEYS["adam_eps"]))
    out = []
    if not math.isclose(scheduler.base_lrs[0], float(train_cfg.lr), rel_tol=1e-9):
        out.append(f"lr={scheduler.base_lrs[0]:g} (training.lr={float(train_cfg.lr):g})")
    if tuple(float(b) for b in g["betas"]) != betas:
        out.append(f"betas={tuple(g['betas'])} (training.adam_betas={list(betas)})")
    if not math.isclose(g["eps"], eps, rel_tol=1e-9):
        out.append(f"eps={g['eps']:g} (training.adam_eps={eps:g})")
    if not math.isclose(g["weight_decay"], float(train_cfg.weight_decay),
                        rel_tol=1e-9, abs_tol=1e-12):
        out.append(f"weight_decay={g['weight_decay']:g} "
                   f"(training.weight_decay={float(train_cfg.weight_decay):g})")
    return out


def restore_optimizer(model, train_cfg, optimizer, scheduler, lr_lambda, ckpt):
    saved = ("matrices" if len(ckpt["optimizer_state_dict"]["param_groups"]) == 2
             else "single")
    rebuilt = None
    if saved != optimizer_layout(train_cfg):
        optimizer = build_optimizer(model, train_cfg, saved)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        rebuilt = saved
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return (optimizer, scheduler, rebuilt,
            optimizer_overrides_ignored(optimizer, scheduler, train_cfg))


def sample_logit_normal(batch_size, device, t_min, t_max, mean=0.0, std=1.0):
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u).clamp(t_min, t_max)


T_SAMPLERS = ("logit_normal", "truncated_logit")
T_SAMPLER_RENAMED = {"sa3": "truncated_logit"}

T_TRUNC = 0.075
SHIFT_MU_MIN, SHIFT_MU_MAX = 0.5, 1.15
SHIFT_LEN_MIN, SHIFT_LEN_MAX = 256, 4096


def length_shift_mu(seq_len: int) -> float:
    L = min(max(int(seq_len), SHIFT_LEN_MIN), SHIFT_LEN_MAX)
    return SHIFT_MU_MIN + ((SHIFT_MU_MAX - SHIFT_MU_MIN) * (L - SHIFT_LEN_MIN)
                         / (SHIFT_LEN_MAX - SHIFT_LEN_MIN))


def sample_t_truncated_logit(batch_size, device, t_min, t_max, seq_len):
    lo = torch.special.ndtr(torch.logit(torch.tensor(T_TRUNC, dtype=torch.float64)))
    c = lo + (1.0 - lo) * torch.rand(batch_size, device=device, dtype=torch.float64)
    s = torch.sigmoid(torch.special.ndtri(c))
    u = (s - T_TRUNC) / (1.0 - T_TRUNC)
    u = u.clamp(1e-12, 1.0 - 1e-12)
    t = torch.sigmoid(torch.logit(u) - length_shift_mu(seq_len))
    return t.float().clamp(t_min, t_max)


def t_sampler_cdf(t_sampler, x, seq_len):
    x = torch.tensor(float(x), dtype=torch.float64)
    if t_sampler == "truncated_logit":
        lo = torch.special.ndtr(torch.logit(torch.tensor(T_TRUNC, dtype=torch.float64)))
        u_x = torch.sigmoid(torch.logit(x) + length_shift_mu(seq_len))
        s_x = T_TRUNC + (1.0 - T_TRUNC) * u_x
        return float((torch.special.ndtr(torch.logit(s_x)) - lo) / (1.0 - lo))
    return float(torch.special.ndtr(torch.logit(x)))


class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def copy_from(self, model):
        ema_params = dict(self.model.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].copy_(p.data)
        ema_buffers = dict(self.model.named_buffers())
        for name, b in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(b.data)

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.model.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].lerp_(p.data, 1.0 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


def apply_cfg_dropout(frame_cond, global_cond, device, global_configs, B,
                      p_drop_all, p_drop_frame, p_drop_global,
                      p_drop_each_frame=0.0, text_ctx_mask=None):
    r = torch.rand(B, device=device)
    drop_all    = r < p_drop_all
    drop_frame  = (r >= p_drop_all) & (r < p_drop_all + p_drop_frame)
    drop_global = (r >= p_drop_all + p_drop_frame) & \
                  (r < p_drop_all + p_drop_frame + p_drop_global)

    drop_f = drop_all | drop_frame
    drop_g = drop_all | drop_global

    if frame_cond:
        for k in frame_cond:
            drop_k = drop_f
            if p_drop_each_frame > 0.0:
                drop_k = drop_f | (torch.rand(B, device=device)
                                   < p_drop_each_frame)
            keep_mask = (~drop_k).view(B, 1, 1).to(frame_cond[k].dtype)
            frame_cond[k] = frame_cond[k] * keep_mask

    if global_cond:
        null_g = make_null_global_conditions(B, global_configs, device)
        for k in global_cond:
            keep_mask = (~drop_g).view(B, 1).to(global_cond[k].dtype)
            global_cond[k] = global_cond[k] * keep_mask \
                             + null_g[k] * (1 - keep_mask)

    if text_ctx_mask is not None:
        text_ctx_mask = text_ctx_mask & (~drop_g).view(B, 1)

    return frame_cond, global_cond, text_ctx_mask


VALIDATION_PROTOCOL = "fixed_subset_common_noise_sample_weighted_v2"


def compute_loss(model, batch, device, use_amp, t_min, t_max,
                 global_configs, p_drop_all, p_drop_frame, p_drop_global,
                 training=True, x0=None, t=None, p_drop_each_frame=0.0,
                 t_sampler="logit_normal"):
    frames, frame_cond, _labels, text_embs, image_embs, text_ctx = batch

    x1 = frames.to(device).float()
    B = x1.shape[0]
    if x0 is None:
        x0 = torch.randn_like(x1)
    else:
        if tuple(x0.shape) != tuple(x1.shape):
            raise ValueError(
                f"fixed x0 has shape {tuple(x0.shape)}, expected {tuple(x1.shape)}")
        x0 = x0.to(device=device, dtype=x1.dtype, non_blocking=True)

    if t is None:
        if t_sampler == "truncated_logit":
            t = sample_t_truncated_logit(B, device, t_min, t_max, seq_len=x1.shape[1])
        else:
            t = sample_logit_normal(B, device, t_min, t_max)
    else:
        if t.ndim != 1 or t.shape[0] != B:
            raise ValueError(f"fixed t has shape {tuple(t.shape)}, expected ({B},)")
        t = t.to(device=device, dtype=x1.dtype, non_blocking=True)
    t_expand = t.view(B, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0

    fc = {k: v.to(device).float() for k, v in frame_cond.items()}
    gc = {}
    if "text" in global_configs:
        gc["text"] = text_embs.to(device)
    if "image" in global_configs:
        gc["image"] = image_embs.to(device)

    ctx = ctx_mask = None
    if (getattr(getattr(model, "module", model), "text_cross_layers", None)
            and text_ctx is not None):
        ctx = text_ctx["tokens"].to(device, non_blocking=True)
        ctx_mask = text_ctx["mask"].to(device, non_blocking=True)

    if training:
        fc, gc, ctx_mask = apply_cfg_dropout(
            fc, gc, device, global_configs, B,
            p_drop_all=p_drop_all,
            p_drop_frame=p_drop_frame,
            p_drop_global=p_drop_global,
            p_drop_each_frame=p_drop_each_frame,
            text_ctx_mask=ctx_mask,
        )

    with torch.amp.autocast('cuda', enabled=use_amp):
        pred = model(xt, t, frame_conditions=fc, global_conditions=gc,
                     text_context=ctx, text_context_mask=ctx_mask)
        loss = F.mse_loss(pred, target)
    return loss


def t_schedule_of(sampling_cfg) -> str:
    v = "uniform"
    if sampling_cfg is not None:
        v = str(sampling_cfg.get("t_schedule", None) or "uniform")
    if v not in T_SCHEDULES:
        raise SystemExit(
            f"[config] sampling.t_schedule = {v!r}, must be one of "
            f"{list(T_SCHEDULES)}. Nothing was run.")
    return v


@torch.no_grad()
def euler_sample_cfg(model, n_frames, device, steps, t_min, t_max, use_amp,
                      frame_cond, global_cond, guidance,
                      frame_dims, global_configs, gen_rng=None,
                      text_ctx=None, t_schedule="uniform"):
    model.eval()
    x = torch.randn(1, n_frames, lc.active().latent_dim, device=device, generator=gen_rng)

    null_fc = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc = make_null_global_conditions(1, global_configs or {}, device)

    ctx = ctx_mask = None
    if getattr(model, "text_cross_layers", None) and text_ctx is not None:
        ctx = text_ctx["tokens"].to(device)
        ctx_mask = text_ctx["mask"].to(device)

    has_cond = bool(frame_cond) or bool(global_cond) or (ctx is not None)
    use_cfg = (guidance > 1.0) and has_cond

    for tv, dt in euler_grid(steps, t_min, t_max, t_schedule):
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


@torch.no_grad()
def euler_sample_cfg_paired(model, n_frames, device, steps, t_min, t_max, use_amp,
                            frame_cond, global_cond, guidance,
                            frame_dims, global_configs, gen_rng=None,
                            text_ctx=None, x0=None, t_schedule="uniform"):
    model.eval()
    null_fc_1 = make_null_frame_conditions(1, n_frames, frame_dims or {}, device)
    null_gc_1 = make_null_global_conditions(1, global_configs or {}, device)
    fc_c = frame_cond if frame_cond else {}
    gc_c = global_cond if global_cond else {}

    any_t = next(iter(fc_c.values()), None)
    if any_t is None:
        any_t = next(iter(gc_c.values()), None)
    if any_t is None:
        raise RuntimeError("euler_sample_cfg_paired needs at least one condition "
                           "to infer the batch size; use euler_sample_cfg instead.")
    B = int(any_t.shape[0])

    null_fc = {k: v.expand(B, *v.shape[1:]).contiguous() for k, v in null_fc_1.items()}
    null_gc = {k: v.expand(B, *v.shape[1:]).contiguous() for k, v in null_gc_1.items()}

    if x0 is None:
        x0 = torch.stack([
            torch.randn(n_frames, lc.active().latent_dim, device=device, generator=gen_rng)
            for _ in range(B)
        ])
    else:
        x0 = x0.to(device)
        if tuple(x0.shape) != (B, n_frames, lc.active().latent_dim):
            raise ValueError(f"x0 has shape {tuple(x0.shape)}, expected "
                             f"{(B, n_frames, lc.active().latent_dim)}")
    x_cond = x0.clone()
    x_unc = x0.clone()

    frame_batch = {n: torch.cat([fc_c.get(n, null_fc[n]), null_fc[n], null_fc[n]],
                                dim=0) for n in null_fc}
    global_batch = {n: torch.cat([gc_c.get(n, null_gc[n]), null_gc[n], null_gc[n]],
                                 dim=0) for n in null_gc}

    ctx_batch = ctx_mask_batch = None
    if getattr(model, "text_cross_layers", None) and text_ctx is not None:
        tok = text_ctx["tokens"].to(device)
        msk = text_ctx["mask"].to(device)
        if tok.shape[0] != B:
            raise ValueError(
                f"text_ctx has batch {tok.shape[0]}, the conditions say B={B}")
        ctx_batch = torch.cat([tok, tok, tok], dim=0)
        ctx_mask_batch = torch.cat(
            [msk, torch.zeros_like(msk), torch.zeros_like(msk)], dim=0)

    for tv, dt in euler_grid(steps, t_min, t_max, t_schedule):
        t = torch.ones(3 * B, device=device) * tv
        xb = torch.cat([x_cond, x_cond, x_unc], dim=0)
        with torch.amp.autocast('cuda', enabled=use_amp):
            vb = model(xb, t, frame_conditions=frame_batch,
                       global_conditions=global_batch,
                       text_context=ctx_batch,
                       text_context_mask=ctx_mask_batch)
        v_c    = vb[0:B].float()
        v_null = vb[B:2 * B].float()
        v_u    = vb[2 * B:3 * B].float()
        v_guided = v_null + guidance * (v_c - v_null)
        x_cond = x_cond + v_guided * dt
        x_unc = x_unc + v_u * dt

    return ([x_cond[b].cpu() for b in range(B)],
            [x_unc[b].cpu() for b in range(B)])


UNCOND_AUDIO_GROUP = "uncond generation"
REAL_AUDIO_GROUP = "ground truth"


def norm_wav(x):
    a = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    a = np.asarray(a, dtype=np.float32).reshape(1, -1)
    return torch.from_numpy(a / (np.abs(a).max() + 1e-8))


def uncond_card_tag(k):
    return f"{UNCOND_AUDIO_GROUP}/uncond_{int(k):02d}"


def audio_panel_tags(family, idx, active_conditions=(), suffix="",
                     conditioned=None):
    active = sorted(active_conditions)
    lead = "f0" if "f0" in active else (active[0] if active else None)
    if conditioned is None:
        conditioned = lead is not None
    block = f"{family}_{idx:02d}{suffix}"
    tail = f"{family}_{idx:02d}"

    panel_real = (family == "test" and conditioned)
    first = 2 if panel_real else 1

    ordered = ([lead] + [n for n in active if n != lead]) if lead else []
    conditions = {c: f"{block}/{j + first}_{c}_{tail}"
                  for j, c in enumerate(ordered)}
    gen_slot = len(ordered) + first

    real = (f"{block}/1_real_{tail}" if panel_real
            else f"{REAL_AUDIO_GROUP}/real_{tail}")

    return {
        "conditions": conditions,
        "generation": f"{block}/{gen_slot}_generation_{tail}",
        "real": real,
    }


def subset_generation_tag(family, idx, label, suffix="", n_conditions=0,
                          has_real=False):
    return (f"{family}_{idx:02d}{suffix}"
            f"/{int(n_conditions) + (2 if has_real else 1)}"
            f"_gen_{label}_{family}_{idx:02d}")


FIXED_CARDS_FILE = ".fixed_cards_logged"


def fixed_card_pending(writer, tag):
    d = getattr(writer, "log_dir", None)
    if not d:
        return True
    cache = getattr(fixed_card_pending, "_cache", None)
    if cache is None:
        cache = fixed_card_pending._cache = {}
    seen = cache.get(d)
    if seen is None:
        try:
            with open(os.path.join(d, FIXED_CARDS_FILE), encoding="utf-8") as fh:
                seen = {line.rstrip("\n") for line in fh}
        except OSError:
            seen = set()
        cache[d] = seen
    key = tag.replace("\n", " ")
    if key in seen:
        return False
    seen.add(key)
    try:
        with open(os.path.join(d, FIXED_CARDS_FILE), "a", encoding="utf-8") as fh:
            fh.write(key + "\n")
    except OSError:
        pass
    return True


def resolve_influence_subsets(spec, active_names):
    active = list(active_names or [])
    if not spec or not active:
        return []
    out = []

    def add(label, names):
        picked = tuple(a for a in active if a in set(names))
        if not picked:
            return
        if any(lbl == label for lbl, _ in out):
            return
        out.append((label, picked))

    def label_for(names):
        if len(names) == len(active):
            return "all"
        if len(names) == 1:
            return f"only_{names[0]}"
        if len(names) == len(active) - 1:
            missing = [a for a in active if a not in set(names)]
            return f"no_{missing[0]}"
        return "+".join(names)

    for entry in spec:
        if isinstance(entry, str):
            key = entry.strip().lower()
            if key == "all":
                add("all", active)
            elif key in ("loo", "leave_one_out"):
                for n in active:
                    rest = [a for a in active if a != n]
                    add(label_for(tuple(rest)), rest)
            elif key in ("singletons", "each"):
                for n in active:
                    add(f"only_{n}", [n])
            else:
                raise ValueError(
                    f"sampling.influence_subsets: unknown keyword '{entry}'. "
                    f"Use 'all', 'loo', 'singletons', or an explicit list of "
                    f"condition names from {active}.")
        else:
            names = [str(n) for n in entry]
            unknown = [n for n in names if n not in active]
            if unknown:
                raise ValueError(
                    f"sampling.influence_subsets: condition(s) {unknown} are "
                    f"not active in this run. Active: {active}.")
            picked = tuple(a for a in active if a in set(names))
            add(label_for(picked), picked)
    return out


PANEL_KEYS = {"test": "n_test_panels", "probe": "n_probe_panels"}
PANEL_DEFAULT = 8

_PANELS_NOTE = (
    "The Condition_influence table is computed on all the "
    "sampling.n_metrics_samples generations; the listening panels are "
    "sampling.n_test_panels (test samples) and sampling.n_probe_panels "
    "(probe stimuli).")
OBSOLETE_SAMPLING_KEYS = {
    "n_influence_samples": _PANELS_NOTE,
    "n_influence_samples_valid": _PANELS_NOTE,
    "n_influence_samples_probe": _PANELS_NOTE,
    "n_similarity_samples": (
        "The text / image similarity is a column of the Condition_influence "
        "table, on the same generations as FD-DAC / KL / FAD; it has no "
        "curve of its own any more."),
    "n_val_save": (
        "The validation generations are no longer written to disk: "
        "runs/<run>/audio/ holds what the Audio window shows -- test/, "
        "probe/ (each panel with everything it was conditioned on) and "
        "uncond/."),
}


def drop_obsolete_sampling_keys(cfg):
    smp = cfg.get("sampling", None)
    if smp is None:
        return
    gone = [k for k in OBSOLETE_SAMPLING_KEYS if k in smp]
    for k in gone:
        del smp[k]
    for note in dict.fromkeys(OBSOLETE_SAMPLING_KEYS[k] for k in gone):
        keys = [k for k in gone if OBSOLETE_SAMPLING_KEYS[k] == note]
        print(f"[config] sampling.{', sampling.'.join(keys)}: no longer read. "
              f"{note}")
    for k in PANEL_KEYS.values():
        if smp.get(k, None) is None:
            smp[k] = PANEL_DEFAULT
    if smp.get("t_schedule", None) is None:
        smp["t_schedule"] = "uniform"
    _old = smp.get("t_schedule")
    if _old in T_SCHEDULE_RENAMED:
        smp["t_schedule"] = T_SCHEDULE_RENAMED[_old]
        print(f"[config] sampling.t_schedule: {_old!r} is now called "
              f"{smp['t_schedule']!r}.")
    trn = cfg.get("training", None)
    if trn is not None and trn.get("t_sampler", None) in T_SAMPLER_RENAMED:
        _old = trn.get("t_sampler")
        trn["t_sampler"] = T_SAMPLER_RENAMED[_old]
        print(f"[config] training.t_sampler: {_old!r} is now called "
              f"{trn['t_sampler']!r}.")


def panel_count(sampling_cfg, which) -> int:
    if sampling_cfg is None:
        return PANEL_DEFAULT
    v = sampling_cfg.get(PANEL_KEYS[which], None)
    return PANEL_DEFAULT if v is None else max(0, int(v))


def test_panel_indices(n_test, n_panels):
    n = min(max(0, int(n_panels or 0)), max(0, int(n_test or 0)))
    if n <= 0:
        return []
    return torch.linspace(0, int(n_test) - 1, n).long().tolist()


@torch.no_grad()
def generate_and_log_uncond_cards(
    model, normalizer, n_frames, step, writer, device, output_dir, n_cards,
    sampling_cfg, use_amp, frame_dims, global_configs, metrics_seed=None,
    prefix="EMA",
):
    n_cards = max(0, int(n_cards or 0))
    if n_cards <= 0:
        return
    dac_model = get_dac()
    gen_rng = None
    if metrics_seed is not None:
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(int(metrics_seed))
    for k in range(n_cards):
        gen = euler_sample_cfg(
            model, n_frames, device,
            steps=sampling_cfg.euler_steps,
            t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
            t_schedule=t_schedule_of(sampling_cfg),
            use_amp=use_amp,
            frame_cond=None, global_cond=None, guidance=1.0,
            frame_dims=frame_dims, global_configs=global_configs,
            gen_rng=gen_rng,
        )
        if not torch.isfinite(gen).all():
            print(f"    [uncond] uncond_{k:02d}: non-finite generation, "
                  f"not logged")
            continue
        wav = decode_frames_to_wav(gen, normalizer, dac_model)
        writer.add_audio(uncond_card_tag(k), norm_wav(wav), global_step=step,
                         sample_rate=lc.active().sample_rate)
        sf.write(os.path.join(_panel_dir(output_dir, "uncond", k),
                              _generation_name(step, prefix)),
                 wav.numpy(), lc.active().sample_rate)


@torch.no_grad()
def log_real_audio_samples(test_dataset, normalizer, writer,
                           n_samples, sampling_cfg=None, frame_dims=None,
                           global_configs=None, output_dir=None):
    dac_model = get_dac()
    conditioned = bool(frame_dims or global_configs)
    if conditioned:
        ds, family = test_dataset, "test"
        indices = test_panel_indices(len(ds) if ds is not None else 0,
                                     panel_count(sampling_cfg, "test"))
        sfx = panel_suffixes(ds, indices, global_configs)
    else:
        ds, family = test_dataset, "test"
        total = len(ds) if ds is not None else 0
        n = max(0, min(int(n_samples or 0), total))
        indices = (torch.linspace(0, total - 1, n).long().tolist()
                   if n else [])
        sfx = {}

    written = 0
    for k, idx in enumerate(indices):
        tag = audio_panel_tags(family, k, suffix=sfx.get(k, ""),
                               conditioned=conditioned)["real"]
        pending = fixed_card_pending(writer, tag)
        disk = (os.path.join(output_dir, "ground_truth",
                             f"real_test_{k:02d}.wav")
                if output_dir is not None and not conditioned else None)
        if not pending and disk is None:
            continue
        frames = ds[idx][0]
        waveform = decode_frames_to_wav(frames, normalizer, dac_model).unsqueeze(0)
        if disk is not None:
            os.makedirs(os.path.dirname(disk), exist_ok=True)
            sf.write(disk, waveform[0].numpy(), lc.active().sample_rate)
        if not pending:
            continue
        wn = waveform / (waveform.abs().max() + 1e-8)

        writer.add_audio(tag, wn, global_step=0, sample_rate=lc.active().sample_rate)
        written += 1

    print(f"  {written} real audios logged on TensorBoard"
          + ("" if written == len(indices)
             else f" ({len(indices) - written} already on this board)"))


@torch.no_grad()
def run_joint_probe(probe_sets, model, normalizer, n_frames,
                    step, writer, device,
                    output_dir, use_amp, sampling_cfg, guidance,
                    frame_dims, global_configs, fidelity_evaluator,
                    dac_model, prefix, n_plot,
                    metrics_seed=None):
    from probe_conditions import plot_condition_comparison

    names = [c for c in (frame_dims or {}) if c in (probe_sets or {})]
    gnames = [c for c in (global_configs or {}) if c in (probe_sets or {})]
    if not names and not gnames:
        return
    missing = [c for c in list(frame_dims or {}) + list(global_configs or {})
               if c not in (probe_sets or {})]
    if missing:
        print(f"    [probe] WARNING: no bank for {missing}; those conditions "
              f"go in NULL, so this probe is a partial subset")

    n_probe = min(len(probe_sets[c]) for c in names + gnames)
    n_plot = max(0, min(int(n_plot), n_probe))
    if n_plot == 0:
        return

    targets = {c: [np.asarray(t, dtype=np.float32)
                   for t in probe_sets[c].targets] for c in names}
    gtargets = {c: [np.asarray(t, dtype=np.float32).reshape(-1)
                    for t in probe_sets[c].targets] for c in gnames}

    def _frame_cond(idxs):
        fc = make_null_frame_conditions(len(idxs), n_frames, frame_dims or {},
                                        device)
        for c in names:
            fc[c] = torch.from_numpy(
                np.stack([targets[c][i] for i in idxs])).to(device).float()
        return fc

    def _global_cond(idxs):
        gc = make_null_global_conditions(len(idxs), global_configs or {}, device)
        for c in gnames:
            gc[c] = torch.from_numpy(
                np.stack([gtargets[c][i] for i in idxs])).to(device).float()
        return gc

    _probe_ctx_warned = []

    def _text_ctx(idxs):
        if not getattr(model, "text_cross_layers", None):
            return None
        ps = (probe_sets or {}).get("text")
        if ps is None or "text" not in gnames:
            return None
        items = [ps.context(i) for i in idxs]
        if any(c is None for c in items):
            if not _probe_ctx_warned:
                _probe_ctx_warned.append(True)
                print("    [probe] WARNING: the text bank carries no token "
                      "sequences, so the cross-attention sees only its null "
                      "token on every probe. Delete the probe_text cache and "
                      "let it rebuild.")
            return None
        return {"tokens": torch.cat([c["tokens"] for c in items], dim=0),
                "mask":   torch.cat([c["mask"]   for c in items], dim=0)}

    gen_rng = None
    if metrics_seed is not None:
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(int(metrics_seed))

    spf = max(1, int(sampling_cfg.get("metrics_samples_per_forward", 1) or 1))
    cond_lat = []
    if guidance > 1.0:
        for s in range(0, n_plot, spf):
            grp = list(range(s, min(s + spf, n_plot)))
            gc, _gu = euler_sample_cfg_paired(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                t_schedule=t_schedule_of(sampling_cfg),
                use_amp=use_amp,
                frame_cond=_frame_cond(grp), global_cond=_global_cond(grp),
                guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=_text_ctx(grp),
            )
            cond_lat.extend(gc)
    else:
        for i in range(n_plot):
            cond_lat.append(euler_sample_cfg(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                t_schedule=t_schedule_of(sampling_cfg),
                use_amp=use_amp,
                frame_cond=_frame_cond([i]), global_cond=_global_cond([i]),
                guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=_text_ctx([i]),
            ))

    fidelity_evaluator.reset()
    fidelity_evaluator.keep_contours_for(range(n_plot))
    wavs = []
    for i, lat in enumerate(cond_lat):
        wav = decode_frames_to_wav(lat, normalizer, dac_model)
        if names:
            fidelity_evaluator.add_sample(
                wav.numpy(), lc.active().sample_rate, n_frames,
                {c: targets[c][i] for c in names}, sample_id=i)
        wavs.append(wav)
    ps_cond = fidelity_evaluator.per_sample() if names else {}
    cont_cond = ({c: fidelity_evaluator.contours(c) for c in names}
                 if names else {})

    for i in range(n_plot):
        _suffix = (f" [{probe_sets['text'].text(i)[:48]}]"
                   if "text" in gnames else "")
        _blk = f"probe_{i:02d}{_suffix}"

        for c in names:
            _mkeys = sorted(k for k in ps_cond if k.startswith(f"{c}/"))
            corr_map = ps_cond.get(_mkeys[0], {}) if _mkeys else {}
            _mname = _mkeys[0].partition("/")[2] if _mkeys else None
            if i in cont_cond.get(c, {}):
                tgt, gen = cont_cond[c][i]
                writer.add_image(
                    f"{_blk}/{c}_target_vs_gen",
                    plot_condition_comparison(
                        c, tgt, gen, kind="probe",
                        label=f"'{probe_sets[c].names[i]}'", step=step,
                        prefix=prefix, guidance=guidance,
                        score=corr_map.get(i), score_name=_mname),
                    global_step=step)

        for c in gnames:
            if c != "image":
                continue
            try:
                _arr = np.asarray(probe_sets[c].image(i)).transpose(2, 0, 1)
                _tag = f"{_blk}/image_condition"
                if fixed_card_pending(writer, _tag):
                    writer.add_image(_tag, _arr, global_step=0)
            except Exception as _e:
                print(f"    [probe] image card {i:02d} unavailable: "
                      f"{type(_e).__name__}: {_e}")

        tags = audio_panel_tags("probe", i, names, suffix=_suffix,
                                conditioned=bool(names or gnames))

        for c in names:
            if not fixed_card_pending(writer, tags["conditions"][c]):
                continue
            writer.add_audio(tags["conditions"][c],
                             norm_wav(probe_sets[c].wav(i)),
                             global_step=0, sample_rate=probe_sets[c].sr)
        writer.add_audio(tags["generation"], norm_wav(wavs[i]),
                         global_step=step, sample_rate=lc.active().sample_rate)

        arrays = {c: targets[c][i] for c in names}
        arrays.update({c: gtargets[c][i] for c in gnames})
        text_lines = ([probe_sets["text"].text(i)] if "text" in gnames
                      else None)
        _ctx = _text_ctx([i])
        if _ctx is not None:
            arrays["text_tokens"] = _ctx["tokens"][0].cpu().numpy()
            arrays["text_mask"] = _ctx["mask"][0].cpu().numpy()
        elif text_lines and getattr(model, "text_cross_layers", None):
            text_lines.append("Cross-attention: the learned null token (this "
                              "probe bank has no token sequences).")
        info = {"panel": f"probe_{i:02d}", "tensorboard_block": _blk,
                "stimuli": {c: probe_sets[c].text(i) for c in names + gnames},
                "frame_conditions": names, "global_conditions": gnames,
                "null_conditions": missing, "guidance": guidance,
                "noise": {"metrics_seed": metrics_seed, "draw": i}}
        pdir = _panel_dir(output_dir, "probe", i)
        write_panel_conditions(
            pdir, info, arrays,
            {c: (probe_sets[c].wav(i), probe_sets[c].sr) for c in names},
            text_lines=text_lines,
            image_file=(probe_sets["image"].image_path(i) if "image" in gnames
                        else None))
        sf.write(os.path.join(pdir, _generation_name(step, prefix)),
                 wavs[i].numpy(), lc.active().sample_rate)


def load_text_label_vocab(latent_root):
    try:
        d = Path(latent_root).parent / "global_conditions"
        js, npy = d / "text_vocab.json", d / "text_vocab.npy"
        if not (js.exists() and npy.exists()):
            return None, None
        phrases = json.loads(js.read_text(encoding="utf-8"))["phrases"]
        return list(phrases), np.load(str(npy)).astype(np.float32)
    except Exception as e:
        print(f"[metrics] label vocabulary unreadable ({type(e).__name__}: {e})")
        return None, None


def load_text_captions(latent_root):
    try:
        p = Path(latent_root).parent / "global_conditions" / "text_labels.jsonl"
        if not p.exists():
            return {}
        out = {}
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("chunk"):
                    out[r["chunk"]] = r.get("caption") or r.get("class") or ""
        return out
    except Exception as e:
        print(f"[metrics] text_labels.jsonl unreadable "
              f"({type(e).__name__}: {e})")
        return {}


def load_caption_conditions(latent_root):
    t = load_caption_table(latent_root)
    if not t or "emb" not in t:
        return {}, None
    return t.get("ids", {}), t["emb"]


def _chunk_key_of(val_dataset, cond_path):
    try:
        if cond_path is None:
            return None
        root = Path(val_dataset.latent_root).parent / "conditions"
        return Path(cond_path).relative_to(root).with_suffix("").as_posix()
    except Exception:
        return None


def describe_validation_sample(val_dataset, ds_idx, captions=None,
                               phrases=None, vocab_emb=None, k=1):
    try:
        if not (0 <= ds_idx < len(val_dataset.samples)):
            return ""
        npy_path, cond_path, _start, _lab, class_name = \
            val_dataset.samples[ds_idx]
        if captions:
            key = _chunk_key_of(val_dataset, cond_path)
            if key is not None and key in captions:
                return captions[key]
        if not phrases or vocab_emb is None or cond_path is None:
            return str(class_name)
        with np.load(str(cond_path)) as z:
            if "text" not in z:
                return str(class_name)
            vec = z["text"]
        from conditions import nearest_phrases
        near = nearest_phrases(vec, vocab_emb, phrases, k=k)
        if not near:
            return str(class_name)
        return (f"{class_name} · "
                + ", ".join(f"\"{p}\" ({c:+.2f})" for p, c in near))
    except Exception:
        return ""


def panel_suffixes(dataset, ds_indices, global_configs):
    if "text" not in (global_configs or {}) or dataset is None:
        return {}
    key = (id(dataset), tuple(int(i) for i in ds_indices))
    cache = getattr(panel_suffixes, "_cache", None)
    if cache is None:
        cache = panel_suffixes._cache = {}
    if key in cache:
        return cache[key]

    captions = load_text_captions(dataset.latent_root)
    phrases, vocab = ((None, None) if captions
                      else load_text_label_vocab(dataset.latent_root))
    out = {}
    for k, idx in enumerate(ds_indices):
        desc = describe_validation_sample(dataset, int(idx), captions,
                                          phrases, vocab)
        out[k] = f" [{desc}]" if desc else ""
    cache[key] = out
    return out


def _panel_dir(output_dir, family, k):
    d = os.path.join(output_dir, family, f"{family}_{int(k):02d}")
    os.makedirs(d, exist_ok=True)
    return d


def _generation_name(step, prefix, label=None):
    return (f"step_{int(step):07d}_{prefix}"
            + (f"_{label}" if label else "") + ".wav")


def write_panel_conditions(pdir, info, arrays, sonified, real_wav=None,
                           text_lines=None, image_file=None):
    if arrays:
        np.savez(os.path.join(pdir, "conditions.npz"), **arrays)
    for name, (wav, sr) in sonified.items():
        sf.write(os.path.join(pdir, f"cond_{name}.wav"),
                 np.asarray(wav, dtype=np.float32), int(sr))
    if real_wav is not None:
        sf.write(os.path.join(pdir, "real.wav"), real_wav, lc.active().sample_rate)
    if text_lines:
        with open(os.path.join(pdir, "text.txt"), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write("\n".join(text_lines) + "\n")
    if image_file is not None:
        import shutil
        try:
            shutil.copyfile(image_file, os.path.join(
                pdir, "image" + Path(image_file).suffix.lower()))
        except OSError as e:
            print(f"    [panels] {os.path.basename(pdir)}: picture not copied "
                  f"({type(e).__name__}: {e})")
    with open(os.path.join(pdir, "info.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def test_text_given(dataset, idx, vec, wants_ctx):
    table = (getattr(dataset, "_captions", None)
             or load_caption_table(dataset.latent_root) or {})
    key = _chunk_key_of(dataset, dataset.samples[idx][1])
    cid = (table.get("ids") or {}).get(key) if key is not None else None
    strings = table.get("captions_text") or table.get("captions") or []
    sentence = (strings[cid] if cid is not None and cid < len(strings)
                else None)
    emb = table.get("emb")
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if (cid is not None and emb is not None and cid < len(emb)
            and np.array_equal(v, np.asarray(emb[cid]).reshape(-1))):
        slot = "caption"
        lines = [sentence if sentence is not None
                 else "[caption embedding, its sentence is not stored in "
                      "the dataset]"]
    elif not v.any():
        slot = "null"
        lines = ["[no sentence] The text slot received the null vector "
                 "(zeros): this chunk has no text vector."]
    else:
        slot = "audio_embedding"
        lines = ["[no sentence] The text slot received the CLAP audio "
                 "embedding of this chunk, not a text: the vector is 'text' "
                 "in conditions.npz."]
    info = {"slot": slot, "sentence": sentence if slot == "caption" else None}
    if wants_ctx:
        ctx = (sentence if table.get("tok") is not None and cid is not None
               else None)
        info["cross_attention_tokens"] = ctx
        if ctx is None:
            lines.append("Cross-attention: the learned null token (no caption "
                         "tokens for this chunk).")
        elif slot != "caption":
            lines.append(f"Cross-attention tokens: {ctx}")
    return lines, info


@torch.no_grad()
def run_test_panels(test_dataset, ds_indices, model, normalizer, n_frames,
                    step, writer, device, output_dir, use_amp, sampling_cfg,
                    guidance, frame_dims, global_configs, fidelity_evaluator,
                    dac_model, prefix, metrics_seed=None, text_vec_for=None,
                    subset_specs=(), image_root=None):
    from probe_conditions import plot_condition_comparison
    from condition_metrics import sonify_condition

    n = len(ds_indices)
    if n == 0:
        return
    names = list(frame_dims or {})
    gnames = list(global_configs or {})
    sfx = panel_suffixes(test_dataset, ds_indices, global_configs)
    wants_ctx = bool(getattr(model, "text_cross_layers", None))

    items = []
    for idx in ds_indices:
        frames_real, fcond, _lab, text_emb, image_emb, text_ctx = \
            test_dataset[idx]
        if text_vec_for is not None:
            text_emb = text_vec_for(test_dataset, idx, text_emb)
        items.append((frames_real, fcond, text_emb, image_emb, text_ctx))

    def _fc(grp, subset=None):
        return {k: torch.stack([items[j][1][k] for j in grp]).to(device).float()
                for k in names if subset is None or k in subset}

    def _gc(grp):
        gc = {}
        if "text" in gnames:
            gc["text"] = torch.stack([items[j][2] for j in grp]).to(device)
        if "image" in gnames:
            gc["image"] = torch.stack([items[j][3] for j in grp]).to(device)
        return gc

    def _ctx(grp):
        if not wants_ctx:
            return None
        return {"tokens": torch.stack([items[j][4]["tokens"] for j in grp]),
                "mask":   torch.stack([items[j][4]["mask"] for j in grp])}

    spf = int(sampling_cfg.get("metrics_samples_per_forward", 1))

    def _generate(subset=None):
        gen_rng = None
        if metrics_seed is not None:
            gen_rng = torch.Generator(device=device)
            gen_rng.manual_seed(int(metrics_seed))
        out = []
        if spf >= 1 and guidance > 1.0:
            for s in range(0, n, spf):
                grp = list(range(s, min(s + spf, n)))
                gen_c, _gen_u = euler_sample_cfg_paired(
                    model, n_frames, device,
                    steps=sampling_cfg.euler_steps,
                    t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                    t_schedule=t_schedule_of(sampling_cfg),
                    use_amp=use_amp,
                    frame_cond=_fc(grp, subset), global_cond=_gc(grp),
                    guidance=guidance,
                    frame_dims=frame_dims, global_configs=global_configs,
                    gen_rng=gen_rng, text_ctx=_ctx(grp),
                )
                out.extend(gen_c)
        else:
            for j in range(n):
                out.append(euler_sample_cfg(
                    model, n_frames, device,
                    steps=sampling_cfg.euler_steps,
                    t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                    t_schedule=t_schedule_of(sampling_cfg),
                    use_amp=use_amp,
                    frame_cond=_fc([j], subset), global_cond=_gc([j]),
                    guidance=guidance,
                    frame_dims=frame_dims, global_configs=global_configs,
                    gen_rng=gen_rng, text_ctx=_ctx([j]),
                ))
        return out

    lat = _generate()
    use_ev = bool(names) and fidelity_evaluator is not None \
        and fidelity_evaluator.active
    if use_ev:
        fidelity_evaluator.reset()
        fidelity_evaluator.keep_contours_for(range(n))
    wavs = []
    for j, z in enumerate(lat):
        wav = decode_frames_to_wav(z, normalizer, dac_model)
        if use_ev:
            fidelity_evaluator.add_sample(
                wav.numpy(), lc.active().sample_rate, n_frames,
                {k: v.cpu().numpy() for k, v in items[j][1].items()},
                sample_id=j)
        wavs.append(wav)
    ps = fidelity_evaluator.per_sample() if use_ev else {}
    cont = fidelity_evaluator.contours() if use_ev else {}

    ladder = {}
    _full = tuple(names)
    for _lab, _names in (subset_specs or ()):
        if tuple(_names) == _full:
            continue
        ladder[_lab] = [decode_frames_to_wav(z, normalizer, dac_model)
                        for z in _generate(subset=set(_names))]

    img_failed, img_shown = None, 0
    for k in range(n):
        blk = f"test_{k:02d}{sfx.get(k, '')}"
        img_file = None

        for c in sorted(names):
            if k not in cont.get(c, {}):
                continue
            mk = sorted(key for key in ps if key.startswith(f"{c}/"))
            score_map = ps.get(mk[0], {}) if mk else {}
            tgt, gen = cont[c][k]
            writer.add_image(
                f"{blk}/{c}_target_vs_gen",
                plot_condition_comparison(
                    c, tgt, gen, kind="test",
                    label=f"test sample #{ds_indices[k]}",
                    step=step, prefix=prefix, guidance=guidance,
                    score=score_map.get(k),
                    score_name=mk[0].partition("/")[2] if mk else None),
                global_step=step)

        if "image" in gnames and image_root:
            ref = test_dataset.image_file_for(ds_indices[k])
            if ref is not None:
                try:
                    from PIL import Image as _PILImage
                    _ipath = Path(image_root) / ref[0] / ref[1]
                    im = np.asarray(_PILImage.open(_ipath).convert("RGB"))
                    img_file = _ipath
                    tag = f"{blk}/image_condition"
                    if fixed_card_pending(writer, tag):
                        writer.add_image(tag, im.transpose(2, 0, 1),
                                         global_step=0)
                    img_shown += 1
                except Exception as e:
                    img_failed = f"{ref[0]}/{ref[1]}: {type(e).__name__}: {e}"

        tags = audio_panel_tags("test", k, names, suffix=sfx.get(k, ""),
                                conditioned=True)
        sons = {}
        for cname, carr in sorted(items[k][1].items()):
            if cname not in tags["conditions"]:
                continue
            son = sonify_condition(cname, carr.cpu().numpy(), lc.active().sample_rate)
            if son is None:
                continue
            sons[cname] = (son, lc.active().sample_rate)
            if fixed_card_pending(writer, tags["conditions"][cname]):
                writer.add_audio(tags["conditions"][cname], norm_wav(son),
                                 global_step=0, sample_rate=lc.active().sample_rate)
        real = decode_frames_to_wav(items[k][0], normalizer, dac_model)
        if fixed_card_pending(writer, tags["real"]):
            writer.add_audio(tags["real"], norm_wav(real),
                             global_step=0, sample_rate=lc.active().sample_rate)
        writer.add_audio(tags["generation"], norm_wav(wavs[k]),
                         global_step=step, sample_rate=lc.active().sample_rate)
        for _lab, _w in ladder.items():
            writer.add_audio(
                subset_generation_tag("test", k, _lab, suffix=sfx.get(k, ""),
                                      n_conditions=len(names), has_real=True),
                norm_wav(_w[k]), global_step=step, sample_rate=lc.active().sample_rate)

        idx = int(ds_indices[k])
        npy_path, _cp, start, _lab_i, class_name = test_dataset.samples[idx]
        try:
            latent = Path(npy_path).relative_to(
                test_dataset.latent_root).as_posix()
        except ValueError:
            latent = Path(npy_path).name
        arrays = {c: items[k][1][c].cpu().numpy() for c in names}
        text_lines = None
        info = {"panel": f"test_{k:02d}", "tensorboard_block": blk,
                "split": "test", "dataset_index": idx, "latent": latent,
                "start_frame": int(start), "class": class_name,
                "frame_conditions": names, "global_conditions": gnames}
        if "text" in gnames:
            arrays["text"] = items[k][2].cpu().numpy()
            text_lines, info["text"] = test_text_given(
                test_dataset, idx, items[k][2], wants_ctx)
        if "image" in gnames:
            arrays["image"] = items[k][3].cpu().numpy()
            _iref = test_dataset.image_file_for(idx)
            info["image"] = f"{_iref[0]}/{_iref[1]}" if _iref else None
        if wants_ctx:
            arrays["text_tokens"] = items[k][4]["tokens"].cpu().numpy()
            arrays["text_mask"] = items[k][4]["mask"].cpu().numpy()
        info.update({
            "guidance": guidance,
            "noise": {"metrics_seed": metrics_seed, "draw": k},
            "subsets": {lab: list(nm) for lab, nm in (subset_specs or ())
                        if lab in ladder}})
        pdir = _panel_dir(output_dir, "test", k)
        write_panel_conditions(pdir, info, arrays, sons, real_wav=real.numpy(),
                               text_lines=text_lines, image_file=img_file)
        sf.write(os.path.join(pdir, _generation_name(step, prefix)),
                 wavs[k].numpy(), lc.active().sample_rate)
        for _lab, _w in ladder.items():
            sf.write(os.path.join(pdir, _generation_name(step, prefix, _lab)),
                     _w[k].numpy(), lc.active().sample_rate)
    if img_failed is not None and img_shown == 0:
        print(f"    [test panels] image cards unavailable ({img_failed}); "
              f"is paths.image_root still pointing at the right folder?")


def caption_text_vec_fn(latent_root, sampling_cfg, global_configs):
    cap_ids, cap_emb = ({}, None)
    if (sampling_cfg is not None
            and bool(sampling_cfg.get("validation_text_from_caption", False))
            and "text" in (global_configs or {})):
        cap_ids, cap_emb = load_caption_conditions(latent_root)
        if cap_emb is None:
            print("    [metrics] validation_text_from_caption is on but this "
                  "dataset carries no caption embeddings -- re-run the "
                  "preprocessing with --global text. Falling back to the "
                  "chunk's own CLAP vector.")
        else:
            print(f"    [metrics] text conditioned on the DESCRIPTION "
                  f"({cap_emb.shape[0]} distinct caption(s))")

    def text_vec_for(dataset, idx, text_emb):
        if cap_emb is None:
            return text_emb
        try:
            key = _chunk_key_of(dataset, dataset.samples[idx][1])
            cid = cap_ids.get(key)
            if cid is None:
                return text_emb
            return torch.from_numpy(cap_emb[cid].copy())
        except Exception:
            return text_emb
    return text_vec_for


@torch.no_grad()
def log_listening_panels(model, normalizer, step, writer, device, output_dir,
                         sampling_cfg, conditioning_cfg, use_amp, frame_dims,
                         global_configs, fidelity_evaluator, test_dataset,
                         probe_sets, n_frames, prefix="EMA", metrics_seed=None,
                         image_root=None, text_vec_for=None):
    guidance = float(conditioning_cfg.guidance_scale)
    any_cond_active = bool(frame_dims) or bool(global_configs)
    dac_model = get_dac()
    subset_specs = []
    if fidelity_evaluator is not None and fidelity_evaluator.active:
        subset_specs = resolve_influence_subsets(
            sampling_cfg.get("influence_subsets", None), list(frame_dims or {}))

    if any_cond_active and test_dataset is not None:
        _t_idx = test_panel_indices(len(test_dataset),
                                    panel_count(sampling_cfg, "test"))
        if _t_idx:
            if text_vec_for is None:
                text_vec_for = caption_text_vec_fn(
                    test_dataset.latent_root, sampling_cfg, global_configs)
            print(f"    test panels: {len(_t_idx)} samples of the test split...")
            try:
                run_test_panels(
                    test_dataset, _t_idx, model=model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer, device=device,
                    output_dir=output_dir, use_amp=use_amp,
                    sampling_cfg=sampling_cfg, guidance=guidance,
                    frame_dims=frame_dims, global_configs=global_configs,
                    fidelity_evaluator=fidelity_evaluator, dac_model=dac_model,
                    prefix=prefix, metrics_seed=metrics_seed,
                    text_vec_for=text_vec_for, subset_specs=subset_specs,
                    image_root=image_root)
            except Exception as _e:
                print(f"    [test panels] SKIPPED at step {step}: "
                      f"{type(_e).__name__}: {_e}")
            if device == "cuda":
                torch.cuda.empty_cache()

    if probe_sets:
        _pnames = [c for c in list(frame_dims or {}) + list(global_configs or {})
                   if c in probe_sets]
        _npanel = min(panel_count(sampling_cfg, "probe"),
                      min([len(probe_sets[c]) for c in _pnames], default=0))
        if _npanel:
            print(f"    joint probe: {_npanel} panels over {_pnames}...")
            try:
                run_joint_probe(
                    probe_sets,
                    model=model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer, device=device,
                    output_dir=output_dir, use_amp=use_amp,
                    sampling_cfg=sampling_cfg,
                    guidance=guidance, frame_dims=frame_dims,
                    global_configs=global_configs,
                    fidelity_evaluator=fidelity_evaluator, dac_model=dac_model,
                    prefix=prefix, n_plot=_npanel,
                    metrics_seed=metrics_seed,
                )
            except Exception as _e:
                print(f"    [probe] SKIPPED at step {step}: "
                      f"{type(_e).__name__}: {_e}")
            if device == "cuda":
                torch.cuda.empty_cache()

    generate_and_log_uncond_cards(
        model=model, normalizer=normalizer, n_frames=n_frames, step=step,
        writer=writer, device=device, output_dir=output_dir,
        n_cards=int(sampling_cfg.get("n_audio_samples", 4) or 0),
        sampling_cfg=sampling_cfg, use_amp=use_amp,
        frame_dims=frame_dims, global_configs=global_configs,
        metrics_seed=metrics_seed, prefix=prefix)

    if device == "cuda":
        torch.cuda.empty_cache()


MIR_CURVE_METRICS = {
    "f0":     ("raw_pitch_accuracy", "raw_chroma_accuracy", "overall_accuracy"),
    "chroma": ("chroma_precision", "chroma_recall", "chroma_accuracy"),
    "chord":  ("chroma_precision", "chroma_recall", "chroma_accuracy"),
    "rhythm": ("beat_f_measure", "beat_cmlt", "beat_amlt"),
    "midi":   ("note_f1", "drum_f1", "note_precision"),
}


@torch.no_grad()
def evaluate_and_log_metrics(
    model, normalizer, val_dataset, step, writer, device, output_dir,
    fd_dac_ref_stats, n_samples,
    sampling_cfg, conditioning_cfg, use_amp,
    frame_dims, global_configs,
    fidelity_evaluator=None, global_embedders=None, compute_uncond=True,
    prefix="EMA", metrics_seed=None, metrics_enabled=COND_METRICS,
    fad_embedder=None, fad_ref_stats=None, n_fad=0, fad_device="cuda",
    probe_sets=None, image_root=None, test_dataset=None,
    tag_prefix="Validation", split_name="validation", report=None,
    dist_info=None, mir_curves=False,
):
    guidance = float(conditioning_cfg.guidance_scale)
    n_frames = val_dataset.n_frames
    total = len(val_dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()
    _di = dist_info if (dist_info is not None and dist_info.enabled) else None
    is_main = _di is None or _di.is_main

    def _mine(j):
        return _di is None or (j % _di.world) == _di.rank

    frame_active = (fidelity_evaluator is not None and fidelity_evaluator.active)
    _text_vec_for = caption_text_vec_fn(val_dataset.latent_root, sampling_cfg,
                                        global_configs)

    gsim_names = sorted(c for c in (global_configs or {})
                        if (global_embedders or {}).get(c) is not None)
    scoring_active = frame_active or bool(gsim_names)
    any_cond_active = bool(frame_dims) or bool(global_configs)

    subset_specs = []
    if scoring_active and frame_active:
        subset_specs = resolve_influence_subsets(
            sampling_cfg.get("influence_subsets", None), list(frame_dims or {}))

    ref_frames = (fd_dac_ref_stats["n_total"]
                  if fd_dac_ref_stats is not None else "n/a")
    print(f"\n  Compute metrics @ step {step}: {n_samples} generations "
          f"(cond guidance={guidance}"
          f"{', + uncond' if compute_uncond else ''}) "
          f"vs reference ({ref_frames} frames)...")

    _wants_ctx = bool(getattr(model, "text_cross_layers", None))

    def _stack_ctx(items):
        if not _wants_ctx or not items:
            return None
        return {"tokens": torch.stack([c["tokens"] for c in items]),
                "mask":   torch.stack([c["mask"]   for c in items])}

    def _generate(conditioned, subset=None):
        lat_list = []
        targets = []
        global_targets = []
        gen_rng = None
        if metrics_seed is not None:
            gen_rng = torch.Generator(device=device)
            gen_rng.manual_seed(int(metrics_seed))
        for j, idx in enumerate(indices):
            if not _mine(j):
                if gen_rng is not None:
                    torch.randn(1, n_frames, lc.active().latent_dim, device=device,
                                generator=gen_rng)
                lat_list.append(None)
                if conditioned:
                    targets.append(None)
                    global_targets.append(None)
                continue
            (_frames_real, frame_cond_real, _lab, text_emb, image_emb,
             text_ctx) = val_dataset[idx]
            text_emb = _text_vec_for(val_dataset, idx, text_emb)
            if conditioned:
                fc = {k: v.unsqueeze(0).to(device).float()
                      for k, v in frame_cond_real.items()
                      if subset is None or k in subset}
                gc = {}
                if "text" in global_configs:
                    gc["text"] = text_emb.unsqueeze(0).to(device)
                if "image" in global_configs:
                    gc["image"] = image_emb.unsqueeze(0).to(device)
                g = guidance
                targets.append({k: v.cpu().numpy()
                                for k, v in frame_cond_real.items()})
                global_targets.append({
                    "text":  text_emb.cpu().numpy()  if "text"  in global_configs else None,
                    "image": image_emb.cpu().numpy() if "image" in global_configs else None,
                })
                tctx = _stack_ctx([text_ctx])
            else:
                fc, gc, g, tctx = None, None, 1.0, None
            gen = euler_sample_cfg(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                t_schedule=t_schedule_of(sampling_cfg),
                use_amp=use_amp,
                frame_cond=fc, global_cond=gc, guidance=g,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=tctx,
            )
            lat_list.append(gen)
        return lat_list, targets, global_targets

    def _generate_paired_dist(spf):
        n = len(indices)
        cond_list, unc_list = [None] * n, [None] * n
        targets, global_targets = [None] * n, [None] * n
        gen_rng = None
        if metrics_seed is not None:
            gen_rng = torch.Generator(device=device)
            gen_rng.manual_seed(int(metrics_seed))
        x0_of = {}
        for j in range(n):
            x = torch.randn(n_frames, lc.active().latent_dim, device=device, generator=gen_rng)
            if _mine(j):
                x0_of[j] = x
        own = [j for j in range(n) if _mine(j)]
        for start in range(0, len(own), spf):
            grp = own[start:start + spf]
            fcs, gcs, ctxs = [], [], []
            for j in grp:
                idx = indices[j]
                (_frames_real, frame_cond_real, _lab, text_emb, image_emb,
                 text_ctx) = val_dataset[idx]
                text_emb = _text_vec_for(val_dataset, idx, text_emb)
                fcs.append(frame_cond_real)
                gcs.append((text_emb, image_emb))
                ctxs.append(text_ctx)
                targets[j] = {k: v.cpu().numpy() for k, v in frame_cond_real.items()}
                global_targets[j] = {
                    "text":  text_emb.cpu().numpy()  if "text"  in global_configs else None,
                    "image": image_emb.cpu().numpy() if "image" in global_configs else None,
                }
            fc = {k: torch.stack([d[k] for d in fcs]).to(device).float()
                  for k in (fcs[0].keys() if fcs else [])}
            gc = {}
            if "text" in global_configs:
                gc["text"] = torch.stack([t for t, _ in gcs]).to(device)
            if "image" in global_configs:
                gc["image"] = torch.stack([i for _, i in gcs]).to(device)
            gen_c, gen_u = euler_sample_cfg_paired(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                t_schedule=t_schedule_of(sampling_cfg),
                use_amp=use_amp,
                frame_cond=fc, global_cond=gc, guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=_stack_ctx(ctxs),
                x0=torch.stack([x0_of.pop(j) for j in grp]),
            )
            for j, c, u in zip(grp, gen_c, gen_u):
                cond_list[j], unc_list[j] = c, u
        return cond_list, targets, global_targets, unc_list

    def _generate_paired(spf):
        if _di is not None:
            return _generate_paired_dist(spf)
        cond_list, unc_list = [], []
        targets, global_targets = [], []
        gen_rng = None
        if metrics_seed is not None:
            gen_rng = torch.Generator(device=device)
            gen_rng.manual_seed(int(metrics_seed))
        for start in range(0, len(indices), spf):
            group = indices[start:start + spf]
            fcs, gcs, ctxs = [], [], []
            for idx in group:
                (_frames_real, frame_cond_real, _lab, text_emb, image_emb,
                 text_ctx) = val_dataset[idx]
                text_emb = _text_vec_for(val_dataset, idx, text_emb)
                fcs.append(frame_cond_real)
                gcs.append((text_emb, image_emb))
                ctxs.append(text_ctx)
                targets.append({k: v.cpu().numpy() for k, v in frame_cond_real.items()})
                global_targets.append({
                    "text":  text_emb.cpu().numpy()  if "text"  in global_configs else None,
                    "image": image_emb.cpu().numpy() if "image" in global_configs else None,
                })
            fc = {k: torch.stack([d[k] for d in fcs]).to(device).float()
                  for k in (fcs[0].keys() if fcs else [])}
            gc = {}
            if "text" in global_configs:
                gc["text"] = torch.stack([t for t, _ in gcs]).to(device)
            if "image" in global_configs:
                gc["image"] = torch.stack([i for _, i in gcs]).to(device)
            gen_c, gen_u = euler_sample_cfg_paired(
                model, n_frames, device,
                steps=sampling_cfg.euler_steps,
                t_min=sampling_cfg.t_min, t_max=sampling_cfg.t_max,
                t_schedule=t_schedule_of(sampling_cfg),
                use_amp=use_amp,
                frame_cond=fc, global_cond=gc, guidance=guidance,
                frame_dims=frame_dims, global_configs=global_configs,
                gen_rng=gen_rng, text_ctx=_stack_ctx(ctxs),
            )
            cond_list.extend(gen_c)
            unc_list.extend(gen_u)
        return cond_list, targets, global_targets, unc_list

    has_ref = fd_dac_ref_stats is not None

    spf = int(sampling_cfg.get("metrics_samples_per_forward", 1))
    paired = (spf >= 1 and (guidance > 1.0) and any_cond_active)
    _unc_pre = None
    if paired:
        cond_lat, cond_targets, cond_globals, _unc_pre = _generate_paired(spf)
    else:
        cond_lat, cond_targets, cond_globals = _generate(conditioned=True)
        if compute_uncond and not any_cond_active:
            _unc_pre = cond_lat
    if _di is not None:
        _m = (dist_dac_metrics(cond_lat, fd_dac_ref_stats, metrics_enabled,
                               device, _di) if has_ref else {})
        fd_dac_cond = _m.get("fd_dac")
        kl_cond = {"kl_real_gen": _m.get("kl_real_gen"),
                   "kl_gen_real": _m.get("kl_gen_real")}
    else:
        cond_stack = torch.stack(cond_lat)
        if has_ref:
            _m = compute_dac_metrics(cond_stack, fd_dac_ref_stats,
                                     enabled=metrics_enabled, device=device)
            fd_dac_cond = _m["fd_dac"]
            kl_cond = {"kl_real_gen": _m["kl_real_gen"], "kl_gen_real": _m["kl_gen_real"]}
        else:
            fd_dac_cond = None
            kl_cond = {"kl_real_gen": None, "kl_gen_real": None}
        del cond_stack
    if device == "cuda":
        torch.cuda.empty_cache()

    fd_dac_uncond = None
    kl_uncond = {"kl_real_gen": None, "kl_gen_real": None}
    unc_lat = _unc_pre if _unc_pre is not None else []
    need_null = scoring_active and any_cond_active
    if not unc_lat and (compute_uncond or need_null):
        unc_lat = _generate(conditioned=False)[0]
    if compute_uncond and unc_lat and _di is not None:
        if has_ref:
            _mu = dist_dac_metrics(unc_lat, fd_dac_ref_stats, metrics_enabled,
                                   device, _di)
            fd_dac_uncond = _mu.get("fd_dac")
            kl_uncond = {"kl_real_gen": _mu.get("kl_real_gen"),
                         "kl_gen_real": _mu.get("kl_gen_real")}
        if device == "cuda":
            torch.cuda.empty_cache()
    elif compute_uncond and unc_lat:
        unc_stack = torch.stack(unc_lat)
        if has_ref:
            _mu = compute_dac_metrics(unc_stack, fd_dac_ref_stats,
                                      enabled=metrics_enabled, device=device)
            fd_dac_uncond = _mu["fd_dac"]
            kl_uncond = {"kl_real_gen": _mu["kl_real_gen"],
                         "kl_gen_real": _mu["kl_gen_real"]}
        del unc_stack
        if device == "cuda":
            torch.cuda.empty_cache()

    dac_model = get_dac()

    fad_on = fad_embedder is not None and fad_ref_stats is not None
    n_fad_use = min(int(n_fad or 0), len(cond_lat)) if fad_on else 0
    fad_pos = (set(torch.linspace(0, len(cond_lat) - 1, n_fad_use)
                   .round().long().tolist()) if n_fad_use > 0 else set())

    def _stream_audio(lat_list, measure, fad, desc):
        n_lat = len(lat_list)
        need = set(range(n_lat)) if measure else set()
        if fad:
            need |= {p for p in fad_pos if p < n_lat}
        need = {p for p in need if _mine(p)}
        if measure and frame_active:
            fidelity_evaluator.reset()
            fidelity_evaluator.keep_contours_for(())
        gsims = {c: {} for c in gsim_names} if measure else {}
        fst = {"sx": None, "sxx": None, "n": 0}
        for i in tqdm(sorted(need), desc=desc, leave=False,
                      disable=not is_main):
            wav = decode_frames_to_wav(lat_list[i], normalizer, dac_model)
            if measure:
                wn = wav.numpy()
                if frame_active:
                    fidelity_evaluator.add_sample(
                        wn, lc.active().sample_rate, n_frames, cond_targets[i],
                        sample_id=i)
                for c in gsim_names:
                    t = cond_globals[i].get(c) if i < len(cond_globals) else None
                    if t is None:
                        continue
                    try:
                        emb = global_embedders[c].embed(wn, lc.active().sample_rate)
                        gsims[c][i] = float(np.dot(emb, np.asarray(t).reshape(-1)))
                    except Exception as _e:
                        if not gsims[c]:
                            print(f"    [metrics] {c} similarity unavailable: "
                                  f"{type(_e).__name__}: {_e}")
            if fad and i in fad_pos:
                e = fad_embedder.embed(wav.view(1, 1, -1), lc.active().sample_rate).to(
                    device=fad_device, dtype=torch.float64)
                if fst["sx"] is None:
                    d = e.shape[-1]
                    fst["sx"] = torch.zeros(d, dtype=torch.float64, device=fad_device)
                    fst["sxx"] = torch.zeros(d, d, dtype=torch.float64,
                                             device=fad_device)
                fst["sx"] = fst["sx"] + e.sum(dim=0)
                fst["sxx"] = fst["sxx"] + e.T @ e
                fst["n"] = fst["n"] + e.shape[0]
                del e
        if _di is not None:
            if measure and frame_active:
                _st = dist_all_gather_object(
                    fidelity_evaluator.export_state(), _di)
                if is_main:
                    for _r, _s in enumerate(_st):
                        if _r != _di.rank:
                            fidelity_evaluator.merge_state(_s)
            if measure:
                for _r, _g in enumerate(dist_all_gather_object(gsims, _di)):
                    if is_main and _r != _di.rank:
                        for _c, _d in _g.items():
                            gsims.setdefault(_c, {}).update(_d)
            if fad:
                _mine_f = (None if fst["sx"] is None else
                           (fst["sx"].cpu(), fst["sxx"].cpu(), fst["n"]))
                _all_f = dist_all_gather_object(_mine_f, _di)
                if is_main:
                    _got = [f for f in _all_f if f is not None]
                    if _got:
                        fst = {"sx": sum(f[0] for f in _got).to(fad_device),
                               "sxx": sum(f[1] for f in _got).to(fad_device),
                               "n": sum(f[2] for f in _got)}
            if not is_main:
                return ({}, {}, {}, {"sx": None, "sxx": None, "n": 0})
        per = fidelity_evaluator.per_sample() if (measure and frame_active) else {}
        cov = fidelity_evaluator.coverage() if (measure and frame_active) else {}
        return per, cov, gsims, fst

    print("    decoding each generation once: "
          + (f"all {len(cond_lat)} measured for the table" if scoring_active
             else "nothing to measure")
          + (f", {len(fad_pos)} for FAD" if fad_pos else "")
          + "...")
    per_cond, cov_cond, gsim_cond, fad_c = _stream_audio(
        cond_lat, measure=scoring_active, fad=fad_on,
        desc="metrics: decode + measure")
    fad_u = None
    per_null, gsim_null, have_null = {}, {}, False
    if unc_lat:
        if not any_cond_active:
            fad_u = fad_c
        else:
            per_null, _, gsim_null, fad_u = _stream_audio(
                unc_lat, measure=need_null, fad=(fad_on and compute_uncond),
                desc=("metrics: decode + measure (no conditions)" if need_null
                      else "metrics: decode (no conditions)"))
            have_null = need_null

    subset_entries = []
    _full = tuple(frame_dims or {})
    for _lab, _names in subset_specs:
        if _names == _full:
            subset_entries.append((_lab, _names, per_cond, cov_cond, gsim_cond))
            continue
        print(f"    subset '{_lab}' [{'+'.join(_names)}]: "
              f"{n_samples} generations...")
        _slat = _generate(conditioned=True, subset=set(_names))[0]
        _sper, _scov, _sg, _ = _stream_audio(
            _slat, measure=True, fad=False,
            desc=f"metrics: subset {_lab}")
        subset_entries.append((_lab, _names, _sper, _scov, _sg))
        del _slat
        if device == "cuda":
            torch.cuda.empty_cache()

    fad_cond = fad_uncond = None
    if fad_c is not None and fad_c["n"] > 0:
        _mu, _sig, _ = compute_mu_sigma(fad_c["sx"], fad_c["sxx"], fad_c["n"])
        fad_cond = compute_fad(_mu, _sig, fad_ref_stats, device=fad_device)
        print(f"    FAD-VGGish cond: {len(fad_pos)} clips -> {fad_c['n']} "
              f"embedding vectors (128-D)")
        if compute_uncond and unc_lat:
            if not any_cond_active:
                fad_uncond = fad_cond
            elif fad_u is not None and fad_u["n"] > 0:
                _mu, _sig, _ = compute_mu_sigma(fad_u["sx"], fad_u["sxx"],
                                                fad_u["n"])
                fad_uncond = compute_fad(_mu, _sig, fad_ref_stats,
                                         device=fad_device)
        del _mu, _sig
        if device == "cuda":
            torch.cuda.empty_cache()

    del cond_lat, unc_lat
    if device == "cuda":
        torch.cuda.empty_cache()

    from condition_metrics import pair_influence, pair_scalar

    def _no_metric_free(d):
        return {k: v for k, v in (d or {}).items()
                if not k.endswith("/<no metric>")}

    influence, cov_paired = {}, {}
    if frame_active and scoring_active:
        influence, cov_paired = pair_influence(
            _no_metric_free(per_cond), _no_metric_free(per_null),
            coverage_cond=_no_metric_free(cov_cond), have_null=have_null)

    _gmetric = {"text": "clap_sim", "image": "clip_sim"}

    def _global_rows(gsims, into_inf, into_cov):
        for c in gsim_names:
            _gc = (gsims or {}).get(c, {})
            if not _gc:
                continue
            _cm, _nm, _dm, _npair = pair_scalar(
                _gc, gsim_null.get(c, {}), have_null=have_null)
            key = _gmetric.get(c, "sim")
            into_inf[c] = {key: {"cond": _cm, "null": _nm, "delta": _dm}}
            into_cov[f"{c}/{key}"] = {
                "valid": _npair,
                "attempted": len(_gc),
                "unpaired": (len(set(_gc) ^ set(gsim_null.get(c, {})))
                             if have_null else 0),
            }

    if scoring_active:
        _global_rows(gsim_cond, influence, cov_paired)

    subset_tables = []
    for _lab, _names, _sps, _scov, _sgsim in subset_entries:
        _sinf, _scovp = pair_influence(
            _no_metric_free(_sps), _no_metric_free(per_null),
            coverage_cond=_no_metric_free(_scov), have_null=have_null)
        _global_rows(_sgsim, _sinf, _scovp)
        subset_tables.append((_lab, _sinf, _scovp))

    if scoring_active:
        for c in (global_configs or {}):
            if c in influence or c in gsim_names:
                continue
            influence[c] = {_gmetric.get(c, "sim"): {
                "cond": None, "null": None, "delta": None,
                "note": ("no audio->CLIP embedder: pip install wav2clip"
                         if c == "image" else "no embedder for this condition"),
            }}

    def _row_label(name):
        return f"{name}_{split_name}"

    _inf_named = {_row_label(k): v for k, v in influence.items()}
    _cov_named = {}
    for _k, _v in (cov_paired or {}).items():
        _c, _, _m = _k.partition("/")
        _cov_named[f"{_row_label(_c)}/{_m}"] = _v

    if influence:
        from condition_metrics import (format_influence_panel,
                                       format_influence_matrix,
                                       format_influence_legend)
        if subset_tables:
            panel_md = format_influence_matrix(
                subset_tables, step=step, prefix=prefix,
                guidance=guidance, n_samples=int(n_samples),
                always_given=set(gsim_names),
            )
        else:
            panel_md = format_influence_panel(
                _inf_named, step=step, prefix=prefix,
                guidance=guidance,
                n_samples=int(n_samples),
                coverage=_cov_named,
            )
        writer.add_text(f"{tag_prefix}/Condition_influence", panel_md, step)
        _legend_tag = f"{tag_prefix}/Condition_influence_legend"
        if fixed_card_pending(writer, _legend_tag):
            writer.add_text(_legend_tag, format_influence_legend(), 0)

    _m = f"{tag_prefix}/Metrics"
    _fd = "dac" if lc.active().name == "dac_44khz" else "encodec"
    if any_cond_active:
        if fd_dac_cond is not None:
            writer.add_scalar(f"{_m}/Fd_{_fd}_cond", fd_dac_cond, step)
        if kl_cond["kl_real_gen"] is not None:
            writer.add_scalar(f"{_m}/Kl_cond/real_gen",
                              kl_cond["kl_real_gen"], step)
            writer.add_scalar(f"{_m}/Kl_cond/gen_real",
                              kl_cond["kl_gen_real"], step)
        if fd_dac_uncond is not None:
            writer.add_scalar(f"{_m}/Fd_{_fd}_uncond", fd_dac_uncond, step)
        if kl_uncond["kl_real_gen"] is not None:
            writer.add_scalar(f"{_m}/Kl_uncond/real_gen",
                              kl_uncond["kl_real_gen"], step)
            writer.add_scalar(f"{_m}/Kl_uncond/gen_real",
                              kl_uncond["kl_gen_real"], step)
        if fad_cond is not None:
            writer.add_scalar(f"{_m}/Fad_vggish_cond", fad_cond, step)
        if fad_uncond is not None:
            writer.add_scalar(f"{_m}/Fad_vggish_uncond", fad_uncond, step)
    else:
        if fd_dac_cond is not None:
            writer.add_scalar(f"{_m}/Fd_{_fd}", fd_dac_cond, step)
        if kl_cond["kl_real_gen"] is not None:
            writer.add_scalar(f"{_m}/Kl_real_gen",
                              kl_cond["kl_real_gen"], step)
            writer.add_scalar(f"{_m}/Kl_gen_real",
                              kl_cond["kl_gen_real"], step)
        if fad_cond is not None:
            writer.add_scalar(f"{_m}/Fad_vggish", fad_cond, step)

    if mir_curves and frame_active:
        for _c, _names in MIR_CURVE_METRICS.items():
            if fidelity_evaluator.families.get(_c) != "mir_influence_metrics":
                continue
            for _k in _names:
                _v = influence.get(_c, {}).get(_k, {}).get("cond")
                if _v is not None:
                    writer.add_scalar(f"{_m}/{_c.capitalize()}_{_k}_cond",
                                      _v, step)

    def _f(x):
        return f"{x:.4f}" if x is not None else "n/a"
    print(f"  [cond]   FD-{lc.active().label}: {_f(fd_dac_cond)} | "
          f"KL(real||gen): {_f(kl_cond['kl_real_gen'])} | "
          f"KL(gen||real): {_f(kl_cond['kl_gen_real'])}"
          + (f" | FAD-VGGish: {_f(fad_cond)}" if fad_cond is not None else ""))
    if compute_uncond:
        print(f"  [uncond] FD-{lc.active().label}: {_f(fd_dac_uncond)} | "
              f"KL(real||gen): {_f(kl_uncond['kl_real_gen'])} | "
              f"KL(gen||real): {_f(kl_uncond['kl_gen_real'])}"
              + (f" | FAD-VGGish: {_f(fad_uncond)}" if fad_uncond is not None else ""))
    if influence:
        print("  [influence] with-cond / null / delta (paired/measured)")
        for cname, metrics in _inf_named.items():
            for m, vals in metrics.items():
                cv = _cov_named.get(f"{cname}/{m}", {})
                d = vals.get("delta")
                print(f"    {cname}/{m}: {_f(vals.get('cond'))} / "
                      f"{_f(vals.get('null'))} / "
                      + (f"{d:+.4f}" if d is not None else "n/a")
                      + f"  ({cv.get('valid', 0)}/{cv.get('attempted', 0)})")

    if report is not None:
        report.update({
            "n_generations": int(n_samples),
            "fd_dac_cond": fd_dac_cond,
            "kl_cond_real_gen": kl_cond["kl_real_gen"],
            "kl_cond_gen_real": kl_cond["kl_gen_real"],
            "fd_dac_uncond": fd_dac_uncond,
            "kl_uncond_real_gen": kl_uncond["kl_real_gen"],
            "kl_uncond_gen_real": kl_uncond["kl_gen_real"],
            "fad_vggish_cond": fad_cond,
            "fad_vggish_uncond": fad_uncond,
            "influence": _inf_named,
            "influence_coverage": _cov_named,
        })

    if not is_main:
        return fd_dac_cond, kl_cond["kl_real_gen"], kl_cond["kl_gen_real"]
    log_listening_panels(
        model=model, normalizer=normalizer, step=step, writer=writer,
        device=device, output_dir=output_dir, sampling_cfg=sampling_cfg,
        conditioning_cfg=conditioning_cfg, use_amp=use_amp,
        frame_dims=frame_dims, global_configs=global_configs,
        fidelity_evaluator=fidelity_evaluator, test_dataset=test_dataset,
        probe_sets=probe_sets, n_frames=n_frames, prefix=prefix,
        metrics_seed=metrics_seed, image_root=image_root,
        text_vec_for=_text_vec_for)

    return fd_dac_cond, kl_cond["kl_real_gen"], kl_cond["kl_gen_real"]


def read_influence_settings(metrics_cfg):
    from condition_metrics import INFLUENCE_FAMILIES
    influence_family = str(
        metrics_cfg.get("influence_family", "influence_metrics")
        if metrics_cfg is not None else "influence_metrics")
    if influence_family not in INFLUENCE_FAMILIES:
        raise SystemExit(
            f"[metrics] metrics.influence_family='{influence_family}' is not a "
            f"valid choice. Use 'influence_metrics' (ours) or "
            f"'mir_influence_metrics' (mir_eval, for the conditions it covers).")
    mir_threshold = float(
        metrics_cfg.get("mir_threshold", 0.5)
        if metrics_cfg is not None else 0.5)
    if not 0.0 < mir_threshold <= 1.0:
        raise SystemExit(
            f"[metrics] metrics.mir_threshold={mir_threshold} is not in (0, 1]: "
            f"it is a fraction of the frame's loudest chroma class, and a "
            f"crema probability for chord.")
    return influence_family, mir_threshold


def build_condition_scorers(cfg, frame_dims, global_configs, registry,
                            influence_family, mir_threshold):
    from condition_metrics import ConditionFidelityEvaluator
    _fid_device = str(cfg.metrics.get("fidelity_device", "cuda"))
    if _fid_device.startswith("cuda") and not torch.cuda.is_available():
        print("[metrics] fidelity_device='cuda' but no GPU is available "
              "-> falling back to CPU for the re-extraction.")
        _fid_device = "cpu"
    fidelity_evaluator = ConditionFidelityEvaluator(
        enabled_frame=list(frame_dims.keys()),
        device=_fid_device,
        registry=registry,
        family=influence_family,
        mir_threshold=mir_threshold,
    )
    for _name, _extractor in fidelity_evaluator.extractors.items():
        _actual_device = getattr(
            _extractor, "device", getattr(_extractor, "_device", "cpu"))
        _batch = getattr(_extractor, "batch_size", None)
        _batch_msg = f" | batch_size={_batch}" if _batch is not None else ""
        _thr = getattr(fidelity_evaluator.fidelity_fns[_name], "keywords",
                       {}).get("threshold")
        _thr_msg = f" (ON at >= {_thr})" if _thr is not None else ""
        print(f"[metrics] extractor={_name} | device={_actual_device}{_batch_msg}"
              f" | metrics={fidelity_evaluator.families[_name]}{_thr_msg}")
    global_embedders = {}
    if "text" in global_configs:
        from conditions import ClapAudioEmbedder
        clap_model_name = CONDITION_CONFIG["global"]["text"]["kwargs"].get(
            "model_name", "laion/clap-htsat-unfused")
        global_embedders["text"] = ClapAudioEmbedder(model_name=clap_model_name,
                                                     device=_fid_device)
        print(f"Text-influence (CLAP audio) enabled: {clap_model_name}")
    if "image" in global_configs:
        try:
            from conditions import Wav2ClipAudioEmbedder
            _w2c = Wav2ClipAudioEmbedder(device=_fid_device)
            _w2c._load()
            global_embedders["image"] = _w2c
            print("Image-influence (Wav2CLIP audio->CLIP space) enabled")
        except Exception as _e:
            print(f"Image-influence DISABLED: {type(_e).__name__}: {_e}")
    return fidelity_evaluator, global_embedders


def build_probe_sets(cfg, registry, fidelity_evaluator, dataset,
                     frame_dims, global_configs):
    probe_sets = {}
    _n_probes = panel_count(cfg.sampling, "probe")
    _rng_guard = (
        torch.get_rng_state(),
        np.random.get_state(),
        random.getstate(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    _probe_names = list(frame_dims) + list(global_configs)
    for _cond in _probe_names:
        if _n_probes <= 0:
            continue
        try:
            from probe_conditions import (build_condition_probe_set,
                                          text_probe_bank)
            if _cond == "f0":
                _probe_root = (cfg.paths.get("f0_probe_dir", None)
                               or os.path.join(cfg.paths.cache_dir, "f0_probe"))
            else:
                _probe_root = os.path.join(cfg.paths.cache_dir,
                                           f"probe_{_cond}")
            if _cond in global_configs:
                _probe_extractor = registry.global_extractors[_cond]
            else:
                _probe_extractor = fidelity_evaluator.extractors.get(
                    _cond, registry.frame_extractors[_cond])
            _probe_dev = getattr(_probe_extractor, "device",
                                 getattr(_probe_extractor, "_device", "cpu"))
            _bank = None
            if _cond == "text":
                _bank, _why = text_probe_bank(
                    load_caption_table(dataset.latent_root),
                    n_panels=_n_probes)
                print(f"[text-probe] {_why}")
            _ps = build_condition_probe_set(
                _cond, _probe_root, dataset.n_frames,
                extractor=_probe_extractor,
                n_probes=_n_probes,
                duration_s=float(cfg.model.duration_s),
                sr=lc.active().sample_rate,
                bank=_bank,
            )
            probe_sets[_cond] = _ps
            print(f"{_cond} probe: {len(_ps)} elementary stimuli, "
                  f"all plotted on TB | device={_probe_dev} | dir={_ps.dir}")
        except Exception as e:
            print(f"[{_cond}-probe] disabled -- could not build it: {e}")
    torch.set_rng_state(_rng_guard[0])
    np.random.set_state(_rng_guard[1])
    random.setstate(_rng_guard[2])
    if _rng_guard[3] is not None:
        torch.cuda.set_rng_state_all(_rng_guard[3])
    for _ext in getattr(registry, "global_extractors", {}).values():
        if hasattr(_ext, "unload"):
            try:
                _ext.unload()
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probe_sets


class MetricsAdapter:
    def __init__(self, cond_dataset):
        self._ds = cond_dataset
        self.n_frames = cond_dataset.n_frames
        self.idx_to_label = cond_dataset.idx_to_label
        self.samples = [
            (npy, start, label)
            for (npy, _cond, start, label, _class) in cond_dataset.samples
        ]

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        frames, _frame_cond, label_idx, _text, _image, _ctx = self._ds[idx]
        return frames, label_idx


def infinite_loader(loader, sampler=None, first_epoch=0):
    epoch = int(first_epoch)
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _capture_rng_state(data_generator=None):
    state = {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        if dist.is_available() and dist.is_initialized():
            state["torch_cuda_current"] = torch.cuda.get_rng_state()
        else:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def _restore_rng_state(state, data_generator=None):
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            if dist.is_available() and dist.is_initialized():
                torch.cuda.set_rng_state(state["torch_cuda"][0])
            else:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        elif "torch_cuda_current" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["torch_cuda_current"])
        if data_generator is not None and state.get("data_generator") is not None:
            data_generator.set_state(state["data_generator"])
        print("[SEED] RNG streams restored from checkpoint (x0, t, CFG dropout). "
              "Batch ORDER restarts on a fresh epoch, so a resumed run is "
              "statistically equivalent, not bit-identical.")
    except Exception as e:
        print(f"[SEED] WARNING: could not fully restore RNG state ({type(e).__name__}: "
              f"{e}); continuing with the freshly seeded RNG.")


def build_ckpt_data(model, ema, optimizer, scheduler, scaler, step,
                    val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                    frame_cond_dims, frame_cond_out_dims, global_configs,
                    data_generator=None, ema_ready=False):
    data = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "step":                 step,
        "val_loss":             val_loss,
        "best_val_loss":        best_val_loss,
        "model_kind":           cfg.model.kind,
        "frame_reinject_every": int(cfg.model.get("frame_reinject_every", 0)),
        "text_cross_every":     int(cfg.model.get("text_cross_every", 0)),
        "attention":            str(cfg.model.get("attention", "standard")),
        "qk_norm":              bool(cfg.model.get("qk_norm", False)),
        "codec":                lc.active().name,
        "config":               OmegaConf.to_container(cfg, resolve=True),
        "label_map":            label_map,
        "frame_cond_dims":      frame_cond_dims,
        "frame_cond_out_dims":  frame_cond_out_dims,
        "global_configs":       global_configs,
        "n_frames":             n_frames,
        "run_name":             run_name,
        "validation_protocol":  VALIDATION_PROTOCOL,
        "rng_state":            _capture_rng_state(data_generator),
    }
    if cfg.training.use_ema and ema is not None:
        data["ema_state_dict"] = ema.state_dict()
        data["ema_ready"] = bool(ema_ready)
    return data


DIST_TIMEOUT_MIN = 180


class DistInfo:
    def __init__(self, rank=0, world=1, local_rank=0, backend=None):
        self.rank = int(rank)
        self.world = int(world)
        self.local_rank = int(local_rank)
        self.backend = backend

    @property
    def enabled(self):
        return self.world > 1

    @property
    def is_main(self):
        return self.rank == 0


def init_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1") or 1)
    if world <= 1:
        return DistInfo()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    timeout = timedelta(minutes=DIST_TIMEOUT_MIN)
    if not torch.cuda.is_available():
        dist.init_process_group(backend="gloo", timeout=timeout)
        return DistInfo(rank, world, 0, "gloo")
    if not dist.is_nccl_available() or sys.platform == "win32":
        raise RuntimeError(
            f"[rank {rank}] multi-GPU training needs NCCL, which this "
            f"PyTorch / platform does not have ({sys.platform}). Run it on a "
            f"Linux server, or with the GPU hidden (CUDA_VISIBLE_DEVICES=-1) to "
            f"test the multi-process code on the CPU.")
    n_dev = torch.cuda.device_count()
    if local_rank >= n_dev:
        raise RuntimeError(
            f"[rank {rank}] LOCAL_RANK={local_rank} but only {n_dev} GPU(s) are "
            f"visible (CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}): one GPU per rank.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timeout)
    return DistInfo(rank, world, local_rank, "nccl")


def dist_barrier(di):
    if not di.enabled:
        return
    if di.backend == "nccl":
        dist.barrier(device_ids=[di.local_rank])
    else:
        dist.barrier()


def dist_all_gather_object(obj, di):
    if not di.enabled:
        return [obj]
    out = [None] * di.world
    dist.all_gather_object(out, obj)
    return out


def dist_broadcast_object(obj, di, src=0):
    if not di.enabled:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def dist_sum_tensors(tensors, di):
    if not di.enabled:
        return list(tensors)
    out = []
    for t in tensors:
        x = (t.detach().clone() if di.backend == "nccl"
             else t.detach().cpu().clone())
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        out.append(x.to(t.device))
    return out


def dist_sum(values, di):
    if not di.enabled:
        return [float(v) for v in values]
    dev = f"cuda:{di.local_rank}" if di.backend == "nccl" else "cpu"
    t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.cpu().tolist()


def dist_broadcast_tensor_(t, di, src=0):
    if not di.enabled:
        return
    if di.backend == "nccl":
        dist.broadcast(t, src=src)
    else:
        x = t.detach().cpu().clone()
        dist.broadcast(x, src=src)
        t.copy_(x.to(t.device))


def dist_dac_metrics(lat_list, ref_stats, enabled, device, di, block_size=16):
    out = {"fd_dac": None, "kl_real_gen": None, "kl_gen_real": None}
    want_fd = "fd_dac" in enabled
    want_kl = "kl_dac" in enabled
    if not (want_fd or want_kl):
        return out
    lats = [x for x in lat_list if x is not None]
    sum_x = torch.zeros(lc.active().latent_dim, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(lc.active().latent_dim, lc.active().latent_dim, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, len(lats), block_size):
        block = torch.stack(lats[start:start + block_size])
        block = block.reshape(-1, lc.active().latent_dim).to(device=device, dtype=torch.float64)
        sum_x = sum_x + block.sum(dim=0)
        sum_xx = sum_xx + block.T @ block
        count = count + block.shape[0]
        del block
    sum_x, sum_xx, count = dist_sum_tensors([sum_x, sum_xx, count], di)
    if not di.is_main:
        return out
    mu_gen, sigma_gen, _ = compute_mu_sigma(sum_x, sum_xx, count)
    mu_ref = ref_stats["mu"].to(device)
    sigma_ref = ref_stats["sigma"].to(device)
    if want_fd:
        out["fd_dac"] = float(
            compute_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen).item())
    if want_kl:
        out["kl_real_gen"] = gaussian_kl_fullcov(mu_ref, sigma_ref, mu_gen, sigma_gen)
        out["kl_gen_real"] = gaussian_kl_fullcov(mu_gen, sigma_gen, mu_ref, sigma_ref)
    return out


class NullWriter:
    log_dir = None

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop


if __name__ == "__main__":
    cfg, run_name = load_config()
    DIST = init_distributed()
    if DIST.enabled:
        run_name = dist_broadcast_object(run_name, DIST)
        cfg.paths.run_name = run_name
        if not DIST.is_main:
            sys.stdout = open(os.devnull, "w")
    print(f"[RUN NAME] {run_name}")

    set_dac_device(cfg.metrics.get("dac_device", "cpu"))

    CODEC = lc.activate(lc.dataset_codec(cfg.paths.dataset_root))
    print(f"[codec] {CODEC.name} (from the dataset): {CODEC.sample_rate} Hz, "
          f"{CODEC.frames_per_s:.2f} frames/s, {CODEC.latent_dim}-d latents")

    run_dir   = os.path.join(cfg.paths.runs_dir, run_name)
    ckpt_dir  = os.path.join(run_dir, "checkpoints")
    audio_dir = os.path.join(run_dir, "audio")
    cache_dir = cfg.paths.cache_dir
    os.makedirs(run_dir,   exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    if DIST.enabled and not DIST.is_main:
        sys.stdout = open(os.path.join(run_dir, f"rank{DIST.rank}.log"), "w",
                          buffering=1, encoding="utf-8",
                          errors="backslashreplace")

    normalizer_path   = os.path.join(cache_dir, "normalizer.pt")
    fd_dac_cache_path = os.path.join(cache_dir, "fd_dac_ref_stats.pt")
    _fad_ref_mode = str((cfg.get("metrics", None) or {}).get("fad_reference", "wav"))
    fad_cache_path = os.path.join(cache_dir, f"fad_vggish_ref_{_fad_ref_mode}.pt")

    _n_frames_fp = frames_per_chunk(cfg.paths.dataset_root, cfg.model.duration_s)
    if DIST.enabled and not DIST.is_main:
        dist_barrier(DIST)
    _validate_cache(cache_dir, _cache_fingerprint(cfg, _n_frames_fp),
                    guarded_files=[normalizer_path, fd_dac_cache_path,
                                   fad_cache_path])
    if DIST.enabled and DIST.is_main:
        dist_barrier(DIST)

    config_dump_path = os.path.join(run_dir, "config.yaml")
    if DIST.is_main:
        OmegaConf.save(cfg, config_dump_path)
    print(f"[CONFIG DUMP] {config_dump_path}")
    print(f"[RUN DIR]     {run_dir}")
    print(f"[CACHE DIR]   {cache_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    run_seed = cfg.training.get("seed", None)
    data_generator = None
    if run_seed is not None:
        run_seed = int(run_seed)
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        data_generator = torch.Generator()
        data_generator.manual_seed(run_seed)
        print(f"[SEED] Global training seed = {run_seed}")

    metrics_cfg = cfg.get("metrics", None)
    metrics_seed = (metrics_cfg.get("seed", None) if metrics_cfg is not None else None)
    validation_seed = int(cfg.data.get("validation_seed", 12345))
    validation_shuffle = bool(cfg.data.get("validation_shuffle", True))

    metrics_enabled = list(
        metrics_cfg.get("enabled", list(COND_METRICS))
        if metrics_cfg is not None else list(COND_METRICS))
    _unknown = [m for m in metrics_enabled if m not in COND_METRICS]
    if _unknown:
        raise SystemExit(
            f"[metrics] metrics.enabled contains {_unknown}, which the CONDITIONED "
            f"pipeline does not provide. Available: {list(COND_METRICS)}. "
            f"(fad_encodec needs the Encodec embedder and is not wired here; "
            f"fad_vggish is.)")
    print(f"[metrics] enabled: {metrics_enabled or '(none: distributional metrics off)'}")

    influence_family, mir_threshold = read_influence_settings(metrics_cfg)

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")


    enabled_pool = {name for name, c in CONDITION_CONFIG["frame_level"].items()
                    if c.get("enabled", False)}

    enabled_f = cfg.conditioning.get("enabled_frame",  None)
    enabled_g = cfg.conditioning.get("enabled_global", None)
    if enabled_f is not None:
        enabled_f = list(enabled_f)
    if enabled_g is not None:
        enabled_g = list(enabled_g)

    if enabled_f is None:
        cond_scan = _scan_frame_conditions(cfg.paths.condition_root)
    else:
        cond_scan = {"total": 0, "present": {}}
    available_all = {n for n, c in cond_scan["present"].items()
                     if cond_scan["total"] > 0 and c == cond_scan["total"]}
    available = available_all & enabled_pool

    if enabled_f is None:
        enabled_f = sorted(available)
        print(f"[conditions] enabled_frame not set -> using the frame conditions "
              f"present in ALL train .npz: {enabled_f}")
        if not enabled_f:
            raise RuntimeError(
                f"No frame conditions found on disk under "
                f"{cfg.paths.condition_root}. Extract them first, e.g.:\n"
                f"  python extract_conditions.py <dataset_root> "
                f"--conditions f0 --device cuda")
    elif enabled_f:
        print(f"[conditions] enabled_frame={enabled_f} given explicitly -> the "
              f".npz are not scanned at startup; each sample is checked when it "
              f"is loaded (training.strict_conditions).")

    strict_conditions = bool(cfg.training.get("strict_conditions", True))
    if enabled_f and cfg.paths.condition_root and cond_scan["total"] > 0:
        problems = []
        for name in enabled_f:
            cnt = int(cond_scan["present"].get(name, 0))
            miss = cond_scan["total"] - cnt
            status = "OK" if miss == 0 else f"MISSING in {miss}/{cond_scan['total']}"
            print(f"[conditions] '{name}': "
                  f"{cnt}/{cond_scan['total']} present  [{status}]")
            if miss > 0:
                problems.append((name, miss, cond_scan["total"]))
        if problems:
            lines = "\n".join(f"    - {nm}: missing in {mi}/{to} .npz"
                              for nm, mi, to in problems)
            msg = (f"[conditions] Some requested conditions are NOT present in every "
                   f".npz:\n{lines}\n  Re-run preprocess_stream.py / "
                   f"extract_conditions.py for them.")
            if strict_conditions:
                raise RuntimeError(
                    msg + "\n(training.strict_conditions=True: refusing to train "
                          "with samples that would fall back to NULL conditions. "
                          "Set training.strict_conditions=false to allow it.)")
            print(msg + "\n[conditions] WARNING (strict=false): affected samples "
                        "will use NULL (zero) conditions.")

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

    print("Loading conditioned datasets...")
    if DIST.enabled and not DIST.is_main:
        dist_barrier(DIST)
    cond_root = cfg.paths.condition_root if Path(cfg.paths.condition_root).exists() else None
    img_root  = cfg.paths.image_root     if Path(cfg.paths.image_root).exists()     else None

    train_dataset, val_dataset, test_dataset, normalizer, label_map, split_info = \
        build_conditioned_datasets(
            latent_root=cfg.paths.dataset_root,
            condition_root=cond_root,
            image_root=img_root,
            duration_s=cfg.model.duration_s,
            normalizer_path=(normalizer_path
                             if os.path.exists(normalizer_path) else None),
            registry=registry,
            preload=False,
            strict_conditions=strict_conditions,
            text_source=cfg.conditioning.get("text_source", "audio"),
            text_mix_p=cfg.conditioning.get("text_mix_p", 0.5),
            splits_path=cfg.paths.get("splits_path", None),
        )

    n_classes = len(label_map)
    print(f"\nDetected {n_classes} classes: {list(label_map.keys())}")
    print(f"[split] files  -> train {split_info['file_counts']['train']} | "
          f"val {split_info['file_counts']['val']} | "
          f"test {split_info['file_counts']['test']}")
    if split_info["manifest_path"]:
        print(f"[split] read from: {split_info['manifest_path']}")

    if DIST.is_main and not os.path.exists(normalizer_path):
        normalizer.save(normalizer_path)
    if DIST.enabled and DIST.is_main:
        dist_barrier(DIST)

    n_workers = int(cfg.data.get("num_workers", 4))
    train_sampler = None
    if DIST.enabled:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=DIST.world, rank=DIST.rank,
            shuffle=True, seed=int(run_seed if run_seed is not None else 0),
            drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.data.train_batch_size,
            sampler=train_sampler,
            num_workers=n_workers, pin_memory=(device == "cuda"),
            persistent_workers=(n_workers > 0),
            drop_last=True, collate_fn=collate_conditioned,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.data.train_batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=(device == "cuda"),
            persistent_workers=(n_workers > 0),
            drop_last=True, collate_fn=collate_conditioned,
            generator=data_generator,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.val_batch_size,
        shuffle=validation_shuffle,
        num_workers=n_workers, pin_memory=(device == "cuda"),
        persistent_workers=False,
        drop_last=False, collate_fn=collate_conditioned,
        generator=torch.Generator().manual_seed(validation_seed),
    )

    train_iter = infinite_loader(train_loader)

    requested_val_batches = int(cfg.data.num_val_batches)
    if requested_val_batches <= 0:
        raise ValueError(
            f"data.num_val_batches must be > 0, got {requested_val_batches}.")
    fixed_val_batches = []
    for vb in val_loader:
        fixed_val_batches.append(vb)
        if len(fixed_val_batches) >= requested_val_batches:
            break
    del val_loader
    if not fixed_val_batches:
        raise RuntimeError("Validation dataset produced no batches.")

    val_rng = torch.Generator(device="cpu")
    val_rng.manual_seed(validation_seed)
    fixed_val_inputs = []
    for vb in fixed_val_batches:
        val_frames = vb[0]
        fixed_x0 = torch.randn(
            val_frames.shape, dtype=val_frames.dtype, generator=val_rng)
        u = torch.randn(val_frames.shape[0], generator=val_rng)
        fixed_t = torch.sigmoid(u).clamp(
            float(cfg.sampling.t_min), float(cfg.sampling.t_max))
        fixed_val_inputs.append((fixed_x0, fixed_t))
    n_fixed_val_samples = sum(int(vb[0].shape[0]) for vb in fixed_val_batches)
    print(f"[validation] fixed protocol={VALIDATION_PROTOCOL} | "
          f"shuffle={validation_shuffle} | seed={validation_seed} | "
          f"batches={len(fixed_val_batches)} | "
          f"samples={n_fixed_val_samples}")

    print("\nPre-computation of reference statistics for the metrics...")
    metrics_val_ds = MetricsAdapter(val_dataset)

    if metrics_enabled and not DIST.is_main:
        fd_dac_ref_stats = None
    elif metrics_enabled:
        fd_dac_ref_stats = precompute_latent_reference(
            metrics_val_ds,
            cache_path=fd_dac_cache_path,
        )
        print(f"Reference stats ready: FD-{lc.active().label} + KL on "
              f"{fd_dac_ref_stats['n_total']} latent frames "
              f"({len(val_dataset)} val samples)\n")
    else:
        fd_dac_ref_stats = None
        print("Reference stats SKIPPED: no distributional metric enabled "
              "(metrics.enabled is empty).\n")

    fad_embedder = None
    fad_ref_stats = None
    n_fad = 0
    fad_device = "cpu"
    if "fad_vggish" in metrics_enabled:
        from metrics import VGGishEmbedder
        fad_device = str(cfg.metrics.get("fad_device", "cuda"))
        if fad_device.startswith("cuda") and not torch.cuda.is_available():
            print("[metrics] fad_device='cuda' but no GPU is available "
                  "-> FAD falls back to CPU.")
            fad_device = "cpu"
        n_fad = int(cfg.sampling.get("n_fad_samples", 512) or 0)
        fad_embedder = VGGishEmbedder(device=fad_device)
        print(f"[metrics] FAD-VGGish enabled | device={fad_device} | "
              f"n_fad_samples={n_fad} | reference={_fad_ref_mode}")

        _val_npy = sorted({str(smp[0]) for smp in val_dataset.samples})
        if not DIST.is_main:
            pass
        elif _fad_ref_mode == "wav":
            _lat_root = Path(cfg.paths.dataset_root)
            _wav_root = Path(cfg.paths.wav_root)
            _val_wavs = [_wav_root / Path(f).relative_to(_lat_root).with_suffix(".wav")
                         for f in _val_npy]
            _missing = [w for w in _val_wavs if not w.exists()]
            if _missing:
                raise SystemExit(
                    f"[metrics] fad_vggish with metrics.fad_reference='wav' needs the "
                    f"real validation wavs, but {len(_missing)}/{len(_val_wavs)} are "
                    f"missing under {_wav_root} (first: {_missing[0]}).\n"
                    f"  Either re-run preprocess_stream.py with --save_wav (it does "
                    f"NOT re-encode the latents), or set metrics.fad_reference="
                    f"'decoded' to build the reference by decoding the validation "
                    f"latents through DAC instead. NB: 'decoded' compares two "
                    f"DAC-decoded distributions, which isolates the model from the "
                    f"codec but is NOT comparable with published FAD values.")
            fad_ref_stats = precompute_audio_reference(
                _val_wavs, fad_embedder, cache_path=fad_cache_path,
                device=fad_device)
        elif _fad_ref_mode == "decoded":
            _dac = get_dac()

            def _real_clips():
                for i in range(len(val_dataset)):
                    frames = val_dataset[i][0]
                    yield (decode_frames_to_wav(frames, normalizer, _dac)
                           .view(1, 1, -1), lc.active().sample_rate)

            if os.path.exists(fad_cache_path):
                print(f"[Audio ref] loading cache: {fad_cache_path}")
                fad_ref_stats = torch.load(fad_cache_path, map_location="cpu",
                                           weights_only=False)
            else:
                print(f"[Audio ref] embedding {len(val_dataset)} DAC-decoded val "
                      f"latents (metrics.fad_reference='decoded')...")
                _mu, _sig, _n = compute_audio_mu_sigma(
                    _real_clips(), len(val_dataset), fad_embedder,
                    device=fad_device, desc="FAD ref")
                fad_ref_stats = {"mu": _mu.cpu(), "sigma": _sig.cpu(), "n_total": _n}
                _tmp = fad_cache_path + ".tmp"
                torch.save(fad_ref_stats, _tmp)
                os.replace(_tmp, fad_cache_path)
                print(f"[Audio ref] cache saved: {fad_cache_path}")
        else:
            raise SystemExit(
                f"[metrics] metrics.fad_reference='{_fad_ref_mode}' is not a valid "
                f"choice. Use 'wav' (real validation wavs, comparable with the "
                f"literature) or 'decoded' (validation latents decoded through "
                f"DAC, no wavs needed).")
        if DIST.is_main:
            print(f"FAD reference ready: {fad_ref_stats['n_total']} embedding "
                  f"vectors (128-D)\n")
    if DIST.enabled:
        fd_dac_ref_stats, fad_ref_stats = dist_broadcast_object(
            (fd_dac_ref_stats, fad_ref_stats), DIST)

    fidelity_evaluator, global_embedders = build_condition_scorers(
        cfg, FRAME_COND_DIMS, GLOBAL_CONFIGS, registry,
        influence_family, mir_threshold)

    _n_probes = panel_count(cfg.sampling, "probe")
    if FRAME_COND_DIMS or GLOBAL_CONFIGS:
        _n_test_panels = len(test_panel_indices(
            len(test_dataset), panel_count(cfg.sampling, "test")))
        print(f"[panels] the table: all {int(cfg.sampling.n_metrics_samples)} "
              f"validation generations of the metrics step | to listen to: "
              f"test {_n_test_panels or 'off'}"
              + (" (the test split is empty)"
                 if panel_count(cfg.sampling, "test") and not len(test_dataset)
                 else "")
              + f" | probe {_n_probes if _n_probes > 0 else 'off'}")
    probe_sets = (build_probe_sets(cfg, registry, fidelity_evaluator,
                                   val_dataset, FRAME_COND_DIMS,
                                   GLOBAL_CONFIGS)
                  if DIST.is_main else None)
    if probe_sets:
        print()

    metrics_uncond = bool(cfg.sampling.get("metrics_uncond", True))

    n_uncond_cards = max(0, int(cfg.sampling.get("n_audio_samples", 4) or 0))
    print(f"Uncond cards: {n_uncond_cards}"
          + (f" (uncond_00..uncond_{n_uncond_cards - 1:02d}), refreshed every "
             f"intervals.audio and every intervals.metrics from the same noise"
             if n_uncond_cards else "")
          + "\n")
    print(f"Conditioning influence: frame={list(FRAME_COND_DIMS.keys()) or 'none'}"
          f"{' + text(CLAP)' if 'text' in global_embedders else ''}"
          f"{' + image(Wav2CLIP)' if 'image' in global_embedders else ''}"
          f"{' + image(n/a)' if ('image' in GLOBAL_CONFIGS and 'image' not in global_embedders) else ''} "
          f"| uncond metrics: {'on' if metrics_uncond else 'off'}\n")

    FRAME_REINJECT_EVERY = int(cfg.model.get("frame_reinject_every", 0))

    TEXT_CROSS_EVERY = int(cfg.model.get("text_cross_every", 0))
    TEXT_CTX_DIM = int(getattr(train_dataset, "_ctx_dim", 0) or 0)
    if TEXT_CROSS_EVERY > 0 and "text" in GLOBAL_CONFIGS and TEXT_CTX_DIM <= 0:
        raise RuntimeError(
            "model.text_cross_every > 0 but this dataset carries no caption "
            "TOKEN sequences (global_conditions/text_labels_tok.npy), so the "
            "cross-attention would attend to the null token at every step and "
            "learn nothing -- silently, which is why this refuses to start "
            "instead of warning.\n"
            "  Re-run the preprocessing on the SAME output dir with "
            "--global_conds text: it rewrites only the sidecar, reads back the CLAP "
            "vectors already on disk and does not touch a single audio file."
        )
    ATTENTION = str(cfg.model.get("attention", "standard"))
    QK_NORM = bool(cfg.model.get("qk_norm", False))
    print(f"[MODEL] Building ConditionedAudioDiT-{cfg.model.kind} "
          f"| frame={list(FRAME_COND_DIMS)} | global={list(GLOBAL_CONFIGS)} "
          f"| frame_reinject_every={FRAME_REINJECT_EVERY} "
          f"| text_cross_every={TEXT_CROSS_EVERY} "
          f"| attention={ATTENTION} | qk_norm={QK_NORM}")
    model = ConditionedAudioDiT(
        kind=cfg.model.kind,
        drop=cfg.model.get("drop", 0.0),
        frame_cond_dims=FRAME_COND_DIMS,
        frame_cond_out_dims=FRAME_COND_OUT_DIMS,
        global_cond_configs=GLOBAL_CONFIGS,
        frame_reinject_every=FRAME_REINJECT_EVERY,
        text_cross_every=TEXT_CROSS_EVERY,
        text_ctx_dim=TEXT_CTX_DIM,
        attention=ATTENTION,
        qk_norm=QK_NORM,
        token_dim=lc.active().latent_dim,
    ).to(device)
    ema = EMAModel(model, decay=cfg.training.ema_decay) if cfg.training.use_ema else None

    optimizer = build_optimizer(model, cfg.training, optimizer_layout(cfg.training))
    lr_lambda = lr_lambda_from_cfg(cfg.training)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda', enabled=cfg.training.use_amp)

    writer = SummaryWriter(run_dir) if DIST.is_main else NullWriter()

    _cfg_log = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(_cfg_log, False)
    if "data" not in _cfg_log:
        _cfg_log.data = {}
    if _cfg_log.data.get("split", None) is None:
        _cfg_log.data.split = {}
    _cfg_log.data.split.composition = {
        "file_counts":  dict(split_info["file_counts"]),
        "n_classes":    int(split_info["n_classes"]),
        "splits_file":  split_info["manifest_path"],
        "params":       dict(split_info.get("params", {})),
    }

    _n_params_total = sum(p.numel() for p in model.parameters())
    _n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if "model" not in _cfg_log:
        _cfg_log.model = {}
    _cfg_log.model.n_params_total     = int(_n_params_total)
    _cfg_log.model.n_params_total_M   = round(_n_params_total / 1e6, 2)
    _cfg_log.model.n_params_trainable_M = round(_n_params_train / 1e6, 2)

    writer.add_text(
        "config",
        "```yaml\n" + OmegaConf.to_yaml(_cfg_log) + "\n```",
        global_step=0,
    )
    print(f"[params] total={_n_params_total/1e6:.2f}M | "
          f"trainable={_n_params_train/1e6:.2f}M")

    best_val_loss = float("inf")
    start_step = 0

    init_from = cfg.paths.get("init_from", None)
    if init_from and cfg.paths.resume_from:
        raise RuntimeError(
            "paths.init_from and paths.resume_from are both set. They are "
            "different things: resume continues a run (same architecture, same "
            "optimizer, same step), init_from STARTS one from another run's "
            "weights. Pick one.")
    if init_from:
        if not os.path.exists(init_from):
            raise FileNotFoundError(f"paths.init_from: {init_from} not found")
        print(f"Warm start from: {init_from}")
        _ck = torch.load(init_from, map_location="cpu", weights_only=False)
        check_ckpt_reinject_gate(_ck, init_from)
        _ck_codec = lc.codec_from_ckpt(_ck)
        if _ck_codec != lc.active().name:
            raise RuntimeError(
                f"{init_from} was trained on {_ck_codec} latents, this dataset "
                f"is {lc.active().name}. The two codecs have different latent "
                f"widths and frame rates: the weights cannot be carried over.")
        _ck_att = ckpt_attention(_ck)
        if _ck_att != ATTENTION:
            raise RuntimeError(
                f"{init_from} was trained with model.attention='{_ck_att}', "
                f"this run builds '{ATTENTION}'. A warm start may only ADD "
                f"zero-initialised weights; changing the self-attention changes "
                f"what the existing ones compute, so it needs a run from "
                f"scratch (leave paths.init_from empty), or set "
                f"model.attention={_ck_att} to warm-start as before.")
        _ck_qk = ckpt_qk_norm(_ck)
        if _ck_qk != QK_NORM:
            raise RuntimeError(
                f"{init_from} was trained with model.qk_norm={_ck_qk}, this run "
                f"builds qk_norm={QK_NORM}. A warm start may only ADD "
                f"zero-initialised weights; the QK-norm changes what the existing "
                f"q/k weights compute, so it needs a run from scratch (leave "
                f"paths.init_from empty), or set model.qk_norm={_ck_qk} to "
                f"warm-start as before.")
        _sd = (_ck.get("ema_state_dict") if _ck.get("ema_ready", True)
               and "ema_state_dict" in _ck else _ck.get("model_state_dict"))
        if not isinstance(_sd, dict):
            raise RuntimeError(f"{init_from} carries no model weights")
        _res = model.load_state_dict(_sd, strict=False)
        _new = tuple(["cross_attn.", "text_null", "norm_cross."])
        _bad = [k for k in _res.missing_keys if not any(n in k for n in _new)]
        if _bad or _res.unexpected_keys:
            raise RuntimeError(
                f"{init_from} does not fit this model.\n"
                f"  missing and NOT part of the new cross-attention: "
                f"{_bad[:8]}{' ...' if len(_bad) > 8 else ''}\n"
                f"  present in the checkpoint but not in this model: "
                f"{list(_res.unexpected_keys)[:8]}"
                f"{' ...' if len(_res.unexpected_keys) > 8 else ''}\n"
                f"A warm start may only ADD zero-initialised weights. Rebuild "
                f"this run with the same model.kind, conditions and "
                f"frame_reinject_every as the checkpoint.")
        print(f"  {len(_sd) - len(_res.missing_keys)} tensor(s) loaded, "
              f"{len(_res.missing_keys)} left at their zero/random init "
              f"(the new cross-attention). Step counter starts at 0.")
        if ema is not None:
            ema.copy_from(model)
        del _ck, _sd

    resume_from = cfg.paths.resume_from
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from: {resume_from}")
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)

        ckpt_kind = ckpt.get("model_kind", None)
        if ckpt_kind is not None and ckpt_kind != cfg.model.kind:
            raise RuntimeError(
                f"Checkpoint was trained with model.kind='{ckpt_kind}' but the "
                f"model was built as '{cfg.model.kind}'. They must match to "
                f"resume. (Normally the kind is restored automatically from the "
                f"checkpoint; if you passed model.kind on the command line, "
                f"remove it or set it to '{ckpt_kind}'.)"
            )
        ckpt_cross = int(ckpt.get("text_cross_every", 0))
        if ckpt_cross != TEXT_CROSS_EVERY:
            raise RuntimeError(
                f"Checkpoint was trained with model.text_cross_every="
                f"{ckpt_cross} but the model was built with "
                f"{TEXT_CROSS_EVERY}. They must match to resume: the "
                f"cross-attention adds four tensors per selected block plus "
                f"the learned null token, so the two state_dicts cannot be "
                f"put into one another. To START a NEW run with a different "
                f"value, leave paths.resume_from empty -- a zero-initialised "
                f"cross-attention computes the same function as none at all, "
                f"so nothing is lost by warm-starting from this checkpoint "
                f"with paths.init_from instead, if the run supports it."
            )
        ckpt_reinject = int(ckpt.get("frame_reinject_every", 0))
        if ckpt_reinject != FRAME_REINJECT_EVERY:
            raise RuntimeError(
                f"Checkpoint was trained with model.frame_reinject_every="
                f"{ckpt_reinject} but the model was built with "
                f"{FRAME_REINJECT_EVERY}. They must match to resume: the "
                f"per-block frame re-injection changes the weights themselves, "
                f"not just a training setting. (Normally the value is restored "
                f"automatically from the checkpoint config; if you passed "
                f"model.frame_reinject_every on the command line, remove it or "
                f"set it to {ckpt_reinject}. To START a NEW run with a "
                f"different value, do not use --resume: this is a different "
                f"architecture and needs its own run.)"
            )
        check_ckpt_reinject_gate(ckpt, cfg.paths.resume_from)
        ckpt_codec = lc.codec_from_ckpt(ckpt)
        if ckpt_codec != lc.active().name:
            raise RuntimeError(
                f"Checkpoint was trained on {ckpt_codec} latents but this "
                f"dataset is {lc.active().name} (its dataset_meta.json). A "
                f"resume needs the dataset it was trained on: point "
                f"paths.dataset_root at it.")
        ckpt_att = ckpt_attention(ckpt)
        if ckpt_att != ATTENTION:
            raise RuntimeError(
                f"Checkpoint was trained with model.attention='{ckpt_att}' but "
                f"the model was built with '{ATTENTION}'. They must match to "
                f"resume: the two read the same qkv weights differently, and "
                f"the differential one has its own tensors (lambda vectors, "
                f"per-head RMSNorm) in every block. Pass "
                f"model.attention={ckpt_att} to resume this run; a different "
                f"attention is a NEW run from scratch.")
        ckpt_qk = ckpt_qk_norm(ckpt)
        if ckpt_qk != QK_NORM:
            raise RuntimeError(
                f"Checkpoint was trained with model.qk_norm={ckpt_qk} but the "
                f"model was built with qk_norm={QK_NORM}. They must match to "
                f"resume: the QK-norm changes what the q/k weights compute and "
                f"adds its own gains in every attention layer. Pass "
                f"model.qk_norm={ckpt_qk} to resume this run (--resume reads it "
                f"off the checkpoint weights by itself); a different qk_norm is "
                f"a NEW run from scratch.")

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
        optimizer, scheduler, _rebuilt, _ignored = restore_optimizer(
            model, cfg.training, optimizer, scheduler, lr_lambda, ckpt)
        if _rebuilt:
            print(f"[RESUME] optimizer rebuilt with the checkpoint's parameter "
                  f"grouping ({_rebuilt}): the saved Adam moments are "
                  f"indexed by it.")
        if _ignored:
            print(f"[RESUME] NOTE: the optimizer restored from the checkpoint "
                  f"keeps {', '.join(_ignored)}; these training.* values are not "
                  f"applied on resume (grad_clip and the schedule shape are).")
        if cfg.training.use_ema:
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            else:
                ema = EMAModel(model, decay=cfg.training.ema_decay)
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        _restore_rng_state(ckpt.get("rng_state"), data_generator)
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        saved_val_protocol = ckpt.get("validation_protocol", None)
        if saved_val_protocol != VALIDATION_PROTOCOL:
            print("[validation] Checkpoint best_val_loss was measured with a "
                  "different/stochastic protocol; resetting best_val_loss so it "
                  "is not compared with the new fixed validation curve.")
            best_val_loss = float("inf")
        print(f"  -> Step {start_step} | best_val_loss: {best_val_loss:.6f}")
        writer.add_text("resumed_from", resume_from, global_step=start_step)

        if device == "cuda":
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        _resumed_ema_ready = bool(
            ckpt.get("ema_ready", start_step > cfg.training.ema_start))

        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        _resumed_ema_ready = False
        print("Training from zero.")

    train_model = model
    if DIST.enabled:
        from torch.nn.parallel import DistributedDataParallel as DDP
        _on_gpu = (device == "cuda")
        train_model = DDP(
            model,
            device_ids=[DIST.local_rank] if _on_gpu else None,
            output_device=DIST.local_rank if _on_gpu else None,
            gradient_as_bucket_view=True,
            find_unused_parameters=bool(getattr(model, "text_cross_layers", None)))
        if ema is not None:
            for _t in list(ema.model.parameters()) + list(ema.model.buffers()):
                dist_broadcast_tensor_(_t.data, DIST)
        if run_seed is not None and not DIST.is_main:
            _rs = run_seed + 1000003 * DIST.rank + start_step
            random.seed(_rs)
            np.random.seed(_rs % (2 ** 32))
            torch.manual_seed(_rs)
        train_iter = infinite_loader(train_loader, train_sampler,
                                     first_epoch=start_step)
        print(f"[multi-GPU] rank {DIST.rank} of {DIST.world} on "
              f"{f'cuda:{DIST.local_rank}' if _on_gpu else 'cpu'} | backend "
              f"{DIST.backend} | batch per optimizer step: "
              f"{cfg.data.effective_bs} x {DIST.world} processes = "
              f"{cfg.data.effective_bs * DIST.world}")

    n_frames = train_dataset.n_frames
    print(f"\n{'='*60}")
    print(f"Training on {device} | ConditionedAudioDiT-{cfg.model.kind}")
    print(f"Steps: {cfg.training.num_steps} | "
          f"Effective Batch: {cfg.data.effective_bs}")
    print(f"LR: {cfg.training.lr} | "
          f"EMA: {'on (decay=' + str(cfg.training.ema_decay) + ')' if cfg.training.use_ema else 'off'} | "
          f"AMP: {cfg.training.use_amp}")
    _sch = str(cfg.training.get("lr_schedule", "cosine"))
    print(f"LR schedule: {_sch} | warmup {cfg.training.warmup_steps} steps, then "
          + (f"lr * (1 + step / {float(cfg.training.get('lr_inv_gamma', 1.0e6)):g})"
             f"^-{float(cfg.training.get('lr_power', 0.5)):g} (Stable Audio 3)"
             if _sch == "inverse_power" else
             f"constant until {cfg.training.decay_start_frac:g} of the run, then "
             f"cosine to 0"))
    _g0 = optimizer.param_groups[0]
    print(f"Optimizer: AdamW | betas={tuple(_g0['betas'])} | eps={_g0['eps']:g} | "
          f"weight_decay={_g0['weight_decay']:g} "
          + ("on the weight matrices only" if len(optimizer.param_groups) == 2
             else "on every parameter")
          + f" | grad_clip={cfg.training.grad_clip or 'off'}")
    del _sch, _g0
    print(f"Sequence: {n_frames} frame = {n_frames} token of dim {lc.active().latent_dim}")
    print(f"Train: {len(train_dataset)} chunk | Val: {len(val_dataset)} chunk")
    print(f"Audio every {cfg.intervals.audio} step | "
          f"Metrics every {cfg.intervals.metrics} step")
    if fd_dac_ref_stats is not None:
        print(f"Metrics: {cfg.sampling.n_metrics_samples} generated vs "
              f"{fd_dac_ref_stats['n_total']} reference frames")
    else:
        print("Metrics: distributional metrics DISABLED (metrics.enabled: [])")
    print(f"Codec: {lc.active().name} ({lc.active().sample_rate} Hz, "
          f"{lc.active().frames_per_s:.2f} frames/s, "
          f"{lc.active().latent_dim}-d latents)")
    print(f"{lc.active().label} decoder device: {_DAC_DEVICE}"
          + ("  (~2.5 s per 5 s clip; 'cuda' is ~13x faster but adds a ~1 GB "
             "peak at the metrics step)" if _DAC_DEVICE == "cpu" else
             "  (~0.2 s per 5 s clip; ~1 GB peak at the metrics step)"))
    P_DROP_EACH_FRAME = float(cfg.conditioning.get("p_drop_each_frame", 0.0))
    print(f"CFG dropout: all={cfg.conditioning.p_drop_all} "
          f"frame={cfg.conditioning.p_drop_frame} "
          f"global={cfg.conditioning.p_drop_global} "
          f"each_frame={P_DROP_EACH_FRAME}"
          + ("  (partial subsets ON)" if P_DROP_EACH_FRAME > 0 else ""))
    print(f"CFG guidance scale (validation): {cfg.conditioning.guidance_scale}")
    _reinj_txt = (f"every {FRAME_REINJECT_EVERY} block(s)"
                  if FRAME_REINJECT_EVERY > 0 else "OFF (input concat only)")
    print(f"Frame re-injection (per block): {_reinj_txt}")
    print(f"Self-attention: {ATTENTION}"
          + (" (DIFF Transformer: two maps per head, lambda, per-head RMSNorm)"
             if ATTENTION == "differential" else ""))
    T_SAMPLER = str(cfg.training.get("t_sampler", "logit_normal"))
    _t_lo = t_sampler_cdf(T_SAMPLER, 0.1, n_frames)
    _t_hi = 1.0 - t_sampler_cdf(T_SAMPLER, 0.8, n_frames)
    print(f"Training t sampler: {T_SAMPLER} "
          + (f"(Stable Audio 3: logit-normal truncated at {T_TRUNC} and "
             f"rescaled, shift mu={length_shift_mu(n_frames):.4f} for {n_frames} "
             f"frames, alpha={math.exp(length_shift_mu(n_frames)):.2f})"
             if T_SAMPLER == "truncated_logit" else "(logit-normal (0, 1))")
          + f" | draws with t < 0.1 (noise end): {100 * _t_lo:.1f}%, "
            f"t > 0.8 (data end): {100 * _t_hi:.1f}% | the val loss keeps its "
            f"fixed logit-normal t, comparable across samplers")
    T_SCHEDULE = t_schedule_of(cfg.sampling)
    _grid = euler_grid(int(cfg.sampling.euler_steps), float(cfg.sampling.t_min),
                       float(cfg.sampling.t_max), T_SCHEDULE)
    print(f"Sampling t schedule: {T_SCHEDULE} "
          + ("(Stable Audio 3: steps equally spaced in log-SNR from -6.2 to "
             "2.0, from pure noise t=0 to the data t=1)"
             if T_SCHEDULE == "logsnr_uniform" else
             f"(equal steps from t={cfg.sampling.t_min} to {cfg.sampling.t_max})")
          + f" | {len(_grid)} Euler steps, "
            f"{sum(1 for _tv, _ in _grid if _tv < 0.1)} of them at t < 0.1 "
            f"(noise end), the last at t={_grid[-1][0]:.3f}")
    del _grid
    print(f"DATASET_ROOT:   {cfg.paths.dataset_root}")
    print(f"WAV_ROOT:       {cfg.paths.wav_root}")
    print(f"CONDITION_ROOT: {cfg.paths.condition_root}")
    print(f"IMAGE_ROOT:     {cfg.paths.image_root}")
    print(f"RUN DIR:        {run_dir}")
    print(f"{'='*60}\n")

    if DIST.is_main:
        log_real_audio_samples(
            test_dataset=test_dataset,
            normalizer=normalizer,
            writer=writer,
            n_samples=cfg.sampling.n_audio_samples,
            sampling_cfg=cfg.sampling,
            frame_dims=FRAME_COND_DIMS,
            global_configs=GLOBAL_CONFIGS,
            output_dir=audio_dir,
        )


    val_loss = None
    pbar = tqdm(range(start_step, cfg.training.num_steps),
                initial=start_step, total=cfg.training.num_steps,
                desc="Training", unit="step", disable=not DIST.is_main)
    last_step = start_step
    ema_ready = _resumed_ema_ready
    loop_completed = False

    try:
        for step in pbar:
            last_step = step
            model.train()

            accum_loss = 0.0
            for _micro in range(cfg.data.grad_accum):
                batch = next(train_iter)
                _sync = (train_model.no_sync()
                         if DIST.enabled and _micro < cfg.data.grad_accum - 1
                         else nullcontext())
                with _sync:
                    loss = compute_loss(
                        train_model, batch, device,
                        use_amp=cfg.training.use_amp,
                        t_min=cfg.sampling.t_min,
                        t_max=cfg.sampling.t_max,
                        global_configs=GLOBAL_CONFIGS,
                        p_drop_all=cfg.conditioning.p_drop_all,
                        p_drop_frame=cfg.conditioning.p_drop_frame,
                        p_drop_global=cfg.conditioning.p_drop_global,
                        p_drop_each_frame=P_DROP_EACH_FRAME,
                        training=True,
                        t_sampler=T_SAMPLER,
                    ) / cfg.data.grad_accum
                    scaler.scale(loss).backward()
                accum_loss += loss.item()

                del loss, batch

            scaler.unscale_(optimizer)
            _clip = cfg.training.grad_clip
            max_norm = _clip if (_clip is not None and _clip > 0) else float('inf')
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if cfg.training.use_ema and step >= cfg.training.ema_start:
                if not ema_ready:
                    ema.copy_from(model)
                    ema_ready = True
                    pbar.write(f"  -> EMA seeded from the live model @ step {step}")
                else:
                    ema.update(model)

            if DIST.enabled:
                accum_loss = dist_sum([accum_loss], DIST)[0] / DIST.world
            writer.add_scalar("Train/Loss", accum_loss, step)
            writer.add_scalar("Train/Learning rate",
                              scheduler.get_last_lr()[0], step)
            writer.add_scalar("Train/Grad_norm", grad_norm.item(), step)
            pbar.set_postfix(loss=f"{accum_loss:.4f}",
                              lr=f"{scheduler.get_last_lr()[0]:.1e}")

            if step % cfg.intervals.val == 0:
                model.eval()

                if device == "cuda":
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    val_loss_sum = 0.0
                    ema_val_loss_sum = 0.0
                    val_sample_count = 0
                    ema_active = (cfg.training.use_ema
                                  and step >= cfg.training.ema_start)
                    for _vbi, (vb, (fixed_x0, fixed_t)) in enumerate(zip(
                            fixed_val_batches, fixed_val_inputs)):
                        if DIST.enabled and _vbi % DIST.world != DIST.rank:
                            continue
                        batch_sample_count = int(vb[0].shape[0])
                        vl = compute_loss(
                            model, vb, device,
                            use_amp=cfg.training.use_amp,
                            t_min=cfg.sampling.t_min,
                            t_max=cfg.sampling.t_max,
                            global_configs=GLOBAL_CONFIGS,
                            p_drop_all=cfg.conditioning.p_drop_all,
                            p_drop_frame=cfg.conditioning.p_drop_frame,
                            p_drop_global=cfg.conditioning.p_drop_global,
                            p_drop_each_frame=P_DROP_EACH_FRAME,
                            training=False,
                            x0=fixed_x0,
                            t=fixed_t,
                        ).item()
                        val_loss_sum += vl * batch_sample_count
                        val_sample_count += batch_sample_count

                        if ema_active:
                            evl = compute_loss(
                                ema.model, vb, device,
                                use_amp=cfg.training.use_amp,
                                t_min=cfg.sampling.t_min,
                                t_max=cfg.sampling.t_max,
                                global_configs=GLOBAL_CONFIGS,
                                p_drop_all=cfg.conditioning.p_drop_all,
                                p_drop_frame=cfg.conditioning.p_drop_frame,
                                p_drop_global=cfg.conditioning.p_drop_global,
                                p_drop_each_frame=P_DROP_EACH_FRAME,
                                training=False,
                                x0=fixed_x0,
                                t=fixed_t,
                            ).item()
                            ema_val_loss_sum += evl * batch_sample_count

                    if DIST.enabled:
                        val_loss_sum, ema_val_loss_sum, val_sample_count = \
                            dist_sum([val_loss_sum, ema_val_loss_sum,
                                      val_sample_count], DIST)
                    val_loss = val_loss_sum / val_sample_count
                    ema_val_loss = val_loss
                    if ema_active:
                        ema_val_loss = ema_val_loss_sum / val_sample_count
                        writer.add_scalar("Validation/Loss_ema", ema_val_loss, step)

                writer.add_scalar("Validation/Loss", val_loss, step)

                if device == "cuda":
                    torch.cuda.empty_cache()

                ema_str = (f" | EMA Val {ema_val_loss:.6f}"
                           if cfg.training.use_ema and step >= cfg.training.ema_start else "")
                pbar.write(f"Step {step:7d} | Train {accum_loss:.6f} | "
                           f"Val {val_loss:.6f}{ema_str} | "
                           f"LR {scheduler.get_last_lr()[0]:.2e}")

                check_loss = (ema_val_loss
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else val_loss)
                if check_loss < best_val_loss and not DIST.is_main:
                    best_val_loss = check_loss
                elif check_loss < best_val_loss:
                    best_val_loss = check_loss
                    save_path = os.path.join(ckpt_dir, f"best_model_step{step}.pt")
                    ckpt_data = build_ckpt_data(
                        model, ema, optimizer, scheduler, scaler, step,
                        val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                        FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS,
                        data_generator=data_generator, ema_ready=ema_ready)
                    torch.save(ckpt_data, save_path)
                    for old in Path(ckpt_dir).glob("best_model_step*.pt"):
                        if old.resolve() != Path(save_path).resolve():
                            old.unlink()
                    pbar.write(f"  -> Best model: {save_path}")

            if (step > 0 and step % cfg.intervals.audio == 0
                    and step % cfg.intervals.metrics != 0 and n_uncond_cards
                    and DIST.is_main):
                pbar.write(f"\n  Uncond preview step {step}...")
                gen_model = (ema.model
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else model)
                generate_and_log_uncond_cards(
                    model=gen_model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer,
                    device=device, output_dir=audio_dir,
                    n_cards=n_uncond_cards,
                    sampling_cfg=cfg.sampling,
                    use_amp=cfg.training.use_amp,
                    frame_dims=FRAME_COND_DIMS,
                    global_configs=GLOBAL_CONFIGS,
                    metrics_seed=metrics_seed,
                    prefix=("EMA"
                            if cfg.training.use_ema and step >= cfg.training.ema_start
                            else "Model"),
                )
                pbar.write(f"  Uncond preview logged (step {step})\n")
                model.train()

            if step % cfg.intervals.ckpt == 0 and step > 0 and DIST.is_main:
                p = os.path.join(ckpt_dir, f"checkpoint_step{step}.pt")
                ckpt_data = build_ckpt_data(
                    model, ema, optimizer, scheduler, scaler, step,
                    val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                    FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS,
                        data_generator=data_generator, ema_ready=ema_ready)
                torch.save(ckpt_data, p)
                pbar.write(f"  -> Checkpoint: {p}")

                keep_n = cfg.intervals.get("keep_last_n_ckpts", 4)
                periodic_ckpts = sorted(
                    Path(ckpt_dir).glob("checkpoint_step*.pt"),
                    key=lambda x: int(x.stem.replace("checkpoint_step", "")),
                )
                for old in periodic_ckpts[:-keep_n]:
                    old.unlink()
                    pbar.write(f"  -> Removed old periodic checkpoint: {old.name}")

            if step > 0 and step % cfg.intervals.metrics == 0:
                gen_model = (ema.model
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else model)
                fd_dac_cond, kl_cond_rg, kl_cond_gr = evaluate_and_log_metrics(
                    model=gen_model,
                    normalizer=normalizer,
                    val_dataset=val_dataset,
                    step=step,
                    writer=writer,
                    device=device,
                    output_dir=audio_dir,
                    fd_dac_ref_stats=fd_dac_ref_stats,
                    n_samples=cfg.sampling.n_metrics_samples,
                    sampling_cfg=cfg.sampling,
                    conditioning_cfg=cfg.conditioning,
                    use_amp=cfg.training.use_amp,
                    frame_dims=FRAME_COND_DIMS,
                    global_configs=GLOBAL_CONFIGS,
                    fidelity_evaluator=fidelity_evaluator,
                    global_embedders=global_embedders,
                    compute_uncond=metrics_uncond,
                    prefix=("EMA"
                            if cfg.training.use_ema and step >= cfg.training.ema_start
                            else "Model"),
                    metrics_seed=metrics_seed,
                    metrics_enabled=metrics_enabled,
                    fad_embedder=fad_embedder,
                    fad_ref_stats=fad_ref_stats,
                    n_fad=n_fad,
                    fad_device=fad_device,
                    probe_sets=probe_sets,
                    image_root=cfg.paths.get("image_root", None),
                    test_dataset=test_dataset,
                    dist_info=DIST,
                    mir_curves=True,
                )
                _parts = []
                if fd_dac_cond is not None:
                    _parts.append(f"FD-{lc.active().label}={fd_dac_cond:.4f}")
                if kl_cond_rg is not None:
                    _parts.append(f"KL(real||gen)={kl_cond_rg:.4f}")
                if kl_cond_gr is not None:
                    _parts.append(f"KL(gen||real)={kl_cond_gr:.4f}")
                if _parts:
                    pbar.write("  Metrics [cond]: " + " | ".join(_parts) + "\n")
                model.train()
        loop_completed = True

    finally:
        last_path = os.path.join(ckpt_dir, f"checkpoint_last_step{last_step}.pt")

        def _try_save():
            if not DIST.is_main:
                return
            ckpt_data = build_ckpt_data(
                model, ema, optimizer, scheduler, scaler, last_step,
                val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                FRAME_COND_DIMS, FRAME_COND_OUT_DIMS, GLOBAL_CONFIGS,
                        data_generator=data_generator, ema_ready=ema_ready)
            torch.save(ckpt_data, last_path)

        saved = False
        try:
            _try_save()
            saved = True
            if DIST.is_main:
                print(f"\n  -> Last checkpoint saved: {last_path}")
        except Exception as e_gpu:
            print(f"\n  [WARN] Normal checkpoint save failed ({type(e_gpu).__name__}: "
                  f"{e_gpu}). Retrying from CPU after freeing GPU memory...")
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
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
        if DIST.enabled and loop_completed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        print("Training concluded." if saved else "Training ended (see warnings above).")
