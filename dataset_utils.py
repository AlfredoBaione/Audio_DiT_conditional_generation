# Latent normalizer and split reader.

import json
import random
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

import latent_codec as lc

MAX_FRAMES = 4096

SUPPORTED_EXTS   = {".npy"}


def frames_per_chunk(latent_root, duration_s: float) -> int:
    meta_path = Path(latent_root).parent / "dataset_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    real_T = meta.get("latent_frames_per_chunk")
    chunk_s = meta.get("chunk_duration_s")
    fps = lc.get_spec(lc.codec_from_meta(meta)).frames_per_s

    if chunk_s and duration_s > float(chunk_s) + 1e-6:
        raise RuntimeError(
            f"[frames_per_chunk] model.duration_s={duration_s}s exceeds the "
            f"dataset chunk length ({chunk_s}s, from {meta_path}). The latents "
            f"only hold {chunk_s}s each: re-run preprocess_stream.py with "
            f"--chunk_duration {duration_s} into a NEW output dir (and a fresh "
            f"cache_dir), or set model.duration_s <= {chunk_s}.")

    if real_T:
        if chunk_s is None or abs(duration_s - float(chunk_s)) <= 1e-6:
            return int(real_T)
        return int(duration_s * fps)
    return int(duration_s * fps)

class LatentNormalizer:
    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None

    def fit_from_chunks(
        self,
        chunks: List[Tuple[Path, int]],
        n_frames: int,
        device: Optional[str] = None,
        batch_accum: int = 50,
        io_workers: int = 16,
        latent_dim: Optional[int] = None,
        merge=None,
        progress: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if latent_dim is None:
            latent_dim = lc.active().latent_dim

        n_chunks = len(chunks)
        print(f"[Normalizer] Welford batched on {n_chunks:,} chunk "
              f"(device={device}, batch_accum={batch_accum}, "
              f"io_workers={io_workers})...")

        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor

        def _read(item):
            path, start = item
            try:
                z = np.load(str(path), mmap_mode="r")[:, start:start + n_frames]
                z = np.ascontiguousarray(z, dtype=np.float32)
            except Exception:
                return "unreadable"
            if z.ndim != 2 or z.shape[0] != latent_dim:
                return "bad_shape"
            if z.shape[1] != n_frames:
                return "short"
            if not np.isfinite(z).all():
                return "non_finite"
            return z

        mean_acc = None
        m2_acc   = None
        n_total  = 0
        n_used   = 0
        rejected = {}
        rejected_paths = []

        with ThreadPoolExecutor(max_workers=max(1, io_workers)) as ex:
            with tqdm(total=n_chunks, desc="Normalizer fit", unit="chunk",
                      disable=not progress) as pbar:
                for b0 in range(0, n_chunks, batch_accum):
                    block = chunks[b0:b0 + batch_accum]
                    arrs = []
                    for (path, _start), r in zip(block, ex.map(_read, block)):
                        if isinstance(r, str):
                            rejected[r] = rejected.get(r, 0) + 1
                            if len(rejected_paths) < 10:
                                rejected_paths.append(f"{path}  [{r}]")
                        else:
                            arrs.append(r)
                    pbar.update(len(block))
                    if not arrs:
                        continue
                    n_used += len(arrs)

                    batch = torch.from_numpy(np.concatenate(arrs, axis=1)).to(
                        device=device, dtype=torch.float64)
                    n_new = batch.shape[1]

                    if mean_acc is None:
                        dim = batch.shape[0]
                        mean_acc = torch.zeros(dim, 1, dtype=torch.float64, device=device)
                        m2_acc   = torch.zeros(dim, 1, dtype=torch.float64, device=device)

                    n_total_new = n_total + n_new
                    batch_mean  = batch.mean(dim=1, keepdim=True)
                    delta       = batch_mean - mean_acc
                    mean_acc    = mean_acc + delta * (n_new / n_total_new)
                    batch_m2    = ((batch - batch_mean) ** 2).sum(dim=1, keepdim=True)
                    m2_acc      = m2_acc + batch_m2 + (delta ** 2) * (n_total * n_new / n_total_new)
                    n_total     = n_total_new

        if merge is not None:
            if mean_acc is None:
                mean_acc = torch.zeros(latent_dim, 1, dtype=torch.float64, device=device)
                m2_acc   = torch.zeros(latent_dim, 1, dtype=torch.float64, device=device)
            (n_total, mean_acc, m2_acc, n_used, n_chunks, rejected,
             rejected_paths) = merge(n_total, mean_acc, m2_acc, n_used, n_chunks,
                                     rejected, rejected_paths)
            if n_total == 0:
                mean_acc = None

        if mean_acc is None:
            raise RuntimeError(
                f"Normalizer fit read no usable chunk out of {n_chunks:,} "
                f"(rejected: {rejected or 'none'}).")

        var = (m2_acc / n_total).float().cpu()
        self.mean = mean_acc.float().cpu()
        self.std  = (var + 1e-6).sqrt()

        if device == "cuda":
            torch.cuda.empty_cache()

        if not (torch.isfinite(self.mean).all() and torch.isfinite(self.std).all()):
            raise RuntimeError(
                "Normalizer produced non-finite mean/std. The latents feeding it "
                "are corrupt; inspect the dataset before training.")

        print(f"[Normalizer] fitted on {n_used:,}/{n_chunks:,} chunk "
              f"({n_total:,} frames)")
        if rejected:
            print(f"[Normalizer] rejected chunks: {rejected} "
                  f"-- these did NOT contribute to mean/std")
            for p in rejected_paths:
                print(f"             {p}")
            if sum(rejected.values()) > len(rejected_paths):
                print(f"             ... and "
                      f"{sum(rejected.values()) - len(rejected_paths)} more")
            print("             NOTE: these files are still indexed by the "
                  "datasets; training will stop on them.")
        print(f"[Normalizer] mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
        print(f"[Normalizer] std range:  [{self.std.min():.3f}, {self.std.max():.3f}]")

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None, "Call fit_from_chunks() before normalize()"
        return (z - self.mean.to(z.device)) / self.std.to(z.device)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def save(self, path: str):
        import os as _os
        path = str(path)
        tmp = f"{path}.tmp"
        torch.save({"mean": self.mean, "std": self.std}, tmp)
        _os.replace(tmp, path)
        print(f"[Normalizer] saved in {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "mean" not in ckpt or "std" not in ckpt:
            raise RuntimeError(
                f"{path} is not a valid normalizer file (missing mean/std). "
                f"Delete it and let the training recompute it.")
        mean, std = ckpt["mean"], ckpt["std"]
        if mean.shape != std.shape:
            raise RuntimeError(
                f"{path}: mean{tuple(mean.shape)} and std{tuple(std.shape)} "
                f"have different shapes.")
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise RuntimeError(
                f"{path} contains non-finite mean/std (it was computed before "
                f"the latent validation existed, or from corrupt latents). "
                f"Delete this file so it is recomputed.")
        if (std <= 0).any():
            raise RuntimeError(
                f"{path} contains a non-positive std, which would divide by "
                f"zero on normalize(). Delete this file so it is recomputed.")
        self.mean = mean
        self.std  = std
        print(f"[Normalizer] loaded from {path}")


def _class_of_file(npy_path: Path) -> str:
    return npy_path.parent.name


def _source_group_of(npy_path: Path, latent_root: Path) -> str:
    rel_parent = npy_path.parent.relative_to(latent_root)
    stem = npy_path.stem.split("__")[0]
    return (rel_parent / stem).as_posix()


def _seed_for(seed: int, key: str) -> int:
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h, 16) % (2 ** 32)


def _allocate_three(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    r_tr, r_val, r_te = ratios
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    n_te = round(r_te * n)
    n_val = round(r_val * n)
    if r_te > 0 and n_te == 0:
        n_te = 1
    if r_val > 0 and n_val == 0 and (n - n_te) >= 2:
        n_val = 1
    n_tr = n - n_te - n_val
    if n_tr < 0:
        over = -n_tr
        take = min(over, n_val); n_val -= take; over -= take
        if over > 0:
            n_te -= over
        n_tr = n - n_te - n_val
    return (n_tr, n_val, n_te)


SPLITS_NAME = "splits.json"


def load_source_split(latent_root, splits_path=None) -> dict:
    latent_root = Path(latent_root)
    p = Path(splits_path) if splits_path else latent_root.parent / SPLITS_NAME
    if not p.exists():
        raise FileNotFoundError(
            f"[split] {p} not found. Create it with:\n"
            f"    python preprocess_stream.py SRC {latent_root.parent} --split_only")
    payload = json.loads(p.read_text())
    groups = payload.get("groups")
    if not groups:
        raise RuntimeError(f"[split] {p} has no 'groups' entry.")

    all_files = sorted(latent_root.rglob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"No .npy latents under {latent_root}")

    splits = {"train": [], "val": [], "test": []}
    unassigned = []
    for f in all_files:
        name = groups.get(_source_group_of(f, latent_root))
        if name in splits:
            splits[name].append(f)
        else:
            unassigned.append(f)
    if unassigned:
        raise RuntimeError(
            f"[split] {len(unassigned)} latent(s) under {latent_root} belong to "
            f"source(s) with no assignment in {p} (first: {unassigned[0].name}). "
            f"The split is older than the dataset: re-run preprocess_stream.py "
            f"(--split_only is enough) to assign the new sources.")

    classes = sorted({_class_of_file(f) for f in all_files})
    params = dict(payload.get("params", {}))
    params["source"] = payload.get("source", "assigned")
    print(f"[split] loaded {p.name} ({payload.get('source', 'assigned')}): "
          f"{ {k: len(v) for k, v in splits.items()} } chunks over "
          f"{len(groups)} recorded source(s)")
    return {
        "splits": {k: sorted(v) for k, v in splits.items()},
        "classes": classes,
        "file_counts": {k: len(v) for k, v in splits.items()},
        "manifest_path": str(p),
        "params": params,
    }


NORMALIZER_MAX_CHUNKS = None


def _meta_latent_frames(latent_root) -> Optional[int]:
    p = Path(latent_root).parent / "dataset_meta.json"
    if not p.exists():
        return None
    try:
        v = json.loads(p.read_text()).get("latent_frames_per_chunk")
        return int(v) if v else None
    except Exception:
        return None


def _chunks_from_files(files: List[Path], n_frames: int,
                       uniform_frames: Optional[int] = None,
                       max_chunks: Optional[int] = None,
                       seed: int = 0) -> List[Tuple[Path, int]]:
    n_files = len(files)
    if n_files == 0:
        return []

    per_file = 0
    if uniform_frames and uniform_frames >= n_frames:
        probe = sorted({int(round(i)) for i in
                        np.linspace(0, n_files - 1, min(64, n_files))})
        ok = True
        for i in probe:
            try:
                if np.load(str(files[i]), mmap_mode="r").shape[1] != uniform_frames:
                    ok = False
                    break
            except Exception:
                ok = False
                break
        if ok:
            per_file = uniform_frames // n_frames
        else:
            print("[normalizer] dataset_meta.json declares "
                  f"latent_frames_per_chunk={uniform_frames} but the sampled "
                  "files disagree -> falling back to the exact per-file scan "
                  "(slower).")

    if per_file > 0:
        sel = files
        if max_chunks and n_files * per_file > max_chunks:
            keep = max(1, max_chunks // per_file)
            idx = random.Random(seed).sample(range(n_files), min(keep, n_files))
            sel = [files[i] for i in sorted(idx)]
            print(f"[normalizer] fitting on {len(sel):,} of {n_files:,} files "
                  f"({min(len(sel) * per_file, max_chunks):,} chunks, "
                  f"seed={seed}): a per-channel mean/std does not need the full "
                  f"corpus.")
        out = [(f, k * n_frames) for f in sel for k in range(per_file)]
        if max_chunks and len(out) > max_chunks:
            out = out[:max_chunks]
        return out

    chunks = []
    for f in files:
        try:
            file_frames = np.load(str(f), mmap_mode="r").shape[1]
        except Exception:
            continue
        for k in range(file_frames // n_frames):
            chunks.append((f, k * n_frames))
    if max_chunks and len(chunks) > max_chunks:
        chunks = random.Random(seed).sample(chunks, max_chunks)
        chunks.sort()
        print(f"[normalizer] fitting on {len(chunks):,} sampled chunks "
              f"(seed={seed}).")
    return chunks
