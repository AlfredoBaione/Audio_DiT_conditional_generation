# Latent dataset and normalizer.

import json
import random
import hashlib
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict

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

_dac_model = None

def get_dac_model(device: str = "cpu"):
    global _dac_model
    if _dac_model is None:
        try:
            import dac
            _dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
            _dac_model.to(device)
            _dac_model.eval()
            print(f"[DAC] Modello caricato su {device}")
        except ImportError:
            raise ImportError("DAC not found. Install it with: pip install descript-audio-codec")
    return _dac_model


@torch.no_grad()
def decode_latents(latents: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    if lc.active().name != "dac_44khz":
        model = lc.load_model(lc.active().name, device)
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        return lc.decode(model, latents.to(device)).squeeze(0)
    model = get_dac_model(device)
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    latents = latents.to(device)
    z_q, _, _ = model.quantizer.from_latents(latents)
    waveform = model.decode(z_q)
    return waveform.squeeze(0)


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
            with tqdm(total=n_chunks, desc="Normalizer fit", unit="chunk") as pbar:
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


def _split_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


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


def _split_two(remaining: List[str], r_tr: float, r_val: float) -> Tuple[List[str], List[str]]:
    n = len(remaining)
    if n == 0:
        return [], []
    denom = r_tr + r_val
    n_val = 0 if denom <= 0 else round((r_val / denom) * n)
    if r_val > 0 and n_val == 0 and n >= 2:
        n_val = 1
    n_val = min(n_val, n)
    return remaining[n_val:], remaining[:n_val]


SPLITS_NAME = "splits.json"


def load_source_split(latent_root, splits_path=None) -> dict:
    latent_root = Path(latent_root)
    p = Path(splits_path) if splits_path else latent_root.parent / SPLITS_NAME
    if not p.exists():
        raise FileNotFoundError(
            f"[split] {p} not found. The train/val/test split is now decided by "
            f"preprocess_stream.py and recorded in that file.\n"
            f"  For a dataset built before this change, create it once with:\n"
            f"    python preprocess_stream.py SRC {latent_root.parent} "
            f"--import_legacy_split\n"
            f"  (reproduces the split the training used to compute in-code, so a "
            f"run already in flight keeps the exact same val/test sets), or\n"
            f"    python preprocess_stream.py SRC {latent_root.parent} --split_only\n"
            f"  for a fresh one.")
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


def compute_split(
    latent_root,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    group_by_source: bool = True,
    stratify_by_class: bool = True,
    save_test_manifest: bool = True,
    manifest_dir: Optional[str] = None,
) -> dict:
    latent_root = Path(latent_root)
    all_files = sorted(latent_root.rglob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"No .npy latents under {latent_root}")

    groups: Dict[str, dict] = {}
    for f in all_files:
        gk = _source_group_of(f, latent_root) if group_by_source else f.as_posix()
        g = groups.setdefault(gk, {"class": _class_of_file(f), "files": []})
        g["files"].append(f)

    classes = sorted({g["class"] for g in groups.values()})

    buckets: Dict[str, List[str]] = defaultdict(list)
    for gk, g in groups.items():
        buckets[g["class"] if stratify_by_class else "__all__"].append(gk)

    params = {
        "ratios": list(ratios), "seed": seed,
        "group_by_source": group_by_source,
        "stratify_by_class": stratify_by_class,
    }
    if manifest_dir is None:
        manifest_dir = latent_root.parent / "splits"
    manifest_dir = Path(manifest_dir)
    manifest_path = manifest_dir / f"test_split_{_split_hash(params)}.json"

    fixed_test = None
    if manifest_path.exists():
        try:
            man = json.loads(manifest_path.read_text())
            fixed_test = set(man.get("test_groups", [])) & set(groups.keys())
            print(f"[split] reusing persisted test set from {manifest_path.name} "
                  f"({len(fixed_test)} groups present)")
        except Exception as e:
            print(f"[split] WARNING: could not read {manifest_path} ({e}); recomputing")
            fixed_test = None

    r_tr, r_val, r_te = ratios
    train_g, val_g, test_g = [], [], []
    for bucket, gks in buckets.items():
        gks_sorted = sorted(gks)
        random.Random(_seed_for(seed, bucket)).shuffle(gks_sorted)
        if fixed_test is not None:
            cls_test = [gk for gk in gks_sorted if gk in fixed_test]
            remaining = [gk for gk in gks_sorted if gk not in fixed_test]
            cls_train, cls_val = _split_two(remaining, r_tr, r_val)
        else:
            n_tr, n_val, n_te = _allocate_three(len(gks_sorted), ratios)
            cls_train = gks_sorted[:n_tr]
            cls_val = gks_sorted[n_tr:n_tr + n_val]
            cls_test = gks_sorted[n_tr + n_val:]
        train_g += cls_train
        val_g += cls_val
        test_g += cls_test

    if fixed_test is None and save_test_manifest:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(
            {"params": params, "test_groups": sorted(test_g)}, indent=2))
        print(f"[split] wrote test manifest -> {manifest_path}")

    def _files(group_keys):
        out = []
        for gk in group_keys:
            out.extend(groups[gk]["files"])
        return sorted(out)

    splits = {"train": _files(train_g), "val": _files(val_g), "test": _files(test_g)}

    for _name, _r in (("val", r_val), ("test", r_te)):
        if _r > 0 and len(splits[_name]) == 0:
            raise RuntimeError(
                f"[split] the '{_name}' split is EMPTY despite ratio={_r} > 0. Every "
                f"class has too few source groups (a class with 1 source goes "
                f"entirely to train). Add more data, reduce single-source classes, "
                f"or set the '{_name}' ratio to 0 to run train-only on purpose.")

    return {
        "splits": splits,
        "classes": classes,
        "file_counts": {k: len(v) for k, v in splits.items()},
        "manifest_path": str(manifest_path) if save_test_manifest else None,
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


class AudioLatentDataset(Dataset):
    def __init__(
        self,
        files:        List[Path],
        label_to_idx: Dict[str, int],
        split:        str   = "train",
        latent_root:  Optional[str] = None,
        duration_s:   float = 5.0,
        normalizer:   Optional[LatentNormalizer] = None,
        device:       str   = "cpu",
        preload:      bool  = False,
    ):
        self.files       = [Path(f) for f in files]
        self.split       = split
        self.latent_root = Path(latent_root) if latent_root else None
        self.normalizer  = normalizer
        self.duration_s  = duration_s
        self.preload     = preload

        self.n_frames = frames_per_chunk(latent_root, duration_s)
        self.codec = lc.get_spec(lc.dataset_codec(latent_root)) if latent_root             else lc.active()
        self.latent_dim = self.codec.latent_dim

        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}

        self.samples: List[Tuple[Path, int, int]] = []
        self._actual_file_frames = None

        self._build_samples()

        self._cache: dict = {}
        if preload:
            self._preload_all()

        chunks_per_file = self._actual_file_frames // self.n_frames if self._actual_file_frames else "?"
        print(f"[Dataset/{split}] duration_s={duration_s}s → "
              f"n_frames={self.n_frames} (= token sequence) | "
              f"token_dim={self.latent_dim} | "
              f"file_frames={self._actual_file_frames} | "
              f"chunks per file={chunks_per_file} | "
              f"tot samples={len(self.samples)} | "
              f"preload={'ON' if preload else 'OFF'}")

    def _detect_file_frames(self) -> int:
        for f in self.files:
            if f.suffix.lower() in SUPPORTED_EXTS:
                z = np.load(str(f), mmap_mode='r')
                n_frames = z.shape[1]
                print(f"[Dataset/{self.split}] Self-detected: {n_frames} frame per file "
                      f"({n_frames / self.codec.frames_per_s:.1f}s) from {f.name}")
                return n_frames
        raise FileNotFoundError(f"No .npy file in the {self.split} split file list")

    def _build_samples(self):
        if not self.files:
            print(f"[Dataset/{self.split}] WARNING: no files for this split")
            self._actual_file_frames = 0
            return

        self._actual_file_frames = self._detect_file_frames()

        n_chunks_ref = self._actual_file_frames // self.n_frames
        if n_chunks_ref == 0:
            raise ValueError(
                f"Files have {self._actual_file_frames} frames but "
                f"duration_s={self.duration_s}s requires {self.n_frames} frames. "
                f"Files are too short!")

        for f in self.files:
            if f.suffix.lower() not in SUPPORTED_EXTS:
                continue
            label_idx = self.label_to_idx.get(_class_of_file(f))
            if label_idx is None:
                continue
            try:
                file_frames = np.load(str(f), mmap_mode="r").shape[1]
            except Exception:
                continue
            for k in range(file_frames // self.n_frames):
                start = k * self.n_frames
                if start + self.n_frames <= file_frames:
                    self.samples.append((f, start, label_idx))

        print(f"[Dataset/{self.split}] {len(self.samples)} total chunks | "
              f"classes: {len(self.label_to_idx)}")

    @staticmethod
    def _load_latent_static(npy_path: Path) -> torch.Tensor:
        z = np.load(str(npy_path)).astype(np.float32)
        return torch.from_numpy(z)

    def _load_slice_mmap(self, npy_path: Path, start: int) -> torch.Tensor:
        arr = np.load(str(npy_path), mmap_mode="r")
        try:
            sl = np.array(arr[:, start : start + self.n_frames], dtype=np.float32)
        finally:
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()
            del arr
        return torch.from_numpy(sl)

    def _preload_all(self):
        unique_paths = set(str(p) for p, _, _ in self.samples)
        print(f"[Dataset/{self.split}] Preloading {len(unique_paths)} files in RAM (float32)...")
        from tqdm import tqdm
        for path_str in tqdm(sorted(unique_paths), desc=f"Preload {self.split}"):
            self._cache[path_str] = self._load_latent_static(Path(path_str))
        size_gb = sum(t.nelement() * 4 for t in self._cache.values()) / 1e9
        print(f"[Dataset/{self.split}] Preloaded: {size_gb:.2f} GB in RAM (float32)")

    def get_chunks_for_normalizer(self) -> List[Tuple[Path, int]]:
        return [(path, start) for path, start, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        npy_path, start, label_idx = self.samples[idx]

        key = str(npy_path)
        if key in self._cache:
            z = self._cache[key][:, start : start + self.n_frames].float()
        else:
            z = self._load_slice_mmap(npy_path, start)

        if self.normalizer is not None:
            z = self.normalizer.normalize(z)

        z = z.T

        if z.shape[0] != self.n_frames:
            raise RuntimeError(
                f"Sample {npy_path.name} @ start={start}: "
                f"expected shape ({self.n_frames}, {self.latent_dim}), obtained {tuple(z.shape)}. "
                f"The file has less frames than expected."
            )

        return z, label_idx


def build_datasets(
    latent_root:     str,
    duration_s:      float = 5.0,
    device:          str   = "cpu",
    normalizer_path: Optional[str] = None,
    preload:         bool  = False,
    split_ratios:    Tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_seed:      int = 42,
    group_by_source: bool = True,
    stratify_by_class: bool = True,
    save_test_manifest: bool = True,
):
    latent_root = str(latent_root)

    split = compute_split(
        latent_root, ratios=split_ratios, seed=split_seed,
        group_by_source=group_by_source, stratify_by_class=stratify_by_class,
        save_test_manifest=save_test_manifest,
    )
    splits = split["splits"]
    classes = split["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}

    normalizer = LatentNormalizer()
    if normalizer_path and Path(normalizer_path).exists():
        normalizer.load(normalizer_path)
    else:
        print("[build_datasets] Computing the normalizer on the train split...")
        n_frames = frames_per_chunk(latent_root, duration_s)
        chunks = _chunks_from_files(
            splits["train"], n_frames,
            uniform_frames=_meta_latent_frames(latent_root),
            max_chunks=NORMALIZER_MAX_CHUNKS,
        )
        if not chunks:
            raise RuntimeError("No train chunks available to fit the normalizer.")
        normalizer.fit_from_chunks(chunks, n_frames=n_frames)

    common = dict(label_to_idx=label_to_idx, latent_root=latent_root,
                  duration_s=duration_s, device=device)

    train_dataset = AudioLatentDataset(files=splits["train"], split="train",
                                       normalizer=normalizer, preload=preload, **common)
    val_dataset = AudioLatentDataset(files=splits["val"], split="val",
                                     normalizer=normalizer, preload=False, **common)
    test_dataset = AudioLatentDataset(files=splits["test"], split="test",
                                      normalizer=normalizer, preload=False, **common)

    split_info = {
        "file_counts": split["file_counts"],
        "chunk_counts": {"train": len(train_dataset), "val": len(val_dataset),
                         "test": len(test_dataset)},
        "n_classes": len(classes),
        "manifest_path": split["manifest_path"],
        "params": split["params"],
    }

    print(f"[build_datasets] Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"Test: {len(test_dataset)} | duration_s={duration_s}s → "
          f"{train_dataset.n_frames} frame/token per chunk")

    return train_dataset, val_dataset, test_dataset, normalizer, label_to_idx, split_info


