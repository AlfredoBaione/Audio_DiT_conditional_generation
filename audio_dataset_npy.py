# audio_dataset_npy.py
#
# Dataset for DAC latents pre-computed in .npy (unconditional training).
# No patching — every frame DAC is a token for the transformer.
#
# Auto-detection of the file .npy length:
#   -  30s → 2584 frame → 6 chunk of 5s
#   -  5s  → 431 frame  → 1 chunk of 5s
#   -  10s → 862 frame  → 2 chunk of 5s
#   - ecc.
#
# SPLIT-LESS layout (produced by preprocess_stream.py): the dataset mirrors the
# source class tree and has NO train/val/test directories on disk. The split is
# decided IN CODE (compute_split below): stratified by class, grouped by source
# file (leakage-safe), deterministic from a seed, with the test set persisted to
# <dataset_root>/../splits/test_split_<hash>.json. This module also hosts the
# split machinery so the conditioned dataset (audio_dataset_cond.py) can reuse it
# without a circular import.
#
#   dataset_root/
#       latents/<class...>/*.npy   ← shape (72, T), dtype float32

import json
import random
import hashlib
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Dict


# ============================================================
# CONSTANTS
# ============================================================
DAC_SAMPLE_RATE  = 44100
DAC_LATENT_DIM   = 72      # 9 codebook * 8 = latents DAC PRE-quantizzazione (continui).
                           # Lo z del decoder e' 1024-d: ci si torna con
                           # quantizer.from_latents() dentro decode_latents().
DAC_HOP_LENGTH   = 512
DAC_FRAMES_PER_S = DAC_SAMPLE_RATE / DAC_HOP_LENGTH   # ~86.13

# Upper bound for RoPE precomputation in the network.
# It does not limit the real length of the files — it pre-allocates
# positional frequencies in the transformer.
MAX_FRAMES = 4096

SUPPORTED_EXTS   = {".npy"}


def frames_per_chunk(latent_root, duration_s: float) -> int:
    """
    The latent length (frames per chunk) to use for sub-chunking.

    Prefers the REAL DAC frame count recorded by preprocess_stream.py in
    dataset_meta.json (`latent_frames_per_chunk`, e.g. 431 for 5 s), so training
    uses the exact geometry of the latents on disk instead of the truncating
    estimate int(duration_s * DAC_FRAMES_PER_S) (which gives 430 and drops one
    frame). Falls back to that estimate for legacy datasets without the field.

    `duration_s` (model.duration_s) is CHECKED against the dataset's own chunk
    length rather than ignored:
      * equal            -> the recorded real T (the normal case);
      * shorter          -> the estimate: training on sub-windows of each chunk;
      * longer           -> hard error: the latents simply do not contain that
                            much audio, and silently training on 5 s while the
                            config says 10 s would be invisible in every log.
    """
    meta_path = Path(latent_root).parent / "dataset_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    real_T = meta.get("latent_frames_per_chunk")
    chunk_s = meta.get("chunk_duration_s")

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
        # duration_s < chunk_duration_s -> deliberate sub-window training
        return int(duration_s * DAC_FRAMES_PER_S)
    return int(duration_s * DAC_FRAMES_PER_S)

# ============================================================
# LAZY DAC LOADER
# ============================================================
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


# ============================================================
# DECODING
# ============================================================
@torch.no_grad()
def decode_latents(latents: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    # latents pre-quant di DAC (9*8 = 72 dim, continui) -> waveform.
    # Il decoder DAC accetta SOLO lo z quantizzato a 1024-d, quindi proiettiamo e
    # quantizziamo i 72-d in z con quantizer.from_latents() -- esattamente cio' che
    # DAC fa internamente quando codifica audio vero.
    # from_latents ritorna (z_q, z_p, codes); ci serve z_q (1024-d).
    model = get_dac_model(device)
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)                      # (1, 72, T)
    latents = latents.to(device)
    z_q, _, _ = model.quantizer.from_latents(latents)       # (1, 1024, T)
    waveform = model.decode(z_q)
    return waveform.squeeze(0)


# ============================================================
# NORMALIZER
# ============================================================

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
    ):
        """
        Compute mean and std per-channel with parallel Welford batched.

        Improvements vs naive version:
          - Single-pass (not two: it uses Welford online)
          - Accelerated GPU (float64 for stability)
          - Cache to avoid multiple readings of the same file
          - Batch accumulation before updating the stats
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        n_chunks = len(chunks)
        print(f"[Normalizer] Welford batched on {n_chunks} chunk "
              f"(device={device}, batch_accum={batch_accum})...")

        from tqdm import tqdm

        mean_acc = None   # (dim, 1) float64
        m2_acc   = None   # (dim, 1) float64
        n_total  = 0

        buffer = []

        for i, (path, start) in enumerate(tqdm(chunks, desc="Normalizer fit")):
            # Read only the necessary chunk, without caching (avoid OOM)
            z_arr = np.load(str(path), mmap_mode='r')[:, start:start + n_frames]
            z = torch.from_numpy(np.ascontiguousarray(z_arr).astype(np.float32))

            if z.shape[1] != n_frames:
                continue  # skip short chunks 

            buffer.append(z)

            # Flush with batch_accum or end of dataset
            if len(buffer) >= batch_accum or i == n_chunks - 1:
                # Concatena sul tempo: (dim, batch_accum * n_frames)
                batch = torch.cat(buffer, dim=1).to(device=device, dtype=torch.float64)
                buffer = []

                n_new = batch.shape[1]

                if mean_acc is None:
                    dim = batch.shape[0]
                    mean_acc = torch.zeros(dim, 1, dtype=torch.float64, device=device)
                    m2_acc   = torch.zeros(dim, 1, dtype=torch.float64, device=device)

                # Welford parallel (Chan et al., 1979)
                n_total_new = n_total + n_new
                batch_mean  = batch.mean(dim=1, keepdim=True)
                delta       = batch_mean - mean_acc
                mean_acc    = mean_acc + delta * (n_new / n_total_new)
                batch_m2    = ((batch - batch_mean) ** 2).sum(dim=1, keepdim=True)
                m2_acc      = m2_acc + batch_m2 + (delta ** 2) * (n_total * n_new / n_total_new)
                n_total     = n_total_new

        var = (m2_acc / n_total).float().cpu()
        self.mean = mean_acc.float().cpu()
        self.std  = (var + 1e-6).sqrt()

        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"[Normalizer] mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
        print(f"[Normalizer] std range:  [{self.std.min():.3f}, {self.std.max():.3f}]")

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None, "Call fit_from_chunks() before normalize()"
        return (z - self.mean.to(z.device)) / self.std.to(z.device)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def save(self, path: str):
        torch.save({"mean": self.mean, "std": self.std}, path)
        print(f"[Normalizer] saved in {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.mean = ckpt["mean"]
        self.std  = ckpt["std"]
        print(f"[Normalizer] loaded from {path}")


# ============================================================
# SPLIT (stratified by class, grouped by source, deterministic,
# persisted test) -- shared with audio_dataset_cond.py
# ============================================================
def _class_of_file(npy_path: Path) -> str:
    """Class label = the leaf directory containing the latent (matches the class
    folders of the raw dataset and the leaf key used for global conditions)."""
    return npy_path.parent.name


def _source_group_of(npy_path: Path, latent_root: Path) -> str:
    """
    Leakage-safe source id: the file's directory (relative to latents/) plus the
    source stem, with the channel/chunk suffix stripped. All chunks and BOTH
    stereo channels of one source share this key, so they never split apart.
    """
    rel_parent = npy_path.parent.relative_to(latent_root)
    stem = npy_path.stem.split("__")[0]          # before __ch{n}/__c{idx}
    return (rel_parent / stem).as_posix()


def _split_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _seed_for(seed: int, key: str) -> int:
    """Deterministic per-class RNG seed (independent of PYTHONHASHSEED)."""
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h, 16) % (2 ** 32)


def _allocate_three(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    Split n groups into (train, val, test) by ratios, with small-class guards:
    guarantee >=1 in test (then val) when the ratio is > 0 and there are enough
    groups; n==1 -> all train (cannot hold out).
    """
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
    if n_tr < 0:                                  # tiny class over-allocated
        over = -n_tr
        take = min(over, n_val); n_val -= take; over -= take
        if over > 0:
            n_te -= over
        n_tr = n - n_te - n_val
    return (n_tr, n_val, n_te)


def _split_two(remaining: List[str], r_tr: float, r_val: float) -> Tuple[List[str], List[str]]:
    """Split the non-test groups into (train, val), renormalizing tr:val."""
    n = len(remaining)
    if n == 0:
        return [], []
    denom = r_tr + r_val
    n_val = 0 if denom <= 0 else round((r_val / denom) * n)
    if r_val > 0 and n_val == 0 and n >= 2:
        n_val = 1
    n_val = min(n_val, n)
    return remaining[n_val:], remaining[:n_val]


def compute_split(
    latent_root,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    group_by_source: bool = True,
    stratify_by_class: bool = True,
    save_test_manifest: bool = True,
    manifest_dir: Optional[str] = None,
) -> dict:
    """
    Returns:
        {
          "splits":  {"train": [Path...], "val": [Path...], "test": [Path...]},
          "classes": [sorted class names],
          "file_counts": {"train": n, "val": n, "test": n},
          "manifest_path": str or None,
          "params": {...},
        }
    """
    latent_root = Path(latent_root)
    all_files = sorted(latent_root.rglob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"No .npy latents under {latent_root}")

    # group files (leakage-safe unit)
    groups: Dict[str, dict] = {}
    for f in all_files:
        gk = _source_group_of(f, latent_root) if group_by_source else f.as_posix()
        g = groups.setdefault(gk, {"class": _class_of_file(f), "files": []})
        g["files"].append(f)

    classes = sorted({g["class"] for g in groups.values()})

    # stratification buckets: per-class, or one global bucket
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

    # honor a persisted test set if present
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

    # persist the test set (once) so it is stable across dataset growth / re-runs
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

    # Report #3: a class with a single source group goes entirely to train
    # (leakage-safe), so on tiny/imbalanced datasets val or test can end up empty
    # even though their ratio is > 0 -> a later val_dataset[0] would IndexError.
    # Fail early and clearly, but allow an intentionally-zero ratio.
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


def _chunks_from_files(files: List[Path], n_frames: int) -> List[Tuple[Path, int]]:
    """Build (path, start) sub-chunks for the normalizer, from a file list."""
    chunks = []
    for f in files:
        try:
            file_frames = np.load(str(f), mmap_mode="r").shape[1]
        except Exception:
            continue
        for k in range(file_frames // n_frames):
            chunks.append((f, k * n_frames))
    return chunks


# ============================================================
# DATASET
# ============================================================

class AudioLatentDataset(Dataset):
    """
    Dataset that loads chunks from a given list of .npy latent files (one split).
    Self-detection of the file length (assumes uniform duration). No patching:
    every DAC frame is a token. label_to_idx is GLOBAL (shared across splits).
    """

    def __init__(
        self,
        files:        List[Path],
        label_to_idx: Dict[str, int],
        split:        str   = "train",
        latent_root:  Optional[str] = None,
        duration_s:   float = 5.0,
        normalizer:   Optional[LatentNormalizer] = None,
        device:       str   = "cpu",
        preload:      bool  = False,   # default False to avoid OOM
    ):
        self.files       = [Path(f) for f in files]
        self.split       = split
        self.latent_root = Path(latent_root) if latent_root else None
        self.normalizer  = normalizer
        self.duration_s  = duration_s
        self.preload     = preload

        # Number of frames per chunk
        self.n_frames = frames_per_chunk(latent_root, duration_s)

        # GLOBAL label mapping (identical across train/val/test)
        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}

        # Every sample: (npy_path, start_frame, label_idx)
        self.samples: List[Tuple[Path, int, int]] = []
        self._actual_file_frames = None  # self-detected

        self._build_samples()

        # Per-file dense cache, only if preload=True: {npy_path_str: tensor (72, T)}.
        self._cache: dict = {}
        if preload:
            self._preload_all()

        chunks_per_file = self._actual_file_frames // self.n_frames if self._actual_file_frames else "?"
        print(f"[Dataset/{split}] duration_s={duration_s}s → "
              f"n_frames={self.n_frames} (= token sequence) | "
              f"token_dim={DAC_LATENT_DIM} | "
              f"file_frames={self._actual_file_frames} | "
              f"chunks per file={chunks_per_file} | "
              f"tot samples={len(self.samples)} | "
              f"preload={'ON' if preload else 'OFF'}")

    def _detect_file_frames(self) -> int:
        """Detect the frame count from the first readable .npy in the file list."""
        for f in self.files:
            if f.suffix.lower() in SUPPORTED_EXTS:
                z = np.load(str(f), mmap_mode='r')
                n_frames = z.shape[1]
                print(f"[Dataset/{self.split}] Self-detected: {n_frames} frame per file "
                      f"({n_frames / DAC_FRAMES_PER_S:.1f}s) from {f.name}")
                return n_frames
        raise FileNotFoundError(f"No .npy file in the {self.split} split file list")

    def _build_samples(self):
        """Build (npy_path, start, label_idx) samples from the file list."""
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
                continue   # class not in the global mapping (should not happen)
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
        """Default low-RAM path: memory-map the .npy (float32 on disk), read ONLY
        the requested chunk, then RELEASE the mmap so its file descriptor is
        closed immediately. The OS page cache still caches the file content
        (shared and reclaimable), so resident RAM stays low even when the dataset
        does not fit in memory (e.g. museart), with NO loss of precision, while
        open descriptors stay near zero. Caching the mmap instead (one live handle
        per distinct file) leaks one fd per file and makes large one-chunk-per-file
        datasets (e.g. birds/instrumental) hit 'Too many open files' (Errno 24)."""
        arr = np.load(str(npy_path), mmap_mode="r")    # float32 on disk, lazy paging
        try:
            # np.array(..., copy=True) materialises an INDEPENDENT contiguous copy
            # of just the slice, so it stays valid after the mmap is closed.
            sl = np.array(arr[:, start : start + self.n_frames], dtype=np.float32)
        finally:
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()                             # release the fd deterministically
            del arr
        return torch.from_numpy(sl)

    def _preload_all(self):
        """Optional DENSE preload, in FLOAT32, of every unique .npy into RAM.
        Use only when the whole dataset comfortably fits in RAM (small datasets);
        otherwise keep preload=False and rely on the mmap path above."""
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
                f"expected shape ({self.n_frames}, {DAC_LATENT_DIM}), obtained {tuple(z.shape)}. "
                f"The file has less frames than expected."
            )

        return z, label_idx


# ============================================================
# BUILD DATASETS
# ============================================================

def build_datasets(
    latent_root:     str,
    duration_s:      float = 5.0,
    device:          str   = "cpu",
    normalizer_path: Optional[str] = None,
    preload:         bool  = False,
    # ---- split configuration (leakage-safe defaults) ----
    split_ratios:    Tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_seed:      int = 42,
    group_by_source: bool = True,
    stratify_by_class: bool = True,
    save_test_manifest: bool = True,
):
    """
    Split-less unconditional dataset builder. Computes the split (stratified,
    source-grouped, seeded, with a persisted test manifest) and the normalizer.

    NOTE: the return shape changed from the old 4-tuple to
        (train, val, test, normalizer, label_to_idx, split_info)
    to match build_conditioned_datasets. Update any uncond caller accordingly.
    """
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
        chunks = _chunks_from_files(splits["train"], n_frames)
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


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    import sys

    root       = sys.argv[1] if len(sys.argv) > 1 else "./dataset_ready_cond/latents"
    duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    norm_path  = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Test AudioLatentDataset on: {root}")
    print(f"duration_s={duration_s}s | normalizer_path={norm_path}\n")

    train_dataset, val_dataset, test_dataset, normalizer, label_map, split_info = \
        build_datasets(latent_root=root, duration_s=duration_s,
                       normalizer_path=norm_path, preload=False)

    if norm_path is None:
        import os
        os.makedirs("checkpoints_v2", exist_ok=True)
        normalizer.save("checkpoints_v2/normalizer.pt")

    print(f"\nSplit: {split_info['file_counts']} files | "
          f"{split_info['chunk_counts']} chunks | {split_info['n_classes']} classes")
    sample, label = train_dataset[0]
    print(f"\nSingle sample:")
    print(f"  shape   : {sample.shape}  (n_frames, token_dim)")
    print(f"  label   : {label} ({train_dataset.idx_to_label[label]})")
    print(f"  Mean    : {sample.mean():.4f}")
    print(f"  Std     : {sample.std():.4f}")