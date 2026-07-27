# audio_dataset_cond.py
#
# Multi-modal conditioned dataset for training the ConditionedAudioDiT.
#
# Loads for each sample:
#   - frames:       (n_frames, 72)   normalized DAC pre-quantizer latents
#   - frame_conds:  {melody, chroma, rhythm, ...} pre-extracted conditions
#   - label_idx:    int                 class index
#   - text_emb:     (text_dim,)         CLAP text embedding (pre-computed)
#   - image_emb:    (image_dim,)        CLIP embedding (pre-computed)
#
# ------------------------------------------------------------------------------
# NEW on-disk contract (produced by preprocess_stream.py): the dataset is
# SPLIT-LESS on disk and mirrors the source class tree:
#
#   dataset_root/
#       latents/<class...>/*.npy       <- (72, T) float32 pre-quant DAC latents
#       conditions/<class...>/*.npz    <- melody, chroma, rhythm, energy, f0, ...
#       wav/<class...>/*.wav           <- optional (val/test used for FAD)
#       splits/test_split_<hash>.json  <- persisted test set (written here)
#       dataset_meta.json              <- chunk/acoustic params
#
# The train/val/test split is now decided IN CODE (compute_split below), not by
# on-disk directories, because the split is a training-time concern:
#   * STRATIFIED by class (each class contributes to every split by ratio);
#   * GROUPED by source file (all chunks AND both stereo channels of one source
#     go to the SAME split) -> no train/test leakage across chunks of a track;
#   * DETERMINISTIC from a seed;
#   * the TEST set is persisted to splits/test_split_<hash>.json so every run on
#     the same dataset+params reuses the same test set (comparable conditioned
#     generations across checkpoints); train/val are re-derived deterministically.
#
# Chunk file names are `<stem>[__ch{n}]__c{idx}.npy`; the source group key is the
# part before the first `__` (sanitize_filename never emits `__` inside a stem).

import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_dataset_npy import (
    LatentNormalizer, DAC_LATENT_DIM, SUPPORTED_EXTS,
    frames_per_chunk,
    # split machinery lives in the base module to avoid a circular import;
    # re-exported here so callers can `from audio_dataset_cond import compute_split`.
    compute_split, _class_of_file, _chunks_from_files,
)
from conditions import (
    ConditionRegistry, ImageDatasetManager,
    CLAPTextCondition, ImageCondition,
)


# ============================================================
# CONDITIONED DATASET
# ============================================================
class ConditionedAudioDataset(Dataset):
    """
    Multi-modal dataset for conditioned training. Built from an explicit list of
    latent files (one split), with a GLOBAL label_to_idx shared across splits.

    For each sample returns:
        frames:      (n_frames, 72)
        frame_conds: Dict[str, Tensor] — e.g. {"melody": (n_frames, 88), ...}
        label_idx:   int
        text_emb:    (text_dim,) — embedding of the class name
        image_emb:   (image_dim,) — random image embedding of the class
    """

    def __init__(
        self,
        files:           List[Path],
        label_to_idx:    Dict[str, int],
        split:           str,
        latent_root:     str,
        condition_root:  Optional[str] = None,
        image_root:      Optional[str] = None,
        duration_s:      float = 5.0,
        normalizer:      Optional[LatentNormalizer] = None,
        registry:        Optional[ConditionRegistry] = None,
        image_manager:   Optional[ImageDatasetManager] = None,
        preload_latents: bool  = True,
        strict_conditions: bool = True,
    ):
        self.files          = [Path(f) for f in files]
        self.latent_root    = Path(latent_root)
        self.condition_root = Path(condition_root) if condition_root else None
        self.split          = split
        self.normalizer     = normalizer
        self.duration_s     = duration_s
        self.registry       = registry
        self.image_manager  = image_manager
        self.preload_latents = preload_latents
        self.strict_conditions = strict_conditions
        self._cond_warned = False

        self.n_frames = frames_per_chunk(latent_root, duration_s)

        # GLOBAL label mapping (identical across train/val/test)
        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}

        # (npy_path, cond_path, start, label_idx, class_name)
        self.samples: List[Tuple[Path, Optional[Path], int, int, str]] = []
        self._actual_file_frames = None
        self._present_classes: List[str] = []

        self._build_samples()

        self._latent_cache: dict = {}
        if preload_latents:
            self._preload_latents()

        self._text_embeddings: Dict[str, np.ndarray] = {}
        self._text_dim: int = 0
        self._precompute_text_embeddings()

        self._image_embeddings: Dict[str, List[np.ndarray]] = {}
        self._image_dim: int = 0
        self._precompute_image_embeddings()

        self._print_summary()

    def _print_summary(self):
        has_frame_conds = self.condition_root is not None and len(self._get_frame_names()) > 0
        print(f"[CondDataset/{self.split}] {len(self.samples)} samples | "
              f"n_frames={self.n_frames} | "
              f"frame_conds={'ON' if has_frame_conds else 'OFF'} ({self._get_frame_names()}) | "
              f"text={'ON' if self._text_embeddings else 'OFF'} | "
              f"image={'ON' if self._image_embeddings else 'OFF'}")

    def _get_frame_names(self) -> List[str]:
        if self.registry is None:
            return []
        return self.registry.frame_names

    def _build_samples(self):
        if not self.files:
            print(f"[CondDataset/{self.split}] WARNING: no files for this split")
            return

        self._actual_file_frames = np.load(str(self.files[0]), mmap_mode="r").shape[1]

        n_chunks_ref = self._actual_file_frames // self.n_frames
        if n_chunks_ref == 0:
            raise ValueError(
                f"File has {self._actual_file_frames} frames, "
                f"duration_s={self.duration_s}s requires {self.n_frames} frames")

        n_files_total = 0
        n_files_short = 0
        n_files_unreadable = 0
        n_files_no_cond = 0
        want_cond = (self.condition_root is not None and len(self._get_frame_names()) > 0)
        present = set()

        for f in self.files:
            if f.suffix.lower() not in SUPPORTED_EXTS:
                continue
            n_files_total += 1
            class_name = _class_of_file(f)
            label_idx = self.label_to_idx.get(class_name)
            if label_idx is None:
                # class not in the global mapping (should not happen): skip safely
                continue

            try:
                file_frames = np.load(str(f), mmap_mode="r").shape[1]
            except Exception as e:
                n_files_unreadable += 1
                print(f"[CondDataset/{self.split}] WARNING: unreadable latent "
                      f"{f.name} ({type(e).__name__}: {e}) -> skipped")
                continue

            n_chunks_file = file_frames // self.n_frames
            if n_chunks_file == 0:
                n_files_short += 1
                continue

            cond_path = None
            if self.condition_root:
                rel = f.relative_to(self.latent_root).with_suffix(".npz")
                cand = self.condition_root / rel
                if cand.exists():
                    cond_path = cand
                elif want_cond:
                    n_files_no_cond += 1

            for k in range(n_chunks_file):
                start = k * self.n_frames
                if start + self.n_frames <= file_frames:
                    self.samples.append((f, cond_path, start, label_idx, class_name))
                    present.add(class_name)

        self._present_classes = sorted(present)

        if n_files_short > 0:
            print(f"[CondDataset/{self.split}] WARNING: {n_files_short}/{n_files_total} "
                  f"files too short for {self.n_frames} frames -> skipped")

        # U4 guard: latent files whose .npz of conditions is entirely missing.
        if want_cond and n_files_no_cond > 0:
            msg = (f"[CondDataset/{self.split}] {n_files_no_cond}/{n_files_total} latent "
                   f"files have NO corresponding .npz under {self.condition_root}. "
                   f"Re-run preprocess_stream.py / extract_conditions.py with "
                   f"--conditions {','.join(self._get_frame_names())}")
            if self.strict_conditions:
                raise RuntimeError(
                    msg + "\n(strict_conditions=True: refusing to train with samples "
                          "that would fall back to NULL conditions. Set "
                          "training.strict_conditions=false to allow it.)")
            print("[CondDataset/" + self.split + "] WARNING: " + msg
                  + " -> these samples will use NULL (zero) conditions.")

    def _warn_cond_once(self, msg: str):
        if not self._cond_warned:
            print(f"[CondDataset/{self.split}] WARNING (non-strict): {msg} "
                  f"(further condition warnings suppressed)")
            self._cond_warned = True

    def _preload_latents(self):
        unique = set(str(p) for p, _, _, _, _ in self.samples)
        print(f"[CondDataset/{self.split}] Preloading {len(unique)} latents (fp32)...")
        from tqdm import tqdm
        for p in tqdm(sorted(unique), desc=f"Preload {self.split}"):
            z = np.load(p)
            self._latent_cache[p] = torch.from_numpy(z.astype(np.float32))
        gb = sum(t.nelement() * 4 for t in self._latent_cache.values()) / 1e9
        print(f"[CondDataset/{self.split}] {gb:.2f} GB in RAM")

    def _precompute_text_embeddings(self):
        if self.registry is None or "text" not in self.registry.global_extractors:
            return
        text_ext: CLAPTextCondition = self.registry.global_extractors["text"]
        self._text_dim = text_ext.dim

        class_names = self._present_classes or list(self.label_to_idx.keys())
        prompts = [c.replace("_", " ") for c in class_names]

        embs = text_ext.encode_batch(prompts)   # (n_classes, dim)
        for class_name, emb in zip(class_names, embs):
            self._text_embeddings[class_name] = emb

        print(f"[CondDataset/{self.split}] CLAP text embeddings: "
              f"{len(self._text_embeddings)} classes (dim={self._text_dim})")
        text_ext.unload()

    def _precompute_image_embeddings(self, max_per_class: int = 10):
        if (self.registry is None
                or "image" not in self.registry.global_extractors
                or self.image_manager is None):
            return

        img_ext: ImageCondition = self.registry.global_extractors["image"]
        self._image_dim = img_ext.dim

        orphan_classes = []
        for class_name in (self._present_classes or list(self.label_to_idx.keys())):
            images = self.image_manager.get_all_images(class_name)[:max_per_class]
            embs = []
            for img_path in images:
                try:
                    embs.append(img_ext.encode_image(str(img_path)))
                except Exception as e:
                    print(f"  [WARN] Image not loaded {img_path}: {e}")
            if embs:
                self._image_embeddings[class_name] = embs
            else:
                orphan_classes.append(class_name)

        total = sum(len(v) for v in self._image_embeddings.values())
        print(f"[CondDataset/{self.split}] Image embeddings: "
              f"{total} images in {len(self._image_embeddings)}/{len(self._present_classes)} classes "
              f"(dim={self._image_dim})")
        if orphan_classes:
            print(f"[CondDataset/{self.split}] WARNING: {len(orphan_classes)} classes without "
                  f"images -> will use null fallback (image=zeros): {orphan_classes}")
        # CLIP is large: offload after pre-computing.
        if hasattr(img_ext, "unload"):
            img_ext.unload()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, cond_path, start, label_idx, class_name = self.samples[idx]

        # 1. LATENTI
        key = str(npy_path)
        if key in self._latent_cache:
            z = self._latent_cache[key][:, start:start + self.n_frames].float()
        else:
            z = torch.from_numpy(np.load(key).astype(np.float32))
            z = z[:, start:start + self.n_frames]

        # Validate the latent slice (report #17): shape (72, n_frames) and finite.
        # Cheap (~30k floats) and catches corrupt latents at the source rather than
        # letting NaN/Inf silently contaminate the normalizer / metrics / loss.
        if z.ndim != 2 or z.shape[0] != DAC_LATENT_DIM or z.shape[1] != self.n_frames:
            raise RuntimeError(
                f"Latent {npy_path.name} slice has shape {tuple(z.shape)}, "
                f"expected ({DAC_LATENT_DIM}, {self.n_frames}).")
        if not torch.isfinite(z).all():
            raise RuntimeError(f"Latent {npy_path.name} contains NaN/Inf.")

        if self.normalizer:
            z = self.normalizer.normalize(z)
        frames = z.T  # (n_frames, 72)

        if frames.shape[0] != self.n_frames:
            raise RuntimeError(
                f"Sample {npy_path.name} @ start={start}: "
                f"expected shape ({self.n_frames}, {DAC_LATENT_DIM}), got {tuple(frames.shape)}. "
                f"The file has fewer frames than expected.")

        # 2. FRAME CONDITIONS (da .npz)
        frame_cond = {}
        frame_names = self._get_frame_names()

        if cond_path is not None and frame_names:
            try:
                data = np.load(str(cond_path))
            except Exception as e:
                if self.strict_conditions:
                    raise RuntimeError(
                        f"Failed to load conditions {cond_path} "
                        f"({type(e).__name__}: {e}). Re-extract them or set "
                        f"training.strict_conditions=false.") from e
                self._warn_cond_once(f"load failed for {cond_path.name}: {e}")
                data = None
            if data is not None:
                for name in frame_names:
                    if name not in data:
                        continue
                    c = data[name].astype(np.float32)
                    expected_dim = self.registry.frame_cond_dims[name]

                    # Validate the RAW array (report #2): in strict mode a
                    # malformed/short/non-finite condition must FAIL, not be
                    # silently repaired with zero-padding.
                    if c.ndim != 2 or c.shape[1] != expected_dim:
                        if self.strict_conditions:
                            raise RuntimeError(
                                f"Condition '{name}' for {npy_path.name} has shape "
                                f"{c.shape}, expected (*, {expected_dim}). Re-extract "
                                f"it, or set training.strict_conditions=false.")
                        self._warn_cond_once(
                            f"'{name}' wrong shape {c.shape} -> zero-filled")
                        continue
                    if not np.isfinite(c).all():
                        if self.strict_conditions:
                            raise RuntimeError(
                                f"Condition '{name}' for {npy_path.name} contains "
                                f"NaN/Inf. Re-extract it, or set "
                                f"training.strict_conditions=false.")
                        self._warn_cond_once(f"'{name}' has NaN/Inf -> zero-filled")
                        continue

                    c = c[start:start + self.n_frames]
                    if c.shape[0] < self.n_frames:
                        if self.strict_conditions:
                            raise RuntimeError(
                                f"Condition '{name}' for {npy_path.name} @ start="
                                f"{start} yields {c.shape[0]} frames < n_frames="
                                f"{self.n_frames} (condition shorter than the latent). "
                                f"Re-extract it, or set training.strict_conditions="
                                f"false to zero-pad.")
                        pad = np.zeros((self.n_frames - c.shape[0], c.shape[1]),
                                        dtype=np.float32)
                        c = np.concatenate([c, pad], axis=0)
                    frame_cond[name] = torch.from_numpy(c)
        elif cond_path is None and frame_names and self.strict_conditions:
            raise RuntimeError(
                f"Sample {npy_path.name} @ start={start} has no conditions .npz "
                f"but the model requires {frame_names}. Re-extract conditions or "
                f"set training.strict_conditions=false.")

        for name in frame_names:
            if name not in frame_cond:
                if self.strict_conditions:
                    raise RuntimeError(
                        f"Condition '{name}' missing for {npy_path.name} "
                        f"(cond_path={cond_path}). The .npz does not contain this "
                        f"key. Re-run extraction for '{name}', or set "
                        f"training.strict_conditions=false to zero-fill it.")
                self._warn_cond_once(f"'{name}' missing -> zero-filled")
                dim = self.registry.frame_cond_dims[name]
                frame_cond[name] = torch.zeros(self.n_frames, dim)

        # 3. TEXT EMBEDDING
        if class_name in self._text_embeddings:
            text_emb = torch.from_numpy(self._text_embeddings[class_name])
        elif self._text_dim > 0:
            text_emb = torch.zeros(self._text_dim)
        else:
            text_emb = torch.zeros(1)

        # 4. IMAGE EMBEDDING (train: random -> augmentation; val/test: first -> deterministic)
        if class_name in self._image_embeddings:
            if self.split == "train":
                img_emb = torch.from_numpy(random.choice(self._image_embeddings[class_name]))
            else:
                img_emb = torch.from_numpy(self._image_embeddings[class_name][0])
        elif self._image_dim > 0:
            img_emb = torch.zeros(self._image_dim)
        else:
            img_emb = torch.zeros(1)

        return frames, frame_cond, label_idx, text_emb, img_emb


# ============================================================
# COLLATE
# ============================================================
def collate_conditioned(batch):
    """Custom collate for the DataLoader."""
    frames_l, conds_l, labels_l, text_l, image_l = zip(*batch)

    frames = torch.stack(frames_l)
    labels = torch.tensor(labels_l, dtype=torch.long)
    text_embs = torch.stack(text_l)
    image_embs = torch.stack(image_l)

    frame_conds = {}
    if conds_l and conds_l[0]:
        for name in conds_l[0].keys():
            frame_conds[name] = torch.stack([c[name] for c in conds_l])

    return frames, frame_conds, labels, text_embs, image_embs


# ============================================================
# BUILDER
# ============================================================
def build_conditioned_datasets(
    latent_root:     str,
    condition_root:  Optional[str] = None,
    image_root:      Optional[str] = None,
    duration_s:      float = 5.0,
    normalizer_path: Optional[str] = None,
    registry:        Optional[ConditionRegistry] = None,
    preload:         bool = True,
    strict_conditions: bool = True,
    # ---- split configuration (from cond_default.yaml: data.split) ----
    split_ratios:    Tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_seed:      int = 42,
    group_by_source: bool = True,
    stratify_by_class: bool = True,
    save_test_manifest: bool = True,
):
    """
    Builds the conditioned train/val/test datasets from the SPLIT-LESS dataset,
    computing the split (stratified, source-grouped, seeded) and the normalizer.

    Returns:
        (train, val, test, normalizer, label_to_idx, split_info)
    where split_info = {
        "file_counts": {train,val,test}, "chunk_counts": {train,val,test},
        "n_classes": int, "manifest_path": str|None, "params": {...},
    }
    """
    # 1. split (shared across train/val/test)
    split = compute_split(
        latent_root, ratios=split_ratios, seed=split_seed,
        group_by_source=group_by_source, stratify_by_class=stratify_by_class,
        save_test_manifest=save_test_manifest,
    )
    splits = split["splits"]
    classes = split["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}

    # 2. normalizer (fit on TRAIN files only)
    normalizer = LatentNormalizer()
    if normalizer_path and Path(normalizer_path).exists():
        normalizer.load(normalizer_path)
    else:
        print("[build_conditioned_datasets] Computing normalizer on the train split...")
        n_frames = frames_per_chunk(latent_root, duration_s)
        chunks = _chunks_from_files(splits["train"], n_frames)
        if not chunks:
            raise RuntimeError("No train chunks available to fit the normalizer.")
        normalizer.fit_from_chunks(chunks, n_frames=n_frames)

    # 3. image managers only if image conditioning is active
    image_active = (registry is not None
                    and "image" in getattr(registry, "global_extractors", {}))
    image_mgr = None
    if image_active and image_root and Path(image_root).exists():
        image_mgr = ImageDatasetManager(image_root, split=None)

    common = dict(
        label_to_idx=label_to_idx,
        latent_root=latent_root,
        condition_root=condition_root,
        image_root=image_root,
        duration_s=duration_s,
        normalizer=normalizer,
        registry=registry,
        image_manager=image_mgr,
        strict_conditions=strict_conditions,
    )

    train = ConditionedAudioDataset(files=splits["train"], split="train",
                                    preload_latents=preload, **common)
    val = ConditionedAudioDataset(files=splits["val"], split="val",
                                  preload_latents=False, **common)
    test = ConditionedAudioDataset(files=splits["test"], split="test",
                                   preload_latents=False, **common)

    split_info = {
        "file_counts": split["file_counts"],
        "chunk_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "n_classes": len(classes),
        "manifest_path": split["manifest_path"],
        "params": split["params"],
    }

    print(f"[build_conditioned_datasets] "
          f"Train: {len(train)} chunks ({split['file_counts']['train']} files) | "
          f"Val: {len(val)} ({split['file_counts']['val']}) | "
          f"Test: {len(test)} ({split['file_counts']['test']}) | "
          f"classes={len(classes)}")

    return train, val, test, normalizer, label_to_idx, split_info
