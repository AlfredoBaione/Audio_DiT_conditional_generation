# audio_dataset_cond.py
#
# Multi-modal conditioned dataset for training the ConditionedAudioDiT.
#
# Loads for each sample:
#   - frames:       (n_frames, 1024)  normalized DAC latents
#   - frame_conds:  {melody, chroma, rhythm, ...} pre-extracted conditions
#   - label_idx:    int                 class index
#   - text_emb:     (text_dim,)         sentence-transformer embedding (pre-computed)
#   - image_emb:    (image_dim,)        CLIP embedding (pre-computed)
#
# Text and image embeddings are pre-computed at init.
# Frame-level conditions are loaded from .npz.
#
# Compatible with the dataset structure:
#   dataset_root/
#       latents/train|val|test/class/*.npy
#       wav/train|val|test/class/*.wav         <- val/test used for FAD
#       conditions/train|val|test/class/*.npz  <- melody, chroma, rhythm, ...

import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_dataset_npy import (
    LatentNormalizer, DAC_FRAMES_PER_S, SUPPORTED_EXTS,
    AudioLatentDataset,
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
    Multi-modal dataset for conditioned training.

    For each sample returns:
        frames:      (n_frames, 1024)
        frame_conds: Dict[str, Tensor] — e.g. {"melody": (n_frames, 88), "chroma": (n_frames, 12), "rhythm": (n_frames, 2)}
        label_idx:   int
        text_emb:    (text_dim,) — embedding of the class name
        image_emb:   (image_dim,) — random image embedding of the class
    """

    def __init__(
        self,
        latent_root:     str,
        condition_root:  Optional[str] = None,
        image_root:      Optional[str] = None,
        split:           str   = "train",
        duration_s:      float = 5.0,
        normalizer:      Optional[LatentNormalizer] = None,
        registry:        Optional[ConditionRegistry] = None,
        image_manager:   Optional[ImageDatasetManager] = None,
        preload_latents: bool  = True,
    ):
        self.latent_root    = Path(latent_root)
        self.condition_root = Path(condition_root) if condition_root else None
        self.split          = split
        self.normalizer     = normalizer
        self.duration_s     = duration_s
        self.registry       = registry
        self.image_manager  = image_manager
        self.preload_latents = preload_latents

        self.n_frames = int(duration_s * DAC_FRAMES_PER_S)

        # (npy_path, cond_path, start, label_idx, class_name)
        self.samples: List[Tuple[Path, Optional[Path], int, int, str]] = []
        self.label_to_idx: dict = {}
        self.idx_to_label: dict = {}
        self._actual_file_frames = None

        self._scan()

        # Latent cache in fp16
        self._latent_cache: dict = {}
        if preload_latents:
            self._preload_latents()

        # Pre-compute text embeddings (one per class)
        self._text_embeddings: Dict[str, np.ndarray] = {}
        self._text_dim: int = 0
        self._precompute_text_embeddings()

        # Pre-compute image embeddings (N per class, random choice at getitem)
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

    def _detect_file_frames(self, split_dir: Path) -> int:
        for d in sorted(split_dir.iterdir()):
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    z = np.load(str(f), mmap_mode='r')
                    return z.shape[1]
        raise FileNotFoundError(f"No .npy in {split_dir}")

    def _scan(self):
        split_dir = self.latent_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Not found: {split_dir}")

        self._actual_file_frames = self._detect_file_frames(split_dir)

        label_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.label_to_idx = {d.name: i for i, d in enumerate(label_dirs)}
        self.idx_to_label = {i: d.name for i, d in enumerate(label_dirs)}

        # Controllo di riferimento
        n_chunks_ref = self._actual_file_frames // self.n_frames
        if n_chunks_ref == 0:
            raise ValueError(
                f"File has {self._actual_file_frames} frames, "
                f"duration_s={self.duration_s}s requires {self.n_frames} frames"
            )

        # Per-file scan: check the actual length of each file
        n_files_total = 0
        n_files_short = 0

        for label_dir in label_dirs:
            label_idx = self.label_to_idx[label_dir.name]
            class_name = label_dir.name

            for f in sorted(label_dir.iterdir()):
                if f.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                n_files_total += 1

                # Read only the shape without loading into RAM
                try:
                    file_frames = np.load(str(f), mmap_mode='r').shape[1]
                except Exception:
                    continue

                n_chunks_file = file_frames // self.n_frames
                if n_chunks_file == 0:
                    n_files_short += 1
                    continue

                # Path corrispondente nel cond_root
                cond_path = None
                if self.condition_root:
                    rel = f.relative_to(self.latent_root).with_suffix(".npz")
                    cand = self.condition_root / rel
                    if cand.exists():
                        cond_path = cand

                for k in range(n_chunks_file):
                    start = k * self.n_frames
                    if start + self.n_frames <= file_frames:
                        self.samples.append((f, cond_path, start, label_idx, class_name))

        if n_files_short > 0:
            print(f"[CondDataset/{self.split}] WARNING: {n_files_short}/{n_files_total} "
                  f"files too short for {self.n_frames} frames -> skipped")

    def _preload_latents(self):
        unique = set(str(p) for p, _, _, _, _ in self.samples)
        print(f"[CondDataset/{self.split}] Preloading {len(unique)} latents (fp16)...")
        from tqdm import tqdm
        for p in tqdm(sorted(unique), desc=f"Preload {self.split}"):
            z = np.load(p)
            self._latent_cache[p] = torch.from_numpy(z.astype(np.float16))
        gb = sum(t.nelement() * 2 for t in self._latent_cache.values()) / 1e9
        print(f"[CondDataset/{self.split}] {gb:.2f} GB in RAM")

    def _precompute_text_embeddings(self):
        if self.registry is None or "text" not in self.registry.global_extractors:
            return
        text_ext: CLAPTextCondition = self.registry.global_extractors["text"]
        self._text_dim = text_ext.dim

        # Build the prompts: class name -> readable text
        # (a inferenza si possono passare prompt liberi; qui partiamo
        # from the class for the first training)
        class_names = list(self.label_to_idx.keys())
        prompts = [c.replace("_", " ") for c in class_names]

        # Batch-encode all prompts at once (more efficient)
        embs = text_ext.encode_batch(prompts)   # (n_classes, dim)
        for class_name, emb in zip(class_names, embs):
            self._text_embeddings[class_name] = emb

        print(f"[CondDataset/{self.split}] CLAP text embeddings: "
              f"{len(self._text_embeddings)} classes (dim={self._text_dim})")

        # CLAP is large (~300MB on GPU): offload after pre-computing.
        # The embeddings are cached as np.ndarray; the model is no longer needed.
        text_ext.unload()

    def _precompute_image_embeddings(self, max_per_class: int = 10):
        if (self.registry is None
                or "image" not in self.registry.global_extractors
                or self.image_manager is None):
            return

        img_ext: ImageCondition = self.registry.global_extractors["image"]
        self._image_dim = img_ext.dim

        orphan_classes = []
        for class_name in self.label_to_idx.keys():
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
              f"{total} images in {len(self._image_embeddings)}/{len(self.label_to_idx)} classes "
              f"(dim={self._image_dim})")
        if orphan_classes:
            print(f"[CondDataset/{self.split}] WARNING: {len(orphan_classes)} classes without "
                  f"images -> will use null fallback (image=zeros): {orphan_classes}")

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

        if self.normalizer:
            z = self.normalizer.normalize(z)
        frames = z.T  # (n_frames, 1024)

        # Controllo di sicurezza sulla shape finale
        if frames.shape[0] != self.n_frames:
            raise RuntimeError(
                f"Sample {npy_path.name} @ start={start}: "
                f"expected shape ({self.n_frames}, 1024), got {tuple(frames.shape)}. "
                f"The file has fewer frames than expected."
            )

        # 2. FRAME CONDITIONS (da .npz)
        frame_cond = {}
        frame_names = self._get_frame_names()

        if cond_path is not None and frame_names:
            try:
                data = np.load(str(cond_path))
                for name in frame_names:
                    if name in data:
                        c = data[name].astype(np.float32)
                        c = c[start:start + self.n_frames]
                        # Padding se troppo corto
                        if c.shape[0] < self.n_frames:
                            pad = np.zeros((self.n_frames - c.shape[0], c.shape[1]),
                                            dtype=np.float32)
                            c = np.concatenate([c, pad], axis=0)
                        frame_cond[name] = torch.from_numpy(c)
            except Exception:
                pass

        # Zero fallback for missing conditions
        for name in frame_names:
            if name not in frame_cond:
                dim = self.registry.frame_cond_dims[name]
                frame_cond[name] = torch.zeros(self.n_frames, dim)

        # 3. TEXT EMBEDDING
        if class_name in self._text_embeddings:
            text_emb = torch.from_numpy(self._text_embeddings[class_name])
        elif self._text_dim > 0:
            text_emb = torch.zeros(self._text_dim)
        else:
            text_emb = torch.zeros(1)  # placeholder

        # 4. IMAGE EMBEDDING
        # train: random among the pre-computed ones (effectively data augmentation)
        # val/test: always the first -> deterministic val loss
        if class_name in self._image_embeddings:
            if self.split == "train":
                img_emb = torch.from_numpy(
                    random.choice(self._image_embeddings[class_name])
                )
            else:
                img_emb = torch.from_numpy(self._image_embeddings[class_name][0])
        elif self._image_dim > 0:
            img_emb = torch.zeros(self._image_dim)
        else:
            img_emb = torch.zeros(1)  # placeholder

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

    # Frame conditions: stack per key
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
) -> Tuple[ConditionedAudioDataset, ConditionedAudioDataset, LatentNormalizer, dict]:
    """
    Builds the conditioned train + val datasets, computing the normalizer
    if it does not exist.
    """
    # Normalizer
    normalizer = LatentNormalizer()
    if normalizer_path and Path(normalizer_path).exists():
        normalizer.load(normalizer_path)
    else:
        print("[build_conditioned_datasets] Computing normalizer on the train set...")
        raw = AudioLatentDataset(
            root_dir=latent_root, split="train",
            duration_s=duration_s, normalizer=None, preload=False,
        )
        chunks = raw.get_chunks_for_normalizer()
        normalizer.fit_from_chunks(chunks, n_frames=raw.n_frames)

    # Image manager: one per split (respects image_root/{train,val}/<class>/)
    train_image_mgr = None
    val_image_mgr = None
    if image_root and Path(image_root).exists():
        train_image_mgr = ImageDatasetManager(image_root, split="train")
        val_image_mgr = ImageDatasetManager(image_root, split="val")

    common_kwargs = dict(
        condition_root=condition_root,
        image_root=image_root,
        duration_s=duration_s,
        normalizer=normalizer,
        registry=registry,
    )

    train = ConditionedAudioDataset(
        latent_root=latent_root, split="train",
        image_manager=train_image_mgr,
        preload_latents=preload, **common_kwargs,
    )
    val = ConditionedAudioDataset(
        latent_root=latent_root, split="val",
        image_manager=val_image_mgr,
        preload_latents=False, **common_kwargs,
    )

    print(f"[build_conditioned_datasets] Train: {len(train)} | Val: {len(val)}")

    return train, val, normalizer, train.label_to_idx
