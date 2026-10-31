# Conditioned dataset: latents with their frame and global conditions.

import hashlib
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_utils import (
    LatentNormalizer, SUPPORTED_EXTS,
    frames_per_chunk,
    load_source_split, _class_of_file, _source_group_of, _chunks_from_files,
    _meta_latent_frames, NORMALIZER_MAX_CHUNKS,
)
from conditions import ConditionRegistry
import latent_codec as lc


TEXT_SOURCES = ("audio", "label", "text")


def load_text_table(latent_root):
    d = Path(latent_root).parent / "global_conditions"
    p = d / "text_table.json"
    if not p.exists():
        return {}
    table = json.loads(p.read_text(encoding="utf-8"))
    for kind in ("label", "text"):
        if table[f"{kind}s"]:
            table[f"{kind}_emb"] = np.load(str(d / f"{kind}_emb.npy"))
            table[f"{kind}_tok"] = np.load(str(d / f"{kind}_tok.npy"), mmap_mode="r")
            table[f"{kind}_tok_len"] = np.load(str(d / f"{kind}_tok_len.npy"))
    return table


class ConditionedAudioDataset(Dataset):
    def __init__(
        self,
        files:           List[Path],
        label_to_idx:    Dict[str, int],
        split:           str,
        latent_root:     str,
        condition_root:  Optional[str] = None,
        duration_s:      float = 5.0,
        normalizer:      Optional[LatentNormalizer] = None,
        registry:        Optional[ConditionRegistry] = None,
        preload_latents: bool  = True,
        strict_conditions: bool = True,
        text_source:     str   = "label",
        generation_text_source: Optional[str] = None,
        file_checks_dist=None,
    ):
        self.files          = [Path(f) for f in files]
        self._file_checks_dist = file_checks_dist
        self.latent_root    = Path(latent_root)
        self.condition_root = Path(condition_root) if condition_root else None
        self.split          = split
        self.normalizer     = normalizer
        self.duration_s     = duration_s
        self.registry       = registry
        self.preload_latents = preload_latents
        self.strict_conditions = strict_conditions
        self._cond_warned = False

        self.n_frames = frames_per_chunk(latent_root, duration_s)
        self.codec = lc.get_spec(lc.dataset_codec(latent_root))
        self.latent_dim = self.codec.latent_dim

        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}

        self.samples: List[Tuple[Path, Optional[Path], int, int, str]] = []
        self._actual_file_frames = None
        self._present_classes: List[str] = []

        self._build_samples()

        self._latent_cache: dict = {}
        if preload_latents:
            self._preload_latents()

        self.text_source = None
        self.generation_text_source = None
        self._text_table = {}
        self._text_dim = self._ctx_dim = self._ctx_len = 0
        if registry is not None and "text" in registry.global_extractors:
            self._setup_text(text_source, generation_text_source or text_source)

        self._image_embeddings: Dict[str, List[np.ndarray]] = {}
        self._image_files: Dict[str, List[str]] = {}
        self._image_dim: int = 0
        self._load_image_bank()

        self._print_summary()

    def _print_summary(self):
        has_frame_conds = self.condition_root is not None and len(self._get_frame_names()) > 0
        print(f"[CondDataset/{self.split}] {len(self.samples)} samples | "
              f"n_frames={self.n_frames} | "
              f"frame_conds={'ON' if has_frame_conds else 'OFF'} ({self._get_frame_names()}) | "
              f"text={'ON' if self._text_dim else 'OFF'} | "
              f"image={'ON' if self._image_embeddings else 'OFF'}")
        if self._text_dim:
            print(f"[CondDataset/{self.split}] text slot: {self.text_source}"
                  + (f" | generations: {self.generation_text_source}"
                     if self.generation_text_source != self.text_source else "")
                  + ("" if self.text_source == "audio" else
                     f" | {sum(not self._text_ids(f, self.text_source) for f in self.files)}"
                     f"/{len(self.files)} files without it -> null vector"))

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

        uniform = _meta_latent_frames(self.latent_root) if self.latent_root else None
        if uniform:
            probe = sorted({int(round(i)) for i in
                            np.linspace(0, len(self.files) - 1,
                                        min(64, len(self.files)))})
            for i in probe:
                try:
                    if np.load(str(self.files[i]), mmap_mode="r").shape[1] != uniform:
                        uniform = None
                        break
                except Exception:
                    uniform = None
                    break
            if uniform is None:
                print(f"[CondDataset/{self.split}] dataset_meta.json's "
                      f"latent_frames_per_chunk does not match the sampled files "
                      f"-> reading every latent header (slower).")

        checks = self._file_checks(uniform)
        for i, f in enumerate(self.files):
            if f.suffix.lower() not in SUPPORTED_EXTS:
                continue
            n_files_total += 1
            class_name = _class_of_file(f)
            label_idx = self.label_to_idx.get(class_name)
            if label_idx is None:
                continue

            file_frames, has_cond = checks[i]
            if isinstance(file_frames, str):
                n_files_unreadable += 1
                print(f"[CondDataset/{self.split}] WARNING: unreadable latent "
                      f"{f.name} ({file_frames}) -> skipped")
                continue

            n_chunks_file = file_frames // self.n_frames
            if n_chunks_file == 0:
                n_files_short += 1
                continue

            cond_path = None
            if self.condition_root:
                if has_cond:
                    cond_path = (self.condition_root
                                 / f.relative_to(self.latent_root).with_suffix(".npz"))
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

    def _file_checks(self, uniform):
        def check(i):
            f = self.files[i]
            if (f.suffix.lower() not in SUPPORTED_EXTS
                    or self.label_to_idx.get(_class_of_file(f)) is None):
                return None
            frames = uniform
            if frames is None:
                try:
                    frames = np.load(str(f), mmap_mode="r").shape[1]
                except Exception as e:
                    return (f"{type(e).__name__}: {e}", None)
            has_cond = None
            if self.condition_root and frames // self.n_frames > 0:
                has_cond = (self.condition_root
                            / f.relative_to(self.latent_root).with_suffix(".npz")).exists()
            return (frames, has_cond)

        if self._file_checks_dist is None:
            return [check(i) for i in range(len(self.files))]
        rank, world, gather = self._file_checks_dist
        out = [None] * len(self.files)
        for part in gather({i: check(i) for i in range(rank, len(self.files), world)}):
            for i, v in part.items():
                out[i] = v
        return out

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

    def _setup_text(self, source, generation_source):
        for s in (source, generation_source):
            if s not in TEXT_SOURCES:
                raise ValueError(f"the text source must be one of "
                                 f"{list(TEXT_SOURCES)}, got {s!r}.")
        self.text_source = source
        self.generation_text_source = generation_source
        self._text_table = load_text_table(self.latent_root)
        for s in {source, generation_source} - {"audio"}:
            if f"{s}_emb" not in self._text_table:
                raise RuntimeError(
                    f"[CondDataset/{self.split}] text source '{s}': this dataset "
                    f"has no {s} embeddings. Write them with:\n"
                    f"    python preprocess_stream.py SRC "
                    f"{self.latent_root.parent} --global_conds text")
        if source == "audio":
            self._probe_text_dim()
        else:
            self._text_dim = int(self._text_table[f"{source}_emb"].shape[1])
            _, self._ctx_len, self._ctx_dim = self._text_table[f"{source}_tok"].shape

    def _probe_text_dim(self):
        for _npy, cond_path, _s, _l, _c in self.samples:
            if cond_path is None:
                continue
            try:
                with np.load(str(cond_path)) as data:
                    if "text" in data:
                        self._text_dim = int(data["text"].shape[0])
                        return
            except Exception:
                continue
        raise RuntimeError(
            f"[CondDataset/{self.split}] text source 'audio': no chunk carries "
            f"the CLAP audio embedding. Extract it with:\n"
            f"    python preprocess_stream.py SRC {self.latent_root.parent} "
            f"--global_conds text")

    def _text_ids(self, npy_path, kind):
        key = _source_group_of(Path(npy_path), self.latent_root)
        return self._text_table["sources"].get(key, {}).get(f"{kind}s", [])

    def _text_id(self, npy_path, idx, kind):
        ids = self._text_ids(npy_path, kind)
        if not ids:
            return None
        if len(ids) == 1:
            return ids[0]
        return random.choice(ids) if self.split == "train" else ids[idx % len(ids)]

    def generation_text(self, idx, text_emb):
        kind = self.generation_text_source
        if kind == self.text_source:
            return text_emb
        npy_path, cond_path = self.samples[idx][:2]
        if kind == "audio":
            with np.load(str(cond_path)) as data:
                return torch.from_numpy(data["text"].astype(np.float32))
        cid = self._text_id(npy_path, idx, kind)
        if cid is None:
            return torch.zeros(self._text_dim)
        return torch.from_numpy(self._text_table[f"{kind}_emb"][cid].copy())

    def generation_string(self, idx):
        kind = self.generation_text_source
        if kind in (None, "audio"):
            return None
        cid = self._text_id(self.samples[idx][0], idx, kind)
        return None if cid is None else self._text_table[f"{kind}s"][cid]

    @staticmethod
    def _image_split_of(file_name: str) -> str:
        h = int(hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:8], 16) % 10
        return "train" if h < 8 else ("val" if h == 8 else "test")

    def _load_image_bank(self):
        if (self.registry is None
                or "image" not in self.registry.global_extractors):
            return
        bank_dir = self.latent_root.parent / "global_conditions" / "image"
        if not bank_dir.exists():
            raise RuntimeError(
                f"[CondDataset/{self.split}] the 'image' global condition is "
                f"active but {bank_dir} does not exist. Build it with:\n"
                f"    python preprocess_stream.py SRC {self.latent_root.parent} "
                f"--global_conds image --image_root <folder of <class>/*.jpg>\n"
                f"(it encodes only the images; no latent is touched).")

        wanted = set(self._present_classes or list(self.label_to_idx.keys()))
        found, empty_split, missing = 0, [], []
        for jp in sorted(bank_dir.glob("*.json")):
            try:
                meta = json.loads(jp.read_text(encoding="utf-8"))
                class_name = meta["class"]
                files = list(meta["files"])
            except Exception as e:
                print(f"[CondDataset/{self.split}] WARNING: unreadable "
                      f"{jp.name} ({type(e).__name__}: {e}) -> skipped")
                continue
            if class_name not in wanted:
                continue
            bank = np.load(str(jp.with_suffix(".npy"))).astype(np.float32)
            if bank.shape[0] != len(files):
                raise RuntimeError(
                    f"[CondDataset/{self.split}] {jp.name} lists {len(files)} "
                    f"files but the bank holds {bank.shape[0]} rows. The two "
                    f"are written together; re-run the extraction with --force.")
            self._image_dim = int(bank.shape[1])
            keep = [i for i, f in enumerate(files)
                    if self._image_split_of(f) == self.split]
            if not keep:
                empty_split.append(f"{class_name}({len(files)})")
                keep = list(range(len(files)))
            self._image_embeddings[class_name] = [bank[i] for i in keep]
            self._image_files[class_name] = [files[i] for i in keep]
            found += 1

        missing = sorted(wanted - set(self._image_embeddings))
        total = sum(len(v) for v in self._image_embeddings.values())
        print(f"[CondDataset/{self.split}] image: {total} embeddings over "
              f"{found}/{len(wanted)} classes (dim={self._image_dim}), "
              f"read from {bank_dir}")
        if empty_split:
            print(f"[CondDataset/{self.split}] WARNING: too few images to hold "
                  f"back a '{self.split}' share for {len(empty_split)} class(es) "
                  f"-> the whole bank is used, so a picture here may also have "
                  f"been seen in training: {empty_split[:8]}")
        if missing:
            print(f"[CondDataset/{self.split}] WARNING: no image bank for "
                  f"{len(missing)} class(es) -> those samples get a NULL image "
                  f"(zeros), i.e. no image conditioning at all: {missing[:8]}"
                  f"{' ...' if len(missing) > 8 else ''}")

    def image_file_for(self, idx: int) -> Optional[Tuple[str, str]]:
        if self.split == "train" or not (0 <= idx < len(self.samples)):
            return None
        class_name = self.samples[idx][4]
        files = self._image_files.get(class_name)
        if not files:
            return None
        return class_name, files[idx % len(files)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, cond_path, start, label_idx, class_name = self.samples[idx]

        key = str(npy_path)
        if key in self._latent_cache:
            z = self._latent_cache[key][:, start:start + self.n_frames].float()
        else:
            z = torch.from_numpy(np.load(key).astype(np.float32))
            z = z[:, start:start + self.n_frames]

        if z.ndim != 2 or z.shape[0] != self.latent_dim or z.shape[1] != self.n_frames:
            raise RuntimeError(
                f"Latent {npy_path.name} slice has shape {tuple(z.shape)}, "
                f"expected ({self.latent_dim}, {self.n_frames}).")
        if not torch.isfinite(z).all():
            raise RuntimeError(f"Latent {npy_path.name} contains NaN/Inf.")

        if self.normalizer:
            z = self.normalizer.normalize(z)
        frames = z.T

        if frames.shape[0] != self.n_frames:
            raise RuntimeError(
                f"Sample {npy_path.name} @ start={start}: "
                f"expected shape ({self.n_frames}, {self.latent_dim}), got {tuple(frames.shape)}. "
                f"The file has fewer frames than expected.")

        frame_cond = {}
        frame_names = self._get_frame_names()

        want_audio_text = self.text_source == "audio"
        text_raw = None
        if cond_path is not None and (frame_names or want_audio_text):
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
                text_raw = data["text"] if "text" in data else None
                for name in frame_names:
                    if name not in data:
                        continue
                    c = data[name].astype(np.float32)
                    expected_dim = self.registry.frame_cond_dims[name]

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

        cid = None
        if not self._text_dim:
            text_emb = torch.zeros(1)
        elif not want_audio_text:
            cid = self._text_id(npy_path, idx, self.text_source)
            text_emb = (torch.zeros(self._text_dim) if cid is None else
                        torch.from_numpy(self._text_table[f"{self.text_source}_emb"][cid].copy()))
        elif text_raw is not None:
            t = np.asarray(text_raw, dtype=np.float32).reshape(-1)
            if t.shape[0] != self._text_dim or not np.isfinite(t).all():
                if self.strict_conditions:
                    raise RuntimeError(
                        f"'text' for {npy_path.name} has shape {t.shape} "
                        f"(expected ({self._text_dim},))"
                        f"{' and contains NaN/Inf' if not np.isfinite(t).all() else ''}. "
                        f"Re-extract it, or set training.strict_conditions=false.")
                self._warn_cond_once("'text' malformed -> zero-filled")
                text_emb = torch.zeros(self._text_dim)
            else:
                text_emb = torch.from_numpy(t)
        else:
            if self.strict_conditions:
                raise RuntimeError(
                    f"'text' missing for {npy_path.name} (cond_path={cond_path}). "
                    f"Re-run: python preprocess_stream.py SRC "
                    f"{self.latent_root.parent} --global_conds text  "
                    f"-- or set training.strict_conditions=false to zero-fill it.")
            self._warn_cond_once("'text' missing -> zero-filled")
            text_emb = torch.zeros(self._text_dim)

        if class_name in self._image_embeddings:
            bank = self._image_embeddings[class_name]
            if self.split == "train":
                img_emb = torch.from_numpy(random.choice(bank))
            else:
                img_emb = torch.from_numpy(bank[idx % len(bank)])
        elif self._image_dim > 0:
            img_emb = torch.zeros(self._image_dim)
        else:
            img_emb = torch.zeros(1)

        text_ctx = self.text_context_for(cid)

        return frames, frame_cond, label_idx, text_emb, img_emb, text_ctx

    def text_context_for(self, cid):
        L = max(1, self._ctx_len)
        D = max(1, self._ctx_dim)
        if cid is None:
            return {"tokens": torch.zeros(L, D),
                    "mask": torch.zeros(L, dtype=torch.bool)}
        n = int(self._text_table[f"{self.text_source}_tok_len"][cid])
        mask = torch.zeros(L, dtype=torch.bool)
        mask[:n] = True
        tok = self._text_table[f"{self.text_source}_tok"][cid]
        return {"tokens": torch.from_numpy(tok.astype(np.float32)), "mask": mask}


def collate_conditioned(batch):
    frames_l, conds_l, labels_l, text_l, image_l, ctx_l = zip(*batch)

    frames = torch.stack(frames_l)
    labels = torch.tensor(labels_l, dtype=torch.long)
    text_embs = torch.stack(text_l)
    image_embs = torch.stack(image_l)

    frame_conds = {}
    if conds_l and conds_l[0]:
        for name in conds_l[0].keys():
            frame_conds[name] = torch.stack([c[name] for c in conds_l])

    text_ctx = {"tokens": torch.stack([c["tokens"] for c in ctx_l]),
                "mask":   torch.stack([c["mask"]   for c in ctx_l])}

    return frames, frame_conds, labels, text_embs, image_embs, text_ctx


def build_conditioned_datasets(
    latent_root:     str,
    condition_root:  Optional[str] = None,
    duration_s:      float = 5.0,
    normalizer_path: Optional[str] = None,
    registry:        Optional[ConditionRegistry] = None,
    preload:         bool = True,
    strict_conditions: bool = True,
    splits_path:     Optional[str] = None,
    text_source:     str = "label",
    validation_text_source: Optional[str] = None,
    normalizer_shard: Optional[Tuple[int, int]] = None,
    normalizer_merge=None,
    file_checks_dist=None,
):
    split = load_source_split(latent_root, splits_path=splits_path)
    splits = split["splits"]
    classes = split["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}

    normalizer = LatentNormalizer()
    if normalizer_path and Path(normalizer_path).exists():
        normalizer.load(normalizer_path)
    else:
        _over = (f", over {normalizer_shard[1]} GPUs"
                 if normalizer_shard is not None else "")
        print(f"[build_conditioned_datasets] Computing normalizer on the train "
              f"split{_over}...")
        n_frames = frames_per_chunk(latent_root, duration_s)
        chunks = _chunks_from_files(
            splits["train"], n_frames,
            uniform_frames=_meta_latent_frames(latent_root),
            max_chunks=NORMALIZER_MAX_CHUNKS,
        )
        if not chunks:
            raise RuntimeError("No train chunks available to fit the normalizer.")
        if normalizer_shard is not None:
            chunks = chunks[normalizer_shard[0]::normalizer_shard[1]]
        normalizer.fit_from_chunks(
            chunks, n_frames=n_frames, merge=normalizer_merge,
            progress=normalizer_shard is None or normalizer_shard[0] == 0)
        del chunks

    common = dict(
        label_to_idx=label_to_idx,
        latent_root=latent_root,
        condition_root=condition_root,
        duration_s=duration_s,
        normalizer=normalizer,
        registry=registry,
        strict_conditions=strict_conditions,
        text_source=text_source,
        file_checks_dist=file_checks_dist,
    )

    train = ConditionedAudioDataset(files=splits["train"], split="train",
                                    preload_latents=preload, **common)
    val = ConditionedAudioDataset(files=splits["val"], split="val",
                                  preload_latents=False,
                                  generation_text_source=validation_text_source,
                                  **common)
    test = ConditionedAudioDataset(files=splits["test"], split="test",
                                   preload_latents=False,
                                   generation_text_source=validation_text_source,
                                   **common)

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
