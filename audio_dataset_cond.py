# Conditioned dataset: latents with their frame and global conditions.

import hashlib
import json
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from audio_dataset_npy import (
    LatentNormalizer, DAC_LATENT_DIM, SUPPORTED_EXTS,
    frames_per_chunk,
    load_source_split, compute_split, _class_of_file, _chunks_from_files,
    _meta_latent_frames, NORMALIZER_MAX_CHUNKS,
)
from conditions import ConditionRegistry
import latent_codec as lc


def caption_dir(latent_root):
    return Path(latent_root).parent / "global_conditions"


def chunk_key_for(latent_root, cond_path):
    try:
        if cond_path is None:
            return None
        root = Path(latent_root).parent / "conditions"
        return Path(cond_path).relative_to(root).with_suffix("").as_posix()
    except Exception:
        return None


def load_caption_table(latent_root):
    d = caption_dir(latent_root)
    meta_p, jsonl_p = d / "text_labels.json", d / "text_labels.jsonl"
    if not (meta_p.exists() and jsonl_p.exists()):
        return {}
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        captions = list(meta.get("captions", []))
        n_cap = len(captions)
        if n_cap == 0:
            return {}
        ids = {}
        with open(jsonl_p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                cid = r.get("caption_id")
                if r.get("chunk") and cid is not None and 0 <= cid < n_cap:
                    ids[r["chunk"]] = int(cid)
        out = {"ids": ids, "captions": captions}
        ctext = list(meta.get("captions_text", []))
        if len(ctext) == n_cap:
            out["captions_text"] = ctext
        if meta.get("n_terms") is not None:
            out["n_terms"] = int(meta["n_terms"])

        emb_p = d / "text_labels_emb.npy"
        if emb_p.exists():
            emb = np.load(str(emb_p)).astype(np.float32)
            if emb.ndim == 2 and emb.shape[0] == n_cap:
                out["emb"] = emb
            else:
                print(f"[captions] text_labels_emb.npy is {emb.shape} for "
                      f"{n_cap} caption(s) -> ignored")

        tok_p, len_p = (d / "text_labels_tok.npy", d / "text_labels_tok_len.npy")
        if tok_p.exists() and len_p.exists():
            tok = np.load(str(tok_p)).astype(np.float32)
            tlen = np.load(str(len_p)).astype(np.int64)
            if (tok.ndim == 3 and tok.shape[0] == n_cap
                    and tlen.shape == (n_cap,)
                    and int(tlen.max()) <= tok.shape[1]):
                out["tok"], out["tok_len"] = tok, tlen
            else:
                print(f"[captions] text_labels_tok.npy is {tok.shape} / "
                      f"{tlen.shape} for {n_cap} caption(s) -> ignored")
        return out
    except Exception as e:
        print(f"[captions] sidecar unreadable ({type(e).__name__}: {e})")
        return {}


class ConditionedAudioDataset(Dataset):
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
        preload_latents: bool  = True,
        strict_conditions: bool = True,
        text_source:     str   = "audio",
        text_mix_p:      float = 0.5,
    ):
        self.files          = [Path(f) for f in files]
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

        self._text_dim: int = 0
        self._probe_text_dim()

        self.text_source = str(text_source or "audio").lower()
        if self.text_source not in ("audio", "caption", "mix"):
            raise ValueError(
                f"text_source must be 'audio', 'caption' or 'mix', got "
                f"{text_source!r}")
        self.text_mix_p = float(text_mix_p)
        self._captions = load_caption_table(self.latent_root) if self._text_dim else {}
        self._cap_warned = False
        if self._text_dim and self.text_source != "audio" and "emb" not in self._captions:
            raise RuntimeError(
                f"conditioning.text_source='{self.text_source}' needs the CLAP "
                f"TEXT embedding of each caption, and this dataset has none "
                f"(global_conditions/text_labels_emb.npy). Re-run the "
                f"preprocessing on the same output dir with --global_conds text: it "
                f"rewrites only the sidecar and does not touch the audio.")
        self._ctx_dim = int(self._captions["tok"].shape[2]) if "tok" in self._captions else 0
        self._ctx_len = int(self._captions["tok"].shape[1]) if "tok" in self._captions else 0

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
                  + (f" (p={self.text_mix_p})" if self.text_source == "mix" else "")
                  + f" | {len(self._captions.get('captions', []))} distinct "
                    f"caption(s) | cross-attention context: "
                  + (f"{self._ctx_len} token(s) x {self._ctx_dim}"
                     if self._ctx_dim else "NOT in this dataset"))

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

        for f in self.files:
            if f.suffix.lower() not in SUPPORTED_EXTS:
                continue
            n_files_total += 1
            class_name = _class_of_file(f)
            label_idx = self.label_to_idx.get(class_name)
            if label_idx is None:
                continue

            if uniform is not None:
                file_frames = uniform
            else:
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

    def _probe_text_dim(self):
        if self.registry is None or "text" not in self.registry.global_extractors:
            return
        for _npy, cond_path, _s, _l, _c in self.samples:
            if cond_path is None:
                continue
            try:
                with np.load(str(cond_path)) as data:
                    if "text" in data:
                        arr = data["text"]
                        if arr.ndim != 1:
                            raise RuntimeError(
                                f"'text' in {cond_path} has shape {arr.shape}; "
                                f"a global condition is one vector, (dim,). "
                                f"This .npz was written by an older "
                                f"preprocess_stream.py -- re-extract it.")
                        self._text_dim = int(arr.shape[0])
                        print(f"[CondDataset/{self.split}] text: per-chunk CLAP "
                              f"embeddings from the .npz (dim={self._text_dim})")
                        return
            except RuntimeError:
                raise
            except Exception:
                continue
        raise RuntimeError(
            f"[CondDataset/{self.split}] the 'text' global condition is active "
            f"but no chunk carries it. Extract it with:\n"
            f"    python preprocess_stream.py SRC {self.latent_root.parent} "
            f"--global_conds text\n"
            f"(it re-reads the audio but re-encodes no latent, and keeps every "
            f"condition already on disk).")

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

        want_text = self._text_dim > 0
        text_raw = None
        if cond_path is not None and (frame_names or want_text):
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

        if not want_text:
            text_emb = torch.zeros(1)
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

        cid = self._caption_id_for(cond_path)
        if self._text_dim and self._use_caption_for(idx):
            cap = self._captions.get("emb")
            if cap is not None and cid is not None:
                text_emb = torch.from_numpy(cap[cid].copy())
            elif self.strict_conditions:
                raise RuntimeError(
                    f"text_source='{self.text_source}' but {npy_path.name} has "
                    f"no caption in global_conditions/text_labels.jsonl "
                    f"(key={chunk_key_for(self.latent_root, cond_path)!r}). "
                    f"Re-run the preprocessing with --global_conds text, or set "
                    f"training.strict_conditions=false to fall back to the "
                    f"chunk's own audio vector.")
            else:
                self._warn_cap_once("no caption for this chunk -> audio vector")

        text_ctx = self.text_context_for(cid)

        return frames, frame_cond, label_idx, text_emb, img_emb, text_ctx

    def _warn_cap_once(self, msg):
        if not self._cap_warned:
            self._cap_warned = True
            print(f"[CondDataset/{self.split}] captions: {msg} "
                  f"(warned once)")

    def _caption_id_for(self, cond_path):
        key = chunk_key_for(self.latent_root, cond_path)
        if key is None:
            return None
        return self._captions.get("ids", {}).get(key)

    def _use_caption_for(self, idx: int) -> bool:
        if self.text_source == "audio":
            return False
        if self.text_source == "caption":
            return True
        if self.split == "train":
            return random.random() < self.text_mix_p
        return (idx % 2) == 0

    def text_context_for(self, cid):
        L = max(1, self._ctx_len)
        D = max(1, self._ctx_dim)
        tok = self._captions.get("tok")
        if tok is None or cid is None:
            return {"tokens": torch.zeros(L, D),
                    "mask": torch.zeros(L, dtype=torch.bool)}
        n = int(self._captions["tok_len"][cid])
        mask = torch.zeros(L, dtype=torch.bool)
        mask[:n] = True
        return {"tokens": torch.from_numpy(tok[cid].copy()), "mask": mask}


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
    image_root:      Optional[str] = None,
    duration_s:      float = 5.0,
    normalizer_path: Optional[str] = None,
    registry:        Optional[ConditionRegistry] = None,
    preload:         bool = True,
    strict_conditions: bool = True,
    splits_path:     Optional[str] = None,
    text_source:     str = "audio",
    text_mix_p:      float = 0.5,
):
    split = load_source_split(latent_root, splits_path=splits_path)
    splits = split["splits"]
    classes = split["classes"]
    label_to_idx = {c: i for i, c in enumerate(classes)}

    normalizer = LatentNormalizer()
    if normalizer_path and Path(normalizer_path).exists():
        normalizer.load(normalizer_path)
    else:
        print("[build_conditioned_datasets] Computing normalizer on the train split...")
        n_frames = frames_per_chunk(latent_root, duration_s)
        chunks = _chunks_from_files(
            splits["train"], n_frames,
            uniform_frames=_meta_latent_frames(latent_root),
            max_chunks=NORMALIZER_MAX_CHUNKS,
        )
        if not chunks:
            raise RuntimeError("No train chunks available to fit the normalizer.")
        normalizer.fit_from_chunks(chunks, n_frames=n_frames)
        del chunks

    if image_root:
        print("[build_conditioned_datasets] note: image_root is no longer read "
              "at training time. The image condition comes from the dataset's "
              "own global_conditions/image/ bank (preprocess_stream.py "
              "--global_conds image --image_root ...).")

    common = dict(
        label_to_idx=label_to_idx,
        latent_root=latent_root,
        condition_root=condition_root,
        image_root=image_root,
        duration_s=duration_s,
        normalizer=normalizer,
        registry=registry,
        strict_conditions=strict_conditions,
        text_source=text_source,
        text_mix_p=text_mix_p,
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
