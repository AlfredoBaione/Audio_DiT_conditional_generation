# audio_dataset_cond.py
#
# Multi-modal conditioned dataset for training the ConditionedAudioDiT.
#
# Loads for each sample:
#   - frames:       (n_frames, 72)   normalized DAC pre-quantizer latents
#   - frame_conds:  {f0, chroma, rhythm, ...} pre-extracted conditions
#   - label_idx:    int                 class index
#   - text_emb:     (text_dim,)         CLAP embedding OF THIS CHUNK's audio
#   - image_emb:    (image_dim,)        CLIP embedding of one image of the class
#
# ------------------------------------------------------------------------------
# NEW on-disk contract (produced by preprocess_stream.py): the dataset is
# SPLIT-LESS on disk and mirrors the source class tree:
#
#   dataset_root/
#       latents/<class...>/*.npy       <- (72, T) float32 pre-quant DAC latents
#       conditions/<class...>/*.npz    <- f0, chroma, ... AND the per-chunk
#                                         global 'text' (one (dim,) vector)
#       global_conditions/image/       <- <class>.npy  (n_images, dim) CLIP bank
#                                         <class>.json the file names, in order
#       wav/<class...>/*.wav           <- optional (val/test used for FAD)
#       splits.json                    <- source -> train/val/test (READ here)
#       dataset_meta.json              <- chunk/acoustic params
#
# NOTHING IS ENCODED AT TRAINING TIME ANY MORE. This module used to load CLAP
# and CLIP at startup, on the training GPU, to embed the class names and up to
# ten images per class. Both are now read from what the preprocessing wrote:
#   * text  -- one vector per CHUNK, in that chunk's own .npz. It is the CLAP
#     embedding of the chunk's AUDIO (AudioLDM-style: the two CLAP towers share
#     a space, so a written prompt can take its place at inference). It is NOT
#     the embedding of the class name, which would have carried the same single
#     bit as the image condition and made the two impossible to tell apart.
#   * image -- the whole per-class CLIP bank, from which a sample draws one
#     image at random per epoch (that draw IS the augmentation). No ten-image
#     cap: the bank holds every picture of the class.
#
# The train/val/test split is DECIDED BY preprocess_stream.py and read back here
# (load_source_split). It is not recomputed at training time, because a split
# that is re-derived on every startup is a split that can silently change --
# after new files land, after a parameter is edited, after a seed is touched --
# and a checkpoint would then be evaluated on material it was trained on. The
# properties are unchanged; only the place they are decided moved:
#   * STRATIFIED by class (each class contributes to every split by ratio);
#   * GROUPED by source file (all chunks AND both stereo channels of one source
#     go to the SAME split) -> no train/test leakage across chunks of a track;
#   * DETERMINISTIC from a seed;
#   * WRITTEN DOWN once and never reshuffled: a dataset that grows gets its new
#     sources assigned into the existing split.
#
# Chunk file names are `<stem>[__ch{n}]__c{idx}.npy`; the source group key is the
# part before the first `__` (sanitize_filename never emits `__` inside a stem),
# which is exactly the key preprocess_stream.py writes into splits.json.

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
    # split machinery lives in the base module to avoid a circular import;
    # re-exported here so callers can `from audio_dataset_cond import ...`.
    # load_source_split READS the split preprocess_stream.py wrote; compute_split
    # is the old in-code one, kept only for --import_legacy_split and test_cond.py.
    load_source_split, compute_split, _class_of_file, _chunks_from_files,
    _meta_latent_frames, NORMALIZER_MAX_CHUNKS,
)
from conditions import ConditionRegistry
# NB: CLAPTextCondition / ImageCondition / ImageDatasetManager are deliberately
# NOT imported. Nothing is encoded here any more -- the embeddings are read from
# what preprocess_stream.py wrote -- and importing them would put the option of
# loading CLAP or CLIP back inside the training process.


# ============================================================
# THE CAPTION SIDECAR
# ============================================================
def caption_dir(latent_root):
    return Path(latent_root).parent / "global_conditions"


def chunk_key_for(latent_root, cond_path):
    """The key text_labels.jsonl is indexed by: the chunk's .npz path relative
    to conditions/, without the extension. None when it cannot be formed.

    The sidecar is keyed by PREPROCESSING chunk, which is the unit the CLAP
    vector belongs to. When the model's n_frames is shorter than a stored chunk,
    several training samples come out of one chunk and share its caption --
    correctly: they were extracted from the very same audio.
    """
    try:
        if cond_path is None:
            return None
        root = Path(latent_root).parent / "conditions"
        return Path(cond_path).relative_to(root).with_suffix("").as_posix()
    except Exception:
        return None


def load_caption_table(latent_root):
    """
    Everything preprocess_stream.write_text_labels stored about the CAPTIONS,
    as one dict -- or {} when the dataset carries none.

        ids      {chunk key: caption index}
        emb      (n_captions, dim) float32   the CLAP TEXT embedding, pooled
        tok      (n_captions, L, ctx) float32  its TOKEN-level states
        tok_len  (n_captions,) int              how many of those are real
        captions [str]                          what they say
        captions_text [str]                     the same, as CLAP was given them
        n_terms  int                            terms per caption (1 = the class)

    ONE loader for both consumers -- the dataset, which hands the vectors to the
    model, and the metrics step, which uses them for
    sampling.validation_text_from_caption -- because the two must never disagree
    about which caption belongs to which chunk. It lives here rather than in the
    training script for the same reason the embeddings themselves do: this is a
    property of the DATASET, and reading it must never require CLAP.

    Missing pieces are simply absent from the returned dict: a dataset
    preprocessed before the token sequences existed still has `emb`, and the
    caller decides whether that is enough for what it wants to run.
    """
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
        # What the text probe needs to speak at the dataset's level of detail
        # (probe_conditions.text_probe_bank): the strings CLAP was given, and
        # how many terms each caption has.
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


# ============================================================
# CONDITIONED DATASET
# ============================================================
class ConditionedAudioDataset(Dataset):
    """
    Multi-modal dataset for conditioned training. Built from an explicit list of
    latent files (one split), with a GLOBAL label_to_idx shared across splits.

    For each sample returns:
        frames:      (n_frames, 72)
        frame_conds: Dict[str, Tensor] — e.g. {"f0": (n_frames, 2), ...}
        label_idx:   int
        text_emb:    (text_dim,) — the vector the text slot is GIVEN, which
                     conditioning.text_source decides (the chunk's own CLAP
                     audio embedding, or its caption's CLAP text one)
        image_emb:   (image_dim,) — random image embedding of the class
        text_ctx:    {'tokens': (L, ctx_dim), 'mask': (L,)} — the caption as
                     a SEQUENCE, for the cross-attention. All-False mask
                     when this dataset has none; the network reads that as
                     its learned null token.
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

        self._text_dim: int = 0
        self._probe_text_dim()

        # WHAT THE TEXT SLOT IS FED, decided here rather than in the model
        # because it is a property of the DATA:
        #   audio   -- the chunk's own CLAP AUDIO embedding. What every run
        #              before 14 Sept 2026 did, and the AudioLDM arrangement.
        #   caption -- the CLAP TEXT embedding of that chunk's written
        #              description. The vector a PROMPT puts there at inference,
        #              so training and test finally see the same distribution.
        #   mix     -- one or the other, per sample. Deterministic on val/test
        #              (by index), so the validation loss does not wobble with
        #              the draw and stays comparable across checkpoints.
        # The cross-attention CONTEXT is not governed by this: it is always the
        # caption's tokens, because a pooled audio embedding is one vector and
        # there is no sequence to attend over. See text_context_for().
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

        # Uniform-length fast path. preprocess_stream.py writes FIXED-length
        # chunks, so dataset_meta.json's `latent_frames_per_chunk` describes every
        # latent: opening each one just to read its shape is O(n_files) network
        # round trips for a number we already know -- hours on a multi-million
        # file split, repeated once per split. The claim is verified on a spread
        # sample; if any probed file disagrees, the exact per-file read is used.
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
                # class not in the global mapping (should not happen): skip safely
                continue

            if uniform is not None:
                file_frames = uniform          # from the verified dataset meta
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

    def _probe_text_dim(self):
        """Learn the text embedding's width from the data, not from the model.

        The width is needed up front to build the null vector a sample falls
        back to, and the obvious way to get it -- ask the extractor for .dim --
        loads CLAP onto the training GPU just to read an integer off a config.
        The .npz already knows: read the key's shape from the first chunk that
        has it. Only the archive's header is touched (np.load on an .npz is
        lazy), so this costs one header read, not an embedding.
        """
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
        # Asked for, never written: this cannot be papered over with zeros --
        # the model would train on a null condition and the run would look fine.
        raise RuntimeError(
            f"[CondDataset/{self.split}] the 'text' global condition is active "
            f"but no chunk carries it. Extract it with:\n"
            f"    python preprocess_stream.py SRC {self.latent_root.parent} "
            f"--global_conds text\n"
            f"(it re-reads the audio but re-encodes no latent, and keeps every "
            f"condition already on disk).")

    @staticmethod
    def _image_split_of(file_name: str) -> str:
        """Which split one picture belongs to, from a hash of its FILE NAME.

        The images have no split of their own -- they are not the audio -- but
        the validation panels must not show a picture the model was conditioned
        on during training, or the panel would be measuring recall of a seen
        image rather than the conditioning.

        Hashed by name, not by position, so the assignment survives the bank
        growing: dropping ten new pictures into a class folder and re-running
        the preprocessing re-sorts the list and would shift every index, moving
        images across splits and quietly invalidating the separation. The
        proportions mirror the audio's 80/10/10.
        """
        h = int(hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:8], 16) % 10
        return "train" if h < 8 else ("val" if h == 8 else "test")

    def _load_image_bank(self):
        """Read the per-class CLIP bank written by preprocess_stream.py.

        No CLIP here: the bank is a plain .npy of L2-normalized rows, and the
        .json beside it names the file of every row. The class name is taken
        from that .json rather than reconstructed from the file name, so this
        never has to replicate the preprocessing's name sanitizer -- and cannot
        drift from it.
        """
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
                # Too few pictures for the hash to reach this split. Showing a
                # training image beats showing none -- but say so, because the
                # separation this method exists for is not holding for it.
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
        """-> (class name, image file name) that sample `idx` is conditioned on,
        or None when there is no image for it.

        Only meaningful for val/test, where the choice is deterministic; in
        train the picture is redrawn at every access, so there is no single
        answer and this returns None. It exists for the panels: they must SHOW
        the image a generation was conditioned on, and an embedding cannot be
        turned back into a picture. The file has to be re-opened from the
        original image folder -- the dataset itself never does, and this hands
        out the name rather than the image so that stays true.

        MUST mirror the choice made in __getitem__ exactly; if one changes, the
        panel starts displaying a different picture from the one that
        conditioned the sound, which is worse than showing none.
        """
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

        # The .npz is opened when ANY per-chunk condition is wanted from it --
        # a frame condition, or the 'text' global, which lives in the same
        # archive. Gating this on frame_names alone (as it used to) meant a run
        # conditioned on text ALONE never opened the file and silently trained
        # on null text.
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

        # 3. TEXT EMBEDDING -- this chunk's own, read from the .npz above.
        # Per CHUNK, not per class: two excerpts of the same piece get different
        # vectors, which is the whole reason the text condition is not simply
        # the class name (see the module header).
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
            # Wanted, and this chunk has none.
            if self.strict_conditions:
                raise RuntimeError(
                    f"'text' missing for {npy_path.name} (cond_path={cond_path}). "
                    f"Re-run: python preprocess_stream.py SRC "
                    f"{self.latent_root.parent} --global_conds text  "
                    f"-- or set training.strict_conditions=false to zero-fill it.")
            self._warn_cond_once("'text' missing -> zero-filled")
            text_emb = torch.zeros(self._text_dim)

        # 4. IMAGE EMBEDDING -- one picture of this sample's class.
        # train: drawn at random every epoch, which IS the augmentation this
        # condition exists for. val/test: deterministic, so a panel shows the
        # same picture at every checkpoint and the curves stay comparable; the
        # index is derived from the CHUNK rather than fixed at 0, so the panels
        # do not all end up conditioned on the same one image of the class.
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

        # 5. THE TEXT SLOT AND ITS CONTEXT.
        # Done last because it may REPLACE text_emb computed above: under
        # text_source 'caption' (or a 'mix' draw) the slot receives the
        # caption's CLAP TEXT vector instead of the chunk's own audio one.
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

    # ---- caption plumbing -------------------------------------------------
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
        """Whether THIS sample's text slot takes the caption vector.

        'mix' is random on train -- that is the point, the model must see both
        distributions -- and DETERMINISTIC elsewhere. A validation loss whose
        conditioning is re-drawn at every evaluation is a loss that moves for
        reasons that have nothing to do with the model, and the whole use of
        that curve is to be compared with itself across checkpoints.
        """
        if self.text_source == "audio":
            return False
        if self.text_source == "caption":
            return True
        if self.split == "train":
            return random.random() < self.text_mix_p
        return (idx % 2) == 0

    def text_context_for(self, cid):
        """{'tokens': (L, ctx_dim), 'mask': (L,) bool} for the cross-attention.

        ALWAYS the caption's tokens, whatever text_source says, because this is
        the one channel that cannot take an audio embedding: CLAP's audio tower
        returns a single vector and an attention over one key is a learned bias.

        A dataset with no token sidecar, or a chunk with no caption, gets a
        single all-False mask. That is not "skip the sub-layer" -- the network
        reads an empty mask as its LEARNED NULL TOKEN, the same value the CFG
        dropout trains -- so the shape is always well-defined and there is never
        a third state. A run that MEANT to use the cross-attention must not get
        here silently, which is why training_cond refuses to start when
        model.text_cross_every > 0 and the dataset has no tokens.
        """
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


# ============================================================
# COLLATE
# ============================================================
def collate_conditioned(batch):
    """Custom collate for the DataLoader."""
    frames_l, conds_l, labels_l, text_l, image_l, ctx_l = zip(*batch)

    frames = torch.stack(frames_l)
    labels = torch.tensor(labels_l, dtype=torch.long)
    text_embs = torch.stack(text_l)
    image_embs = torch.stack(image_l)

    frame_conds = {}
    if conds_l and conds_l[0]:
        for name in conds_l[0].keys():
            frame_conds[name] = torch.stack([c[name] for c in conds_l])

    # The context is padded to a fixed length by the dataset (every caption
    # shares one table), so a plain stack is correct and no per-batch padding
    # is needed. The mask travels with it: without it the model cannot tell a
    # short caption from a long one that happens to end in zeros.
    text_ctx = {"tokens": torch.stack([c["tokens"] for c in ctx_l]),
                "mask":   torch.stack([c["mask"]   for c in ctx_l])}

    return frames, frame_conds, labels, text_embs, image_embs, text_ctx


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
    splits_path:     Optional[str] = None,
    text_source:     str = "audio",
    text_mix_p:      float = 0.5,
):
    """
    Builds the conditioned train/val/test datasets from the SPLIT-LESS dataset,
    READING the split (written by preprocess_stream.py) and fitting the normalizer.

    The split is no longer computed here: it is decided over the source files at
    preprocessing time and recorded in <dataset>/splits.json, so it cannot drift
    between runs, and the normalizer below is fitted on exactly the train split
    the preprocessing declared. `splits_path` overrides where that file is read
    from; None = the dataset's own.

    Returns:
        (train, val, test, normalizer, label_to_idx, split_info)
    where split_info = {
        "file_counts": {train,val,test}, "chunk_counts": {train,val,test},
        "n_classes": int, "manifest_path": str|None, "params": {...},
    }
    """
    # 1. split (shared across train/val/test)
    split = load_source_split(latent_root, splits_path=splits_path)
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
        # uniform_frames lets this skip opening every .npy just to read its shape
        # (hours of idle-GPU network I/O on a multi-million-chunk corpus), and
        # max_chunks bounds how many chunks the fit actually reads.
        chunks = _chunks_from_files(
            splits["train"], n_frames,
            uniform_frames=_meta_latent_frames(latent_root),
            max_chunks=NORMALIZER_MAX_CHUNKS,
        )
        if not chunks:
            raise RuntimeError("No train chunks available to fit the normalizer.")
        normalizer.fit_from_chunks(chunks, n_frames=n_frames)
        # Release the (Path, start) list NOW: it is O(n_chunks) tuples -- roughly
        # a GB on a multi-million-chunk corpus -- and the three dataset objects
        # built right below allocate their own per-sample lists. Holding both at
        # once is a pointless RAM peak at the worst possible moment.
        del chunks

    # 3. NO image manager any more.
    # The per-class CLIP bank is read from the dataset itself
    # (global_conditions/image/), written once by preprocess_stream.py, so the
    # raw image folder is not needed at training time and CLIP is never loaded
    # here. `image_root` is accepted and ignored, so an existing config or
    # caller does not break; it is only used by the preprocessing now.
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
