# Preprocessing: audio -> codec latents, conditions, splits.json.

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ.setdefault("HF_HOME", os.path.join(_cache, "huggingface"))
    os.environ["TORCH_HOME"] = os.path.join(_cache, "torch")

import re
import json
import math
import random
import shutil
import hashlib
import argparse
import subprocess
import tempfile
import faulthandler
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import latent_codec as lc

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import IterableDataset, DataLoader
    _TORCH_OK = True
except Exception:
    torch = None
    F = None
    IterableDataset = object
    DataLoader = None
    _TORCH_OK = False


def _identity_collate(batch):
    return batch


def _worker_init_fn(worker_id: int):
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(f"[worker {worker_id}] pid={os.getpid()} torch_threads=1", flush=True)


def _atomic_save_npy(path: Path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_save_npz(path: Path, arrays: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_save_wav(path: Path, audio, sr: int):
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".part.wav", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        sf.write(str(tmp), audio, sr)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _latent_file_is_valid(path: Path, n_frames: int, verbose: bool = True) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    dim = lc.active().latent_dim
    z = None
    try:
        z = np.load(str(path), mmap_mode="r", allow_pickle=False)
        valid = z.shape == (dim, int(n_frames)) and z.dtype == np.dtype(np.float32)
    except Exception as e:
        if verbose:
            print(f"[resume] invalid latent will be regenerated: {path} ({e})")
        return False
    finally:
        if z is not None:
            mm = getattr(z, "_mmap", None)
            if mm is not None:
                mm.close()
    if not valid and verbose:
        print(f"[resume] invalid latent will be regenerated: {path} "
              f"(expected shape=({dim},{n_frames}), dtype=float32)")
    return valid


SUPPORTED_AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".wma", ".mpc", ".oma", ".ape", ".aac",
}

PREPROCESS_BUILD = "2026-09-04-split-and-config-v1"


def sanitize_class_name(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name, flags=re.ASCII)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name if name else "unknown"


def sanitize_filename(name: str) -> str:
    name = Path(name).stem
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name, flags=re.ASCII)
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name if name else "unknown"


LABEL_SOURCES = ("dir", "csv")

CSV_FILE_COLS = ("file", "filename")
CSV_LABEL_COLS = ("label", "class")

_LABEL_FORBIDDEN = set('/\\:*?"<>|') | {chr(c) for c in range(32)}

_AMBIGUOUS = object()


def _validate_class_label(label: str, where: str) -> str:
    s = str(label).strip()
    if not s:
        raise SystemExit(f"[labels] {where}: empty class label.")
    bad = sorted(set(s) & _LABEL_FORBIDDEN)
    if bad or s in (".", ".."):
        raise SystemExit(
            f"[labels] {where}: the label {s!r} cannot be a directory name "
            f"(offending character(s): {bad or [s]}). The class name IS the "
            f"output folder name here, so it is refused rather than silently "
            f"rewritten -- a rewritten name would no longer match "
            f"image_root/<class>. Fix it in the CSV.")
    return s


def _norm_rel_key(s) -> str:
    s = str(s).strip().strip('"').replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s.lstrip("/")


def _index_put(d: dict, key: str, row: int, label_of) -> None:
    if not key:
        return
    prev = d.get(key)
    if prev is None:
        d[key] = row
    elif prev is _AMBIGUOUS:
        return
    elif label_of(prev) != label_of(row):
        d[key] = _AMBIGUOUS


class CsvLabelResolver:
    def __init__(self, csv_path: Path, rows: List[Tuple[str, str]]):
        self.csv_path = Path(csv_path)
        self.rows = list(rows)
        self.used = set()

        def label_of(i):
            return self.rows[i][1]

        self._by_rel = {}
        self._by_name = {}
        self._by_rel_lc = {}
        self._by_name_lc = {}
        for i, (rel, _lab) in enumerate(self.rows):
            name = rel.rsplit("/", 1)[-1]
            _index_put(self._by_rel, rel, i, label_of)
            _index_put(self._by_name, name, i, label_of)
            _index_put(self._by_rel_lc, rel.lower(), i, label_of)
            _index_put(self._by_name_lc, name.lower(), i, label_of)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def labels(self) -> List[str]:
        return sorted({lab for _, lab in self.rows})

    def unmatched_rows(self) -> List[str]:
        return [rel for i, (rel, _) in enumerate(self.rows) if i not in self.used]

    def lookup(self, rel_posix: str) -> Tuple[Optional[str], str]:
        name = rel_posix.rsplit("/", 1)[-1]
        for d, key in ((self._by_rel, rel_posix),
                       (self._by_name, name),
                       (self._by_rel_lc, rel_posix.lower()),
                       (self._by_name_lc, name.lower())):
            hit = d.get(key)
            if hit is None:
                continue
            if hit is _AMBIGUOUS:
                return None, "ambiguous"
            self.used.add(hit)
            return self.rows[hit][1], "ok"
        return None, "missing"


def _pick_csv_column(path, fields, accepted, what: str) -> str:
    hits = [f for f in fields if f.lower() in accepted]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(
            f"[labels] {Path(path).name} has no {what} column. Accepted names "
            f"(any case): {list(accepted)}. The file has "
            f"{fields or 'no header'}. Rename the column in the CSV -- there is "
            f"deliberately no flag for this.")
    raise SystemExit(
        f"[labels] {Path(path).name} has {len(hits)} columns that could be the "
        f"{what} column: {hits}. Which one is the {what} cannot be guessed, and "
        f"guessing would put a whole corpus in the wrong folders. Rename or "
        f"remove one of them in the CSV.")


def load_label_csv(csv_path: str) -> "CsvLabelResolver":
    import csv as _csv
    p = Path(csv_path)
    if not p.exists():
        raise SystemExit(f"[labels] --label_csv {p} does not exist.")

    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = _csv.DictReader(fh)
        fields = [(f or "").strip() for f in (reader.fieldnames or [])]
        fcol = _pick_csv_column(p, fields, CSV_FILE_COLS, "file")
        lcol = _pick_csv_column(p, fields, CSV_LABEL_COLS, "label")
        rows, n_blank, n_seen = [], 0, 0
        for raw in reader:
            n_seen += 1
            rel = _norm_rel_key(raw.get(fcol) or "")
            lab = str(raw.get(lcol) or "").strip()
            if not rel or not lab:
                n_blank += 1
                continue
            rows.append((rel, _validate_class_label(lab, f"{p.name} row {n_seen}")))

    if not rows:
        raise SystemExit(
            f"[labels] {p}: no usable row ({n_seen} read, {n_blank} with an "
            f"empty cell).")
    res = CsvLabelResolver(p, rows)
    print(f"[labels] {p.name}: {len(rows)} row(s), {len(res.labels)} label(s)"
          + (f", {n_blank} row(s) skipped for an empty cell" if n_blank else ""))
    return res


def build_label_resolver(args):
    src = str(getattr(args, "label_source", "dir") or "dir").strip().lower()
    if src not in LABEL_SOURCES:
        raise SystemExit(f"[labels] label_source must be one of "
                         f"{list(LABEL_SOURCES)}, got {src!r}.")
    if src == "dir":
        if getattr(args, "label_csv", None):
            print("[labels] label_csv is set but label_source is 'dir': the CSV "
                  "is NOT read, classes come from the directory tree.")
        return None
    if not getattr(args, "label_csv", None):
        raise SystemExit("[labels] --label_source csv needs --label_csv "
                         "<metadata.csv>.")
    if getattr(args, "single_class", False):
        raise SystemExit(
            "[labels] --single_class and --label_source csv contradict each "
            "other: the first puts every source in ONE class, the second gives "
            "each file the class its CSV row names. Choose one.")
    return load_label_csv(args.label_csv)


def print_class_histogram(files, max_shown: int = 24) -> None:
    if not files:
        return
    counts = {}
    for _p, _rel_parent, leaf, _h, _rel in files:
        counts[leaf] = counts.get(leaf, 0) + 1
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    body = ", ".join(f"{c} {n}" for c, n in order[:max_shown])
    tail = ""
    if len(order) > max_shown:
        rest = sum(n for _c, n in order[max_shown:])
        tail = (f", ... {len(order) - max_shown} more class(es), "
                f"{rest} file(s)")
    print(f"  classes: {body}{tail}")


def labels_provenance(args, resolver) -> dict:
    prov = {"source": "csv" if resolver is not None else "dir"}
    if resolver is not None:
        prov.update({"csv": str(args.label_csv),
                     "csv_name": resolver.csv_path.name})
    elif getattr(args, "single_class", False):
        prov["single_class"] = str(getattr(args, "class_name", None) or "")
    return prov


def load_labels_provenance(out_root: Path) -> dict:
    p = out_root / MANIFEST_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("labels", {}) or {}
    except Exception:
        return {}


def check_labels_provenance(out_root: Path, prov: dict) -> None:
    old = load_labels_provenance(out_root)
    if not old:
        if not load_source_manifest(out_root):
            return
        old = {"source": "dir"}
    if old.get("source") != prov.get("source"):
        raise SystemExit(
            f"[labels] {out_root} was built with label_source="
            f"{old.get('source')!r} and this run uses {prov.get('source')!r}.\n"
            f"The labelling mode decides the output FOLDER of every latent, so "
            f"re-running with the other one does not overwrite the dataset -- "
            f"it writes a SECOND copy under different class folders, and the "
            f"split and the manifest then describe both at once. Use a FRESH "
            f"output dir.")
    differing = {k: (old.get(k), prov.get(k)) for k in ("csv_name",)
                 if k in old and old.get(k) != prov.get(k)}
    if differing:
        print(f"[labels] WARNING: a different CSV from the one recorded for "
              f"{out_root} (old vs new): {differing}. A source whose label "
              f"changed keeps its OLD latents under the OLD class folder -- "
              f"those become orphans (see --prune_orphans).")


def scan_audio_files(
    source_dir: str,
    single_class: bool = False,
    class_name: Optional[str] = None,
    label_resolver=None,
) -> List[Tuple[Path, Path, str, str, str]]:
    src = Path(source_dir)
    files = sorted(
        p for p in src.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS
    )

    out: List[Tuple[Path, Path, str, str, str]] = []
    unlabelled: List[str] = []
    ambiguous: List[str] = []
    for p in files:
        rel_posix = p.relative_to(src).as_posix()
        if label_resolver is not None:
            label, status = label_resolver.lookup(rel_posix)
            if status == "ambiguous":
                ambiguous.append(rel_posix)
                continue
            if status == "missing":
                unlabelled.append(rel_posix)
                continue
            rel_parent = Path(label)
        elif single_class:
            rel_parent = Path(class_name or src.name)
        else:
            rp = p.parent.relative_to(src)
            rel_parent = rp if str(rp) != "." else Path(class_name or src.name)
        leaf_class = rel_parent.name
        src_hash = hashlib.sha1(rel_posix.encode("utf-8")).hexdigest()[:8]
        out.append((p, rel_parent, leaf_class, src_hash, rel_posix))

    if label_resolver is not None:
        _report_csv_labelling(label_resolver, len(files), len(out),
                              unlabelled, ambiguous)
    return out


def _report_csv_labelling(resolver, n_files: int, n_kept: int,
                          unlabelled: List[str], ambiguous: List[str]) -> None:
    if ambiguous:
        raise SystemExit(
            f"[labels] {len(ambiguous)} source file(s) match SEVERAL rows of "
            f"{resolver.csv_path.name} carrying DIFFERENT labels, so the CSV "
            f"cannot say what class they are: "
            f"{ambiguous[:8]}{' ...' if len(ambiguous) > 8 else ''}\n"
            f"This happens when the file column holds bare names and the same "
            f"name appears in two folders under different labels. Fix by "
            f"putting the path relative to source_dir in the file column, or "
            f"by removing the duplicate rows.")

    if unlabelled:
        raise SystemExit(
            f"[labels] {len(unlabelled)} source file(s) have no row in "
            f"{resolver.csv_path.name}: "
            f"{unlabelled[:8]}{' ...' if len(unlabelled) > 8 else ''}\n"
            f"Every source must be named by the CSV. Add the rows, or point "
            f"--label_csv at a CSV that covers this source_dir.")

    print(f"[labels] {n_kept}/{n_files} source file(s) labelled from "
          f"{resolver.csv_path.name} ({resolver.n_rows} row(s))")
    unmatched = resolver.unmatched_rows()
    if unmatched:
        print(f"[labels] {len(unmatched)}/{resolver.n_rows} CSV row(s) matched "
              f"no file under this source_dir: "
              f"{unmatched[:8]}{' ...' if len(unmatched) > 8 else ''}")


MANIFEST_NAME = "source_manifest.json"


def _source_identity(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def load_source_manifest(out_root: Path) -> dict:
    p = out_root / MANIFEST_NAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("sources", {})
    except Exception:
        print(f"[manifest] WARNING: {p} is unreadable, ignoring it.")
        return {}


def audit_sources(prev: dict, files) -> Tuple[list, list]:
    changed, current = [], {}
    for p, _rp, _leaf, _h, rel in files:
        try:
            current[rel] = _source_identity(p)
        except OSError:
            continue
    for rel, ident in current.items():
        old = prev.get(rel)
        if old and (old.get("size"), old.get("mtime_ns")) != (ident["size"],
                                                             ident["mtime_ns"]):
            changed.append(rel)
    removed = [rel for rel in prev if rel not in current]
    return sorted(changed), sorted(removed)


def write_source_manifest(out_root: Path, prev: dict, produced: dict,
                          removed: list, prune: bool,
                          labels: Optional[dict] = None):
    merged = dict(prev)
    if prune:
        for rel in removed:
            merged.pop(rel, None)
    merged.update(produced)
    p = out_root / MANIFEST_NAME
    payload = {"sources": merged}
    labels = labels or load_labels_provenance(out_root)
    if labels:
        payload["labels"] = labels
    _atomic_write_json(p, payload)
    return len(merged)


def source_outputs_complete(chunks, latent_root: Path, cond_root: Path,
                            wav_root: Path, n_frames: int, cond_names,
                            check_latents: bool, want_wav: bool) -> bool:
    if not chunks:
        return False
    cond_names = list(cond_names or ())
    for rel_chunk in chunks:
        if check_latents and not _latent_file_is_valid(
                latent_root / f"{rel_chunk}.npy", n_frames, verbose=False):
            return False
        if cond_names and _npz_missing(cond_root / f"{rel_chunk}.npz", cond_names):
            return False
        if want_wav and not (wav_root / f"{rel_chunk}.wav").exists():
            return False
    return True


def filter_complete_sources(files, prev_manifest: dict, changed_srcs,
                            latent_root: Path, cond_root: Path, wav_root: Path,
                            n_frames: int, cond_names, check_latents: bool,
                            wav_sources) -> Tuple[list, int]:
    changed = set(changed_srcs or ())
    kept, skipped = [], 0
    for item in files:
        rel_src = item[4]
        entry = prev_manifest.get(rel_src)
        if entry is None or rel_src in changed:
            kept.append(item)
            continue
        want_wav = (wav_sources == "all"
                    or (isinstance(wav_sources, (set, frozenset))
                        and rel_src in wav_sources))
        if source_outputs_complete(
                entry.get("chunks", []), latent_root, cond_root, wav_root,
                n_frames, cond_names, check_latents, want_wav):
            skipped += 1
        else:
            kept.append(item)
    return kept, skipped


def prune_orphans(out_root: Path, prev: dict, removed: list) -> int:
    latents = out_root / "latents"
    conds = out_root / "conditions"
    wavs = out_root / "wav"
    n = 0
    for rel in removed:
        for chunk in prev.get(rel, {}).get("chunks", []):
            for root, ext in ((latents, ".npy"), (conds, ".npz"), (wavs, ".wav")):
                f = root / f"{chunk}{ext}"
                if f.exists():
                    f.unlink()
                    n += 1
    return n


SPLITS_NAME = "splits.json"
SPLIT_NAMES = ("train", "val", "test")


def source_group_key(rel_parent: Path, src_name: str, src_hash: str) -> str:
    return (Path(rel_parent) / f"{sanitize_filename(src_name)}_{src_hash}").as_posix()


def _split_params(ratios, seed: int, stratify_by_class: bool) -> dict:
    return {
        "ratios": [float(r) for r in ratios],
        "seed": int(seed),
        "stratify_by_class": bool(stratify_by_class),
        "group_by_source": True,
        "unit": "source",
    }


def load_splits(out_root: Path) -> Optional[dict]:
    p = Path(out_root) / SPLITS_NAME
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except Exception as e:
        raise SystemExit(
            f"[split] {p} exists but is unreadable ({e}). Refusing to continue: "
            f"silently recomputing the split would change the test set of a "
            f"dataset that already has one. Fix or delete the file."
        )
    if "groups" not in payload:
        raise SystemExit(f"[split] {p} has no 'groups' key -- refusing to guess.")
    return payload


def assign_source_splits(files, ratios, seed: int, stratify_by_class: bool,
                         existing: Optional[dict] = None) -> Tuple[dict, int]:
    from audio_dataset_npy import _allocate_three, _seed_for

    existing = dict(existing or {})
    buckets: dict = {}
    key_class = {}
    for path, rel_parent, leaf_class, src_hash, _rel in files:
        gk = source_group_key(rel_parent, Path(path).name, src_hash)
        bucket = leaf_class if stratify_by_class else "__all__"
        key_class[gk] = bucket
        buckets.setdefault(bucket, set()).add(gk)

    out = {}
    n_new = 0
    for bucket in sorted(buckets):
        keys = sorted(buckets[bucket])
        have = {k: existing[k] for k in keys if k in existing}
        new_keys = [k for k in keys if k not in existing]
        random.Random(_seed_for(seed, bucket)).shuffle(new_keys)
        out.update(have)
        if not new_keys:
            continue
        n_new += len(new_keys)

        if not have:
            n_tr, n_val, _n_te = _allocate_three(len(new_keys), ratios)
            for i, k in enumerate(new_keys):
                if i < n_tr:
                    out[k] = "train"
                elif i < n_tr + n_val:
                    out[k] = "val"
                else:
                    out[k] = "test"
            continue

        total = len(have) + len(new_keys)
        target = dict(zip(SPLIT_NAMES, _allocate_three(total, ratios)))
        current = {s: sum(1 for v in have.values() if v == s) for s in SPLIT_NAMES}
        it = iter(new_keys)
        for s in ("test", "val", "train"):
            deficit = max(0, target[s] - current[s])
            for _ in range(deficit):
                k = next(it, None)
                if k is None:
                    break
                out[k] = s
        for k in it:
            out[k] = "train"

    for k, v in existing.items():
        out.setdefault(k, v)
    return out, n_new


def summarize_splits(groups: dict) -> dict:
    counts = {s: 0 for s in SPLIT_NAMES}
    for v in groups.values():
        if v in counts:
            counts[v] += 1
    return counts


def write_splits(out_root: Path, groups: dict, params: dict, source: str):
    payload = {
        "version": 1,
        "unit": "source",
        "source": source,
        "params": params,
        "counts": summarize_splits(groups),
        "groups": dict(sorted(groups.items())),
    }
    _atomic_write_json(Path(out_root) / SPLITS_NAME, payload)
    return payload


def count_chunks_by_split(out_root: Path, groups: dict) -> Optional[dict]:
    sources = load_source_manifest(Path(out_root))
    if not sources:
        return None
    counts = {s: 0 for s in SPLIT_NAMES}
    unassigned = 0
    for info in sources.values():
        for rel_chunk in info.get("chunks", []):
            parent, _, name = rel_chunk.rpartition("/")
            stem = name.split("__")[0]
            gk = f"{parent}/{stem}" if parent else stem
            split = groups.get(gk)
            if split in counts:
                counts[split] += 1
            else:
                unassigned += 1
    return {"counts": counts, "unassigned": unassigned}


def record_chunk_counts(out_root: Path, groups: dict) -> Optional[dict]:
    res = count_chunks_by_split(out_root, groups)
    payload = load_splits(out_root)
    if res is None or payload is None:
        return None
    payload["chunk_counts"] = res["counts"]
    payload["chunk_counts_unassigned"] = res["unassigned"]
    _atomic_write_json(Path(out_root) / SPLITS_NAME, payload)
    return res


def print_chunk_counts(out_root: Path, groups: dict):
    res = record_chunk_counts(out_root, groups)
    if res is None:
        print("  samples per split: not counted "
              f"({MANIFEST_NAME} records no source yet -- nothing encoded)")
        return
    c = res["counts"]
    print(f"  samples (latent chunks) per split: train {c['train']} | "
          f"val {c['val']} | test {c['test']}  (tot {sum(c.values())})")
    print(f"  recorded as 'chunk_counts' in {Path(out_root) / SPLITS_NAME}")
    if res["unassigned"]:
        print(f"  WARNING: {res['unassigned']} chunk(s) belong to a source with "
              f"no split assignment -- re-run with --split_only to assign it.")


def resolve_splits(out_root: Path, files, ratios, seed: int,
                   stratify_by_class: bool, resplit: bool) -> dict:
    want = _split_params(ratios, seed, stratify_by_class)
    prev = load_splits(out_root)

    if prev is not None and resplit:
        print(f"[split] --resplit: discarding the recorded split "
              f"({summarize_splits(prev['groups'])}) and deciding it again. "
              f"Any checkpoint trained on the previous split is now evaluated "
              f"on data it may have seen.")
        prev = None

    existing = None
    if prev is not None:
        old = prev.get("params", {})
        diff = {k: (old.get(k), want[k]) for k in want if old.get(k) != want[k]}
        if diff:
            raise SystemExit(
                f"[split] {Path(out_root) / SPLITS_NAME} was written with "
                f"different split parameters (recorded vs requested): {diff}\n"
                f"  The recorded split is what the existing latents were (or "
                f"will be) trained against. Honour it by passing the same "
                f"values, or pass --resplit to decide the split again from "
                f"scratch -- which reshuffles train/val/test and invalidates "
                f"every evaluation made against the old one."
            )
        existing = prev["groups"]

    groups, n_new = assign_source_splits(
        files, ratios, seed, stratify_by_class, existing=existing)

    if existing is None:
        write_splits(out_root, groups, want, "assigned")
        print(f"[split] created {Path(out_root) / SPLITS_NAME}: "
              f"{summarize_splits(groups)} source(s)")
    elif n_new:
        write_splits(out_root, groups, want, prev.get("source", "assigned"))
        print(f"[split] {n_new} new source(s) assigned into the existing split "
              f"-> {summarize_splits(groups)}")
    else:
        print(f"[split] reusing {Path(out_root) / SPLITS_NAME}: "
              f"{summarize_splits(groups)} source(s), nothing new to assign")

    counts = summarize_splits(groups)
    r = dict(zip(SPLIT_NAMES, want["ratios"]))
    for name in ("val", "test"):
        if r[name] > 0 and counts[name] == 0:
            raise SystemExit(
                f"[split] the '{name}' split is EMPTY despite ratio={r[name]} > 0. "
                f"Every class has too few source files (a class with a single "
                f"source goes entirely to train, which is what keeps the split "
                f"leakage-safe). Add more sources, disable stratification, or "
                f"set the '{name}' ratio to 0 on purpose.")
    return groups


def import_legacy_split(out_root: Path, latent_root: Path, ratios, seed: int,
                        stratify_by_class: bool) -> dict:
    from audio_dataset_npy import compute_split, _source_group_of

    latent_root = Path(latent_root)
    if not latent_root.exists():
        raise SystemExit(f"[split] --import_legacy_split: no latents under "
                         f"{latent_root}. Nothing to import.")
    res = compute_split(
        latent_root, ratios=tuple(ratios), seed=seed,
        group_by_source=True, stratify_by_class=stratify_by_class,
        save_test_manifest=False,
    )
    groups = {}
    for name in SPLIT_NAMES:
        for f in res["splits"][name]:
            groups[_source_group_of(Path(f), latent_root)] = name
    params = _split_params(ratios, seed, stratify_by_class)
    write_splits(out_root, groups, params, "imported-from-training-split")
    print(f"[split] imported the in-code training split into "
          f"{out_root / SPLITS_NAME}: {summarize_splits(groups)} source(s)")
    return groups


class VendoredChunker:
    def __init__(
        self,
        chunk_length: int,
        chunk_overlap: int = 0,
        duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        audio_sr: int = 44100,
        mono: bool = True,
        silence_threshold: Optional[float] = None,
        max_silence_ratio: Optional[float] = None,
        peak_threshold: Optional[float] = 1.0,
        max_peak_ratio: Optional[float] = None,
        clip: bool = True,
        chunk_min_rms_threshold: Optional[float] = None,
        chunk_min_peak_threshold: Optional[float] = None,
        pad_and_keep_last_chunk: bool = False,
        pad_value: float = 0.0,
        keep_num_chunks_per_file: Optional[int] = None,
    ):
        assert chunk_length > 0
        assert 0 <= chunk_overlap < chunk_length
        self.chunk_length = chunk_length
        self.chunk_overlap = chunk_overlap
        self.hop_length = chunk_length - chunk_overlap
        self.duration = duration
        self.min_duration = min_duration
        self.audio_sr = audio_sr
        self.mono = mono
        self.silence_threshold = silence_threshold
        self.max_silence_ratio = max_silence_ratio
        self.peak_threshold = peak_threshold if peak_threshold is not None else 1.0
        self.max_peak_ratio = max_peak_ratio
        self.clip = clip
        self.chunk_min_rms_threshold = chunk_min_rms_threshold
        self.chunk_min_peak_threshold = chunk_min_peak_threshold
        self.pad_and_keep_last_chunk = pad_and_keep_last_chunk
        self.pad_value = pad_value
        self.keep_num_chunks_per_file = keep_num_chunks_per_file

    def load_audio(self, filepath):
        import numpy as np
        try:
            import soundfile as sf
            data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T.copy())
        except Exception as e:
            wav, sr = _ff_decode(filepath)
            if wav is None:
                print(f"[skip] cannot decode {Path(filepath).name} "
                      f"(soundfile: {type(e).__name__}; ffmpeg failed too)")
                return None

        if self.mono:
            wav = wav.mean(dim=0, keepdim=True)

        if self.min_duration is not None and (wav.shape[-1] / sr < self.min_duration):
            return None

        if self.max_peak_ratio is not None and self.peak_threshold is not None:
            peak_ratio = (wav.abs() >= self.peak_threshold).float().mean().item()
            if peak_ratio > self.max_peak_ratio:
                return None

        if self.max_silence_ratio is not None and self.silence_threshold is not None:
            silence_ratio = (wav.abs() < self.silence_threshold).float().mean().item()
            if silence_ratio >= self.max_silence_ratio:
                return None

        if self.audio_sr is not None and sr != self.audio_sr:
            import librosa
            y = librosa.resample(wav.cpu().numpy(), orig_sr=sr,
                                 target_sr=self.audio_sr, axis=-1)
            wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            sr = self.audio_sr

        if self.duration is not None:
            wav = wav[..., :int(self.duration * sr)]

        if self.clip:
            wav = wav.clamp(-self.peak_threshold, self.peak_threshold)

        return wav

    def _chunk_offsets(self, length: int) -> List[int]:
        if length < self.chunk_length:
            return []
        num_offsets = (length - self.chunk_length) // self.hop_length + 1
        if (self.keep_num_chunks_per_file is not None
                and self.keep_num_chunks_per_file < num_offsets):
            import torch
            bins = torch.linspace(0, num_offsets, steps=self.keep_num_chunks_per_file + 1)
            centers = ((bins[:-1] + bins[1:]) / 2).long().tolist()
            indices = sorted(set(int(c) for c in centers))
        else:
            indices = list(range(num_offsets))
        return [i * self.hop_length for i in indices]

    def _keep_chunk(self, chunk) -> bool:
        rms_thr = self.chunk_min_rms_threshold
        peak_thr = self.chunk_min_peak_threshold
        if rms_thr is None and peak_thr is None:
            return True
        rms_ok = (rms_thr is not None) and bool((chunk ** 2).mean().sqrt() > rms_thr)
        peak_ok = (peak_thr is not None) and bool(chunk.abs().max() > peak_thr)
        return rms_ok or peak_ok

    def iter_file_chunks(self, filepath) -> Iterator[Tuple[int, "object", int]]:
        import torch.nn.functional as F
        wav = self.load_audio(filepath)
        if wav is None:
            return
        length = wav.shape[-1]
        offsets = self._chunk_offsets(length)

        kept = 0
        last_end = 0
        for off in offsets:
            chunk = wav[..., off:off + self.chunk_length]
            last_end = off + self.hop_length
            if self._keep_chunk(chunk):
                yield kept, chunk, off
                kept += 1

        if self.pad_and_keep_last_chunk:
            if not offsets and length > 0:
                pad = self.chunk_length - length
                chunk = F.pad(wav, (0, pad), value=self.pad_value)
                if self._keep_chunk(chunk):
                    yield kept, chunk, 0
            elif offsets and last_end < length:
                pad = self.chunk_length - (length - last_end)
                chunk = F.pad(wav[..., last_end:], (0, pad), value=self.pad_value)
                if self._keep_chunk(chunk):
                    yield kept, chunk, last_end


class DACEncoder:
    def __init__(self, device: str = "cuda", codec: str = lc.DEFAULT_CODEC):
        self.spec = lc.get_spec(codec)
        tag = f"[{self.spec.label}]"
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"{tag} CUDA not available -> CPU")
            device = "cpu"
        self.device = device
        self.torch = torch
        print(f"{tag} loading {self.spec.name} on {device} ...")
        self.model = lc.load_model(self.spec.name, device)
        print(f"{tag} model loaded.")

    def encode(self, chunk, sr: int):
        return self.encode_batch([chunk], sr)[0]

    def encode_batch(self, chunks, sr: int):
        import numpy as np
        torch = self.torch
        mats = []
        for c in chunks:
            w = c
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if w.dim() == 2:
                w = w.unsqueeze(0)
            mats.append(w)
        batch = torch.cat(mats, dim=0).to(self.device)
        latents = lc.encode(self.model, batch, sr)
        latents = latents.cpu().numpy().astype(np.float32)
        return [latents[i] for i in range(latents.shape[0])]

    def n_frames_for(self, chunk_length: int, sr: int) -> int:
        z = self.encode(self.torch.zeros(1, chunk_length), sr)
        return int(z.shape[1])


def _global_is_chunk_level(ext) -> bool:
    return callable(getattr(ext, "encode_audio", None))


def _split_globals_by_stage(registry):
    if registry is None:
        return [], []
    chunk, klass = [], []
    for name, ext in getattr(registry, "global_extractors", {}).items():
        (chunk if _global_is_chunk_level(ext) else klass).append(name)
    return sorted(chunk), sorted(klass)


def _chunk_extractor(registry, name):
    ext = getattr(registry, "frame_extractors", {}).get(name)
    if ext is not None:
        return ext, "frame"
    ext = getattr(registry, "global_extractors", {}).get(name)
    if ext is not None and _global_is_chunk_level(ext):
        return ext, "global"
    raise KeyError(
        f"'{name}' is not a per-chunk condition of this registry "
        f"(frame: {sorted(getattr(registry, 'frame_extractors', {}))}, "
        f"per-chunk global: {_split_globals_by_stage(registry)[0]})")


def _companion_suffixes(ext) -> tuple:
    return tuple(getattr(ext, "companion_suffixes", None) or ())


class CompanionNotes:
    def __init__(self, registry, path: Path):
        self.registry = registry
        self.path = Path(path)
        self.t0 = 0.0
        self._parsed = {}

    def data(self, name):
        if name not in self._parsed:
            ext = self.registry.frame_extractors[name]
            self._parsed[name] = ext.load_companion(self.path)
        return self._parsed[name]


def find_companions(files, suffixes) -> Tuple[dict, list, list]:
    wanted = {s.lower() for s in suffixes}
    listing, found, missing, ambiguous = {}, {}, [], []
    for path, _rp, _leaf, _h, rel in files:
        folder = Path(path).parent
        if folder not in listing:
            index = {}
            for q in folder.iterdir():
                if q.is_file() and q.suffix.lower() in wanted:
                    index.setdefault(q.stem, []).append(q)
            listing[folder] = index
        cands = listing[folder].get(Path(path).stem, [])
        if len(cands) == 1:
            found[rel] = cands[0]
        elif not cands:
            missing.append(rel)
        else:
            ambiguous.append((rel, sorted(cands)))
    return found, missing, ambiguous


def _npz_missing(cond_path: Path, names, force: bool = False) -> bool:
    names = set(names or ())
    if not names:
        return False
    if force or not cond_path.exists():
        return True
    try:
        with np.load(str(cond_path)) as data:
            return not names.issubset(set(data.files))
    except Exception:
        return True


def extract_and_merge_chunk_conditions(
    registry, chunk_audio_np, sr: int, n_frames: int,
    cond_path: Path, force: bool = False, names=None,
    companion: Optional[CompanionNotes] = None,
) -> bool:
    import numpy as np
    if names is None:
        names = list(registry.frame_names) + _split_globals_by_stage(registry)[0]
    required = set(names)
    if not required:
        return False

    existing = {}
    if cond_path.exists():
        try:
            with np.load(str(cond_path)) as data:
                existing = {k: data[k] for k in data.keys()}
        except Exception:
            existing = {}
        if not force and required.issubset(existing.keys()):
            return False

    missing = required if force else (required - set(existing.keys()))
    if not missing:
        return False

    new = {}
    for name in missing:
        ext, kind = _chunk_extractor(registry, name)
        if kind == "frame":
            if _companion_suffixes(ext):
                if companion is None:
                    raise RuntimeError(
                        f"'{name}' is read from the file next to the source "
                        f"audio ({'/'.join(_companion_suffixes(ext))}), not "
                        f"from the chunk: it can only be extracted inside "
                        f"preprocess_stream's chunking loop.")
                arr = ext.from_companion(companion.data(name), companion.t0,
                                         n_frames)
            else:
                arr = ext.extract(chunk_audio_np, sr, n_frames)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != int(n_frames):
                raise RuntimeError(
                    f"frame condition '{name}' produced {arr.shape} for "
                    f"{cond_path.name}; expected ({n_frames}, dim).")
        else:
            arr = np.asarray(ext.encode_audio(chunk_audio_np, sr),
                             dtype=np.float32).reshape(-1)
            if arr.size == 0:
                raise RuntimeError(
                    f"global condition '{name}' produced an empty embedding "
                    f"for {cond_path.name}.")
        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"condition '{name}' contains NaN/Inf for {cond_path.name}.")
        new[name] = arr

    final = {**existing, **new}
    _atomic_save_npz(cond_path, final)
    return True


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def write_text_label_vocab(registry, out_root: Path, vocab_path: Optional[str],
                           force: bool = False) -> int:
    import numpy as np
    ext = getattr(registry, "global_extractors", {}).get("text")
    if ext is None:
        return 0

    if vocab_path:
        p = Path(vocab_path)
        if not p.exists():
            print(f"[global/text] --text_vocab {p} not found -> using the "
                  f"built-in vocabulary")
            phrases = None
        else:
            phrases = [ln.strip() for ln in
                       p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            print(f"[global/text] vocabulary from {p}: {len(phrases)} phrases")
    else:
        phrases = None
    if not phrases:
        from conditions import TEXT_LABEL_VOCAB
        phrases = list(TEXT_LABEL_VOCAB)

    d = out_root / "global_conditions"
    d.mkdir(parents=True, exist_ok=True)
    npy, js = d / "text_vocab.npy", d / "text_vocab.json"
    if npy.exists() and js.exists() and not force:
        try:
            old = json.loads(js.read_text(encoding="utf-8"))
            if list(old.get("phrases", [])) == phrases:
                print(f"[global/text] label vocabulary already current "
                      f"({len(phrases)} phrases)")
                return len(phrases)
        except Exception:
            pass

    embs = np.asarray(ext.encode_batch(phrases), dtype=np.float32)
    _atomic_save_npy(npy, embs)
    _atomic_write_json(js, {"phrases": phrases,
                            "model_name": getattr(ext, "model_name", None)})
    print(f"[global/text] label vocabulary: {len(phrases)} phrases encoded "
          f"-> {npy.name}")
    return len(phrases)


TEXT_LABELS_JSONL = "text_labels.jsonl"
TEXT_LABELS_COS = "text_labels_cos.npy"
TEXT_LABELS_EMB = "text_labels_emb.npy"
TEXT_LABELS_TOK = "text_labels_tok.npy"
TEXT_LABELS_TOKLEN = "text_labels_tok_len.npy"
TEXT_LABELS_META = "text_labels.json"


def _text_labels_fingerprint(phrases, model_name, n_terms) -> str:
    h = hashlib.sha1()
    h.update(repr(list(phrases)).encode("utf-8"))
    h.update(str(model_name).encode("utf-8"))
    h.update(str(int(n_terms)).encode("utf-8"))
    return h.hexdigest()[:16]


def format_text_caption(class_name: str, phrases_cos, quote: bool = True) -> str:
    if not phrases_cos:
        return str(class_name)
    if not quote:
        return ", ".join([str(class_name)] + [str(p) for p, _c in phrases_cos])
    body = ", ".join(f"\"{p}\" ({c:+.2f})" for p, c in phrases_cos)
    return f"{class_name} · {body}"


def write_text_labels(out_root: Path, cond_root: Path, n_terms: int = 2,
                      force: bool = False, text_extractor=None) -> int:
    import numpy as np

    d = out_root / "global_conditions"
    vocab_npy, vocab_js = d / "text_vocab.npy", d / "text_vocab.json"
    if not (vocab_npy.exists() and vocab_js.exists()):
        print("[global/text] no text_vocab on disk -> no per-chunk labels "
              "written (nothing to compare the vectors against).")
        return 0
    try:
        meta = json.loads(vocab_js.read_text(encoding="utf-8"))
        phrases = list(meta.get("phrases", []))
        model_name = meta.get("model_name")
        vocab = np.load(str(vocab_npy)).astype(np.float32)
    except Exception as e:
        print(f"[global/text] text_vocab unreadable ({type(e).__name__}: {e}) "
              f"-> no per-chunk labels written.")
        return 0
    if not phrases or vocab.ndim != 2 or vocab.shape[0] != len(phrases):
        print(f"[global/text] text_vocab is inconsistent "
              f"({len(phrases)} phrases vs {getattr(vocab, 'shape', None)}) "
              f"-> no per-chunk labels written.")
        return 0

    n_terms = max(1, int(n_terms))
    k = min(max(0, n_terms - 1), len(phrases))

    files = sorted(p for p in cond_root.rglob("*.npz")) if cond_root.exists() else []
    if not files:
        print("[global/text] no per-chunk conditions on disk -> no labels.")
        return 0

    fp = _text_labels_fingerprint(phrases, model_name, n_terms)
    jsonl_p, cos_p, meta_p = (d / TEXT_LABELS_JSONL, d / TEXT_LABELS_COS,
                              d / TEXT_LABELS_META)
    if not force and jsonl_p.exists() and cos_p.exists() and meta_p.exists():
        try:
            old = json.loads(meta_p.read_text(encoding="utf-8"))
            wants = []
            if text_extractor is not None:
                wants.append(d / TEXT_LABELS_EMB)
                if hasattr(text_extractor, "encode_tokens"):
                    wants += [d / TEXT_LABELS_TOK, d / TEXT_LABELS_TOKLEN]
            absent = [p.name for p in wants if not p.exists()]
            if (old.get("fingerprint") == fp
                    and int(old.get("n_npz", -1)) == len(files)
                    and not absent):
                print(f"[global/text] per-chunk labels already current "
                      f"({old.get('n_chunks')} chunks, {n_terms} term(s))")
                return int(old.get("n_chunks", 0))
            if absent and old.get("fingerprint") == fp:
                print(f"[global/text] labels are current but {', '.join(absent)} "
                      f"{'is' if len(absent) == 1 else 'are'} missing -> "
                      f"rewriting the sidecar (no audio is read, no latent is "
                      f"re-encoded)")
        except Exception:
            pass

    rows, cos_rows, n_missing = [], [], 0
    for p in files:
        rel = p.relative_to(cond_root).with_suffix("").as_posix()
        try:
            with np.load(str(p)) as z:
                if "text" not in z.files:
                    n_missing += 1
                    continue
                vec = np.asarray(z["text"], dtype=np.float32).reshape(-1)
        except Exception as e:
            print(f"  [global/text] {rel}: {type(e).__name__}: {e} -> skipped")
            n_missing += 1
            continue
        if vec.shape[0] != vocab.shape[1] or not np.isfinite(vec).all():
            n_missing += 1
            continue
        sims = vocab @ vec
        cos_rows.append(sims.astype(np.float16))
        top = np.argsort(-sims)[:k] if k else []
        best = [[phrases[i], round(float(sims[i]), 4)] for i in top]
        cls = Path(rel).parent.name or ""
        rows.append({"chunk": rel, "class": cls, "phrases": best,
                     "caption": format_text_caption(cls, best)})

    if not rows:
        print(f"[global/text] no chunk carries a 'text' vector "
              f"({len(files)} .npz inspected) -> no labels written.")
        return 0

    captions, captions_text, caption_id = [], [], {}
    for r in rows:
        enc = format_text_caption(r["class"], r["phrases"], quote=False)
        if enc not in caption_id:
            caption_id[enc] = len(captions_text)
            captions_text.append(enc)
            captions.append(r["caption"])
        r["caption_id"] = caption_id[enc]
    cap_emb = None
    if text_extractor is not None:
        try:
            cap_emb = np.asarray(text_extractor.encode_batch(captions_text),
                                 dtype=np.float32)
            if cap_emb.ndim != 2 or cap_emb.shape[0] != len(captions):
                raise ValueError(f"encode_batch returned {cap_emb.shape} for "
                                 f"{len(captions)} caption(s)")
        except Exception as e:
            print(f"[global/text] caption embeddings NOT written "
                  f"({type(e).__name__}: {e}). The descriptions are still "
                  f"stored; sampling.validation_text_from_caption will have "
                  f"nothing to read.")
            cap_emb = None

    cap_tok = cap_len = None
    if text_extractor is not None and hasattr(text_extractor, "encode_tokens"):
        try:
            cap_tok, cap_len = text_extractor.encode_tokens(captions_text)
            if cap_tok.ndim != 3 or cap_tok.shape[0] != len(captions):
                raise ValueError(f"encode_tokens returned {cap_tok.shape} for "
                                 f"{len(captions)} caption(s)")
        except Exception as e:
            print(f"[global/text] caption TOKEN sequences NOT written "
                  f"({type(e).__name__}: {e}). A run with "
                  f"model.text_cross_every > 0 will refuse to start on this "
                  f"dataset; everything else is unaffected.")
            cap_tok = cap_len = None

    d.mkdir(parents=True, exist_ok=True)
    tmp = jsonl_p.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(str(tmp), str(jsonl_p))
    _atomic_save_npy(cos_p, np.stack(cos_rows, axis=0))
    if cap_emb is not None:
        _atomic_save_npy(d / TEXT_LABELS_EMB, cap_emb)
    else:
        try:
            (d / TEXT_LABELS_EMB).unlink()
        except FileNotFoundError:
            pass
    if cap_tok is not None:
        _atomic_save_npy(d / TEXT_LABELS_TOK, cap_tok.astype(np.float16))
        _atomic_save_npy(d / TEXT_LABELS_TOKLEN, cap_len.astype(np.int32))
    else:
        for stale in (TEXT_LABELS_TOK, TEXT_LABELS_TOKLEN):
            try:
                (d / stale).unlink()
            except FileNotFoundError:
                pass
    _atomic_write_json(meta_p, {
        "fingerprint": fp, "n_chunks": len(rows), "n_npz": len(files),
        "n_terms": n_terms,
        "n_phrases": len(phrases), "model_name": model_name,
        "phrases": phrases,
        "captions": captions,
        "captions_text": captions_text,
        "caption_emb": bool(cap_emb is not None),
        "caption_tokens": bool(cap_tok is not None),
        "caption_tok_dim": int(cap_tok.shape[2]) if cap_tok is not None else 0,
        "caption_tok_max_len": int(cap_tok.shape[1]) if cap_tok is not None else 0,
        "cos_dtype": "float16",
        "note": ("Row i of text_labels_cos.npy is line i of text_labels.jsonl. "
                 "The phrases are a closed-vocabulary retrieval over the stored "
                 "CLAP vector, not a description: read every one with its "
                 "cosine. The class is exact, read off the chunk's folder."),
    })

    distinct = len({tuple(p for p, _c in r["phrases"]) for r in rows})
    print(f"[global/text] per-chunk labels: {len(rows)} chunk(s), "
          f"{n_terms} term(s) each, {distinct} distinct phrase combination(s) "
          f"over {len(phrases)} phrases -> {jsonl_p.name}")
    print(f"[global/text] {len(captions)} distinct caption(s)"
          + (f", CLAP-text encoded -> {TEXT_LABELS_EMB}" if cap_emb is not None
             else ", NOT encoded (no text extractor)"))
    if cap_tok is not None:
        print(f"[global/text] caption token sequences: "
              f"{cap_tok.shape[0]} x {cap_tok.shape[1]} token(s) x "
              f"{cap_tok.shape[2]} -> {TEXT_LABELS_TOK} "
              f"(lengths {cap_len.min()}..{cap_len.max()}, float16). "
              f"This is what model.text_cross_every reads.")
    if cap_emb is not None:
        if len(captions) > 1:
            M = cap_emb @ cap_emb.T
            off = M[~np.eye(len(captions), dtype=bool)]
            print(f"[global/text] caption embeddings: mean cosine between "
                  f"DIFFERENT captions {float(off.mean()):+.3f} "
                  f"(max {float(off.max()):+.3f}) -- the lower, the more the "
                  f"text slot can distinguish them")
        for _c, _t in list(zip(captions, captions_text))[:4]:
            print(f"    {_c!r}  ->  encoded as {_t!r}")
    if n_missing:
        print(f"[global/text] {n_missing} chunk(s) had no usable 'text' vector "
              f"and are absent from the sidecar.")
    return len(rows)


def extract_class_global_conditions(
    registry, out_root: Path, classes: List[str], image_root: Optional[str],
    force: bool = False,
) -> dict:
    import numpy as np
    _, class_globals = _split_globals_by_stage(registry)
    if not class_globals:
        return {}

    written = {}
    if "image" in class_globals:
        if not image_root or not Path(image_root).exists():
            print(f"[global/image] --image_root {image_root or '(unset)'} "
                  f"missing/not found -> NOTHING WRITTEN. The image condition "
                  f"will have no bank to read.")
            return {}
        ext = registry.global_extractors["image"]
        d = out_root / "global_conditions" / "image"
        d.mkdir(parents=True, exist_ok=True)
        missing_dir, empty_dir, refreshed = [], [], []
        for c in classes:
            p = d / f"{sanitize_class_name(c)}.npy"
            jp = p.with_suffix(".json")
            cls_dir = Path(image_root) / c
            if not cls_dir.exists():
                if p.exists() and jp.exists():
                    try:
                        written[c] = int(np.load(str(p), mmap_mode="r").shape[0])
                        continue
                    except Exception:
                        pass
                missing_dir.append(c)
                continue
            imgs = sorted(q for q in cls_dir.rglob("*")
                          if q.suffix.lower() in IMAGE_EXTS)
            if p.exists() and jp.exists() and not force:
                try:
                    recorded = list(json.loads(
                        jp.read_text(encoding="utf-8")).get("files", []))
                    n_rows = int(np.load(str(p), mmap_mode="r").shape[0])
                    if (len(recorded) == n_rows
                            and recorded == [q.name for q in imgs]):
                        written[c] = n_rows
                        continue
                    if imgs:
                        refreshed.append(f"{c}({len(recorded)}->{len(imgs)})")
                except Exception:
                    pass
            if not imgs:
                empty_dir.append(c)
                continue
            stack, kept = [], []
            for q in imgs:
                try:
                    stack.append(np.asarray(ext.encode_image(str(q)),
                                            dtype=np.float32))
                    kept.append(q.name)
                except Exception as e:
                    print(f"  [global/image] {c}/{q.name}: "
                          f"{type(e).__name__}: {e} -> skipped")
            if not stack:
                empty_dir.append(c)
                continue
            _atomic_save_npy(p, np.stack(stack, axis=0))
            _atomic_write_json(jp, {"class": c, "files": kept})
            written[c] = len(kept)

        if hasattr(ext, "unload"):
            ext.unload()
        total = sum(written.values())
        print(f"[global/image] {total} images over "
              f"{len(written)}/{len(classes)} classes -> {d}")
        if refreshed:
            print(f"[global/image] re-encoded {len(refreshed)} class(es) whose "
                  f"folder changed since the bank was written: "
                  f"{refreshed[:8]}{' ...' if len(refreshed) > 8 else ''}")
        if missing_dir:
            print(f"[global/image] WARNING: no folder under {image_root} for "
                  f"{len(missing_dir)} class(es): "
                  f"{missing_dir[:8]}{' ...' if len(missing_dir) > 8 else ''}")
        if empty_dir:
            print(f"[global/image] WARNING: folder present but no usable image "
                  f"for {len(empty_dir)} class(es): "
                  f"{empty_dir[:8]}{' ...' if len(empty_dir) > 8 else ''}")
    return written


def _ff_duration(path) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def _ff_channels(path) -> int:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channels",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        return int(r.stdout.strip())
    except Exception:
        return 1


def _ff_sample_rate(path) -> Optional[int]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        return int(r.stdout.strip())
    except Exception:
        return None


def _ff_decode(path):
    import numpy as np
    sr = _ff_sample_rate(path)
    ch = _ff_channels(path)
    if not sr or not ch:
        return None, None
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
         "-filter_threads", "1", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0 or not r.stdout:
        return None, None
    a = np.frombuffer(r.stdout, dtype=np.float32)
    n = (a.size // ch) * ch
    if n == 0:
        return None, None
    a = a[:n].reshape(-1, ch).T
    return torch.from_numpy(np.ascontiguousarray(a)), sr


def _detect_trim_points(path, threshold_db: float) -> Tuple[float, float]:
    duration = _ff_duration(path)
    if duration == 0:
        return 0.0, 0.0
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-threads", "1",
             "-filter_threads", "1", "-i", str(path),
             "-af", f"silencedetect=noise={threshold_db}dB:d=0.1",
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
    except Exception:
        return 0.0, duration
    regions = []
    cur = None
    for line in (r.stderr or "").splitlines():
        if "silence_start:" in line:
            try:
                cur = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                cur = None
        elif "silence_end:" in line and cur is not None:
            try:
                regions.append((cur, float(line.split("silence_end:")[1].strip().split()[0])))
            except (ValueError, IndexError):
                pass
            cur = None
    if cur is not None:
        regions.append((cur, duration))
    if not regions:
        return 0.0, duration
    trim_start = regions[0][1] if regions[0][0] < 0.05 else 0.0
    trim_end = regions[-1][0] if regions[-1][1] >= duration - 0.05 else duration
    return trim_start, trim_end


def _analyze_loudness(path, target_lufs, target_tp, target_lra):
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-threads", "1",
             "-filter_threads", "1", "-i", str(path),
             "-af", (f"loudnorm=I={target_lufs}:TP={target_tp}:"
                     f"LRA={target_lra}:print_format=json"),
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        stderr = r.stderr or ""
        js, je = stderr.rfind("{"), stderr.rfind("}") + 1
        if js == -1 or je == 0:
            return None
        data = json.loads(stderr[js:je])
        return float(data.get("input_i", "nan")), float(data.get("input_tp", "nan"))
    except Exception:
        return None


def acoustic_preprocess_file(
    path, tmp_dir, sr, target_lufs, target_tp, target_lra,
    silence_trim_db, min_sec, stereo_split,
) -> List[Tuple[str, int, float]]:
    trim_start, trim_end = _detect_trim_points(path, silence_trim_db)
    dur = trim_end - trim_start
    if dur < min_sec:
        return []

    meas = _analyze_loudness(path, target_lufs, target_tp, target_lra)
    gain_filter = None
    if meas is not None:
        mi, mtp = meas
        if math.isfinite(mi) and math.isfinite(mtp):
            gain_db = min(target_lufs - mi, target_tp - mtp)
            gain_filter = f"volume={gain_db:.2f}dB"

    n_channels = _ff_channels(path)
    channels = [0, 1] if (stereo_split and n_channels >= 2) else [None]

    out: List[Tuple[str, int, float]] = []
    for ch in channels:
        filt = []
        if ch is not None:
            filt.append(f"pan=mono|c0=c{ch}")
        if gain_filter:
            filt.append(gain_filter)
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
        os.close(fd)
        cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-threads", "1",
               "-filter_threads", "1",
               "-ss", str(trim_start), "-i", str(path), "-t", str(dur)]
        if filt:
            cmd += ["-af", ",".join(filt)]
        cmd += ["-ar", str(sr), "-ac", "1", "-loglevel", "error", tmp]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError:
            Path(tmp).unlink(missing_ok=True)
            continue
        if _ff_duration(tmp) < min_sec:
            Path(tmp).unlink(missing_ok=True)
            continue
        out.append((tmp, ch if ch is not None else 0, float(trim_start)))
    return out


def _chunk_mean_dbfs(chunk) -> float:
    x = float(chunk.abs().mean().item())
    return 20.0 * math.log10(x + 1e-7)


def _shard_files(files, worker_id: int, num_workers: int):
    return files[worker_id::num_workers] if num_workers and num_workers > 1 else files


def _extractor_device_attr(ext) -> Optional[str]:
    for attr in ("device", "_device"):
        if hasattr(ext, attr):
            return attr
    return None


def _all_chunk_extractors(registry) -> dict:
    if registry is None:
        return {}
    out = dict(getattr(registry, "frame_extractors", {}))
    chunk_globals, _ = _split_globals_by_stage(registry)
    gexts = getattr(registry, "global_extractors", {})
    out.update({n: gexts[n] for n in chunk_globals})
    return out


def _split_extractors_by_device(registry):
    gpu, cpu = [], []
    for name, ext in _all_chunk_extractors(registry).items():
        on_gpu = _extractor_device_attr(ext) and not _companion_suffixes(ext)
        (gpu if on_gpu else cpu).append(name)
    return sorted(gpu), sorted(cpu)


def _set_extractor_device(registry, names, device: str):
    if registry is None:
        return
    exts = _all_chunk_extractors(registry)
    for name in names or ():
        ext = exts.get(name)
        if ext is None:
            continue
        attr = _extractor_device_attr(ext)
        if attr:
            setattr(ext, attr, device)


def _force_cpu_extractors(registry, names=None):
    if registry is None:
        return
    if names is None:
        names = list(_all_chunk_extractors(registry).keys())
    _set_extractor_device(registry, names, "cpu")


def _process_gpu_batch(dac_enc, batch, sr, n_frames: int,
                       registry=None, gpu_names=None, force=False):
    if not batch:
        return 0, 0

    n_lat = _encode_and_save_batch(
        dac_enc, [it for it in batch if it.get("latent_path")], sr, n_frames)

    n_cond = 0
    if registry is not None and gpu_names:
        for it in batch:
            cp = it.get("cond_path")
            if not cp:
                continue
            audio = it["audio"]
            chunk_np = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
            extract_and_merge_chunk_conditions(
                registry, chunk_np, sr, n_frames, Path(cp),
                force=force, names=gpu_names)
            n_cond += 1
    return n_lat, n_cond


def _encode_and_save_batch(dac_enc, batch, sr, n_frames: int) -> int:
    if not batch or dac_enc is None:
        return 0
    audios = [it["audio"] for it in batch]
    lats = dac_enc.encode_batch(audios, sr)
    tag, dim = f"[{dac_enc.spec.label}]", dac_enc.spec.latent_dim
    if len(lats) != len(batch):
        raise RuntimeError(
            f"{tag} Encoder returned {len(lats)} latents for "
            f"{len(batch)} input chunks. Refusing a partial batch."
        )
    n = 0
    for it, lat in zip(batch, lats):
        p = Path(it["latent_path"])
        lat = np.asarray(lat)
        if lat.shape != (dim, int(n_frames)):
            raise RuntimeError(
                f"{tag} Refusing latent with shape {lat.shape}; expected "
                f"({dim}, {n_frames}) for {p}"
            )
        if lat.dtype != np.dtype(np.float32):
            lat = lat.astype(np.float32)
        if not np.isfinite(lat).all():
            raise RuntimeError(f"{tag} Refusing NaN/Inf latent for {p}")
        _atomic_save_npy(p, lat)
        n += 1
    return n


class StreamingChunkDataset(IterableDataset):
    def __init__(self, files, chunker, registry, latent_root, wav_root, cond_root,
                 sr, wav_sources, force, n_frames_fixed,
                 acoustic, acoustic_kw, silence_thresh_db, tmp_dir,
                 cpu_cond_names=None, gpu_cond_names=None, companions=None,
                 codec_name=lc.DEFAULT_CODEC):
        self.files = files
        self.codec_name = codec_name
        self.chunker = chunker
        self.registry = registry
        self.cpu_cond_names = list(cpu_cond_names or [])
        self.gpu_cond_names = list(gpu_cond_names or [])
        self.companions = dict(companions or {})
        self.companion_names = [
            n for n in self.cpu_cond_names
            if _companion_suffixes(_all_chunk_extractors(registry).get(n))]
        self.latent_root = Path(latent_root)
        self.wav_root = Path(wav_root)
        self.cond_root = Path(cond_root)
        self.sr = sr
        self.wav_sources = wav_sources
        self.force = force
        self.n_frames_fixed = n_frames_fixed
        self.acoustic = acoustic
        self.acoustic_kw = acoustic_kw
        self.silence_thresh_db = silence_thresh_db
        self.tmp_dir = tmp_dir

    def __iter__(self):
        import soundfile as sf
        lc.activate(self.codec_name)
        wi = torch.utils.data.get_worker_info()
        if wi is None:
            shard = self.files
        else:
            shard = _shard_files(self.files, wi.id, wi.num_workers)

        has_cpu_conditions = self.registry is not None and bool(self.cpu_cond_names)
        has_gpu_conditions = self.registry is not None and bool(self.gpu_cond_names)

        for path, rel_parent, _leaf, src_hash, rel_src in shard:
            want_wav = (self.wav_sources == "all"
                        or (isinstance(self.wav_sources, (set, frozenset))
                            and rel_src in self.wav_sources))
            if self.acoustic:
                items = acoustic_preprocess_file(path, self.tmp_dir, self.sr, **self.acoustic_kw)
                cleanup = [w for w, _, _ in items]
            else:
                items = [(str(path), 0, 0.0)]
                cleanup = []
            companion = None
            if has_cpu_conditions and self.companion_names:
                if rel_src not in self.companions:
                    raise RuntimeError(
                        f"no file next to {rel_src} for "
                        f"{self.companion_names} (main checks this before "
                        f"streaming: the source list changed in between?)")
                companion = CompanionNotes(self.registry,
                                           self.companions[rel_src])
            produced_chunks = []
            try:
                for wav_src, channel, trim_start in items:
                    ch_suffix = f"__ch{channel}" if self.acoustic else ""
                    for idx, chunk, offset in self.chunker.iter_file_chunks(wav_src):
                        if self.acoustic and \
                                _chunk_mean_dbfs(chunk) < self.silence_thresh_db:
                            continue
                        if companion is not None:
                            companion.t0 = (trim_start + offset
                                            / float(self.chunker.audio_sr))

                        name = (f"{sanitize_filename(Path(path).name)}_{src_hash}"
                                f"{ch_suffix}__c{idx:04d}")
                        rel_chunk = f"{rel_parent.as_posix()}/{name}"
                        produced_chunks.append(rel_chunk)
                        latent_path = self.latent_root / rel_parent / f"{name}.npy"
                        wav_path = self.wav_root / rel_parent / f"{name}.wav"
                        cond_path = self.cond_root / rel_parent / f"{name}.npz"

                        if has_cpu_conditions:
                            chunk_np = chunk.squeeze(0).cpu().numpy()
                            extract_and_merge_chunk_conditions(
                                self.registry, chunk_np, self.sr,
                                self.n_frames_fixed, cond_path,
                                force=self.force, names=self.cpu_cond_names,
                                companion=companion)

                        if want_wav and (self.force or not wav_path.exists()):
                            _atomic_save_wav(
                                wav_path, chunk.squeeze(0).cpu().numpy(), self.sr
                            )

                        needs_latent = (
                            self.force
                            or not _latent_file_is_valid(
                                latent_path, self.n_frames_fixed
                            )
                        )
                        needs_gpu_cond = has_gpu_conditions and _npz_missing(
                            cond_path, self.gpu_cond_names, force=self.force)
                        if needs_latent or needs_gpu_cond:
                            audio = chunk.squeeze(0).detach().clone()
                            yield {"audio": audio,
                                   "latent_path": (str(latent_path)
                                                   if needs_latent else None),
                                   "cond_path": (str(cond_path)
                                                 if needs_gpu_cond else None)}
            finally:
                for w in cleanup:
                    Path(w).unlink(missing_ok=True)

            try:
                ident = _source_identity(Path(path))
            except OSError:
                ident = {"size": None, "mtime_ns": None}
            yield {"file_done": 1, "src": rel_src,
                   "ident": {**ident, "chunks": produced_chunks}}


def _meta_dict(args, chunk_length, chunk_overlap, latent_frames_per_chunk=None):
    return {
        "sr": args.sr,
        "chunk_length_samples": chunk_length,
        "chunk_overlap_samples": chunk_overlap,
        "chunk_duration_s": args.chunk_duration,
        "chunk_overlap_s": args.chunk_overlap,
        "duration_s": args.duration,
        "min_duration_s": args.min_duration,
        "mono": True,
        "pad_last_chunk": args.pad_last_chunk,
        "keep_num_chunks_per_file": args.keep_num_chunks_per_file,
        "latent_frames_per_chunk": latent_frames_per_chunk,
        "latent_dim": lc.get_spec(args.codec).latent_dim,
        "codec": lc.get_spec(args.codec).name,
        "dac_model": "44khz" if args.codec == "dac_44khz" else None,
        "min_chunk_sec": args.min_chunk_sec,
        "silence_threshold": args.silence_threshold,
        "max_silence_ratio": args.max_silence_ratio,
        "peak_threshold": args.peak_threshold,
        "max_peak_ratio": args.max_peak_ratio,
        "clip": not args.no_clip,
        "chunk_min_rms": args.chunk_min_rms,
        "chunk_min_peak": args.chunk_min_peak,
        "acoustic_rules": args.acoustic_rules,
        "target_lufs": args.target_lufs if args.acoustic_rules else None,
        "target_tp": args.target_tp if args.acoustic_rules else None,
        "target_lra": args.target_lra if args.acoustic_rules else None,
        "silence_trim_db": args.silence_trim_db if args.acoustic_rules else None,
        "silence_thresh_db": args.silence_thresh_db if args.acoustic_rules else None,
        "stereo_split": (not args.no_stereo_split) if args.acoustic_rules else None,
    }


def check_or_write_meta(out_root: Path, meta: dict):
    p = out_root / "dataset_meta.json"
    if p.exists():
        old = json.loads(p.read_text())
        if "codec" not in old and old.get("dac_model") == "44khz":
            old = dict(old, codec="dac_44khz")
        conflicts = {k: (old[k], meta.get(k)) for k in meta
                     if k in old and old[k] != meta.get(k)}
        unverifiable = [k for k in meta if k not in old]
        if conflicts:
            raise SystemExit(
                f"[meta] the parameters differ from the existing dataset in "
                f"{out_root} (old vs new): {conflicts}\n"
                f"Re-running with different parameters would mix incompatible "
                f"chunks/latents in one dataset. Use a FRESH output dir.\n"
                f"(--force does NOT rebuild: it overwrites the chunks it "
                f"re-encounters and LEAVES the others, which is exactly how a "
                f"mixed dataset is produced. It is meant for re-extracting "
                f"conditions with the SAME parameters.)"
            )
        if unverifiable:
            print(f"[meta] WARNING: {len(unverifiable)} parameter(s) are NOT "
                  f"recorded in {p} (dataset built by an older version): "
                  f"{sorted(unverifiable)}.\n"
                  f"      They CANNOT be checked against this run, and they are "
                  f"deliberately not written into the meta (recording today's "
                  f"values would certify them as the original ones without "
                  f"evidence). Make sure you are passing the same values used "
                  f"originally. For a dataset whose provenance is fully "
                  f"verifiable, rebuild into a FRESH output dir with this "
                  f"version.")
    else:
        _atomic_write_json(p, meta)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Streaming preprocessing (chunk -> DAC encode -> latents), "
                    "supervisor-style. Optional per-chunk WAV/conditions, "
                    "incremental, with the train/val/test split decided here "
                    "and recorded in splits.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source_dir", type=str, nargs="?", default=None)
    parser.add_argument("output_dir", type=str, nargs="?", default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML file with any of these options (keys are the "
                             "long flag names without '--'). Precedence: "
                             "built-in defaults < config file < command line, so "
                             "a flag you type always wins over the file.")

    parser.add_argument("--codec", type=str, default=lc.DEFAULT_CODEC,
                        choices=list(lc.CODEC_NAMES),
                        help="The autoencoder that turns each chunk into the "
                             "latents the DiT models: dac_44khz (default; 72-d "
                             "latents, 86.13 frames/s) or encodec_32khz "
                             "(MusicGen's EnCodec; 128-d continuous latents, 50 "
                             "frames/s). Recorded in dataset_meta.json -- the "
                             "training reads it from there. Part of the dataset "
                             "identity: one OUT dir holds one codec.")
    parser.add_argument("--sr", type=int, default=None,
                        help="Target sample rate. Follows --codec when not given "
                             "(44100 for dac_44khz, 32000 for encodec_32khz); "
                             "any other value is refused.")
    parser.add_argument("--chunk_duration", type=float, default=5.0,
                        help="Chunk length in seconds (default: 5.0).")
    parser.add_argument("--chunk_overlap", type=float, default=0.0,
                        help="Chunk overlap in seconds (default: 0).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Trim each source file to N s BEFORE chunking "
                             "(default: None = whole file).")
    parser.add_argument("--min_duration", type=float, default=None,
                        help="Discard source files shorter than N s.")
    parser.add_argument("--pad_last_chunk", action="store_true",
                        help="Pad and keep the last incomplete chunk of each file.")
    parser.add_argument("--keep_num_chunks_per_file", type=int, default=None,
                        help="Keep at most N chunks per file (deterministic "
                             "linspace selection).")

    parser.add_argument("--silence_threshold", type=float, default=None)
    parser.add_argument("--max_silence_ratio", type=float, default=None)
    parser.add_argument("--peak_threshold", type=float, default=1.0)
    parser.add_argument("--max_peak_ratio", type=float, default=None)
    parser.add_argument("--no_clip", action="store_true",
                        help="Disable clamp to [-peak_threshold, peak_threshold].")
    parser.add_argument("--chunk_min_rms", type=float, default=None,
                        help="Drop chunks whose RMS is below this value.")
    parser.add_argument("--chunk_min_peak", type=float, default=None,
                        help="Keep a low-RMS chunk if its peak exceeds this value.")

    parser.add_argument("--acoustic_rules", action="store_true",
                        help="Apply the previous ffmpeg treatment before chunking: "
                             "silence edge-trim + constant-gain loudness "
                             "normalization (true-peak-capped, no compression) + "
                             "per-channel stereo split. Needs ffmpeg/ffprobe.")
    parser.add_argument("--target_lufs", type=float, default=-18.0)
    parser.add_argument("--target_tp", type=float, default=-1.0)
    parser.add_argument("--target_lra", type=float, default=20.0)
    parser.add_argument("--silence_trim_db", type=float, default=-55.0,
                        help="Edge-trim threshold (silencedetect noise).")
    parser.add_argument("--silence_thresh_db", type=float, default=-60.0,
                        help="Per-chunk mean-volume gate: drop chunks below this.")
    parser.add_argument("--no_stereo_split", action="store_true",
                        help="With --acoustic_rules, average stereo to mono "
                             "instead of emitting one example per channel.")
    parser.add_argument("--min_chunk_sec", type=float, default=None,
                        help="Min seconds after trim to keep a file "
                             "(default: --chunk_duration).")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for the DAC encoder (and, unless "
                             "--cond_device says otherwise, for the GPU-capable "
                             "condition extractors).")
    parser.add_argument("--cond_device", type=str, default=None,
                        help="Device for the GPU-capable condition extractors "
                             "(f0 via torchcrepe, rhythm via beat_this). "
                             "Default: follow --device. These run in the MAIN "
                             "process, next to the DAC batch, so no worker ever "
                             "touches CUDA. Pure-DSP conditions (chroma, energy) "
                             "always run on CPU in the workers, "
                             "whatever this says. The device changes speed only, "
                             "never the extracted values.")
    parser.add_argument("--save_wav", type=str, nargs="?", default="none",
                        const="all", metavar="SPLITS",
                        help="Also save the per-chunk WAV under wav/<class>/, for "
                             "the listed splits: 'none' (default), 'all', or a "
                             "comma-separated subset of train,val,test (e.g. "
                             "--save_wav val). A bare --save_wav means 'all', as "
                             "before. Anything but 'none'/'all' needs the split, "
                             "so it is resolved from splits.json. The wavs are "
                             "the REAL source audio (never DAC round-trips): "
                             "'val' is what a standard FAD reference is built "
                             "from, and it costs ~10x less disk than 'all'.")

    parser.add_argument("--split_ratios", type=str, default="0.8,0.1,0.1",
                        help="train,val,test ratios over SOURCE FILES (not "
                             "chunks). Applied per class when stratifying.")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Seed of the split. Part of its identity: changing "
                             "it on an existing dataset is refused without "
                             "--resplit.")
    parser.add_argument("--no_stratify", action="store_true",
                        help="Do NOT balance the split per class (default is to "
                             "stratify, matching the training's old behaviour).")
    parser.add_argument("--resplit", action="store_true",
                        help="Decide the split again from scratch, discarding "
                             "the recorded one. This REASSIGNS sources across "
                             "train/val/test, so any checkpoint trained on the "
                             "old split is then evaluated on data it has seen.")
    parser.add_argument("--split_only", action="store_true",
                        help="Only create/extend splits.json and exit: no "
                             "decoding, no encoding, no conditions. This is how "
                             "an already-preprocessed dataset gets a split.")
    parser.add_argument("--import_legacy_split", action="store_true",
                        help="Write splits.json by reproducing the split the "
                             "TRAINING used to compute in-code, from the latents "
                             "on disk. Use it once on datasets built before the "
                             "split moved here, so runs already in flight keep "
                             "the exact same val/test sets. Implies --split_only.")
    parser.add_argument("--conditions", type=str, default=None,
                        help="Comma-separated frame conditions to extract, e.g. "
                             "'f0' or 'f0,energy'. None = skip conditions. "
                             "'midi' is not computed from the audio: it is read "
                             "from the MIDI file with the same name next to "
                             "each audio file (song.mp3 -> song.mid or "
                             "song.midi), and the run stops if one is missing.")
    parser.add_argument("--global_conds", "--global", dest="global_conds",
                        type=str, default=None,
                        help="Comma-separated global conditions to extract "
                             "('text' and/or 'image'). Independent of "
                             "--conditions: either flag can be used alone, "
                             "both together, and each takes any subset. "
                             "'text' is computed PER CHUNK (the CLAP embedding "
                             "of that chunk's audio) and is stored in the same "
                             ".npz as the frame conditions, so it is extracted "
                             "and resumed exactly like f0. 'image' is computed "
                             "PER CLASS from --image_root into "
                             "global_conditions/image/<class>.npy.")
    parser.add_argument("--text_vocab", type=str, default=None,
                        help="Text file, one phrase per line, replacing the "
                             "built-in label vocabulary (conditions.py "
                             "TEXT_LABEL_VOCAB). It is encoded once into "
                             "global_conditions/text_vocab.npy and is what the "
                             "validation panels use to put WORDS next to a "
                             "stored CLAP vector -- a nearest-phrase retrieval, "
                             "shown with its cosine, not a translation. "
                             "Changing it never re-touches the audio.")
    parser.add_argument("--image_root", type=str, default=None,
                        help="image_root/<class>/*.jpg for --global_conds image. The "
                             "class folder names must match the source audio's "
                             "class folders. EVERY image of a class is encoded: "
                             "the training draws one at random per epoch.")

    parser.add_argument("--single_class", action="store_true")
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--label_source", type=str, default="dir",
                        help="Where the CLASS of each source comes from: 'dir' "
                             "(default) the source subdirectory, as before; "
                             "'csv' a metadata file. The class is the output "
                             "folder either way, so a CSV-labelled corpus can "
                             "live in one flat directory and still be split, "
                             "conditioned and reported per class. Recorded in "
                             "source_manifest.json: changing it on an existing "
                             "output dir is refused (it would relocate every "
                             "latent instead of overwriting it).")
    parser.add_argument("--label_csv", type=str, default=None,
                        help="The metadata CSV for --label_source csv. One "
                             "file column ('file' or 'filename') and one label "
                             "column ('label' or 'class'), any case; every "
                             "source must have a row. Rows are matched to files "
                             "by path relative to source_dir, then by bare file "
                             "name, then by either case-insensitively.")
    parser.add_argument("--text_labels_n", type=int, default=2,
                        help="How many TERMS the per-chunk text description "
                             "holds, the class included: 1 = the class alone, "
                             "2 = class + the nearest vocabulary phrase, 3 = "
                             "class + the two nearest. Written for EVERY chunk "
                             "into global_conditions/text_labels.jsonl (with "
                             "--global_conds text), together with the full cosine "
                             "table, so another N can be re-derived later "
                             "without re-reading a single .npz.")

    parser.add_argument("--skip_dac", action="store_true",
                        help="Do everything except DAC encoding (debug).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute the latents/conditions of the chunks this "
                             "run encounters, instead of skipping the ones already "
                             "on disk. It is NOT a rebuild: outputs that this run "
                             "no longer produces are LEFT in place, and it does "
                             "NOT let you change the chunking/gate/acoustic "
                             "parameters (those are still checked against "
                             "dataset_meta.json -- use a fresh OUT dir for that). "
                             "Intended use: re-extract conditions with the SAME "
                             "parameters, e.g. after fixing an extractor.")

    parser.add_argument("--prune_orphans", action="store_true",
                        help="Delete the outputs of sources listed in "
                             "source_manifest.json that no longer exist on disk "
                             "(deleted/renamed). Without this they stay and keep "
                             "feeding the split/normalizer/training while "
                             "corresponding to nothing. Only files the manifest "
                             "attributes to those sources are removed.")

    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers doing the CPU work (load, "
                             "acoustic, chunking, condition extraction) in "
                             "parallel with the GPU DAC. 0 = single process "
                             "(default). Start with 4 on a large dataset; each "
                             "worker loads one condition model when conditions "
                             "are enabled.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Chunks encoded per DAC forward pass (default: 8).")
    parser.add_argument("--loader_batch_size", type=int, default=8,
                        help="Chunks transported in each worker IPC batch "
                             "(default: 8). Kept separate from --batch_size so "
                             "GPU batching can grow without multiplying shared "
                             "memory in every worker.")
    parser.add_argument("--prefetch_factor", type=int, default=1,
                        help="Batches prefetched by EACH worker (default: 1). "
                             "Higher values multiply shared-memory use by "
                             "num_workers * loader_batch_size.")
    parser.add_argument("--worker_start_method", type=str, default="spawn",
                        choices=("spawn", "forkserver"),
                        help="Safe multiprocessing start method (default: "
                             "spawn). Avoids forking workers from a process "
                             "that already initialized CUDA.")
    return parser


_UNSET = object()


def _explicitly_given(argv=None) -> dict:
    p = build_parser()
    for a in p._actions:
        if a.dest != "help":
            a.default = _UNSET
    ns = vars(p.parse_args(argv))
    return {k: v for k, v in ns.items() if v is not _UNSET}


def _load_preprocess_config(path: str, parser) -> dict:
    import yaml
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[config] no such file: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise SystemExit(f"[config] {p} is not readable YAML: {e}")
    if not isinstance(raw, dict):
        raise SystemExit(f"[config] {p} must contain a mapping of option -> value.")
    known = {a.dest for a in parser._actions if a.dest != "help"} - {"config"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise SystemExit(
            f"[config] {p} has {len(unknown)} unknown option(s): {unknown}\n"
            f"  Keys are the long flags without '--'. Known: {sorted(known)}")
    return raw


def _resolve_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        cfg = _load_preprocess_config(args.config, parser)
        typed = _explicitly_given(argv)
        applied = []
        for k, v in cfg.items():
            if k in typed:
                continue
            setattr(args, k, v)
            applied.append(k)
        print(f"[config] {args.config}: applied {len(applied)} option(s)"
              + (f"; overridden on the command line: "
                 f"{sorted(set(cfg) & set(typed))}"
                 if set(cfg) & set(typed) else ""))
    for name in ("source_dir", "output_dir"):
        if not getattr(args, name):
            raise SystemExit(
                f"[preprocess_stream] {name} is required: pass it positionally "
                f"(SRC OUT) or set '{name}' in the --config file.")
    return args


def _parse_ratios(spec) -> Tuple[float, float, float]:
    if isinstance(spec, (list, tuple)):
        parts = list(spec)
    else:
        parts = [p for p in str(spec).split(",") if p.strip() != ""]
    if len(parts) != 3:
        raise SystemExit(f"[split] --split_ratios needs exactly 3 values "
                         f"(train,val,test), got {spec!r}.")
    try:
        r = tuple(float(x) for x in parts)
    except ValueError:
        raise SystemExit(f"[split] --split_ratios must be numbers, got {spec!r}.")
    if any(x < 0 for x in r):
        raise SystemExit(f"[split] --split_ratios must be >= 0, got {r}.")
    if not math.isclose(sum(r), 1.0, rel_tol=0, abs_tol=1e-6):
        raise SystemExit(f"[split] --split_ratios must sum to 1.0, got {r} "
                         f"(sum={sum(r)}).")
    return r


def _parse_wav_splits(spec) -> frozenset:
    if spec is None or spec is False:
        return frozenset()
    if spec is True:
        return frozenset(SPLIT_NAMES)
    if isinstance(spec, (list, tuple)):
        parts = [str(x).strip().lower() for x in spec]
    else:
        parts = [p.strip().lower() for p in str(spec).split(",") if p.strip()]
    if not parts or parts == ["none"]:
        return frozenset()
    if parts == ["all"]:
        return frozenset(SPLIT_NAMES)
    bad = sorted(set(parts) - set(SPLIT_NAMES))
    if bad:
        raise SystemExit(
            f"[wav] --save_wav got {bad}; expected 'none', 'all', or a "
            f"comma-separated subset of {list(SPLIT_NAMES)}.")
    return frozenset(parts)


def main():
    args = _resolve_args()

    if not _TORCH_OK:
        raise SystemExit("[preprocess_stream] PyTorch is required to run "
                         "(only --help works without it).")
    print(f"[preprocess] build={PREPROCESS_BUILD}")
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    if args.num_workers < 0:
        raise SystemExit("[preprocess_stream] --num_workers must be >= 0.")
    if args.batch_size < 1:
        raise SystemExit("[preprocess_stream] --batch_size must be >= 1.")
    if args.loader_batch_size < 1:
        raise SystemExit("[preprocess_stream] --loader_batch_size must be >= 1.")
    if args.prefetch_factor < 1:
        raise SystemExit("[preprocess_stream] --prefetch_factor must be >= 1.")

    try:
        codec = lc.activate(args.codec)
    except ValueError as e:
        raise SystemExit(f"[preprocess_stream] --codec: {e}")
    if args.sr is None:
        args.sr = codec.sample_rate
    elif int(args.sr) != codec.sample_rate:
        raise SystemExit(
            f"[preprocess_stream] --sr {args.sr}, but {codec.name} encodes "
            f"{codec.sample_rate} Hz audio. Leave --sr unset (it follows "
            f"--codec) or set it to {codec.sample_rate}. The latent frame rate "
            f"and all conditions assume the codec's own rate.")
    args.sr = int(args.sr)
    print(f"[codec] {codec.name}: {codec.sample_rate} Hz, hop {codec.hop_length} "
          f"-> {codec.frames_per_s:.2f} frames/s, {codec.latent_dim}-d latents")

    chunk_length = int(round(args.chunk_duration * args.sr))
    chunk_overlap = int(round(args.chunk_overlap * args.sr))

    out_root = Path(args.output_dir)
    latent_root = out_root / "latents"
    wav_root = out_root / "wav"
    cond_root = out_root / "conditions"

    _meta_p = out_root / "dataset_meta.json"
    if _meta_p.exists():
        try:
            _old_codec = lc.codec_from_meta(json.loads(_meta_p.read_text()))
        except Exception:
            _old_codec = None
        if _old_codec is not None and _old_codec != codec.name:
            raise SystemExit(
                f"[preprocess_stream] {out_root} holds a {_old_codec} dataset, "
                f"this run asks for {codec.name}. To add to it (a condition, the "
                f"wavs), pass --codec {_old_codec} with the other parameters it "
                f"was built with; a dataset in another codec needs a fresh "
                f"output dir.")


    split_ratios = _parse_ratios(args.split_ratios)
    stratify = not args.no_stratify
    wav_splits = _parse_wav_splits(args.save_wav)

    label_resolver = build_label_resolver(args)
    labels_prov = labels_provenance(args, label_resolver)
    check_labels_provenance(out_root, labels_prov)

    if args.import_legacy_split or args.split_only:
        if args.import_legacy_split:
            groups = import_legacy_split(out_root, latent_root, split_ratios,
                                         args.split_seed, stratify)
        else:
            print("Scanning source files ...")
            files = scan_audio_files(args.source_dir, args.single_class,
                                     args.class_name, label_resolver)
            if not files:
                raise SystemExit(f"[ERROR] No audio files under {args.source_dir}")
            print(f"  {len(files)} files")
            print_class_histogram(files)
            groups = resolve_splits(out_root, files, split_ratios,
                                    args.split_seed, stratify, args.resplit)
            write_source_manifest(out_root, load_source_manifest(out_root),
                                  {}, [], False, labels_prov)
        print(f"\nDONE (split only)\n  output: {out_root / SPLITS_NAME}")
        print_chunk_counts(out_root, groups)
        return

    enabled_frame = None
    if args.conditions is not None:
        enabled_frame = [c.strip() for c in args.conditions.split(",") if c.strip()]
    enabled_global = None
    if args.global_conds is not None:
        enabled_global = [c.strip() for c in args.global_conds.split(",") if c.strip()]

    registry = None
    gpu_cond_names, cpu_cond_names = [], []
    chunk_cond_names, class_global_names = [], []
    if enabled_frame or enabled_global:
        from conditions import ConditionRegistry
        registry = ConditionRegistry(
            enabled_frame=enabled_frame if enabled_frame else [],
            enabled_global=enabled_global if enabled_global else [],
        )

        chunk_globals, class_global_names = _split_globals_by_stage(registry)
        chunk_cond_names = sorted(list(registry.frame_names) + chunk_globals)

        cond_device = args.cond_device or args.device
        if cond_device.startswith("cuda") and not torch.cuda.is_available():
            print("[cond] cond_device='cuda' but no GPU is available "
                  "-> condition extraction falls back to CPU.")
            cond_device = "cpu"

        gpu_capable, cpu_only = _split_extractors_by_device(registry)
        if cond_device.startswith("cuda"):
            gpu_cond_names, cpu_cond_names = gpu_capable, cpu_only
            _set_extractor_device(registry, gpu_cond_names, cond_device)
        else:
            gpu_cond_names, cpu_cond_names = [], sorted(gpu_capable + cpu_only)
            _set_extractor_device(registry, cpu_cond_names, "cpu")

        if args.num_workers > 0:
            _force_cpu_extractors(registry, cpu_cond_names)

        if chunk_cond_names:
            print(f"[cond] per-chunk conditions {chunk_cond_names}: "
                  f"GPU({cond_device})={gpu_cond_names or 'none'} in the main "
                  f"process | CPU={cpu_cond_names or 'none'} in the workers")
        if class_global_names:
            print(f"[cond] per-class conditions {class_global_names}: "
                  f"one sidecar per class, after the stream")
        if not chunk_cond_names and not class_global_names:
            print("[cond] a registry was built but it holds no condition "
                  "-> nothing to extract")

    print("Scanning source files ...")
    files = scan_audio_files(args.source_dir, args.single_class, args.class_name,
                             label_resolver)
    if not files:
        print(f"[ERROR] No audio files under {args.source_dir}")
        return
    classes = sorted({leaf for _, _, leaf, _, _ in files})
    print(f"  {len(files)} files, {len(classes)} classes")
    print_class_histogram(files)

    _all_ext = _all_chunk_extractors(registry)
    companion_names = [n for n in cpu_cond_names
                       if _companion_suffixes(_all_ext.get(n))]
    companions = {}
    if companion_names:
        _sfx = sorted({s for n in companion_names
                       for s in _companion_suffixes(_all_ext[n])})
        companions, _missing, _ambig = find_companions(files, _sfx)
        if _missing or _ambig:
            msg = [f"[companion] {companion_names} is read from the file with "
                   f"the SAME NAME as each audio file, next to it "
                   f"({' / '.join(_sfx)}): song.mp3 -> song.mid."]
            if _missing:
                msg.append(f"  {len(_missing)} audio file(s) have none, e.g.:")
                msg += [f"    {r}" for r in _missing[:10]]
            if _ambig:
                msg.append(f"  {len(_ambig)} audio file(s) have more than "
                           f"one, e.g.:")
                msg += [f"    {r}: {[q.name for q in qs]}"
                        for r, qs in _ambig[:10]]
            msg.append("  Put exactly one next to each audio file, or move "
                       "the audio files without one out of the source folder.")
            raise SystemExit("\n".join(msg))
        print(f"[companion] {companion_names}: every source has its file "
              f"({len(companions)}/{len(files)}, {' / '.join(_sfx)})")

    split_groups = resolve_splits(out_root, files, split_ratios, args.split_seed,
                                  stratify, args.resplit)

    if not wav_splits:
        wav_sources = None
    elif set(wav_splits) == set(SPLIT_NAMES):
        wav_sources = "all"
    else:
        wav_sources = {
            rel for path, rel_parent, _leaf, src_hash, rel in files
            if split_groups.get(
                source_group_key(rel_parent, Path(path).name, src_hash)
            ) in wav_splits
        }
    if wav_splits:
        _n = len(files) if wav_sources == "all" else len(wav_sources)
        print(f"[wav] saving the per-chunk WAV of {sorted(wav_splits)} "
              f"-> {_n}/{len(files)} source(s)")

    backend_kw = dict(
        chunk_length=chunk_length, chunk_overlap=chunk_overlap,
        duration=args.duration, min_duration=args.min_duration,
        audio_sr=args.sr, mono=True,
        silence_threshold=args.silence_threshold,
        max_silence_ratio=args.max_silence_ratio,
        peak_threshold=args.peak_threshold,
        max_peak_ratio=args.max_peak_ratio,
        clip=not args.no_clip,
        chunk_min_rms_threshold=args.chunk_min_rms,
        chunk_min_peak_threshold=args.chunk_min_peak,
        pad_and_keep_last_chunk=args.pad_last_chunk,
        pad_value=0.0,
        keep_num_chunks_per_file=args.keep_num_chunks_per_file,
    )
    chunker = VendoredChunker(**backend_kw)

    dac_enc = None
    if not args.skip_dac:
        dac_enc = DACEncoder(device=args.device, codec=codec.name)

    if dac_enc is not None:
        n_frames_fixed = dac_enc.n_frames_for(chunk_length, args.sr)
    else:
        n_frames_fixed = None
        meta_path = out_root / "dataset_meta.json"
        if meta_path.exists():
            try:
                n_frames_fixed = json.loads(meta_path.read_text()).get(
                    "latent_frames_per_chunk")
            except Exception:
                n_frames_fixed = None
        if n_frames_fixed is None:
            for existing in latent_root.rglob("*.npy"):
                z = None
                try:
                    z = np.load(
                        str(existing), mmap_mode="r", allow_pickle=False
                    )
                    if (z.ndim == 2 and z.shape[0] == codec.latent_dim
                            and z.dtype == np.dtype(np.float32)):
                        n_frames_fixed = int(z.shape[1])
                        break
                except Exception:
                    continue
                finally:
                    if z is not None:
                        mm = getattr(z, "_mmap", None)
                        if mm is not None:
                            mm.close()
        if n_frames_fixed is None:
            raise SystemExit(
                "[preprocess_stream] --skip_dac but no latent/meta on disk to read "
                "the latent length from: run once without --skip_dac first.")
    print(f"[latents] T (real {codec.label} frames per chunk) = {n_frames_fixed}")
    if registry is not None and registry.frame_names:
        print(f"[conditions] frame-aligned to T={n_frames_fixed}")
    if registry is not None and _split_globals_by_stage(registry)[0]:
        print(f"[conditions] per-chunk globals "
              f"{_split_globals_by_stage(registry)[0]}: one vector per chunk "
              f"(independent of T)")

    meta = _meta_dict(args, chunk_length, chunk_overlap, n_frames_fixed)
    check_or_write_meta(out_root, meta)

    prev_manifest = load_source_manifest(out_root)
    changed_srcs, removed_srcs = ([], [])
    if prev_manifest:
        changed_srcs, removed_srcs = audit_sources(prev_manifest, files)
        if changed_srcs:
            print(f"[manifest] WARNING: {len(changed_srcs)} source(s) CHANGED since "
                  f"they were encoded (size/mtime differ). Their existing latents "
                  f"are STALE and, without --force, would be silently kept:")
            for rel in changed_srcs[:5]:
                print(f"             {rel}")
            if len(changed_srcs) > 5:
                print(f"             ... and {len(changed_srcs) - 5} more")
            print(f"           -> re-run with --force to re-encode them, or use a "
                  f"fresh OUT dir.")
        if removed_srcs:
            n_orph = sum(len(prev_manifest[r].get("chunks", [])) for r in removed_srcs)
            print(f"[manifest] {len(removed_srcs)} source(s) no longer exist, "
                  f"leaving ~{n_orph} ORPHAN output(s) that would still feed the "
                  f"split/normalizer/training.")
            if args.prune_orphans:
                n_del = prune_orphans(out_root, prev_manifest, removed_srcs)
                print(f"           -> --prune_orphans: deleted {n_del} file(s).")
            else:
                print(f"           -> pass --prune_orphans to delete them.")

    if not args.force and prev_manifest:
        _run_cond_names = list(cpu_cond_names) + list(gpu_cond_names)
        n_before = len(files)
        files, n_skipped = filter_complete_sources(
            files, prev_manifest, changed_srcs,
            latent_root, cond_root, wav_root, n_frames_fixed,
            _run_cond_names, dac_enc is not None, wav_sources)
        if n_skipped:
            print(f"[resume] {n_skipped}/{n_before} source(s) already have every "
                  f"output this run would write -> not decoded at all.")
        if not files:
            print("[resume] no source needs per-chunk work this run.")

    tmp_dir = None
    min_chunk_sec = args.min_chunk_sec if args.min_chunk_sec is not None else args.chunk_duration
    acoustic_kw = None
    if args.acoustic_rules:
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise SystemExit("[acoustic_rules] ffmpeg/ffprobe not found on PATH.")
        base = _IRCAM_LOCAL if os.path.isdir(_IRCAM_LOCAL) else None
        tmp_dir = tempfile.mkdtemp(prefix="preprocess_stream_", dir=base)
        acoustic_kw = dict(
            target_lufs=args.target_lufs, target_tp=args.target_tp,
            target_lra=args.target_lra, silence_trim_db=args.silence_trim_db,
            min_sec=min_chunk_sec, stereo_split=not args.no_stereo_split,
        )
        print(f"[acoustic_rules] ON (LUFS={args.target_lufs}, TP={args.target_tp}, "
              f"trim={args.silence_trim_db}dB, gate={args.silence_thresh_db}dB, "
              f"stereo_split={not args.no_stereo_split}) tmp={tmp_dir}")

    dataset = StreamingChunkDataset(
        files=files, chunker=chunker, registry=registry,
        latent_root=latent_root, wav_root=wav_root, cond_root=cond_root,
        sr=args.sr, wav_sources=wav_sources, force=args.force,
        n_frames_fixed=n_frames_fixed,
        cpu_cond_names=cpu_cond_names, gpu_cond_names=gpu_cond_names,
        acoustic=args.acoustic_rules, acoustic_kw=acoustic_kw,
        silence_thresh_db=args.silence_thresh_db, tmp_dir=tmp_dir,
        companions=companions, codec_name=codec.name,
    )
    loader_kw = dict(
        dataset=dataset,
        batch_size=args.loader_batch_size,
        num_workers=args.num_workers,
        collate_fn=_identity_collate,
        persistent_workers=False,
    )
    if args.num_workers > 0:
        loader_kw.update(
            prefetch_factor=args.prefetch_factor,
            multiprocessing_context=args.worker_start_method,
            worker_init_fn=_worker_init_fn,
        )
    loader = DataLoader(**loader_kw)
    print(f"[run] num_workers={args.num_workers}, "
          f"loader_batch_size={args.loader_batch_size}, "
          f"dac_batch_size={args.batch_size}, "
          f"prefetch_factor={args.prefetch_factor if args.num_workers else 0}, "
          f"start_method={args.worker_start_method if args.num_workers else 'none'}, "
          f"skip_dac={args.skip_dac}")

    try:
        from tqdm import tqdm
        file_bar = tqdm(total=len(files), desc="Files", unit="file")
    except Exception:
        file_bar = None

    n_lat = 0
    n_gcond = 0
    pending = []
    produced = {}
    _dac_on = not (args.skip_dac or dac_enc is None)
    _gpu_work = _dac_on or bool(gpu_cond_names)

    def _flush(items):
        nonlocal n_lat, n_gcond
        a, b = _process_gpu_batch(
            dac_enc if _dac_on else None, items, args.sr, n_frames_fixed,
            registry=registry, gpu_names=gpu_cond_names, force=args.force)
        n_lat += a
        n_gcond += b

    try:
        for batch in loader:
            for it in batch:
                if "file_done" in it:
                    if file_bar is not None:
                        file_bar.update(1)
                    src = it.get("src")
                    if src is not None:
                        produced[src] = it["ident"]
                    continue
                pending.append(it)
            if not _gpu_work:
                pending.clear()
                continue
            while len(pending) >= args.batch_size:
                _flush(pending[:args.batch_size])
                del pending[:args.batch_size]
        if pending and _gpu_work:
            _flush(pending)
            pending.clear()
    finally:
        if file_bar is not None:
            file_bar.close()
        if produced or prev_manifest:
            n_src = write_source_manifest(out_root, prev_manifest, produced,
                                          removed_srcs, args.prune_orphans,
                                          labels_prov)
            print(f"[manifest] {out_root / MANIFEST_NAME}: {n_src} source(s) "
                  f"({len(produced)} seen this run)")
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if class_global_names:
        extract_class_global_conditions(
            registry, out_root, classes, args.image_root, force=args.force
        )
    if "text" in _split_globals_by_stage(registry)[0]:
        write_text_label_vocab(registry, out_root,
                               getattr(args, "text_vocab", None),
                               force=args.force)
        write_text_labels(out_root, cond_root,
                          n_terms=getattr(args, "text_labels_n", 2),
                          force=args.force,
                          text_extractor=registry.global_extractors.get("text"))

    print("\nDONE")
    print(f"  latents encoded this run: {n_lat}")
    if gpu_cond_names:
        print(f"  GPU per-chunk conditions {gpu_cond_names} extracted this run: "
              f"{n_gcond} chunk(s)")
    print("  CPU conditions / wav were written in-stream by the workers "
          "(counts not aggregated across processes).")
    print_chunk_counts(out_root, split_groups)
    print(f"  output: {out_root}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
