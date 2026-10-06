# YourMT3+ transcription, for the midi condition.

import os

_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ.setdefault("HF_HOME", os.path.join(_cache, "huggingface"))
    os.environ["TORCH_HOME"] = os.path.join(_cache, "torch")

import io
import sys
import time
import types
import warnings
import contextlib
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ID = "mimbres/YourMT3"
REPO_TYPE = "space"
REVISION = "5e66c1ea173a8186e0d20432b841d3180cc015b5"
MODEL_NAME = "YPTF.MoE+Multi (noPS)"
EXP_ID = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"
CKPT_NAME = "last.ckpt"
MODEL_ARGS = ["-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
              "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe",
              "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu",
              "-epe", "rope", "-rp", "1", "-ac", "spec", "-hop", "300",
              "-atc", "1", "-pr", "32"]
TRANSFORMERS_SERIES = "4.45"
_FILES = ["amt/src/**/*.py", "model_helper.py",
          f"amt/logs/2024/{EXP_ID}/checkpoints/{CKPT_NAME}"]

Note = Tuple[float, float, int, bool, int]


def check_transformers():
    import transformers
    v = transformers.__version__
    if ".".join(v.split(".")[:2]) != TRANSFORMERS_SERIES:
        raise RuntimeError(
            f"YourMT3+ needs transformers {TRANSFORMERS_SERIES}.x, this "
            f"environment has {v}. Its decoder subclasses transformers' "
            f"internal T5 classes, which 4.46+ changed: transcription dies "
            f"with a TypeError in the decoder. Install the version the "
            f"authors pin:  pip install transformers==4.45.1  (verified: CLAP "
            f"and CLIP give the same vectors under it, see "
            f"midi_transcriber.py).")
    return v


def snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                                  revision=REVISION, allow_patterns=_FILES))


def _import_authors_code(root: Path):
    src = root / "amt" / "src"
    for p in (str(src), str(root)):
        if p not in sys.path:
            sys.path.append(p)
    stub = "wandb" not in sys.modules
    if stub:
        sys.modules["wandb"] = types.SimpleNamespace(Table=lambda **_: None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import model_helper
            import model.ymt3 as ymt3
    finally:
        if stub:
            sys.modules.pop("wandb", None)
    for mod in (model_helper, ymt3):
        f = Path(mod.__file__).resolve()
        if root.resolve() not in f.parents:
            raise RuntimeError(
                f"YourMT3+: '{mod.__name__}' was imported from {f}, not from "
                f"the YourMT3 snapshot {root}. A module with the same generic "
                f"name is installed or already imported in this process.")
    return model_helper


class YourMT3Transcriber:
    _models: dict = {}

    def __init__(self, device: str = "cuda"):
        self.device = str(device)

    def load(self):
        import torch
        key = self.device
        if key in self._models:
            return self._models[key]
        check_transformers()
        root = snapshot_dir()
        mh = _import_authors_code(root)
        from config.config import shared_cfg as default_shared_cfg
        ckpt = root / "amt" / "logs" / "2024" / EXP_ID / "checkpoints" / CKPT_NAME

        def _no_trainer(args, stage="test"):
            args.exp_id = args.exp_id.split("@")[0]
            return (None, None,
                    {"lightning_dir": None, "last_ckpt_path": str(ckpt)},
                    deepcopy(default_shared_cfg))

        prev = torch.get_float32_matmul_precision()
        orig = mh.initialize_trainer
        mh.initialize_trainer = _no_trainer
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = mh.load_model_checkpoint(
                    args=[f"{EXP_ID}@{CKPT_NAME}"] + MODEL_ARGS, device="cpu")
        finally:
            mh.initialize_trainer = orig
            torch.set_float32_matmul_precision(prev)
        model = model.to(self.device).eval()
        self._models[key] = model
        print(f"[midi] YourMT3+ {MODEL_NAME} @ {REVISION[:8]} loaded on "
              f"{self.device}")
        return model

    def transcribe(self, audio: np.ndarray, sr: int) -> List[Note]:
        import torch
        import torchaudio
        model = self.load()
        from utils.audio import slice_padded_array
        from utils.note2event import mix_notes
        from utils.event2note import merge_zipped_note_events_and_ties_to_notes

        tsr = int(model.audio_cfg["sample_rate"])
        seg = int(model.audio_cfg["input_frames"])
        y = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)
                             .reshape(1, -1))
        if int(sr) != tsr:
            y = torchaudio.functional.resample(y, int(sr), tsr)
        segs = slice_padded_array(y.numpy(), seg, seg)
        if segs.shape[0] == 0:
            return []
        x = torch.from_numpy(segs.astype(np.float32)).unsqueeze(1)
        x = x.to(self.device)
        starts = [seg * i / tsr for i in range(segs.shape[0])]
        dev = torch.device(self.device)
        rng_devices = ([dev.index or 0] if dev.type == "cuda" else [])
        prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        try:
            with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
                pred, _ = model.inference_file(bsz=max(8, segs.shape[0]),
                                               audio_segments=x)
        finally:
            torch.set_float32_matmul_precision(prev)
        per_channel = []
        for ch in range(model.task_manager.num_decoding_channels):
            tokens = [a[:, ch, :] for a in pred]
            zipped, _, _ = model.task_manager.detokenize_list_batches(
                tokens, starts, return_events=True)
            notes, _ = merge_zipped_note_events_and_ties_to_notes(zipped)
            per_channel.append(notes)
        return [(float(n.onset), float(n.offset), int(n.pitch),
                 bool(n.is_drum), int(n.program))
                for n in mix_notes(per_channel)]

    def unload(self):
        self._models.pop(self.device, None)


def _scale(sr: int = 44100, dur: float = 5.0):
    rng = np.random.default_rng(0)
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    y = np.zeros(int(dur * sr))
    onsets = []
    for k, m in enumerate(pitches):
        o = 0.2 + 0.55 * k
        f = 440.0 * 2.0 ** ((m - 69) / 12.0)
        n = int(round(sr / f))
        buf = rng.uniform(-1.0, 1.0, n)
        out = np.zeros(int(0.5 * sr))
        for i in range(out.size):
            out[i] = buf[i % n]
            buf[i % n] = 0.999 * 0.5 * (buf[i % n] + buf[(i + 1) % n])
        s = int(o * sr)
        y[s:s + out.size] += out[:y.size - s]
        onsets.append((o, m))
    return (0.5 * y / np.abs(y).max()).astype(np.float32), sr, onsets


def selftest(device: str) -> bool:
    import transformers
    y, sr, ref = _scale()
    tr = YourMT3Transcriber(device)
    t0 = time.time()
    tr.load()
    t_load = time.time() - t0
    t0 = time.time()
    notes = tr.transcribe(y, sr)
    t_one = time.time() - t0
    pitched = sorted((o, p) for o, _e, p, d, _pr in notes if not d)
    hit = sum(1 for o, m in ref
              if any(abs(o - oe) <= 0.05 and pe == m for oe, pe in pitched))
    print(f"transformers {transformers.__version__} | {MODEL_NAME} @ "
          f"{REVISION[:8]} | device {device}")
    print(f"load {t_load:.1f} s | one 5 s chunk {t_one:.2f} s")
    print(f"C major scale: {hit}/{len(ref)} notes found with the right pitch "
          f"within +-50 ms (8/8 on the laptop GPU it was written on); "
          f"{len(pitched)} pitched notes in total")
    for o, p in pitched:
        print(f"   {o:6.3f} s  MIDI {p}")
    ok = hit >= len(ref) - 1
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    import argparse
    import torch
    ap = argparse.ArgumentParser(
        description="YourMT3+ transcription (the midi condition's "
                    "re-extraction).")
    ap.add_argument("wav", nargs="?", help="audio file to transcribe")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available()
                                         else "cpu"))
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest(args.device) else 1)
    if not args.wav:
        ap.error("give a wav file or --selftest")
    import soundfile as sf
    y, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    for o, e, p, d, pr in YourMT3Transcriber(args.device).transcribe(
            y.mean(axis=1), sr):
        kind = "drum" if d else f"program {pr:3d}"
        print(f"{o:8.3f} {e:8.3f}  MIDI {p:3d}  {kind}")


if __name__ == "__main__":
    main()
