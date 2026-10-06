# crema's chord model in PyTorch (weights: crema_chord_weights.npz, next to this file).

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CREMA_SR = 44100
CREMA_HOP = 4096
CREMA_FPS = CREMA_SR / CREMA_HOP
CREMA_N_OCTAVES = 6
CREMA_OVER_SAMPLE = 3
CREMA_BINS_PER_OCTAVE = 12 * CREMA_OVER_SAMPLE
CREMA_N_BINS = CREMA_N_OCTAVES * CREMA_BINS_PER_OCTAVE
CREMA_HARMONICS = (1, 2)
CREMA_FMIN_NOTE = "C1"

DEFAULT_WEIGHTS = Path(__file__).resolve().with_name("crema_chord_weights.npz")
OUTPUT_DIMS = {"chord_tag": 170, "chord_pitch": 12,
               "chord_root": 13, "chord_bass": 13}
BN_EPSILON = 1e-3


def hcqt(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"hcqt expects a mono signal (T,), got {y.shape}")
    if sr != CREMA_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=CREMA_SR)
    n_frames = int(librosa.time_to_frames(
        librosa.get_duration(y=y, sr=CREMA_SR),
        sr=CREMA_SR, hop_length=CREMA_HOP))
    if n_frames <= 0:
        return np.zeros((0, CREMA_N_BINS, len(CREMA_HARMONICS)), np.float32)

    fmin = librosa.note_to_hz(CREMA_FMIN_NOTE)
    mags = []
    for h in CREMA_HARMONICS:
        C = librosa.cqt(y=y, sr=CREMA_SR, hop_length=CREMA_HOP,
                        fmin=fmin * h, n_bins=CREMA_N_BINS,
                        bins_per_octave=CREMA_BINS_PER_OCTAVE)
        C = librosa.util.fix_length(C, size=n_frames)
        mags.append(librosa.amplitude_to_db(np.abs(C), ref=np.max))
    return np.transpose(np.asarray(mags).astype(np.float32), (2, 1, 0))


def _hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class _FrozenBatchNorm(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        for k in ("gamma", "beta", "moving_mean", "moving_variance"):
            self.register_buffer(k, torch.zeros(n))

    def forward(self, x):
        return ((x - self.moving_mean)
                * torch.rsqrt(self.moving_variance + BN_EPSILON)
                * self.gamma + self.beta)


class _Conv2d(nn.Module):
    def __init__(self, n_in: int, n_out: int, size, padding):
        super().__init__()
        self.padding = padding
        self.register_buffer("kernel", torch.zeros(n_out, n_in, *size))
        self.register_buffer("bias", torch.zeros(n_out))

    def forward(self, x):
        return F.conv2d(x, self.kernel, self.bias, padding=self.padding)


class _Dense(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.register_buffer("kernel", torch.zeros(n_in, n_out))
        self.register_buffer("bias", torch.zeros(n_out))

    def forward(self, x):
        return torch.matmul(x, self.kernel) + self.bias


class _GRUWeights(nn.Module):
    def __init__(self, n_in: int, units: int):
        super().__init__()
        self.register_buffer("kernel", torch.zeros(n_in, 3 * units))
        self.register_buffer("recurrent_kernel", torch.zeros(units, 3 * units))
        self.register_buffer("bias", torch.zeros(3 * units))


class _KerasBiGRU(nn.Module):
    def __init__(self, n_in: int, units: int):
        super().__init__()
        self.units = units
        self.fwd = _GRUWeights(n_in, units)
        self.bwd = _GRUWeights(n_in, units)

    def forward(self, x):
        B, T, _ = x.shape
        u = self.units
        xw = torch.stack([torch.matmul(x, self.fwd.kernel) + self.fwd.bias,
                          (torch.matmul(x, self.bwd.kernel)
                           + self.bwd.bias).flip(1)])
        rk = torch.stack([self.fwd.recurrent_kernel,
                          self.bwd.recurrent_kernel])
        u_zr, u_h = rk[..., :2 * u], rk[..., 2 * u:]
        h = x.new_zeros(2, B, u)
        out = x.new_empty(2, B, T, u)
        for t in range(T):
            zr = _hard_sigmoid(torch.baddbmm(xw[:, :, t, :2 * u], h, u_zr))
            z, r = zr[..., :u], zr[..., u:]
            hh = torch.tanh(torch.baddbmm(xw[:, :, t, 2 * u:], r * h, u_h))
            h = torch.lerp(hh, h, z)
            out[:, :, t] = h
        return torch.cat([out[0], out[1].flip(1)], dim=-1)


_KERAS_LAYERS = {
    "bn_in": "batch_normalization_1",
    "conv1": "conv2d_1",
    "bn1": "batch_normalization_2",
    "conv2": "conv2d_2",
    "bn2": "batch_normalization_3",
    "gru1.fwd": "bidirectional_1/forward_gru_1",
    "gru1.bwd": "bidirectional_1/backward_gru_1",
    "bn3": "batch_normalization_4",
    "gru2.fwd": "bidirectional_2/forward_gru_2",
    "gru2.bwd": "bidirectional_2/backward_gru_2",
    "pitch": "chord_pitch",
    "root": "chord_root",
    "bass": "chord_bass",
    "bn4": "batch_normalization_5",
    "tag": "chord_tag",
}


class CremaChordNet(nn.Module):
    def __init__(self):
        super().__init__()
        n_h = len(CREMA_HARMONICS)
        self.bn_in = _FrozenBatchNorm(n_h)
        self.conv1 = _Conv2d(n_h, 1, (5, 5), padding=2)
        self.bn1 = _FrozenBatchNorm(1)
        self.conv2 = _Conv2d(1, 72, (1, CREMA_N_BINS), padding=0)
        self.bn2 = _FrozenBatchNorm(72)
        self.gru1 = _KerasBiGRU(72, 128)
        self.bn3 = _FrozenBatchNorm(256)
        self.gru2 = _KerasBiGRU(256, 128)
        self.pitch = _Dense(256, 12)
        self.root = _Dense(256, 13)
        self.bass = _Dense(256, 13)
        self.bn4 = _FrozenBatchNorm(256 + 12 + 13 + 13)
        self.tag = _Dense(256 + 12 + 13 + 13, 170)
        self.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.bn_in(x)
        x = F.relu(self.conv1(x.permute(0, 3, 1, 2)))
        x = self.bn1(x.permute(0, 2, 3, 1))
        x = F.relu(self.conv2(x.permute(0, 3, 1, 2)))
        x = self.bn2(x.permute(0, 2, 3, 1)).squeeze(2)
        h = self.gru2(self.bn3(self.gru1(x)))
        pitch = torch.sigmoid(self.pitch(h))
        root = torch.softmax(self.root(h), dim=-1)
        bass = torch.softmax(self.bass(h), dim=-1)
        codec = self.bn4(torch.cat([h, pitch, root, bass], -1))
        tag = torch.softmax(self.tag(codec), dim=-1)
        return {"chord_tag": tag, "chord_pitch": pitch,
                "chord_root": root, "chord_bass": bass}

    def load_keras_weights(self, arrays: Dict[str, np.ndarray]) -> None:
        weights = {k: v for k, v in arrays.items() if not k.startswith(
            ("meta/", "selftest/"))}
        used = set()
        for path, layer in _KERAS_LAYERS.items():
            mod = self.get_submodule(path)
            for buf_name, buf in mod.named_buffers(recurse=False):
                key = f"{layer}/{buf_name}"
                if key not in weights:
                    raise KeyError(f"crema weights: '{key}' is missing")
                w = torch.from_numpy(np.asarray(weights[key], np.float32))
                if isinstance(mod, _Conv2d) and buf_name == "kernel":
                    w = w.permute(3, 2, 0, 1)
                if tuple(w.shape) != tuple(buf.shape):
                    raise ValueError(f"crema weights: '{key}' has shape "
                                     f"{tuple(w.shape)}, expected "
                                     f"{tuple(buf.shape)}")
                buf.copy_(w)
                used.add(key)
        extra = sorted(set(weights) - used)
        if extra:
            raise ValueError(f"crema weights: arrays not used by the port: "
                             f"{extra}")


def _frontend_settings() -> dict:
    import librosa
    return {"sr": float(CREMA_SR), "hop_length": int(CREMA_HOP),
            "n_octaves": int(CREMA_N_OCTAVES),
            "over_sample": int(CREMA_OVER_SAMPLE),
            "fmin": float(librosa.note_to_hz(CREMA_FMIN_NOTE)),
            "harmonics": [int(h) for h in CREMA_HARMONICS],
            "log": True, "conv": "tf"}


def read_weights(path=None) -> Dict[str, np.ndarray]:
    path = Path(path) if path else DEFAULT_WEIGHTS
    if not path.exists():
        raise FileNotFoundError(
            f"crema weights not found: {path}. It ships next to "
            f"crema_chord.py; rebuild it with "
            f"'python crema_chord.py --convert <crema>/models/chord'.")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    recorded = json.loads(str(arrays["meta/frontend"]))
    mine = _frontend_settings()
    diff = {k: (recorded.get(k), v) for k, v in mine.items()
            if k not in recorded or (recorded[k] != v and not (
                isinstance(v, float) and abs(recorded[k] - v) < 1e-9))}
    if diff:
        raise ValueError(f"{path}: converted for a different front-end than "
                         f"hcqt() computes (recorded vs here): {diff}")
    return arrays


def file_sha1(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class CremaChord:
    def __init__(self, weights=None, device: str = "cpu"):
        arrays = read_weights(weights)
        self.net = CremaChordNet()
        self.net.load_keras_weights(arrays)
        self.net.to(device)
        self.device = device
        self.vocabulary: List[str] = [str(v) for v in
                                      arrays["meta/vocabulary"]]
        self.model_version = str(arrays["meta/crema_model_version"])

    @torch.inference_mode()
    def outputs_from_features(self, feats: np.ndarray) -> Dict[str, np.ndarray]:
        if feats.shape[0] == 0:
            return {k: np.zeros((0, d), np.float32)
                    for k, d in OUTPUT_DIMS.items()}
        x = torch.from_numpy(np.ascontiguousarray(feats, np.float32))
        out = self.net(x.unsqueeze(0).to(self.device))
        return {k: v[0].float().cpu().numpy() for k, v in out.items()}

    def outputs(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        return self.outputs_from_features(hcqt(y, sr))

    def chord_segments(self, y: np.ndarray, sr: int):
        tags = self.outputs(y, sr)["chord_tag"].argmax(axis=1)
        segs = []
        for k, idx in enumerate(tags):
            label = self.vocabulary[int(idx)]
            if segs and segs[-1][2] == label:
                segs[-1][1] = (k + 1) / CREMA_FPS
            else:
                segs.append([k / CREMA_FPS, (k + 1) / CREMA_FPS, label])
        return [tuple(s) for s in segs]


_SELFTEST_CHORDS = (("C:maj", (48, 60, 64, 67)), ("A:min", (45, 57, 60, 64)),
                    ("F:maj", (41, 57, 60, 65)), ("G:7", (43, 59, 62, 65)))


def selftest_signal(sr: int = CREMA_SR, seconds_per_chord: float = 2.0):
    n = int(round(seconds_per_chord * sr))
    t = np.arange(n, dtype=np.float64) / sr
    fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.02)
    parts = []
    for _, notes in _SELFTEST_CHORDS:
        seg = np.zeros(n)
        for m in notes:
            f = 440.0 * 2.0 ** ((m - 69) / 12.0)
            for k in range(1, 7):
                if k * f < 0.45 * sr:
                    seg += np.sin(2 * np.pi * k * f * t) / k
        parts.append(seg * fade)
    y = np.concatenate(parts)
    return (0.25 * y / np.abs(y).max()).astype(np.float32)


def selftest(weights=None) -> bool:
    arrays = read_weights(weights)
    model = CremaChord(weights)
    feats_ref = arrays["selftest/features"]
    ok = True
    print(f"crema chord model {model.model_version}, reference recorded with "
          f"{str(arrays['selftest/environment'])}")

    out = model.outputs_from_features(feats_ref)
    for k in OUTPUT_DIMS:
        err = float(np.abs(out[k] - arrays[f"selftest/{k}"]).max())
        good = err < 1e-4
        ok &= good
        print(f"  network   {k:<12s} max |port - crema| = {err:.2e}  "
              f"{'OK' if good else 'FAIL'}")

    feats = hcqt(selftest_signal(), CREMA_SR)
    if feats.shape != feats_ref.shape:
        print(f"  front-end shape {feats.shape} != recorded "
              f"{feats_ref.shape}  FAIL")
        return False
    err_db = float(np.abs(feats - feats_ref).max())
    out2 = model.outputs_from_features(feats)
    err_p = float(np.abs(out2["chord_pitch"]
                         - arrays["selftest/chord_pitch"]).max())
    tag_agree = float((out2["chord_tag"].argmax(1)
                       == arrays["selftest/chord_tag"].argmax(1)).mean())
    note = "OK" if err_db < 1e-3 else "DIFFERS (librosa version?)"
    print(f"  front-end features max |here - pumpp| = {err_db:.2e} dB  {note}")
    print(f"  end-to-end chord_pitch max diff = {err_p:.2e}, "
          f"chord_tag argmax agreement = {tag_agree * 100:.1f}%")
    labels = [lab for _, _, lab in model.chord_segments(selftest_signal(),
                                                        CREMA_SR)]
    print(f"  chords heard in the test signal (C, Am, F, G7 played): "
          f"{' '.join(labels)}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def _load_crema_reference():
    import tf_keras
    for sub in ("models", "layers", "backend"):
        sys.modules[f"keras.{sub}"] = getattr(tf_keras, sub)
    sys.modules["keras"] = tf_keras
    hs = float(tf_keras.activations.hard_sigmoid(
        tf_keras.backend.constant([2.0])).numpy()[0])
    if abs(hs - 0.9) > 1e-6:
        raise RuntimeError(f"hard_sigmoid(2) = {hs}, not Keras 2's 0.9")
    from crema.models.chord import ChordModel
    return ChordModel()


def convert(model_dir, out_path=None) -> Path:
    import h5py
    import librosa
    import pumpp
    import tensorflow as tf
    import tf_keras

    ref = _load_crema_reference()
    import crema

    model_dir = Path(model_dir)
    h5_path = model_dir / "model.h5"
    out_path = Path(out_path) if out_path else DEFAULT_WEIGHTS

    arrays: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        keras_version = f.attrs.get("keras_version")
        keras_version = (keras_version.decode()
                         if isinstance(keras_version, bytes)
                         else str(keras_version))

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                parts = name.split("/")
                rest = parts[2:] if parts[1] == parts[0] else parts[1:]
                key = "/".join([parts[0]] + rest)
                if key.endswith(":0"):
                    key = key[:-2]
                arrays[key] = np.asarray(obj, dtype=np.float32)
        f["model_weights"].visititems(visit)

    op = ref.pump["cqt"]
    frontend = {"sr": float(op.sr), "hop_length": int(op.hop_length),
                "n_octaves": int(op.n_octaves),
                "over_sample": int(op.over_sample), "fmin": float(op.fmin),
                "harmonics": [int(h) for h in op.harmonics],
                "log": bool(op.log), "conv": str(op.conv)}
    if type(op).__name__ != "HCQTMag":
        raise RuntimeError(f"crema's pump front-end is {type(op).__name__}")
    vocab = [str(v) for v in ref.pump["chord_tag"].encoder.classes_]

    y = selftest_signal()
    feats = ref.pump.transform(y=y, sr=CREMA_SR)["cqt/mag"][0]
    outs = ref.outputs(y=y, sr=CREMA_SR)
    env = (f"crema {crema.version.version}, tensorflow {tf.__version__}, "
           f"tf_keras {tf_keras.__version__}, pumpp {pumpp.version.version}, "
           f"librosa {librosa.__version__}, numpy {np.__version__}")

    meta = {
        "meta/crema_model_version":
            np.array((model_dir / "version.txt").read_text().strip()),
        "meta/source_sha256": np.array(hashlib.sha256(
            h5_path.read_bytes()).hexdigest()),
        "meta/keras_version": np.array(keras_version),
        "meta/frontend": np.array(json.dumps(frontend, sort_keys=True)),
        "meta/vocabulary": np.array(vocab),
        "selftest/features": feats.astype(np.float32),
        "selftest/environment": np.array(env),
    }
    for k in OUTPUT_DIMS:
        meta[f"selftest/{k}"] = np.asarray(outs[k], np.float32)

    net = CremaChordNet()
    net.load_keras_weights(arrays)
    with torch.inference_mode():
        mine = net(torch.from_numpy(feats).unsqueeze(0))
    feats_mine = hcqt(y, CREMA_SR)
    print(f"front-end: max |hcqt - pumpp| = "
          f"{float(np.abs(feats_mine - feats).max()):.2e} dB")
    worst = 0.0
    for k in OUTPUT_DIMS:
        err = float(np.abs(mine[k][0].numpy() - outs[k]).max())
        worst = max(worst, err)
        print(f"network: {k:<12s} max |port - crema| = {err:.2e}")
    if worst >= 1e-4:
        raise RuntimeError("the port does not reproduce crema; not writing")

    np.savez_compressed(out_path, **arrays, **meta)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB), "
          f"sha1 {file_sha1(out_path)}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="PyTorch port of crema's chord model.")
    ap.add_argument("audio", nargs="*",
                    help="audio file(s) to print a chord timeline for")
    ap.add_argument("--selftest", action="store_true",
                    help="check this machine against the original's outputs")
    ap.add_argument("--convert", metavar="CREMA_CHORD_DIR",
                    help="rebuild the weights file from crema's models/chord "
                         "directory (needs crema's environment)")
    ap.add_argument("--weights", default=None,
                    help=f"weights file (default: {DEFAULT_WEIGHTS.name})")
    args = ap.parse_args()

    if args.convert:
        convert(args.convert, args.weights)
        return
    if args.selftest:
        sys.exit(0 if selftest(args.weights) else 1)
    if not args.audio:
        ap.error("give an audio file, --selftest or --convert")

    import librosa
    model = CremaChord(args.weights)
    for path in args.audio:
        y, sr = librosa.load(path, sr=CREMA_SR, mono=True)
        print(f"== {path}")
        for start, end, label in model.chord_segments(y, sr):
            print(f"  {start:8.2f} {end:8.2f}  {label}")


if __name__ == "__main__":
    main()
