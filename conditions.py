# Conditions: registry and extractors (f0, chroma, rhythm, energy, chord, midi, text, image).

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import inspect
import random
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


DAC_SAMPLE_RATE  = 44100
DAC_HOP_LENGTH   = 512
DAC_FRAMES_PER_S = DAC_SAMPLE_RATE / DAC_HOP_LENGTH

import latent_codec as _lc


class FrameConditionExtractor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        ...

    @staticmethod
    def _resample_to_frames(x: np.ndarray, target_len: int) -> np.ndarray:
        if x.shape[0] == target_len:
            return x
        from scipy.interpolate import interp1d
        f = interp1d(
            np.linspace(0, 1, x.shape[0]),
            x, axis=0, kind='linear', fill_value='extrapolate',
        )
        return f(np.linspace(0, 1, target_len))


def _projection_dim_from_config(model_name: str) -> Optional[int]:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        for obj in (cfg,
                    getattr(cfg, "text_config", None),
                    getattr(cfg, "vision_config", None),
                    getattr(cfg, "audio_config", None)):
            d = getattr(obj, "projection_dim", None) if obj is not None else None
            if isinstance(d, int) and d > 0:
                return int(d)
    except Exception:
        return None
    return None


class GlobalConditionExtractor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class ChromaExtractor(FrameConditionExtractor):
    @property
    def name(self): return "chroma"
    @property
    def dim(self): return 12

    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        import librosa
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=sr, tuning=0.0,
            hop_length=_lc.active().hop_length, n_chroma=12,
        ).T
        return self._resample_to_frames(chroma, n_frames).astype(np.float32)


class RhythmExtractor(FrameConditionExtractor):
    BEAT_THIS_FPS = 50.0

    _model = None

    def __init__(self, checkpoint: str = "final0", device: str = "cpu"):
        self.checkpoint = checkpoint
        self._device = device

    @property
    def name(self) -> str:
        return "rhythm"

    @property
    def dim(self) -> int:
        return 2

    @classmethod
    def _get_model(cls, checkpoint: str, device: str):
        if cls._model is None:
            try:
                from beat_this.inference import Audio2Frames
            except ImportError as e:
                raise ImportError(
                    "beat_this is required for RhythmExtractor. "
                    "Install with: pip install beat_this"
                ) from e
            cls._model = Audio2Frames(checkpoint_path=checkpoint, device=device)
        return cls._model

    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        model = self._get_model(self.checkpoint, self._device)

        beat_logits, downbeat_logits = model(audio.astype(np.float32), sr)

        beat = torch.sigmoid(beat_logits.detach().float()).cpu().numpy()
        downbeat = torch.sigmoid(downbeat_logits.detach().float()).cpu().numpy()

        curves = np.stack([beat, downbeat], axis=-1)
        curves = self._resample_to_frames(curves, n_frames)
        return np.clip(curves, 0.0, 1.0).astype(np.float32)


class EnergyExtractor(FrameConditionExtractor):
    def __init__(self,
                 n_fft: int = 2048,
                 weighting: str = "A",
                 fmin: float = 40.0,
                 top_db: float = 80.0,
                 smooth_sec: float = 1.0,
                 polyorder: int = 3):
        self.n_fft = int(n_fft)
        self.weighting = weighting
        self.fmin = float(fmin)
        self.top_db = float(top_db)
        self.smooth_sec = float(smooth_sec)
        self.polyorder = int(polyorder)
        self._freq_gain = None
        self._freq_gain_sr = None

    @property
    def name(self) -> str:
        return "energy"

    @property
    def dim(self) -> int:
        return 1

    def _frequency_gain(self, sr: int) -> np.ndarray:
        if self._freq_gain is not None and self._freq_gain_sr == sr:
            return self._freq_gain
        import librosa
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        gain = np.ones_like(freqs, dtype=np.float64)
        if self.weighting == "A":
            with np.errstate(divide="ignore"):
                a_db = librosa.A_weighting(freqs)
            gain = gain * (10.0 ** (a_db / 10.0))
        if self.fmin > 0:
            gain[freqs < self.fmin] = 0.0
        self._freq_gain = np.nan_to_num(gain, nan=0.0, posinf=0.0,
                                        neginf=0.0).astype(np.float64)
        self._freq_gain_sr = sr
        return self._freq_gain

    def _savgol(self, x: np.ndarray, fps: Optional[float] = None) -> np.ndarray:
        from scipy.signal import savgol_filter
        win = int(round(self.smooth_sec * _lc.active_fps(fps)))
        if win % 2 == 0:
            win += 1
        win = max(win, self.polyorder + 2)
        if win % 2 == 0:
            win += 1
        if win > len(x):
            win = len(x) if len(x) % 2 == 1 else len(x) - 1
        if win <= self.polyorder:
            return x
        return savgol_filter(x, window_length=win, polyorder=self.polyorder)

    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        import librosa
        hop = _lc.active().hop_length
        S = np.abs(librosa.stft(y=audio.astype(np.float32),
                                n_fft=self.n_fft, hop_length=hop))

        gain = self._frequency_gain(sr)
        S_w = S * np.sqrt(gain)[:, None]

        rms = librosa.feature.rms(S=S_w, frame_length=self.n_fft,
                                  hop_length=hop)[0]

        energy_db = 20.0 * np.log10(rms + 1e-7)

        energy_db = self._savgol(energy_db, fps=sr / hop)

        energy_norm = np.clip((energy_db + self.top_db) / self.top_db, 0.0, 1.0)

        energy_norm = self._resample_to_frames(energy_norm, n_frames)
        return energy_norm.reshape(n_frames, 1).astype(np.float32)


class CrepeF0Extractor(FrameConditionExtractor):
    def __init__(self,
                 fmin: float = 50.0,
                 fmax: float = 1000.0,
                 model: str = "full",
                 voicing_threshold: float = 0.5,
                 with_periodicity: bool = True,
                 hop_ms: float = 10.0,
                 silence_db: float = -60.0,
                 median_win: int = 3,
                 min_voiced_frames: int = 3,
                 voiced_floor: float = 0.05,
                 device: Optional[str] = "cpu",
                 batch_size: int = 64):
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.model = model
        self.voicing_threshold = float(voicing_threshold)
        self.with_periodicity = bool(with_periodicity)
        self.hop_ms = float(hop_ms)
        self.silence_db = float(silence_db)
        self.median_win = int(median_win)
        self.min_voiced_frames = int(min_voiced_frames)
        self.voiced_floor = float(voiced_floor)
        self.device = device
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("CrepeF0Extractor batch_size must be > 0")

    @property
    def name(self) -> str:
        return "f0"

    @property
    def dim(self) -> int:
        return 2 if self.with_periodicity else 1

    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        import torch
        import torchcrepe
        CR = 16000

        y = np.asarray(audio, dtype=np.float32)
        if sr != CR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=CR)
        wav = torch.from_numpy(y).unsqueeze(0)

        hop = max(1, int(round(self.hop_ms / 1000.0 * CR)))
        dev = self.device
        if dev in (None, "auto"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        pitch, periodicity = torchcrepe.predict(
            wav, CR, hop_length=hop, fmin=self.fmin, fmax=self.fmax,
            model=self.model, return_periodicity=True,
            batch_size=self.batch_size, device=dev,
        )
        pitch = pitch.squeeze(0).cpu().numpy().astype(np.float64)
        periodicity = periodicity.squeeze(0).cpu().numpy().astype(np.float64)
        F = pitch.shape[0]
        if F < 2:
            return np.zeros((n_frames, self.dim), dtype=np.float32)

        per_s = self._median_filter(periodicity, self.median_win)
        silent = self._silence_mask(y, hop, F, self.silence_db)
        voiced = (per_s >= self.voicing_threshold) & (~silent)
        voiced = self._remove_short_runs(voiced, self.min_voiced_frames)

        lo, hi = np.log2(self.fmin), np.log2(self.fmax)
        pf = np.clip(pitch, self.fmin, self.fmax)
        pnorm_all = ((np.log2(pf) - lo) / (hi - lo + 1e-8)).astype(np.float64)

        pnorm_r = self._resample_to_frames(pnorm_all.reshape(-1, 1), n_frames)[:, 0]
        mask_r  = self._resample_nearest(voiced.astype(np.float64), n_frames) > 0.5
        per_r   = self._resample_to_frames(per_s.reshape(-1, 1), n_frames)[:, 0]

        pitch_ch = np.where(
            mask_r, self.voiced_floor + (1.0 - self.voiced_floor) * pnorm_r, 0.0)

        if self.with_periodicity:
            per_ch = np.where(mask_r, per_r, 0.0)
            feat = np.stack([pitch_ch, per_ch], axis=-1)
        else:
            feat = pitch_ch.reshape(-1, 1)
        return feat.astype(np.float32)

    @staticmethod
    def _median_filter(x: np.ndarray, win: int) -> np.ndarray:
        if win and win >= 3:
            from scipy.signal import medfilt
            return medfilt(x, kernel_size=win if win % 2 == 1 else win + 1)
        return x

    @staticmethod
    def _silence_mask(y16k: np.ndarray, hop: int, F: int, silence_db: float) -> np.ndarray:
        win = max(hop, 1024)
        energy = np.convolve((y16k.astype(np.float64) ** 2),
                             np.ones(win) / win, mode="same")
        centers = np.clip(np.arange(F) * hop, 0, len(energy) - 1)
        rms = np.sqrt(energy[centers] + 1e-12)
        db = 20.0 * np.log10(rms + 1e-9)
        return db < silence_db

    @staticmethod
    def _remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
        if min_len <= 1 or mask.size == 0:
            return mask
        out = mask.copy()
        n = out.size
        i = 0
        while i < n:
            if out[i]:
                j = i
                while j < n and out[j]:
                    j += 1
                if (j - i) < min_len:
                    out[i:j] = False
                i = j
            else:
                i += 1
        return out

    @staticmethod
    def _resample_nearest(x: np.ndarray, target_len: int) -> np.ndarray:
        if x.shape[0] == target_len:
            return x
        from scipy.interpolate import interp1d
        f = interp1d(np.linspace(0, 1, x.shape[0]), x, axis=0,
                     kind="nearest", fill_value="extrapolate")
        return f(np.linspace(0, 1, target_len))


class CremaChordExtractor(FrameConditionExtractor):
    _models: dict = {}

    def __init__(self, silence_db: Optional[float] = -60.0,
                 weights: Optional[str] = None):
        from crema_chord import DEFAULT_WEIGHTS, file_sha1
        self.silence_db = None if silence_db is None else float(silence_db)
        self._weights = str(weights or DEFAULT_WEIGHTS)
        if not Path(self._weights).is_file():
            raise FileNotFoundError(
                f"CremaChordExtractor: weights not found at {self._weights}. "
                f"crema_chord_weights.npz ships next to crema_chord.py -- copy "
                f"it along with the code.")
        self.weights_sha1 = file_sha1(self._weights)

    @property
    def name(self) -> str:
        return "chord"

    @property
    def dim(self) -> int:
        return 12

    def _get_model(self):
        model = self._models.get(self._weights)
        if model is None:
            from crema_chord import CremaChord
            model = CremaChord(self._weights, device="cpu")
            self._models[self._weights] = model
        return model

    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        from crema_chord import CREMA_FPS
        y = np.asarray(audio, dtype=np.float32)
        pitch = self._get_model().outputs(y, sr)["chord_pitch"]
        if pitch.shape[0] == 0:
            return np.zeros((n_frames, self.dim), dtype=np.float32)
        t_src = np.arange(pitch.shape[0]) / CREMA_FPS
        fps = _lc.active_fps()
        t_dst = np.arange(n_frames) / fps
        out = np.stack([np.interp(t_dst, t_src, pitch[:, c])
                        for c in range(pitch.shape[1])], axis=1)
        if self.silence_db is not None:
            import librosa
            hop = max(1, int(round(sr / fps)))
            rms = librosa.feature.rms(y=y, frame_length=4 * hop,
                                      hop_length=hop)[0]
            rms_db = 20.0 * np.log10(rms + 1e-12)
            silent = np.interp(t_dst, np.arange(rms.size) * hop / sr,
                               rms_db) < self.silence_db
            out[silent] = 0.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)


MIDI_LOWEST_KEY = 21
MIDI_N_KEYS = 88
MIDI_DRUM_CLASSES = (
    ("kick", (36, 35)),
    ("snare", (38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85)),
    ("hihat_closed", (42, 44, 54, 68, 69, 70, 71, 73, 78, 80, 22)),
    ("hihat_open", (46, 67, 72, 74, 79, 81, 26)),
    ("tom_low", (45, 29, 41, 43, 61, 64, 84)),
    ("tom_mid", (48, 47, 60, 63, 77, 86, 87)),
    ("tom_high", (50, 30, 62, 76, 83)),
    ("crash", (49, 52, 55, 57, 58)),
    ("ride", (51, 53, 59, 82)),
)
MIDI_DRUM_CLASS_OF = {p: k for k, (_n, ps) in enumerate(MIDI_DRUM_CLASSES)
                      for p in ps}
MIDI_DIM = 2 * MIDI_N_KEYS + len(MIDI_DRUM_CLASSES)


def read_midi_notes(path) -> list:
    import mido
    mf = mido.MidiFile(str(path))
    t, open_notes, notes = 0.0, {}, []
    for msg in mf:
        t += msg.time
        if msg.type not in ("note_on", "note_off"):
            continue
        key = (msg.channel, msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            open_notes.setdefault(key, []).append(t)
        elif open_notes.get(key):
            notes.append((open_notes[key].pop(0), t, msg.note,
                          msg.channel == 9))
    for (channel, note), starts in open_notes.items():
        notes.extend((s, t, note, channel == 9) for s in starts)
    notes.sort()
    return notes


def midi_notes_to_roll(notes, n_frames: int, t0: float = 0.0,
                       fps: Optional[float] = None) -> np.ndarray:
    fps = _lc.active_fps(fps)
    n_frames = int(n_frames)
    roll = np.zeros((n_frames, MIDI_DIM), dtype=np.float32)
    end = n_frames / float(fps)
    n_keys, drum0 = MIDI_N_KEYS, 2 * MIDI_N_KEYS
    for onset, offset, pitch, is_drum in notes:
        s, e = float(onset) - t0, float(offset) - t0
        if e <= 0.0 or s >= end:
            continue
        a = int(round(s * fps))
        if is_drum:
            k = MIDI_DRUM_CLASS_OF.get(int(pitch))
            if k is not None and s >= 0.0 and a < n_frames:
                roll[a, drum0 + k] = 1.0
            continue
        key = int(pitch) - MIDI_LOWEST_KEY
        if not 0 <= key < n_keys:
            continue
        lo = max(a, 0)
        hi = min(max(int(round(e * fps)), a + 1), n_frames)
        if hi > lo:
            roll[lo:hi, key] = 1.0
        if s >= 0.0 and a < n_frames:
            roll[a, n_keys + key] = 1.0
    return roll


def midi_roll_to_events(roll, fps: Optional[float] = None):
    fps = _lc.active_fps(fps)
    r = np.asarray(roll, dtype=np.float32)
    n_keys, drum0 = MIDI_N_KEYS, 2 * MIDI_N_KEYS
    sounding = r[:, :n_keys] >= 0.5
    onsets = r[:, n_keys:drum0] >= 0.5
    T = r.shape[0]
    pitched = []
    for a, k in zip(*np.nonzero(onsets)):
        b = a + 1
        while b < T and sounding[b, k] and not onsets[b, k]:
            b += 1
        pitched.append((a / fps, b / fps, int(k) + MIDI_LOWEST_KEY))
    pitched.sort()
    drums = sorted((a / fps, int(k))
                   for a, k in zip(*np.nonzero(r[:, drum0:] >= 0.5)))
    return pitched, drums


class MidiExtractor(FrameConditionExtractor):
    companion_suffixes = (".mid", ".midi")
    lowest_key = MIDI_LOWEST_KEY
    n_keys = MIDI_N_KEYS
    n_drum_classes = len(MIDI_DRUM_CLASSES)

    def __init__(self, device: str = "cpu"):
        from midi_transcriber import MODEL_NAME, REVISION
        self.device = str(device)
        self.transcriber = f"YourMT3+ {MODEL_NAME} @ {REVISION[:8]}"

    @property
    def name(self) -> str:
        return "midi"

    @property
    def dim(self) -> int:
        return MIDI_DIM

    def load_companion(self, path) -> list:
        return read_midi_notes(path)

    def from_companion(self, notes, t0: float, n_frames: int) -> np.ndarray:
        return midi_notes_to_roll(notes, n_frames, t0=float(t0))

    def from_notes(self, notes, n_frames: int) -> np.ndarray:
        return midi_notes_to_roll(notes, n_frames, t0=0.0)

    def prepare(self):
        from midi_transcriber import YourMT3Transcriber
        YourMT3Transcriber(self.device).load()

    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        from midi_transcriber import YourMT3Transcriber
        notes = YourMT3Transcriber(self.device).transcribe(audio, sr)
        return midi_notes_to_roll([(o, e, p, d) for o, e, p, d, _prog in notes],
                                  n_frames)


class CLAPTextCondition(GlobalConditionExtractor):
    def __init__(self, model_name: str = "laion/clap-htsat-unfused"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._dim = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._audio_embedder = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import ClapTextModelWithProjection, AutoTokenizer
        self._model = ClapTextModelWithProjection.from_pretrained(self.model_name)
        self._processor = AutoTokenizer.from_pretrained(self.model_name)
        self._model.eval()
        self._dim = int(self._model.config.projection_dim)
        self._model.to(self._device)
        print(f"[CLAPTextCondition] '{self.model_name}' "
              f"loaded on {self._device} (dim={self._dim})")

    @property
    def name(self): return "text"
    @property
    def dim(self):
        if self._dim is None:
            self._dim = _projection_dim_from_config(self.model_name)
        if self._dim is None:
            self._load()
        return self._dim

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        self._load()
        inputs = self._processor([text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        out = self._model(**inputs)
        feat = out.text_embeds
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        self._load()
        inputs = self._processor(list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        out = self._model(**inputs)
        feat = out.text_embeds
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    @property
    def ctx_dim(self) -> int:
        self._load()
        return int(self._model.config.hidden_size)

    @torch.no_grad()
    def encode_tokens(self, texts: List[str]):
        self._load()
        inputs = self._processor(list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        out = self._model(**inputs)
        h = out.last_hidden_state.float()
        mask = inputs["attention_mask"].to(h.dtype)
        h = h * mask.unsqueeze(-1)
        lengths = inputs["attention_mask"].sum(dim=1)
        return (h.cpu().numpy().astype(np.float32),
                lengths.cpu().numpy().astype(np.int32))


    def _audio_side(self):
        if self._audio_embedder is None:
            self._audio_embedder = ClapAudioEmbedder(model_name=self.model_name,
                                                     device=self._device)
        return self._audio_embedder

    @torch.no_grad()
    def encode_audio(self, wav_np: np.ndarray, sr: int) -> np.ndarray:
        return self._audio_side().embed(wav_np, sr)

    def unload(self):
        if self._audio_embedder is not None:
            self._audio_embedder.unload()
            self._audio_embedder = None
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()


class ClapAudioEmbedder:
    def __init__(self, model_name: str = "laion/clap-htsat-unfused",
                 device: Optional[str] = None):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._dim = None
        self._audio_kw = "audio"
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._model is not None:
            return
        from transformers import ClapAudioModelWithProjection, AutoProcessor
        self._model = ClapAudioModelWithProjection.from_pretrained(self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model.eval().to(self._device)
        self._dim = int(self._model.config.projection_dim)
        try:
            params = inspect.signature(self._processor.__call__).parameters
            self._audio_kw = "audio" if "audio" in params else "audios"
        except (TypeError, ValueError):
            self._audio_kw = "audio"
        print(f"[ClapAudioEmbedder] '{self.model_name}' audio encoder "
              f"loaded on {self._device} (dim={self._dim})")

    @torch.no_grad()
    def embed(self, wav_np: np.ndarray, sr: int) -> np.ndarray:
        self._load()
        if wav_np.ndim > 1:
            wav_np = wav_np.squeeze()
        wav_np = np.asarray(wav_np, dtype=np.float32)

        want_sr = int(getattr(self._processor, "feature_extractor",
                              self._processor).sampling_rate)
        if int(sr) != want_sr:
            wav_np = torchaudio.functional.resample(
                torch.from_numpy(wav_np), int(sr), want_sr).numpy()

        inputs = self._processor(sampling_rate=want_sr, return_tensors="pt",
                                 **{self._audio_kw: wav_np})
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        out = self._model(**inputs)
        feat = out.audio_embeds
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def unload(self):
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()


TEXT_LABEL_VOCAB = [
    "solo pipe organ", "a cappella choir", "harpsichord", "solo piano",
    "string quartet", "solo violin", "solo cello", "acoustic guitar",
    "electric guitar", "distorted electric guitar", "electric bass",
    "drum kit", "hand percussion", "brass section", "solo trumpet",
    "woodwinds", "synthesizer", "analog synthesizer bass", "orchestra",
    "a single sustained note", "a dense polyphonic texture",
    "a solo melodic line", "a low bass register", "a high bright register",
    "sparse and quiet", "loud and dense",
    "a steady four on the floor beat", "a fast rhythmic pattern",
    "a slow tempo", "no clear pulse", "a strong groove",
    "a large reverberant church", "a dry close recording",
    "a live concert recording", "a lo-fi noisy recording",
    "baroque sacred music", "gregorian chant", "classical music",
    "romantic orchestral music", "rock music", "heavy metal",
    "electronic dance music", "ambient music", "experimental noise",
    "jazz", "folk music", "film score",
    "a plucked pizzicato attack", "a bowed tremolo", "breathy air noise",
    "silence", "white noise",
]


def nearest_phrases(vec, vocab_emb, phrases, k: int = 2):
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    M = np.asarray(vocab_emb, dtype=np.float32)
    if v.size == 0 or M.size == 0 or M.shape[1] != v.shape[0]:
        return []
    sims = M @ v
    order = np.argsort(-sims)[:max(1, int(k))]
    return [(phrases[i], float(sims[i])) for i in order if i < len(phrases)]


class Wav2ClipAudioEmbedder:
    SR = 16000

    def __init__(self, device: Optional[str] = None):
        self._model = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load(self):
        if self._model is not None:
            return
        try:
            import wav2clip
        except ImportError as e:
            raise ImportError(
                "wav2clip is required to measure the IMAGE condition's "
                "influence (audio<->image similarity). Install it with:\n"
                "    pip install wav2clip\n"
                "or switch the image influence off; nothing else needs it."
            ) from e
        self._model = wav2clip.get_model(device=self._device)
        print(f"[Wav2ClipAudioEmbedder] loaded on {self._device} "
              f"(audio -> CLIP space, dim=512)")

    @torch.no_grad()
    def embed(self, wav_np: np.ndarray, sr: int) -> np.ndarray:
        import wav2clip
        self._load()
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = wav_np.squeeze()
        if int(sr) != self.SR:
            wav_np = torchaudio.functional.resample(
                torch.from_numpy(wav_np), int(sr), self.SR).numpy()
        emb = np.asarray(wav2clip.embed_audio(wav_np, self._model),
                         dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(emb))
        return emb / n if n > 0 else emb

    def unload(self):
        if self._model is not None:
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()


class ImageCondition(GlobalConditionExtractor):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._dim = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import CLIPVisionModelWithProjection, AutoProcessor
        self._model = CLIPVisionModelWithProjection.from_pretrained(self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model.eval()
        self._dim = int(self._model.config.projection_dim)
        print(f"[ImageCondition] CLIP '{self.model_name}' loaded (dim={self._dim})")

    @property
    def name(self): return "image"
    @property
    def dim(self):
        if self._dim is None:
            self._dim = _projection_dim_from_config(self.model_name)
        if self._dim is None:
            self._load()
        return self._dim

    @torch.no_grad()
    def encode_image(self, image_path: str) -> np.ndarray:
        self._load()
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")
        out = self._model(**inputs)
        feat = out.image_embeds
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def unload(self):
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None


class ImageDatasetManager:
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, image_root: str, split: Optional[str] = None):
        self.image_root = Path(image_root)
        self.split = split
        self.class_images: Dict[str, List[Path]] = {}
        self._scan()

    def _scan(self):
        if not self.image_root.exists():
            print(f"[ImageDataset] WARNING: {self.image_root} not found")
            return

        if self.split is not None:
            base = self.image_root / self.split
            if not base.exists():
                print(f"[ImageDataset/{self.split}] {base} does not exist, "
                      f"using legacy layout {self.image_root}")
                base = self.image_root
        else:
            base = self.image_root

        for d in sorted(base.iterdir()):
            if d.is_dir():
                imgs = [f for f in sorted(d.iterdir())
                        if f.suffix.lower() in self.EXTS]
                if imgs:
                    self.class_images[d.name] = imgs
        total = sum(len(v) for v in self.class_images.values())
        tag = f"/{self.split}" if self.split else ""
        print(f"[ImageDataset{tag}] {len(self.class_images)} classes, {total} images")

    def has_class(self, class_name: str) -> bool:
        return class_name in self.class_images

    def get_random_image(self, class_name: str, rng=None) -> Optional[Path]:
        imgs = self.class_images.get(class_name, [])
        if not imgs:
            return None
        return (rng or random).choice(imgs)

    def get_all_images(self, class_name: str) -> List[Path]:
        return self.class_images.get(class_name, [])


CONDITION_CONFIG = {
    "frame_level": {
        "chroma": {
            "class": ChromaExtractor,
            "kwargs": {},
            "out_dim": 64,
            "enabled": True,
        },
        "rhythm": {
            "class": RhythmExtractor,
            "kwargs": {},
            "out_dim": 32,
            "enabled": True,
        },
        "energy": {
            "class": EnergyExtractor,
            "kwargs": {"weighting": "A", "fmin": 40.0},
            "out_dim": 16,
            "enabled": True,
        },
        "f0": {
            "class": CrepeF0Extractor,
            "kwargs": {"fmin": 50.0, "fmax": 1000.0, "model": "full",
                       "voicing_threshold": 0.5, "with_periodicity": True,
                       "silence_db": -60.0, "median_win": 3,
                       "min_voiced_frames": 3, "voiced_floor": 0.05,
                       "device": "cpu", "batch_size": 512},
            "out_dim": 16,
            "enabled": True,
        },
        "chord": {
            "class": CremaChordExtractor,
            "kwargs": {"silence_db": -60.0},
            "out_dim": 64,
            "enabled": True,
        },
        "midi": {
            "class": MidiExtractor,
            "kwargs": {"device": "cpu"},
            "out_dim": 128,
            "enabled": True,
        },
    },
    "global": {
        "text": {
            "class": CLAPTextCondition,
            "kwargs": {"model_name": "laion/clap-htsat-unfused"},
            "enabled": True,
        },
        "image": {
            "class": ImageCondition,
            "kwargs": {"model_name": "openai/clip-vit-base-patch32"},
            "enabled": True,
        },
    },
}


class ConditionRegistry:
    def __init__(self, n_classes: Optional[int] = None,
                 config: Optional[dict] = None,
                 enabled_frame:  Optional[List[str]] = None,
                 enabled_global: Optional[List[str]] = None):
        self.frame_extractors: Dict[str, FrameConditionExtractor] = {}
        self.frame_out_dims: Dict[str, int] = {}
        self.global_extractors: Dict[str, GlobalConditionExtractor] = {}
        self.n_classes = n_classes

        config = config or CONDITION_CONFIG
        self._build(config, enabled_frame, enabled_global)

    def _build(self, config,
               enabled_frame:  Optional[List[str]] = None,
               enabled_global: Optional[List[str]] = None):
        for name, cfg in config.get("frame_level", {}).items():
            if not cfg.get("enabled", False):
                continue
            if enabled_frame is not None and name not in enabled_frame:
                continue
            cls = cfg["class"]
            kwargs = cfg.get("kwargs", {})
            self.frame_extractors[name] = cls(**kwargs)
            self.frame_out_dims[name] = int(
                cfg.get("out_dim", self.frame_extractors[name].dim)
            )

        if enabled_frame is not None:
            missing = set(enabled_frame) - set(self.frame_extractors.keys())
            if missing:
                available = [n for n, c in config.get("frame_level", {}).items()
                             if c.get("enabled", False)]
                raise ValueError(
                    f"enabled_frame requested {sorted(missing)} but these "
                    f"are not enabled=True in CONDITION_CONFIG. "
                    f"Currently active in CONDITION_CONFIG: {available}"
                )

        for name, cfg in config.get("global", {}).items():
            if not cfg.get("enabled", False):
                continue
            if enabled_global is not None and name not in enabled_global:
                continue
            cls = cfg["class"]
            kwargs = dict(cfg.get("kwargs", {}))
            self.global_extractors[name] = cls(**kwargs)

        if enabled_global is not None:
            missing = set(enabled_global) - set(self.global_extractors.keys())
            if missing:
                available = [n for n, c in config.get("global", {}).items()
                             if c.get("enabled", False)]
                raise ValueError(
                    f"enabled_global requested {sorted(missing)} but these "
                    f"are not enabled=True in CONDITION_CONFIG. "
                    f"Currently active in CONDITION_CONFIG: {available}"
                )

    @property
    def frame_names(self) -> List[str]:
        return list(self.frame_extractors.keys())

    @property
    def global_names(self) -> List[str]:
        return list(self.global_extractors.keys())

    @property
    def frame_cond_dims(self) -> Dict[str, int]:
        return {n: e.dim for n, e in self.frame_extractors.items()}

    @property
    def frame_cond_out_dims(self) -> Dict[str, int]:
        return {n: self.frame_out_dims[n] for n in self.frame_extractors.keys()}

    @property
    def global_cond_configs(self) -> Dict[str, dict]:
        return {n: {"dim": e.dim} for n, e in self.global_extractors.items()}

    def extract_frame_conditions(
        self, audio: np.ndarray, sr: int, n_frames: int,
    ) -> Dict[str, np.ndarray]:
        out = {}
        for name, extractor in self.frame_extractors.items():
            out[name] = extractor.extract(audio, sr, n_frames)
        return out

    def __repr__(self):
        f = ", ".join(f"{n}(dim={e.dim})" for n, e in self.frame_extractors.items())
        g = ", ".join(f"{n}(dim={e.dim})" for n, e in self.global_extractors.items())
        return f"ConditionRegistry(frame=[{f}], global=[{g}])"


class FrameConditionEncoder(nn.Module):
    def __init__(self, condition_dims: Dict[str, int], out_dims: Dict[str, int]):
        super().__init__()
        self.names = list(condition_dims.keys())
        missing = set(self.names) - set(out_dims.keys())
        if missing:
            raise ValueError(f"FrameConditionEncoder: missing out_dim for {sorted(missing)}")
        self.projections = nn.ModuleDict({
            name: nn.Linear(condition_dims[name], out_dims[name])
            for name in self.names
        })
        self.total_out_dim = int(sum(out_dims[name] for name in self.names))

    def forward(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.projections[name](conditions[name]) for name in self.names]
        return torch.cat(parts, dim=-1)


class GlobalConditionEncoder(nn.Module):
    def __init__(self, global_configs: Dict[str, dict], hidden_size: int):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for name, cfg in global_configs.items():
            self.encoders[name] = nn.Sequential(
                nn.Linear(cfg["dim"], hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, conditions: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        embs = [enc(conditions[name]) for name, enc in self.encoders.items()
                if name in conditions]
        if not embs:
            return None
        return self.final_norm(torch.stack(embs, dim=0).sum(dim=0))


def make_null_frame_conditions(B: int, n_frames: int,
                                 cond_dims: Dict[str, int],
                                 device) -> Dict[str, torch.Tensor]:
    return {n: torch.zeros(B, n_frames, d, device=device)
            for n, d in cond_dims.items()}


def make_null_global_conditions(B: int,
                                  global_configs: Dict[str, dict],
                                  device) -> Dict[str, torch.Tensor]:
    return {n: torch.zeros(B, cfg["dim"], device=device)
            for n, cfg in global_configs.items()}


if __name__ == "__main__":
    print("=" * 60)
    print("Test ConditionRegistry (CLAP-based, no label)")
    print("=" * 60)

    reg = ConditionRegistry()
    print(reg)
    print(f"\nFrame cond dims:    {reg.frame_cond_dims}")
    print(f"Global cond configs: {reg.global_cond_configs}")

    print("\n--- Test CLAP text encoding (single prompt) ---")
    if "text" in reg.global_extractors:
        t = reg.global_extractors["text"]
        emb = t.encode_text("baroque sacred music")
        print(f"  Embedding shape: {emb.shape}, "
              f"norm: {np.linalg.norm(emb):.4f} (expected ~1.0)")
        t.unload()
        print("  CLAP offloaded from GPU.")
