# conditions.py
#
# Modular conditioning system for the Audio DiT.
#
# Design:
#   - Centralised CONDITION_CONFIG: single source of truth for which
#     conditions are active and how they are configured.
#   - Every script (extract, dataset, training, sampling) reads from this
#     config: change it in one place to change everything.
#
# To add a new condition:
#   1. Write a class extending FrameConditionExtractor or GlobalConditionExtractor
#   2. Add it to CONDITION_CONFIG
#   3. Done -- dataset, training and sampling will use it automatically
#
# Two condition families:
#
#   FRAME-LEVEL (time-aligned, injected by CONCATENATION on the feature
#   dimension at the model input, JASCO-style -- see network_cond.py):
#     - chroma: chromagram CQT (12 pitch classes) -- harmony.
#     - rhythm: per-frame beat + downbeat probability curves (2 channels)
#               from beat_this (Music ControlNet-style rhythm control).
#     - chord:  per-frame chord pitch classes (12) from crema's chord model
#               (PyTorch port, crema_chord.py) -- harmony as a chord
#               recognizer hears it, rather than raw pitch-class energy.
#     [extensible: mfcc, spectral_centroid, ...]
#
#   GLOBAL (single vector per sample, injected via AdaLN as in the official
#   DiT class label -- modulates every block) -- continuous only:
#     - text:  CLAP text encoder
#     - image: CLIP (from an image of the same class)
#     [extensible: mood embedding, tempo embedding, ...]
#
# NB: LabelCondition was REMOVED. The `text` modality with CLAP (fed by the
# class name) takes its place and will later allow free-form prompts without
# any architecture change.
#
# Requirements:
#   pip install librosa scipy transformers Pillow
#   pip install beat_this            # beat/downbeat tracker (PyTorch, ISMIR 2024)

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import inspect
import random
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


# ============================================================
# DAC CONSTANTS (consistent with audio_dataset_npy)
# ============================================================
DAC_SAMPLE_RATE  = 44100
DAC_HOP_LENGTH   = 512
DAC_FRAMES_PER_S = DAC_SAMPLE_RATE / DAC_HOP_LENGTH


# ============================================================
# BASE CLASSES
# ============================================================

class FrameConditionExtractor(ABC):
    """
    Extracts a frame-level condition from audio.
    Output shape: (n_frames, dim)

    Used by extract_conditions.py to pre-compute the conditions and save them
    to disk (.npz), so that training is fast.
    """

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
        """
        Args:
            audio: (T,) waveform mono
            sr:    sample rate
            n_frames: target number of frames (alignment with DAC latents)
        Returns:
            (n_frames, self.dim) float32
        """
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
    """The width of a HuggingFace checkpoint's projected embedding, read off its
    CONFIG -- a small JSON -- without building the model.

    WHY IT EXISTS. `dim` is asked for in places that never encode anything: the
    training reads it to size the AdaLN input projection, and the probe-cache
    fingerprint reads it through dir(). Answering by instantiating the model
    put the CLAP text tower on the TRAINING GPU (measured: 0.56 GB) and left it
    there for the whole run, to obtain an integer that is written in the config
    file. Every caller that actually encodes still loads the weights the usual
    way, so nothing else changes.

    Returns None when the config does not declare it (the caller then falls back
    to loading the model, which is the answer of last resort but always right).
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)
        # Composite configs (ClapConfig, CLIPConfig) carry projection_dim at the
        # top level AND on each tower; the single-tower configs carry only their
        # own. Take the first that answers -- they agree, and this works for both.
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
    """
    Encodes a global condition (one continuous vector per sample).

    NB: with LabelCondition removed, all global conditions are now continuous.
    No categorical branch -> no Embedding lookup.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...


# ============================================================
# FRAME-LEVEL: CHROMAGRAM
# ============================================================

class ChromaExtractor(FrameConditionExtractor):
    """Chromagram CQT. Output: (n_frames, 12) -> distribution over 12 pitch classes."""

    @property
    def name(self): return "chroma"
    @property
    def dim(self): return 12

    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        import librosa
        # tuning=0.0 -- chroma_cqt defaults to tuning=None, which makes librosa
        # ESTIMATE the tuning of every chunk (estimate_tuning -> piptrack -> a
        # numba gufunc that segfaults on the IRCAM nodes, in the worker and in
        # the main process alike). 0.0 means A440, which is also what
        # librosa.cqt itself defaults to -- so crema (crema_chord.py) never
        # took that path and needs no change.
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=sr, tuning=0.0,
            hop_length=DAC_HOP_LENGTH, n_chroma=12,
        ).T
        return self._resample_to_frames(chroma, n_frames).astype(np.float32)


# ============================================================
# FRAME-LEVEL: RHYTHM (beat + downbeat, beat_this)
# ============================================================

class RhythmExtractor(FrameConditionExtractor):
    """
    Music ControlNet-style rhythm control: two per-frame probability curves,
    one for beats and one for downbeats.

    Backbone: beat_this (CPJKU, ISMIR 2024 -- "Beat This! Accurate Beat
    Tracking Without DBN Postprocessing"). It is the modern, pip-installable,
    PyTorch replacement for madmom's beat/downbeat tracker (madmom is pinned to
    Python < 3.10 on PyPI and is painful to install on recent setups). beat_this
    is from the same lab as madmom and needs no DBN/madmom postprocessing.

    Pipeline:
      1. beat_this Audio2Frames returns FRAMEWISE beat and downbeat LOGITS at
         50 fps (its mel spectrogram uses sr=22050, hop=441 -> 22050/441 = 50).
      2. sigmoid(logits) -> per-frame probabilities in [0, 1] (the continuous
         beat/downbeat curves, as in Music ControlNet -- not hard impulses).
      3. The two curves are stacked to (T_native, 2) and linearly interpolated
         on the time axis to n_frames (DAC-aligned), then clamped to [0, 1].

    Output: (n_frames, 2) float32 -- channel 0 = beat prob, channel 1 = downbeat prob.

    Null rhythm = all zeros (no beats / no downbeats), consistent with the CFG
    dropout and make_null_frame_conditions.
    """

    BEAT_THIS_FPS = 50.0  # beat_this framewise rate (22050 / 441)

    _model = None  # process-wide singleton

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

        # beat_this accepts a numpy/torch signal directly and resamples to
        # 22050 internally (via soxr). Returns framewise logits at 50 fps.
        beat_logits, downbeat_logits = model(audio.astype(np.float32), sr)

        beat = torch.sigmoid(beat_logits.detach().float()).cpu().numpy()       # (T_native,)
        downbeat = torch.sigmoid(downbeat_logits.detach().float()).cpu().numpy()  # (T_native,)

        curves = np.stack([beat, downbeat], axis=-1)                           # (T_native, 2)
        curves = self._resample_to_frames(curves, n_frames)
        return np.clip(curves, 0.0, 1.0).astype(np.float32)


# ============================================================
# FRAME-LEVEL: ENERGY / DYNAMICS (frequency-weighted spectral energy in dB)
# ============================================================

class EnergyExtractor(FrameConditionExtractor):
    """
    Music ControlNet-style "dynamics" control: a single per-frame curve that
    tracks the perceived loudness / dynamics of the music (forte vs piano,
    crescendo / diminuendo), NOT the per-note onset transients.

    Convergent recipe across the controllable-music-generation literature
    (Music ControlNet, Wu et al. 2024; MuseControlLite 2025; Audio ControlNet
    2026; Controllable Video-to-Music 2025):

      1. Frequency-weighted SPECTRAL ENERGY. We take the STFT power spectrogram
         (hop = DAC_HOP_LENGTH, so frames align with the DAC latents like the
         chroma) and weight the frequency bins BEFORE summing them, so the curve
         reflects PERCEIVED intensity rather than raw sample energy:
           - a high-pass cutoff (`fmin`) zeroes DC and sub-audible rumble
             (relevant on classical recordings with room/handling noise);
           - optional A-weighting (`weighting="A"`) applies the standard
             perceptual loudness contour (librosa.A_weighting).
         The weighted power is summed over frequency -> per-frame energy.

      2. dB SCALE. The weighted power is averaged over frequency, square-rooted
         to an amplitude-like RMS, and converted to ABSOLUTE dB (20*log10(rms+eps),
         dBFS-like) -- NOT relative to the clip maximum. This keeps the dynamics
         comparable across clips (absolute level is roughly equalised by the
         loudnorm in preprocessing) and lets silence map to the floor.

      3. SMOOTHING. A Savitzky-Golay filter over a ~`smooth_sec` window removes
         the fast onset spikes, leaving the slow dynamic envelope.

      4. NORMALISATION [-top_db, 0] dB -> [0, 1]: silence -> 0, full-scale -> 1.
         This keeps the same non-negative range as the other frame conditions
         and, crucially, makes the NULL condition (all zeros, used by CFG dropout
         and make_null_frame_conditions) read as "silence", consistent with the
         zeros-mean-absence convention of chroma / rhythm.

    Output: (n_frames, 1) float32 in [0, 1].

    Adherence is evaluated (in condition_metrics.py) with Pearson correlation
    between the input curve and the one re-extracted from the generation, exactly
    as Music ControlNet evaluates dynamics control.
    """

    def __init__(self,
                 n_fft: int = 2048,
                 weighting: str = "A",      # "A" (perceptual) or "none"
                 fmin: float = 40.0,        # high-pass cutoff (Hz); 0 disables
                 top_db: float = 80.0,      # dynamic range below the per-clip max
                 smooth_sec: float = 1.0,   # Savitzky-Golay window (seconds)
                 polyorder: int = 3):
        self.n_fft = int(n_fft)
        self.weighting = weighting
        self.fmin = float(fmin)
        self.top_db = float(top_db)
        self.smooth_sec = float(smooth_sec)
        self.polyorder = int(polyorder)
        self._freq_gain = None      # cached linear power gain per FFT bin
        self._freq_gain_sr = None

    @property
    def name(self) -> str:
        return "energy"

    @property
    def dim(self) -> int:
        return 1

    def _frequency_gain(self, sr: int) -> np.ndarray:
        """Per-FFT-bin multiplicative gain applied to the POWER spectrogram.
        Cached per (sr, params). High-pass mask * (optional) A-weighting."""
        if self._freq_gain is not None and self._freq_gain_sr == sr:
            return self._freq_gain
        import librosa
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)  # (n_fft//2+1,)
        gain = np.ones_like(freqs, dtype=np.float64)
        if self.weighting == "A":
            # A_weighting returns dB; convert to a LINEAR POWER gain (10^(dB/10)).
            # At f=0 it is -inf dB (log10(0)); the high-pass below zeroes that
            # bin anyway, so the warning is harmless -- silence it.
            with np.errstate(divide="ignore"):
                a_db = librosa.A_weighting(freqs)
            gain = gain * (10.0 ** (a_db / 10.0))
        if self.fmin > 0:
            gain[freqs < self.fmin] = 0.0                          # high-pass
        self._freq_gain = np.nan_to_num(gain, nan=0.0, posinf=0.0,
                                        neginf=0.0).astype(np.float64)
        self._freq_gain_sr = sr
        return self._freq_gain

    def _savgol(self, x: np.ndarray) -> np.ndarray:
        """Savitzky-Golay smoothing with a window sized in seconds, guarded for
        short clips (window must be odd, > polyorder, and <= len(x))."""
        from scipy.signal import savgol_filter
        win = int(round(self.smooth_sec * DAC_FRAMES_PER_S))
        if win % 2 == 0:
            win += 1                          # must be odd
        win = max(win, self.polyorder + 2)
        if win % 2 == 0:
            win += 1
        if win > len(x):                      # clip too short -> shrink window
            win = len(x) if len(x) % 2 == 1 else len(x) - 1
        if win <= self.polyorder:
            return x                          # not enough frames to smooth
        return savgol_filter(x, window_length=win, polyorder=self.polyorder)

    def extract(self, audio: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
        import librosa
        # 1. STFT magnitude spectrogram, hop aligned to DAC frames (like chroma).
        S = np.abs(librosa.stft(y=audio.astype(np.float32),
                                n_fft=self.n_fft, hop_length=DAC_HOP_LENGTH))  # (freq, T)

        # 2. Frequency weighting on the magnitude. _frequency_gain is a POWER
        #    gain, so we apply sqrt(gain) to the magnitude => magnitude^2 carries
        #    the intended power weighting (high-pass + optional A-weighting).
        gain = self._frequency_gain(sr)                    # power gain, (freq,)
        S_w = S * np.sqrt(gain)[:, None]

        # 3. Per-frame RMS from the weighted spectrogram. librosa.feature.rms(S=)
        #    is correctly normalized (time-domain-consistent, window-aware), so
        #    the dB scale below is properly calibrated -- unlike a raw bin sum.
        rms = librosa.feature.rms(S=S_w, frame_length=self.n_fft,
                                  hop_length=DAC_HOP_LENGTH)[0]   # (T,)

        # 4. ABSOLUTE dB (not relative to the clip max): 20*log10(rms + eps).
        #    Keeps dynamics comparable across clips (level roughly equalised by
        #    the preprocessing loudnorm); silence maps to the floor.
        energy_db = 20.0 * np.log10(rms + 1e-7)

        # 5. Smooth the dynamic envelope (remove onset spikes).
        energy_db = self._savgol(energy_db)

        # 6. Normalise [-top_db, 0] dB -> [0, 1] (silence -> 0, full-scale -> 1).
        #    The NULL condition (all zeros) therefore reads as silence, matching
        #    the zeros-mean-absence convention of chroma / rhythm.
        energy_norm = np.clip((energy_db + self.top_db) / self.top_db, 0.0, 1.0)

        # 7. Align to the DAC-aligned n_frames and shape (n_frames, 1).
        energy_norm = self._resample_to_frames(energy_norm, n_frames)
        return energy_norm.reshape(n_frames, 1).astype(np.float32)


# ============================================================
# FRAME-LEVEL: F0 (CREPE, monophonic pitch contour)
# ============================================================

class CrepeF0Extractor(FrameConditionExtractor):
    """
    Monophonic fundamental-frequency (f0) contour from CREPE, via the torch-native
    `torchcrepe` backend (NOT the TensorFlow `crepe` package -- the project runs
    with USE_TF=0). It is the project's ONLY pitch condition:
      * f0      -> a single continuous pitch curve, for monophonic / lead-line
                   control (bass line, solo voice, lead instrument in front).

    Pipeline (mirrors the other frame extractors' DAC alignment, hardened per the
    f0 review):
      1. resample to 16 kHz mono (torchcrepe's operating rate);
      2. torchcrepe.predict -> per-frame pitch (Hz) + periodicity (voicing conf.);
      3. voicing decision BEFORE any zeroing: median-filter periodicity, gate out
         silent frames (local RMS < silence_db), threshold, and drop voiced runs
         shorter than min_voiced_frames;
      4. normalize pitch on a LOG scale in [0,1] over ALL frames (no zeros yet, so
         interpolation never sees an injected 0 -> no phantom pitch ramps);
      5. resample SEPARATELY to the DAC-aligned n_frames -- pitch linearly, the
         voiced mask with nearest -- then re-apply the mask;
      6. map voiced pitch to [voiced_floor, 1], reserving 0 EXCLUSIVELY for
         "unvoiced/absent" (so 0 is never confused with pitch==fmin).

    Output: (n_frames, dim). dim=2 (default, with_periodicity=True) ->
    [pitch_norm, periodicity]; dim=1 -> [pitch_norm]. Both channels are 0 on
    unvoiced frames, matching the zeros-mean-absence convention of the other
    conditions, so the CFG NULL condition reads as fully unvoiced.

    Requires: pip install torchcrepe
    """

    def __init__(self,
                 fmin: float = 50.0,
                 fmax: float = 1000.0,
                 model: str = "full",              # "full" or "tiny"
                 voicing_threshold: float = 0.5,
                 with_periodicity: bool = True,     # 2 channels [pitch, periodicity]
                 hop_ms: float = 10.0,
                 silence_db: float = -60.0,         # frames quieter than this -> unvoiced
                 median_win: int = 3,               # median filter on periodicity (odd, 0=off)
                 min_voiced_frames: int = 3,        # drop voiced runs shorter than this
                 voiced_floor: float = 0.05,        # voiced pitch mapped to [floor, 1]
                 device: Optional[str] = "cpu",     # set "cuda" in CONFIG for speed
                 batch_size: int = 64):             # CREPE frames per inference batch
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
        wav = torch.from_numpy(y).unsqueeze(0)               # [1, t] @16k

        hop = max(1, int(round(self.hop_ms / 1000.0 * CR)))
        dev = self.device
        if dev in (None, "auto"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        pitch, periodicity = torchcrepe.predict(
            wav, CR, hop_length=hop, fmin=self.fmin, fmax=self.fmax,
            model=self.model, return_periodicity=True,
            batch_size=self.batch_size, device=dev,
        )
        pitch = pitch.squeeze(0).cpu().numpy().astype(np.float64)         # (F,) Hz
        periodicity = periodicity.squeeze(0).cpu().numpy().astype(np.float64)  # (F,)
        F = pitch.shape[0]
        if F < 2:                                            # degenerate -> unvoiced
            return np.zeros((n_frames, self.dim), dtype=np.float32)

        # --- voicing decision (periodicity + silence + cleanup), BEFORE any zeroing ---
        per_s = self._median_filter(periodicity, self.median_win)
        silent = self._silence_mask(y, hop, F, self.silence_db)
        voiced = (per_s >= self.voicing_threshold) & (~silent)
        voiced = self._remove_short_runs(voiced, self.min_voiced_frames)

        # --- pitch normalized on a LOG scale over ALL frames (no zeros injected) ---
        lo, hi = np.log2(self.fmin), np.log2(self.fmax)
        pf = np.clip(pitch, self.fmin, self.fmax)
        pnorm_all = ((np.log2(pf) - lo) / (hi - lo + 1e-8)).astype(np.float64)

        # --- SEPARATE resampling to n_frames (report #3 of the f0 review) ---
        # pitch: linear (smooth contour, interpolated from REAL pitch values, so no
        #        artificial ramps toward zero at voiced/unvoiced boundaries);
        # mask : nearest (crisp voiced/unvoiced boundaries);
        # then RE-APPLY the mask to the pitch. Zero is thus reserved for absence.
        pnorm_r = self._resample_to_frames(pnorm_all.reshape(-1, 1), n_frames)[:, 0]
        mask_r  = self._resample_nearest(voiced.astype(np.float64), n_frames) > 0.5
        per_r   = self._resample_to_frames(per_s.reshape(-1, 1), n_frames)[:, 0]

        # voiced pitch mapped to [voiced_floor, 1] so that 0 means ONLY "unvoiced"
        pitch_ch = np.where(
            mask_r, self.voiced_floor + (1.0 - self.voiced_floor) * pnorm_r, 0.0)

        if self.with_periodicity:
            per_ch = np.where(mask_r, per_r, 0.0)            # 0 at unvoiced too
            feat = np.stack([pitch_ch, per_ch], axis=-1)     # (n_frames, 2)
        else:
            feat = pitch_ch.reshape(-1, 1)                   # (n_frames, 1)
        return feat.astype(np.float32)

    # ---- f0 post-processing helpers (numpy; no unverifiable torchcrepe APIs) ----
    @staticmethod
    def _median_filter(x: np.ndarray, win: int) -> np.ndarray:
        if win and win >= 3:
            from scipy.signal import medfilt
            return medfilt(x, kernel_size=win if win % 2 == 1 else win + 1)
        return x

    @staticmethod
    def _silence_mask(y16k: np.ndarray, hop: int, F: int, silence_db: float) -> np.ndarray:
        """Per-CREPE-frame silence gate: True where the local RMS is below
        silence_db. Local energy is a box-smoothed y^2 sampled at frame centers."""
        win = max(hop, 1024)
        energy = np.convolve((y16k.astype(np.float64) ** 2),
                             np.ones(win) / win, mode="same")
        centers = np.clip(np.arange(F) * hop, 0, len(energy) - 1)
        rms = np.sqrt(energy[centers] + 1e-12)
        db = 20.0 * np.log10(rms + 1e-9)
        return db < silence_db

    @staticmethod
    def _remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
        """Set voiced runs shorter than min_len frames back to unvoiced."""
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


# ============================================================
# FRAME-LEVEL: CHORD (crema's chord model, PyTorch port)
# ============================================================

class CremaChordExtractor(FrameConditionExtractor):
    """
    Harmony as a chord recognizer hears it: per frame, the probability that
    each of the 12 pitch classes (C, C#, ..., B) belongs to the chord being
    played -- crema's `chord_pitch` output. Output (n_frames, 12) in [0, 1],
    the same shape as chroma.

    Versus chroma: chroma measures how much ENERGY each pitch class has, so a
    melody note, an overtone or a drum hit light it up as much as the chord
    does. chord_pitch is the answer of a network trained to name chords
    (McFee & Bello, ISMIR 2017): which notes make up the chord. A frame with
    no chord (crema's N) is all near 0, which is also what the null condition
    of CFG dropout means here.

    Backbone: crema_chord.py, a PyTorch port of crema that loads its original
    weights (crema_chord_weights.npz, next to it). Verified against the
    original (Keras 2) on real chunks, whole tracks and edge cases: features
    bit-identical, outputs within 3e-6. On a new machine:
        python crema_chord.py --selftest

    Pipeline:
      1. crema's HCQT front-end on the chunk (44.1 kHz, 10.77 frames/s);
      2. the network -> chord_pitch, (T_native, 12);
      3. aligned to the DAC frames ON THE TIME AXIS: crema frame k is centred
         on k*4096/44100 s, DAC frame j on j*512/44100 s (the convention the
         chroma condition follows), linear interpolation, first and last crema
         frame held at the ends. Not _resample_to_frames: that maps the first
         and last frame onto the chunk's ends, and crema's last frame of a 5 s
         chunk sits at 4.83 s, not 4.99 s -- an 8x slower rate makes that
         stretch visible;
      4. SILENCE GATE: DAC frames whose RMS is below `silence_db` (dBFS,
         window of 4 DAC hops) are set to 0 = no chord;
      5. clipped to [0, 1].

    WHY THE GATE: crema reads every excerpt relative to its own loudest bin,
    so it has no notion of absolute level -- measured: a C major triad gets
    the same answer at -20 and at -90 dBFS -- and it "hears" a chord in
    silence too: on the silent frames of real chunks (3.3% of the frames of
    the instruments set) it put up to 0.3-0.9 on arbitrary pitch classes.
    Zeros mean absence for every frame condition here (and are the null of
    CFG dropout), so silence has to be zeros. -60 dBFS is the f0 extractor's
    silence_db and the dataset's gate. The gate is on the extractor, not on
    the port: crema_chord.py reproduces crema as it is.

    CONTEXT, measured (3 tracks, 12.8k frames): crema reads whole tracks.
    Given 5 s chunks, as here, it names the same chord as on the whole track
    in 58% of the frames (cosine of the 12-d vectors 0.91). The cause is the
    network's context -- fed full-track features but seeing 5 s it agrees 62%
    of the time -- not the chunk's level normalization or its edges (about 2
    points each). The training target and the adherence re-extraction (from a
    5 s generation) both see 5 s, so they are consistent with each other.

    Cost: CPU, in the preprocessing workers like chroma. The librosa HCQT is
    ~80-150 ms per 5 s chunk (chroma_cqt ~100 ms), the network ~8 ms on one
    thread. Deliberately NO device attribute: preprocess_stream moves
    extractors that have one into the main process, next to the DAC, where
    this CPU-bound work would run serially.
    """

    _models: dict = {}   # weights path -> crema_chord.CremaChord, per process

    def __init__(self, silence_db: Optional[float] = -60.0,
                 weights: Optional[str] = None):
        from crema_chord import DEFAULT_WEIGHTS, file_sha1
        # None disables the gate (crema's raw answer, silence included).
        self.silence_db = None if silence_db is None else float(silence_db)
        self._weights = str(weights or DEFAULT_WEIGHTS)
        if not Path(self._weights).is_file():
            raise FileNotFoundError(
                f"CremaChordExtractor: weights not found at {self._weights}. "
                f"crema_chord_weights.npz ships next to crema_chord.py -- copy "
                f"it along with the code.")
        # Content identity of the weights: the probe cache fingerprint reads
        # public scalar attributes, and a path would differ between machines.
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
        pitch = self._get_model().outputs(y, sr)["chord_pitch"]   # (T, 12)
        if pitch.shape[0] == 0:          # shorter than one crema hop
            return np.zeros((n_frames, self.dim), dtype=np.float32)
        t_src = np.arange(pitch.shape[0]) / CREMA_FPS
        t_dst = np.arange(n_frames) / DAC_FRAMES_PER_S
        out = np.stack([np.interp(t_dst, t_src, pitch[:, c])
                        for c in range(pitch.shape[1])], axis=1)
        if self.silence_db is not None:
            import librosa
            hop = max(1, int(round(sr / DAC_FRAMES_PER_S)))   # one DAC frame
            rms = librosa.feature.rms(y=y, frame_length=4 * hop,
                                      hop_length=hop)[0]
            rms_db = 20.0 * np.log10(rms + 1e-12)
            silent = np.interp(t_dst, np.arange(rms.size) * hop / sr,
                               rms_db) < self.silence_db
            out[silent] = 0.0
        return np.clip(out, 0.0, 1.0).astype(np.float32)


# ============================================================
# GLOBAL: TEXT with CLAP (music-aware)
# ============================================================

class CLAPTextCondition(GlobalConditionExtractor):
    """
    Encodes text with the CLAP text encoder.

    CLAP is a dual-encoder model (audio + text) trained on audio-text pairs.
    The text encoder lives in the same space as the audio encoder, so the
    embedding of "baroque sacred music" is *close* to the audio embeddings of
    baroque sacred music. For conditioning a musical generative model this is
    more appropriate than a generic sentence-transformer.

    Implementation: uses ClapTextModelWithProjection (not ClapModel), which is
    the canonical API to extract the projected text embedding. It exposes an
    explicit .text_embeds field, robust to signature changes of
    ClapModel.get_text_features across transformers versions.

    Available models, with what they scored here (6 Sept 2026, measured on 120
    Museart clips, 40 per class, plus a sanity check on four unmistakable
    synthetic sounds -- sine / white noise / silence / siren -- each against its
    own description). "retrieval" is 3-way class accuracy from the audio
    (chance 33.3%), "margin" the mean gap between the right description's score
    and the best wrong one, "sep" the audio-audio cosine within a class minus
    between classes:

        - 'laion/clap-htsat-unfused'     sanity 4/4  83.3%  +0.192  +0.260  <- default
        - 'laion/larger_clap_general'    sanity 4/4  80.0%  +0.121  +0.221
        - 'laion/clap-htsat-fused'       sanity 4/4  75.0%  +0.079  +0.199
        - 'laion/larger_clap_music'      sanity 1/4  BROKEN -- DO NOT USE

    THE MUSIC-SPECIALISED CHECKPOINT IS BROKEN and was this file's default until
    the day those numbers were measured. Both its towers emit near-constant
    embeddings: its softmax over the four synthetic sounds is a flat 0.250 on
    every cell, and audio-text cosines sit at ~0.01 whatever the pair. It is the
    published checkpoint, not the code: the weights (projections included) load
    correctly, the feature extractor is configured as the model expects
    (enable_fusion=False, rand_trunc, repeatpad), and transformers 4.57.6 and
    5.16.1 behave identically. Nothing here had ever run it -- enabled_global
    was [] -- so no past result is affected, but it would have made both the
    text condition and its influence metric silently meaningless.

    A degenerate checkpoint fails SILENTLY: the embeddings look healthy (finite,
    unit norm, right dtype and shape) and only their spread gives it away. Run
    the synthetic sanity check before trusting any CLAP number from a checkpoint
    that has not been measured here.
    """

    def __init__(self, model_name: str = "laion/clap-htsat-unfused"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._dim = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # The audio tower, built only if encode_audio is ever called (see below).
        self._audio_embedder = None

    def _load(self):
        if self._model is not None:
            return
        # ClapTextModelWithProjection is the canonical API to extract the
        # projected text embedding in the shared audio-text space. It returns an
        # object with a .text_embeds field, independent of the transformers
        # version (ClapModel.get_text_features changed signature across recent
        # versions).
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
        # From the CONFIG first: asking for the width must not put a second
        # checkpoint on the GPU (see _projection_dim_from_config).
        if self._dim is None:
            self._dim = _projection_dim_from_config(self.model_name)
        if self._dim is None:
            self._load()
        return self._dim

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single string -> (dim,) L2-normalized embedding."""
        self._load()
        inputs = self._processor([text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        out = self._model(**inputs)
        feat = out.text_embeds   # (1, dim) — gia' proiettato
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings -> (N, dim) all L2-normalized."""
        self._load()
        inputs = self._processor(list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        out = self._model(**inputs)
        feat = out.text_embeds
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    # ---- AUDIO SIDE: what the preprocessing actually stores ---------------
    #
    # The condition is called "text" because that is the SLOT, and text is what
    # it receives at inference. At TRAINING time the stored value is the CLAP
    # embedding of the chunk's own AUDIO (the AudioLDM arrangement): CLAP's two
    # towers share one space, so a vector from the audio encoder and one from
    # the text encoder are interchangeable in that slot, and a model trained on
    # the first accepts the second.
    #
    # WHY NOT THE CLASS NAME, which is what the preprocessing used to store:
    # one embedding per class makes the text condition carry exactly one bit --
    # the category -- and the image condition, drawn from that same class,
    # carries the same one. Two conditions saying the identical thing cannot be
    # told apart in an ablation. The per-chunk audio embedding makes text a
    # signal that varies WITHIN a class and leaves image as the categorical one.
    #
    # TWO CAVEATS, both accepted deliberately when this was chosen:
    #   * at training the condition is computed from the very audio the model
    #     has to produce, so it contains the answer and the text-influence
    #     metric is flattering by construction. Read it knowing that.
    #   * the two towers share a space but do not sit exactly on top of each
    #     other, so a text vector at inference is not drawn from quite the same
    #     cloud as the audio vectors seen in training.
    #
    # The audio tower is a SECOND checkpoint load, so it is built lazily and
    # only in the process that extracts: a run that merely reads back stored
    # embeddings never pays for it.

    def _audio_side(self):
        if self._audio_embedder is None:
            self._audio_embedder = ClapAudioEmbedder(model_name=self.model_name,
                                                     device=self._device)
        return self._audio_embedder

    @torch.no_grad()
    def encode_audio(self, wav_np: np.ndarray, sr: int) -> np.ndarray:
        """(T,) waveform -> (dim,) L2-normalized embedding, in the SAME space as
        encode_text.

        The PRESENCE of this method is what marks a global condition as
        computable per chunk (preprocess_stream._global_is_chunk_level), so it
        rides the same extraction path as f0 instead of the per-class one. A
        global condition without it is a per-class one, and the two need no
        registry of names to tell apart."""
        return self._audio_side().embed(wav_np, sr)

    def unload(self):
        """Free GPU memory after pre-computing the embeddings."""
        if self._audio_embedder is not None:
            self._audio_embedder.unload()
            self._audio_embedder = None
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()


# ============================================================
# CLAP AUDIO EMBEDDER (audio side of CLAP, for text-condition INFLUENCE)
# ============================================================

class ClapAudioEmbedder:
    """
    Audio encoder of the SAME CLAP checkpoint used by CLAPTextCondition.

    Used at validation only, to measure how much the text condition influenced
    the generation: CLAP's audio and text encoders share one space, so the
    cosine between the audio embedding of a generation and the CLAP-text
    embedding that conditioned it (already stored in the dataset, L2-normalized)
    is a direct text-adherence score. The influence is the delta of this score
    between the with-text and the null-text generations.

    Lazily loaded; only instantiated when 'text' is an active global condition.
    """

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
        # The processor's audio keyword was renamed `audios` -> `audio`:
        # transformers 5 REJECTS the old name, older 4.x releases do not know
        # the new one. Decided ONCE from the signature rather than by calling
        # and catching: the wrong-keyword error is a ValueError, and so is
        # "your audio is at the wrong sampling rate", so a try/except around
        # the call cannot tell the two apart and would report a real data
        # problem as a version problem.
        try:
            params = inspect.signature(self._processor.__call__).parameters
            self._audio_kw = "audio" if "audio" in params else "audios"
        except (TypeError, ValueError):
            self._audio_kw = "audio"
        print(f"[ClapAudioEmbedder] '{self.model_name}' audio encoder "
              f"loaded on {self._device} (dim={self._dim})")

    @torch.no_grad()
    def embed(self, wav_np: np.ndarray, sr: int) -> np.ndarray:
        """(T,) waveform at `sr` -> (dim,) L2-normalized audio embedding in CLAP
        space.

        RESAMPLES to CLAP's own rate first. The feature extractor does NOT do it
        for you: handed 44100 Hz audio it raises ValueError and refuses, which
        is exactly what this project would feed it -- every chunk here is at the
        DAC's 44.1 kHz while CLAP wants 48 kHz. (This docstring used to claim
        the processor resampled internally. It does not, and nothing had ever
        called this method with real audio to find out.)"""
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
        feat = out.audio_embeds                      # (1, dim)
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def unload(self):
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()


# ============================================================
# TEXT LABEL VOCABULARY (reading a stored CLAP vector back as words)
# ============================================================
#
# WHY THIS EXISTS. The 'text' condition of a validation sample is the CLAP
# embedding of its own audio, and CLAP is a one-way street: there is no decoder,
# so those numbers cannot be turned back into the sentence that would describe
# them -- no sentence ever existed. A panel showing "validation sample #37"
# conditioned on 512 anonymous numbers tells the reader nothing about WHAT it
# was conditioned on.
#
# So the label is a RETRIEVAL, not a translation: of the phrases below, which
# sit closest to that sample's vector? It is honest only if read that way --
# "the nearest phrase I know" -- which is why the cosine is always shown beside
# it. A low cosine means the vocabulary has nothing close, not that the audio
# resembles the phrase shown.
#
# It costs nothing to recompute: the per-chunk vector is already on disk and the
# phrase embeddings are cached in the dataset, so the label is one dot product.
# That is deliberate -- edit this list, re-encode the phrases, and every label
# changes without re-preprocessing a single chunk of audio.
TEXT_LABEL_VOCAB = [
    # instrumentation
    "solo pipe organ", "a cappella choir", "harpsichord", "solo piano",
    "string quartet", "solo violin", "solo cello", "acoustic guitar",
    "electric guitar", "distorted electric guitar", "electric bass",
    "drum kit", "hand percussion", "brass section", "solo trumpet",
    "woodwinds", "synthesizer", "analog synthesizer bass", "orchestra",
    # texture and register
    "a single sustained note", "a dense polyphonic texture",
    "a solo melodic line", "a low bass register", "a high bright register",
    "sparse and quiet", "loud and dense",
    # rhythm and motion
    "a steady four on the floor beat", "a fast rhythmic pattern",
    "a slow tempo", "no clear pulse", "a strong groove",
    # space and production
    "a large reverberant church", "a dry close recording",
    "a live concert recording", "a lo-fi noisy recording",
    # style
    "baroque sacred music", "gregorian chant", "classical music",
    "romantic orchestral music", "rock music", "heavy metal",
    "electronic dance music", "ambient music", "experimental noise",
    "jazz", "folk music", "film score",
    # extended technique / non-musical
    "a plucked pizzicato attack", "a bowed tremolo", "breathy air noise",
    "silence", "white noise",
]


def nearest_phrases(vec, vocab_emb, phrases, k: int = 2):
    """(dim,) stored vector + (n_phrases, dim) vocabulary -> [(phrase, cos), ...]

    Both sides are L2-normalized, so the dot product IS the cosine. Returns the
    k nearest, best first. Kept here rather than in the metrics module because
    it is about what the text condition MEANS, not about scoring a model."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    M = np.asarray(vocab_emb, dtype=np.float32)
    if v.size == 0 or M.size == 0 or M.shape[1] != v.shape[0]:
        return []
    sims = M @ v
    order = np.argsort(-sims)[:max(1, int(k))]
    return [(phrases[i], float(sims[i])) for i in order if i < len(phrases)]


# ============================================================
# WAV2CLIP AUDIO EMBEDDER (audio in CLIP's space, for IMAGE influence)
# ============================================================

class Wav2ClipAudioEmbedder:
    """
    Audio encoder distilled INTO CLIP's own embedding space (Wav2CLIP, Wu et
    al. 2022). It is what makes an audio-vs-image number exist at all.

    WHY IT IS NEEDED. CLIP embeds images and text in one space; CLAP embeds
    audio and text in a DIFFERENT one. Between a CLIP image vector and a CLAP
    audio vector there is no meaningful cosine -- they are coordinates in
    unrelated spaces. Wav2CLIP is trained to place audio where CLIP would place
    the matching image, so its output can be compared directly with the CLIP
    image embeddings this project already stores.

    HOW STRONG IS THE SIGNAL, measured here on 6 September 2026 over 36 Museart
    clips and 36 of its images (12 + 12 per class): same-class audio-image
    cosine +0.0748, cross-class +0.0629, i.e. a separation of only +0.0119, and
    audio->class retrieval 41.7% against a 33.3% chance level. That is real but
    SMALL, and the reason is a double domain mismatch: Wav2CLIP is distilled on
    VGGSound (video frames of everyday sound events) while this corpus pairs
    music with album covers and paintings.
    READ THE INFLUENCE ROW ACCORDINGLY: it is a PAIRED delta -- the same image
    scored against the conditioned and the null generation -- which is far more
    sensitive than the cross-class retrieval above, so a consistent positive
    delta still means something. An absolute cosine near 0.07 does not.

    Kept deliberately separate from ImageCondition: that class encodes the
    IMAGES (and is what the dataset's banks were built with), this one encodes
    AUDIO into the same space, and only validation ever needs it.
    """

    SR = 16000          # Wav2CLIP's own rate; anything else must be resampled

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
        """(T,) waveform at `sr` -> (512,) L2-normalized vector in CLIP space,
        directly comparable with an ImageCondition embedding."""
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


# ============================================================
# GLOBAL: IMAGE (CLIP)
# ============================================================

class ImageCondition(GlobalConditionExtractor):
    """
    Encodes an image with CLIP.

    Implementation: uses CLIPVisionModelWithProjection (not CLIPModel), for the
    same reason CLAPTextCondition uses ClapTextModelWithProjection -- it is the
    canonical API to extract the projected embedding, it exposes an explicit
    .image_embeds field, and it is robust to the signature changes that
    CLIPModel.get_image_features has gone through across transformers versions.
    Under transformers 5.x that method no longer returns a tensor at all but a
    BaseModelOutputWithPooling, which broke the previous implementation outright.
    The two produce bit-identical vectors (verified: max abs diff 0.0), and this
    one loads only the vision tower instead of the full dual-encoder.
    """

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
        # Same as CLAPTextCondition.dim: the config knows the width, so reading
        # it never builds the vision tower.
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
        feat = out.image_embeds          # (1, dim) -- already projected
        feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def unload(self):
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None


# ============================================================
# IMAGE DATASET MANAGER
# ============================================================

class ImageDatasetManager:
    """
    Manages an image dataset with a structure parallel to the audio.

    Supported layout (with split):
        image_root/{train,val,test}/<class_name>/*.jpg
    Legacy layout (without split):
        image_root/<class_name>/*.jpg

    If you pass `split`, the split layout is used. If the split folder
    does not exist, it automatically falls back to the legacy layout with a warning.
    """

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


# ============================================================
# CONDITION CONFIG -- UNICO PUNTO DI VERITA
# ============================================================
#
# To enable/disable a condition: change "enabled".
# To add a new condition:
#   1. Create a class extending FrameConditionExtractor or GlobalConditionExtractor
#   2. Aggiungila in CONDITION_CONFIG con "enabled": True
#   3. Done -- everything else adapts automatically
#
# LABEL REMOVAL: the `label` (categorical) condition has been removed.
# The `text` modality with CLAP replaces it: at training it receives the name
# the class name as a string at training; at inference it can take free prompts.
# ============================================================

CONDITION_CONFIG = {
    "frame_level": {
        # out_dim = per-condition projection width (JASCO bottleneck). The
        # extractor produces `raw_dim` channels per frame (ChromaExtractor ->
        # 12 chroma classes via librosa CQT); a single Linear(raw_dim ->
        # out_dim) projects them before they are concatenated with the latent
        # on the feature dim.
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
            # Frequency-weighted spectral-energy dynamics curve (1 channel).
            # raw_dim=1 -> projected to out_dim by the FrameConditionEncoder.
            # weighting="A" + fmin high-pass = perceptual loudness on the
            # audible band. out_dim is small: it is a low-bandwidth 1-D signal.
            "class": EnergyExtractor,
            "kwargs": {"weighting": "A", "fmin": 40.0},
            "out_dim": 16,
            "enabled": True,
        },
        "f0": {
            # Monophonic f0 contour from CREPE (torchcrepe backend): a single
            # continuous pitch curve for monophonic / lead-line control, and the
            # project's only pitch condition. raw_dim=2 by default
            # ([pitch_norm, periodicity]); set with_periodicity=False for raw_dim=1.
            # Set device="cuda" here for speed on large datasets (only honored
            # with --num_workers 0). Requires: pip install torchcrepe
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
            # crema's chord model (PyTorch port, crema_chord.py): per frame,
            # the probability that each pitch class belongs to the chord.
            # raw_dim=12 like chroma, and chroma's projection width. It has no
            # device knob, so it runs on CPU in the preprocessing workers, like
            # chroma. Weights: crema_chord_weights.npz, next to crema_chord.py.
            # LAST in this dict on purpose: the frame conditions are
            # concatenated in this order, so appending keeps every existing
            # run's layout. silence_db: frames quieter than this are "no
            # chord" (crema itself hears chords in silence; None = off).
            "class": CremaChordExtractor,
            "kwargs": {"silence_db": -60.0},
            "out_dim": 64,
            "enabled": True,
        },
        # Example for adding MFCC in the future:
        # "mfcc": {
        #     "class": MFCCExtractor,
        #     "kwargs": {"n_mfcc": 20},
        #     "out_dim": 64,
        #     "enabled": False,
        # },
    },
    "global": {
        "text": {
            "class": CLAPTextCondition,
            # NOT 'laion/larger_clap_music': that checkpoint is broken (flat
            # 0.250 softmax over four unmistakable sounds). See the table in
            # CLAPTextCondition's docstring for the measurements behind this
            # choice. ClapAudioEmbedder reads this same value, so the influence
            # metric always scores in the space the condition was built in.
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


# ============================================================
# CONDITION REGISTRY -- LETTORE DEL CONFIG
# ============================================================

class ConditionRegistry:
    """
    Instantiates the extractors that are both:
      1) marked enabled=True in CONDITION_CONFIG (the project-wide pool
         of conditions that the pipeline knows how to handle), AND
      2) selected by the per-run filters `enabled_frame` / `enabled_global`,
         typically driven by the YAML config of the training run.

    Used by extract_conditions.py, audio_dataset_cond.py, training_cond.py,
    test_cond.py.

    Args:
        n_classes:       kept for back-compat with extract_conditions.py
                         (LabelCondition has been removed).
        config:          alternative dict in place of CONDITION_CONFIG.
        enabled_frame:   per-run filter for frame-level conditions:
                           - None  -> use everything enabled=True in CONDITION_CONFIG
                           - []    -> use NO frame-level condition
                           - list  -> use only the listed names (each must
                                      be enabled=True in CONDITION_CONFIG)
        enabled_global:  same semantics for global conditions.
    """

    def __init__(self, n_classes: Optional[int] = None,
                 config: Optional[dict] = None,
                 enabled_frame:  Optional[List[str]] = None,
                 enabled_global: Optional[List[str]] = None):
        self.frame_extractors: Dict[str, FrameConditionExtractor] = {}
        self.frame_out_dims: Dict[str, int] = {}   # per-condition projection width (JASCO)
        self.global_extractors: Dict[str, GlobalConditionExtractor] = {}
        self.n_classes = n_classes  # ignored, kept for back-compat

        config = config or CONDITION_CONFIG
        self._build(config, enabled_frame, enabled_global)

    def _build(self, config,
               enabled_frame:  Optional[List[str]] = None,
               enabled_global: Optional[List[str]] = None):
        # ---- Frame-level ----
        for name, cfg in config.get("frame_level", {}).items():
            if not cfg.get("enabled", False):
                continue
            if enabled_frame is not None and name not in enabled_frame:
                continue
            cls = cfg["class"]
            kwargs = cfg.get("kwargs", {})
            self.frame_extractors[name] = cls(**kwargs)
            # Per-condition projection width for the JASCO-style concat.
            # Defaults to the raw extractor dim when out_dim is not declared
            # (i.e. an identity-width projection), so older configs still work.
            self.frame_out_dims[name] = int(
                cfg.get("out_dim", self.frame_extractors[name].dim)
            )

        # Sanity check: explicit list must reference conditions that
        # are enabled=True in CONDITION_CONFIG (catches typos early).
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

        # ---- Global (all continuous now) ----
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
        """Per-condition projection width used for the JASCO-style concat.
        Same keys (and order) as frame_cond_dims."""
        return {n: self.frame_out_dims[n] for n in self.frame_extractors.keys()}

    @property
    def global_cond_configs(self) -> Dict[str, dict]:
        """All global conditions are now continuous -> only `dim`."""
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


# ============================================================
# ENCODERS nn.Module (used by the DiT)
# ============================================================

class FrameConditionEncoder(nn.Module):
    """
    JASCO-style frame-condition encoder.

    Each frame condition is projected by a SINGLE Linear (raw_dim -> out_dim),
    exactly as JASCO's MelodyConditioner (output_proj = nn.Linear(card, out_dim),
    audiocraft/modules/jasco_conditioners.py). The projected conditions are
    returned CONCATENATED on the feature dim, in a fixed canonical order. The
    network then concatenates this with the noisy latent on the feature dim and
    applies a single input projection to hidden_size — see
    audiocraft/models/flow_matching.py, forward():
        for cond in temporal_conds: x = torch.concat((x, c), dim=-1)
        input_ = self.emb(x)

    NB: this does NOT project to hidden_size and does NOT sum the conditions.
    The fusion to hidden_size is the network's single input_proj, applied AFTER
    concatenation with the latent.
    """

    def __init__(self, condition_dims: Dict[str, int], out_dims: Dict[str, int]):
        super().__init__()
        # Canonical fixed order = insertion order of condition_dims (driven by
        # CONDITION_CONFIG / the registry). The concat slots are positional, so
        # this order MUST be identical at train and inference time.
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
        """
        conditions: {name: (B, T, raw_dim)} — every expected name must be
                    present (zeros for null conditions). The caller
                    (ConditionedAudioDiT) guarantees this.
        Returns:
            (B, T, total_out_dim) — projected conditions concatenated in
            canonical order.
        """
        parts = [self.projections[name](conditions[name]) for name in self.names]
        return torch.cat(parts, dim=-1)


class GlobalConditionEncoder(nn.Module):
    """
    Projects global conditions (continuous) -> hidden_size.
    Sums the projections and applies a final LayerNorm to balance the scales
    across different modalities (e.g. text CLAP vs image CLIP).

    NB: no more categorical branch (LabelCondition removed).
    """

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
        """
        conditions: {name: (B, dim)} — every expected name must be present
                    (zeros for null conditions), exactly like
                    FrameConditionEncoder. The caller (ConditionedAudioDiT)
                    guarantees this via _gather_global_conditions.

        The contract matters: a name simply MISSING from the dict used to be
        left out of the sum, which is a third state the model never saw. The
        null it is trained against is a ZERO VECTOR through the projection --
        not the absence of that projection's bias and LayerNorm contribution --
        so "give me only a prompt, no image" has to send zeros for the image,
        not nothing.
        """
        embs = [enc(conditions[name]) for name, enc in self.encoders.items()
                if name in conditions]
        if not embs:
            return None
        return self.final_norm(torch.stack(embs, dim=0).sum(dim=0))


# ============================================================
# NULL CONDITIONS (for CFG)
# ============================================================

def make_null_frame_conditions(B: int, n_frames: int,
                                 cond_dims: Dict[str, int],
                                 device) -> Dict[str, torch.Tensor]:
    return {n: torch.zeros(B, n_frames, d, device=device)
            for n, d in cond_dims.items()}


def make_null_global_conditions(B: int,
                                  global_configs: Dict[str, dict],
                                  device) -> Dict[str, torch.Tensor]:
    """
    Create "null" global conditions for CFG: zero vectors.

    text (CLAP) and image (CLIP) are both L2-normalized in the projected
    space, so a zero vector is OOD with respect to any
    real condition and acts as a pseudo-null token. This is the standard
    choice in generative models with continuous embeddings.
    """
    return {n: torch.zeros(B, cfg["dim"], device=device)
            for n, cfg in global_configs.items()}


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Test ConditionRegistry (CLAP-based, no label)")
    print("=" * 60)

    reg = ConditionRegistry()
    print(reg)
    print(f"\nFrame cond dims:    {reg.frame_cond_dims}")
    print(f"Global cond configs: {reg.global_cond_configs}")

    # Test text encoding (requires internet the first time)
    print("\n--- Test CLAP text encoding (single prompt) ---")
    if "text" in reg.global_extractors:
        t = reg.global_extractors["text"]
        emb = t.encode_text("baroque sacred music")
        print(f"  Embedding shape: {emb.shape}, "
              f"norm: {np.linalg.norm(emb):.4f} (expected ~1.0)")
        t.unload()
        print("  CLAP offloaded from GPU.")
