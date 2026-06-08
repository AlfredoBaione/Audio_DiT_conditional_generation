# condition_metrics.py
#
# Conditioning-fidelity metrics for the conditioned Audio DiT.
#
# Idea (paired, NOT distributional):
#   FD-DAC / FAD compare two *distributions* (real vs generated) and need
#   aggregate statistics (mean / covariance). Conditioning fidelity is a
#   *paired* comparison instead: each conditioned generation was produced from
#   ONE specific input condition (a validation melody / chroma / rhythm), so we
#   re-extract that same condition FROM the generated audio and compare it
#   one-to-one with the input condition, then average over the (fixed) metrics
#   subset.
#
# Re-extraction reuses the SAME extractor classes that produced the stored
# targets (conditions.py), so the generated-side representation is identical in
# nature to the target-side one and the comparison is fair.
#
# Per-condition distance (literature-grounded):
#   - melody : Raw Pitch Accuracy + Raw Chroma Accuracy (mir_eval.melody),
#              the standard melody-transcription/adherence metrics. Octave
#              errors lower RPA but not RCA. Computed over voiced target frames.
#   - chroma : mean per-frame cosine similarity between target and generated
#              chromagram (the melody/harmony adherence metric of MusicGen-Melody,
#              Copet et al. 2023, arXiv:2306.05284).
#   - rhythm : Pearson correlation of the beat and downbeat probability curves
#              (robust, hyper-parameter free). A stricter beat F-measure
#              (peak-picking + mir_eval.beat, in the spirit of Music ControlNet,
#              Wu et al. 2023, arXiv:2311.07069) can be added once rhythm
#              conditioning is enabled and validated.
#
# This module is evaluation-only: it never touches the trained weights.

from collections import defaultdict

import numpy as np

from conditions import (
    MelodyExtractor,
    ChromaExtractor,
    RhythmExtractor,
    DAC_FRAMES_PER_S,
)


# ============================================================
# MELODY  -- Raw Pitch / Raw Chroma Accuracy via mir_eval
# ============================================================
# basic-pitch note posteriorgram: 88 bins, bin 0 = MIDI 21 (A0). So
# bin i -> MIDI (21 + i) -> Hz. Unvoiced frames (no active bin) -> 0 Hz, which
# mir_eval treats as "unvoiced".
_MIDI_MIN = 21


def _melody_onehot_to_freqs(onehot: np.ndarray) -> np.ndarray:
    """(T, 88) one-hot -> (T,) frequencies in Hz (0.0 where unvoiced)."""
    active = onehot.sum(axis=1) > 0
    bins = np.argmax(onehot, axis=1)
    midi = _MIDI_MIN + bins
    freqs = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    return np.where(active, freqs, 0.0).astype(np.float64)


def melody_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: float = DAC_FRAMES_PER_S) -> dict:
    """
    Raw Pitch Accuracy (RPA) and Raw Chroma Accuracy (RCA) between the target
    melody and the melody re-extracted from the generated audio.
    Returns NaNs if the target has no voiced frames (sample skipped upstream).
    """
    import mir_eval

    T = target.shape[0]
    times = (np.arange(T) / float(fps)).astype(np.float64)
    ref_freq = _melody_onehot_to_freqs(target)
    est_freq = _melody_onehot_to_freqs(generated)

    if not np.any(ref_freq > 0):
        return {"rpa": float("nan"), "rca": float("nan")}

    # freq_to_voicing returns (abs_frequencies, voicing); the accuracy fns want
    # pitches in CENTS (via hz2cents), not Hz.
    ref_abs, ref_voicing = mir_eval.melody.freq_to_voicing(ref_freq)
    est_abs, est_voicing = mir_eval.melody.freq_to_voicing(est_freq)
    ref_cent = mir_eval.melody.hz2cents(ref_abs)
    est_cent = mir_eval.melody.hz2cents(est_abs)

    rpa = mir_eval.melody.raw_pitch_accuracy(
        ref_voicing, ref_cent, est_voicing, est_cent)
    rca = mir_eval.melody.raw_chroma_accuracy(
        ref_voicing, ref_cent, est_voicing, est_cent)
    return {"rpa": float(rpa), "rca": float(rca)}


# ============================================================
# CHROMA  -- mean per-frame cosine similarity
# ============================================================
def chroma_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: float = DAC_FRAMES_PER_S) -> dict:
    """Mean cosine similarity between target and generated chroma, per frame."""
    num = (target * generated).sum(axis=1)
    den = (np.linalg.norm(target, axis=1) * np.linalg.norm(generated, axis=1))
    valid = den > 1e-8
    if not np.any(valid):
        return {"cosine": float("nan")}
    cos = num[valid] / den[valid]
    return {"cosine": float(cos.mean())}


# ============================================================
# RHYTHM  -- Pearson correlation of beat / downbeat curves
# ============================================================
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-8 else float("nan")


def rhythm_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: float = DAC_FRAMES_PER_S) -> dict:
    """Pearson correlation of the beat (ch 0) and downbeat (ch 1) curves."""
    return {
        "beat_corr":     _pearson(target[:, 0], generated[:, 0]),
        "downbeat_corr": _pearson(target[:, 1], generated[:, 1]),
    }


FIDELITY_FNS = {
    "melody": melody_fidelity,
    "chroma": chroma_fidelity,
    "rhythm": rhythm_fidelity,
}


# ============================================================
# SONIFICATION  -- render a one-hot melody to audible sine tones
# ============================================================
def sonify_melody(onehot: np.ndarray, sr: int,
                  fps: float = DAC_FRAMES_PER_S, amp: float = 0.2) -> np.ndarray:
    """
    Render an (T, 88) one-hot melody to a mono waveform of sine tones, so the
    conditioning/generated melody can be *listened to* on TensorBoard. The
    active pitch of each frame is played for that frame's duration (bin i ->
    MIDI 21+i -> Hz); silent frames are silence. Phase is carried across frames
    of the same note to avoid clicks. Returns float32 in [-amp, amp].
    """
    onehot = np.asarray(onehot)
    T = onehot.shape[0]
    spf = max(1, int(round(sr / float(fps))))
    out = np.zeros(T * spf, dtype=np.float32)
    freqs = _melody_onehot_to_freqs(onehot)   # (T,), 0.0 where unvoiced
    t_local = np.arange(spf) / float(sr)
    phase = 0.0
    for i in range(T):
        f = float(freqs[i])
        seg = slice(i * spf, (i + 1) * spf)
        if f > 0.0:
            ph = phase + 2.0 * np.pi * f * t_local
            out[seg] = (amp * np.sin(ph)).astype(np.float32)
            phase = float(ph[-1] + 2.0 * np.pi * f / sr)
        else:
            phase = 0.0
    return out


_EXTRACTOR_FNS = {
    "melody": MelodyExtractor,
    "chroma": ChromaExtractor,
    "rhythm": RhythmExtractor,
}


# ============================================================
# EVALUATOR
# ============================================================
class ConditionFidelityEvaluator:
    """
    Accumulates per-condition fidelity over the (fixed) metrics subset.

    Usage per metrics step:
        evaluator.reset()
        for gen_wav_np, target_cond in zip(cond_wavs, cond_targets):
            evaluator.add_sample(gen_wav_np, sr, n_frames, target_cond)
        results = evaluator.results()   # {"melody/rpa": ..., "melody/rca": ...}

    `target_cond` is the dict of input conditions for that sample, as numpy
    arrays of shape (n_frames, dim) keyed by condition name (e.g. "melody").
    Only the conditions in `enabled_frame` are evaluated.
    """

    def __init__(self, enabled_frame, device: str = "cpu",
                 fps: float = DAC_FRAMES_PER_S):
        self.fps = fps
        self.device = device
        self.extractors = {}
        for name in enabled_frame:
            if name not in _EXTRACTOR_FNS:
                continue
            # MelodyExtractor / ChromaExtractor take no constructor args here
            # (defaults match extraction time); RhythmExtractor needs a device.
            if name == "rhythm":
                self.extractors[name] = _EXTRACTOR_FNS[name](device=device)
            else:
                self.extractors[name] = _EXTRACTOR_FNS[name]()
        self.reset()

    def reset(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)

    @property
    def active(self) -> bool:
        return len(self.extractors) > 0

    def add_sample(self, gen_wav_np: np.ndarray, sr: int, n_frames: int,
                   target_cond: dict):
        """Re-extract every enabled condition from one generated waveform and
        accumulate its fidelity against the paired target condition."""
        if gen_wav_np.ndim > 1:
            gen_wav_np = gen_wav_np.squeeze()
        gen_wav_np = np.ascontiguousarray(gen_wav_np, dtype=np.float32)

        for name, extractor in self.extractors.items():
            if name not in target_cond:
                continue
            target = np.asarray(target_cond[name], dtype=np.float32)
            try:
                generated = extractor.extract(gen_wav_np, sr, n_frames)
            except Exception as e:
                print(f"  [fidelity] re-extraction failed for '{name}': {e}")
                continue
            metrics = FIDELITY_FNS[name](target, generated, self.fps)
            for k, v in metrics.items():
                if v == v:  # not NaN
                    self._sums[f"{name}/{k}"] += v
                    self._counts[f"{name}/{k}"] += 1

    def results(self) -> dict:
        """Average over the samples that contributed a finite value."""
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }
