"""
probe_conditions.py -- out-of-the-box probe sets for EVERY frame condition
(f0, energy, chroma, rhythm).

One module, one shape: a bank of elementary stimuli per condition, one
synthesizer per condition, one builder, one plotter. Adding a fifth condition
means adding a bank, a synthesizer and (if its shape needs one) a branch in the
plotter -- nothing else. The f0 material lived in probe_f0.py first and was
moved here unchanged when the four were unified.

What makes a probe different from the validation rows:

    The validation rows score the model against REAL recordings. That is the
    honest test, but a hard one to read: a real energy envelope is jittery, a
    real chromagram is smeared across neighbouring pitch classes, a real beat
    grid may not exist at all on this material. A middling score there does not
    separate "the conditioning is weak" from "the target was ambiguous".

    A probe removes the ambiguity. A rising scale, a linear crescendo, a
    sustained C major triad, a 120 bpm click grid: if the model does not follow
    THOSE, the conditioning does not work. It is a proof of concept, NOT a substitute --
    the stimuli are synthetic and far simpler than the training material, so a
    good probe score is necessary but not sufficient.

Every probe target is extracted with THE RUN'S OWN extractor (the registry
instance, same configuration that produced the training targets), so the target
lives in the same space as what the model was trained on. Nothing here
re-implements an extractor.

Cache contract, identical to probe_f0: built once per configuration under
`probe_dir`, keyed by a fingerprint of the stimuli, the synthesis source, the
chunk geometry and the extractor's parameters. Change any of them and the set
rebuilds itself instead of being silently reused stale.

Build one standalone to look at it before training:
    python probe_conditions.py energy ./cache/probe_energy --n_frames 431
    python probe_conditions.py f0     ./cache/f0_probe     --n_frames 431
"""

import os
import json
import hashlib
import inspect
from io import BytesIO

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conditions import DAC_SAMPLE_RATE, DAC_FRAMES_PER_S, CONDITION_CONFIG
from condition_metrics import f0_norm_to_hz


# ============================================================
# THE STIMULI
# ============================================================
# Ordered from "trivially readable" to "slightly less so": only the first
# n_plot are drawn, so the plotted ones must be the ones whose shape is
# unmistakable. The rest still count in the influence row.
#
# ---- f0 -----------------------------------------------------
# Each melody is a list of (midi_note | None, duration_in_beats); None is a
# rest. Moved here verbatim from probe_f0.py when the four banks were
# unified -- the melodies themselves are unchanged.
PROBE_MELODIES = [
    ("scale_up",        [(60, 1), (62, 1), (64, 1), (65, 1),
                         (67, 1), (69, 1), (71, 1), (72, 1)]),
    ("scale_down",      [(72, 1), (71, 1), (69, 1), (67, 1),
                         (65, 1), (64, 1), (62, 1), (60, 1)]),
    ("arpeggio_major",  [(60, 1), (64, 1), (67, 1), (72, 1),
                         (67, 1), (64, 1), (60, 1)]),
    ("octave_leap",     [(60, 2), (72, 2), (60, 2), (72, 2)]),
    ("sustained_a4",    [(69, 8)]),
    ("thirds_alt",      [(60, 1), (64, 1), (60, 1), (64, 1),
                         (60, 1), (64, 1), (60, 1), (64, 1)]),
    ("fifth_leap",      [(60, 1), (67, 1), (60, 1), (67, 1),
                         (60, 1), (67, 1)]),
    ("frere_jacques",   [(60, 1), (62, 1), (64, 1), (60, 1),
                         (60, 1), (62, 1), (64, 1), (60, 1)]),
    ("twinkle",         [(60, 1), (60, 1), (67, 1), (67, 1),
                         (69, 1), (69, 1), (67, 2)]),
    ("pentatonic_up",   [(60, 1), (62, 1), (64, 1), (67, 1),
                         (69, 1), (72, 1)]),
    ("chromatic_up",    [(60, 1), (61, 1), (62, 1), (63, 1),
                         (64, 1), (65, 1), (66, 1), (67, 1)]),
    ("wholetone_up",    [(60, 1), (62, 1), (64, 1), (66, 1),
                         (68, 1), (70, 1)]),
    ("staccato_rests",  [(60, 1), (None, 1), (64, 1), (None, 1),
                         (67, 1), (None, 1), (72, 1), (None, 1)]),
    ("arpeggio_minor",  [(57, 1), (60, 1), (64, 1), (69, 1),
                         (64, 1), (60, 1), (57, 1)]),
    ("two_phrases",     [(60, 1), (62, 1), (64, 1), (None, 2),
                         (67, 1), (65, 1), (64, 1)]),
    ("repeated_note",   [(62, 1), (None, 0.4), (62, 1), (None, 0.4),
                         (62, 1), (None, 0.4), (62, 1)]),
]

# ---- ENERGY -------------------------------------------------
# Each entry is a list of (level, duration_in_beats) breakpoints; the envelope
# is linearly interpolated between the levels and applied to broadband noise.
# Levels are linear amplitude in [0, 1]. A `None` level is silence.
ENERGY_SHAPES = [
    ("ramp_up",         [(0.02, 0), (1.0, 8)]),
    ("ramp_down",       [(1.0, 0), (0.02, 8)]),
    ("four_stabs",      [(1.0, 0.5), (None, 1.5), (1.0, 0.5), (None, 1.5),
                         (1.0, 0.5), (None, 1.5), (1.0, 0.5), (None, 1.5)]),
    ("swell",           [(0.02, 0), (1.0, 4), (0.02, 4)]),
    ("plateau",         [(0.6, 0), (0.6, 8)]),
    ("staircase_up",    [(0.15, 0), (0.15, 2), (0.4, 2), (0.4, 2),
                         (0.7, 2), (0.7, 2), (1.0, 2), (1.0, 2)]),
    ("staircase_down",  [(1.0, 0), (1.0, 2), (0.7, 2), (0.7, 2),
                         (0.4, 2), (0.4, 2), (0.15, 2), (0.15, 2)]),
    ("two_swells",      [(0.02, 0), (1.0, 2), (0.02, 2), (1.0, 2), (0.02, 2)]),
    ("step_up",         [(0.15, 0), (0.15, 4), (1.0, 0.01), (1.0, 4)]),
    ("step_down",       [(1.0, 0), (1.0, 4), (0.15, 0.01), (0.15, 4)]),
    ("half_silent",     [(None, 0), (None, 4), (0.8, 0.01), (0.8, 4)]),
    ("sparse_hits",     [(1.0, 0.4), (None, 3.6), (1.0, 0.4), (None, 3.6)]),
    ("pulse_2hz",       [(1.0, 0.5), (0.05, 0.5)] * 8),
    ("pulse_slow_fast", [(1.0, 1), (0.05, 1), (1.0, 1), (0.05, 1),
                         (1.0, 0.5), (0.05, 0.5), (1.0, 0.5), (0.05, 0.5),
                         (1.0, 0.25), (0.05, 0.25), (1.0, 0.25), (0.05, 0.25)]),
    ("accent_every_4",  [(1.0, 0.4), (0.25, 0.6), (0.25, 1), (0.25, 1),
                         (1.0, 0.4), (0.25, 0.6), (0.25, 1), (0.25, 1)]),
    ("long_decay",      [(1.0, 0), (0.3, 2), (0.1, 3), (0.02, 3)]),
]

# ---- CHROMA -------------------------------------------------
# Each entry is a list of (pitch classes as MIDI note numbers, duration_in_beats)
# segments, rendered as sustained additive chords. Pitch classes are what a
# chromagram sees; the octave is chosen inside the tone generator.
CHROMA_CHORDS = [
    ("C_major_triad",   [([60, 64, 67], 8)]),
    ("A_minor_triad",   [([57, 60, 64], 8)]),
    ("single_pc_C",     [([48, 60, 72], 8)]),
    ("I_IV_V",          [([60, 64, 67], 3), ([65, 69, 72], 3),
                         ([67, 71, 74], 2)]),
    ("C_then_Fsharp",   [([60, 64, 67], 4), ([66, 70, 73], 4)]),
    ("fifth_C_G",       [([60, 67], 8)]),
    ("tritone_C_Fs",    [([60, 66], 8)]),
    ("alternating_C_F", [([60, 64, 67], 2), ([65, 69, 72], 2),
                         ([60, 64, 67], 2), ([65, 69, 72], 2)]),
    ("chromatic_cluster", [([60, 61, 62], 8)]),
    ("whole_tone",      [([60, 62, 64, 66], 8)]),
    ("quartal_C_F_Bb",  [([60, 65, 70], 8)]),
    ("D_major_triad",   [([62, 66, 69], 8)]),
    ("descending_5ths", [([60, 64, 67], 2), ([65, 69, 72], 2),
                         ([58, 62, 65], 2), ([63, 67, 70], 2)]),
    ("pc_sweep",        [([60], 1), ([62], 1), ([64], 1), ([65], 1),
                         ([67], 1), ([69], 1), ([71], 1), ([72], 1)]),
    ("Eb_major_triad",  [([63, 67, 70], 8)]),
    ("cluster_then_triad", [([60, 61, 62, 63], 4), ([60, 64, 67], 4)]),
]

# ---- RHYTHM -------------------------------------------------
# Each entry is (bpm | (bpm_start, bpm_end) for a ramp, beats_per_bar). Rendered
# as a click track: a short bright burst per beat, a brighter+louder one on the
# downbeat, over a quiet sustained bed so the signal is not pure silence between
# clicks (beat trackers are trained on music, not on isolated impulses).
RHYTHM_GRIDS = [
    # ORDER MATTERS: only the first n_plot are drawn, so the reliable stimuli
    # come first. `beat_this` recovers 12 of these 16 grids at the intended
    # tempo; the last four fall into the well-known tempo-OCTAVE ambiguity of
    # beat tracking -- 60 bpm comes back as 120, 160 and 180 come back halved,
    # and the ritardando is not followed through the ramp. Those four are NOT
    # broken probes: the target is whatever the run's own extractor produced, so
    # the model is still scored against a self-consistent goal and they keep
    # counting in the influence row. They are simply confusing to LOOK at, since
    # the picture then disagrees with the name, so they are pushed past the
    # plotted head. (Measured on the 5 s / 431-frame geometry -- re-check the
    # order if the chunk duration changes.)
    ("clicks_120_4",    (120, 4)),
    ("clicks_90_4",     (90, 4)),
    ("clicks_140_4",    (140, 4)),
    ("clicks_80_4",     (80, 4)),
    ("clicks_120_3",    (120, 3)),
    ("clicks_100_2",    (100, 2)),
    ("clicks_110_4",    (110, 4)),
    ("clicks_75_3",     (75, 3)),
    ("clicks_130_2",    (130, 2)),
    ("clicks_95_4",     (95, 4)),
    ("clicks_70_4",     (70, 4)),
    ("accelerando",     ((80, 160), 4)),
    # ---- below: recovered at a different metrical level (see the note above) --
    ("clicks_60_4",     (60, 4)),
    ("clicks_160_4",    (160, 4)),
    ("clicks_180_4",    (180, 4)),
    ("ritardando",      ((160, 80), 4)),
]

PROBE_BANKS = {
    "f0": PROBE_MELODIES,
    "energy": ENERGY_SHAPES,
    "chroma": CHROMA_CHORDS,
    "rhythm": RHYTHM_GRIDS,
}


# ============================================================
# SYNTHESIS
# ============================================================
def _midi_to_hz(midi) -> float:
    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def _harmonic_tone(freq, n, sr, n_harmonics: int = 6) -> np.ndarray:
    """Additive tone: fundamental + partials at 1/k. Not a bare sine -- every
    pitch-aware extractor (chromagram included) keys off the harmonic series,
    and a pure sine is the degenerate case that would put the extractor's
    weakness into the target instead of the model's."""
    t = np.arange(n, dtype=np.float64) / float(sr)
    seg = np.zeros(n, dtype=np.float64)
    for h in range(1, n_harmonics + 1):
        fh = freq * h
        if fh >= 0.45 * sr:
            break
        seg += np.sin(2.0 * np.pi * fh * t) / float(h)
    return seg


def _fade(seg, sr, atk_s=0.015, rel_s=0.040):
    """Short attack/release so segment boundaries do not click. A click is
    broadband and reads as a transient, which perturbs the frames around it."""
    n = len(seg)
    a = min(max(1, int(atk_s * sr)), n)
    r = min(max(1, int(rel_s * sr)), n)
    env = np.ones(n, dtype=np.float64)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[n - r:] = np.linspace(1.0, 0.0, r)
    return seg * env


def _normalize(out, amp=0.25):
    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * amp
    return out.astype(np.float32)


def synthesize_energy(breakpoints, sr: int = DAC_SAMPLE_RATE,
                      duration_s: float = 5.0, amp: float = 0.25,
                      seed: int = 12345) -> np.ndarray:
    """
    Render an (level, beats) breakpoint list as an amplitude envelope applied to
    broadband noise, exactly duration_s long.

    Noise, not a tone, is the carrier on purpose: the energy condition is a
    frequency-weighted loudness curve, so the probe should vary loudness and
    NOTHING else. A pitched carrier would let the model reproduce the envelope
    by tracking pitch instead, and the row would not be measuring what it says.

    The noise is drawn from a FIXED seed so the stimulus is identical on every
    machine and the cached target stays valid.
    """
    n_total = int(round(duration_s * sr))
    rng = np.random.default_rng(seed)
    carrier = rng.standard_normal(n_total)

    total_beats = sum(float(b) for _, b in breakpoints) or 1.0
    # Envelope by linear interpolation between breakpoints, in SAMPLES.
    xs, ys, pos = [], [], 0.0
    for level, beats in breakpoints:
        xs.append(pos / total_beats * n_total)
        ys.append(0.0 if level is None else float(level))
        pos += float(beats)
    if len(xs) == 1:
        xs.append(float(n_total))
        ys.append(ys[0])
    xs[-1] = max(xs[-1], float(n_total))
    env = np.interp(np.arange(n_total, dtype=np.float64), xs, ys)
    return _normalize(carrier * env, amp)


def synthesize_chroma(segments, sr: int = DAC_SAMPLE_RATE,
                      duration_s: float = 5.0, amp: float = 0.25) -> np.ndarray:
    """Render a (midi list, beats) segment list as sustained additive chords."""
    n_total = int(round(duration_s * sr))
    out = np.zeros(n_total, dtype=np.float64)
    total_beats = sum(float(b) for _, b in segments) or 1.0

    pos = 0
    for k, (notes, beats) in enumerate(segments):
        end = int(round(sum(float(b) for _, b in segments[:k + 1])
                        / total_beats * n_total))
        end = min(end, n_total)
        n = end - pos
        if n <= 0:
            pos = end
            continue
        seg = np.zeros(n, dtype=np.float64)
        for m in notes:
            seg += _harmonic_tone(_midi_to_hz(m), n, sr)
        out[pos:end] = _fade(seg / max(1, len(notes)), sr)
        pos = end
    return _normalize(out, amp)


def synthesize_rhythm(grid, sr: int = DAC_SAMPLE_RATE, duration_s: float = 5.0,
                      amp: float = 0.25, seed: int = 54321) -> np.ndarray:
    """
    Render a (bpm, beats_per_bar) grid as a click track over a quiet bed.

    `bpm` may be a (start, end) pair for a linear tempo ramp; the beat times are
    then integrated from the instantaneous tempo rather than spaced evenly, so
    an accelerando really accelerates.

    The bed matters: beat trackers are trained on music, and a signal that is
    pure silence between impulses is out of their distribution. A quiet
    sustained tone keeps the stimulus musical enough to be tracked while leaving
    the clicks as the only rhythmic information.
    """
    n_total = int(round(duration_s * sr))
    rng = np.random.default_rng(seed)
    bpm, per_bar = grid
    per_bar = int(per_bar)

    # ---- beat times ----
    times = []
    if isinstance(bpm, (tuple, list)):
        b0, b1 = float(bpm[0]), float(bpm[1])
        t = 0.0
        while t < duration_s:
            times.append(t)
            frac = min(1.0, t / duration_s)
            t += 60.0 / (b0 + (b1 - b0) * frac)
    else:
        period = 60.0 / float(bpm)
        t = 0.0
        while t < duration_s:
            times.append(t)
            t += period

    out = np.zeros(n_total, dtype=np.float64)

    # ---- quiet sustained bed (a low fifth), well under the clicks ----
    bed = (_harmonic_tone(_midi_to_hz(48), n_total, sr, n_harmonics=4)
           + _harmonic_tone(_midi_to_hz(55), n_total, sr, n_harmonics=4))
    out += _fade(bed, sr, atk_s=0.05, rel_s=0.05) * 0.08

    # ---- the clicks: filtered noise burst, brighter/louder on the downbeat ----
    for i, tsec in enumerate(times):
        start = int(round(tsec * sr))
        down = (i % per_bar) == 0
        n_click = int(round((0.030 if down else 0.018) * sr))
        n_click = min(n_click, n_total - start)
        if n_click <= 0:
            continue
        burst = rng.standard_normal(n_click)
        # one-pole high-pass -> a bright tick; the downbeat gets a stronger one
        alpha = 0.55 if down else 0.75
        for j in range(1, n_click):
            burst[j] = burst[j] - alpha * burst[j - 1]
        decay = np.exp(-np.arange(n_click, dtype=np.float64)
                       / (n_click / 3.5))
        out[start:start + n_click] += burst * decay * (1.0 if down else 0.55)
    return _normalize(out, amp)


def synthesize_melody(notes, sr: int = DAC_SAMPLE_RATE, duration_s: float = 5.0,
                      n_harmonics: int = 6, amp: float = 0.25) -> np.ndarray:
    """
    Render a (midi, beats) note list to a float32 waveform of EXACTLY
    duration_s seconds.

    The tone is additive -- fundamental plus `n_harmonics` partials at 1/k
    amplitude -- rather than a bare sine, because a pitch tracker keys off the
    harmonic series: a pure sine is the degenerate case where octave errors are
    most likely, which would put CREPE's weakness, not the model's, into the
    target. Each note gets a short attack and release so the boundaries do not
    click (a click is broadband and reads as a transient, which can perturb the
    frame around it).

    Beat durations are relative: the whole list is scaled to fill duration_s, so
    a melody's note count sets its tempo and every probe is the same length as
    a training chunk.
    """
    total_beats = sum(float(d) for _, d in notes) or 1.0
    n_total = int(round(duration_s * sr))
    out = np.zeros(n_total, dtype=np.float64)

    atk = max(1, int(0.015 * sr))     # 15 ms
    rel = max(1, int(0.040 * sr))     # 40 ms

    pos = 0
    for k, (midi, beats) in enumerate(notes):
        # Distribute rounding over the whole melody rather than per note, so the
        # last note ends exactly at n_total instead of accumulating drift.
        end = int(round((sum(float(d) for _, d in notes[:k + 1]) / total_beats)
                        * n_total))
        end = min(end, n_total)
        n = end - pos
        if n <= 0:
            pos = end
            continue
        if midi is not None:
            f = _midi_to_hz(midi)
            t = np.arange(n, dtype=np.float64) / float(sr)
            seg = np.zeros(n, dtype=np.float64)
            for h in range(1, n_harmonics + 1):
                fh = f * h
                if fh >= 0.45 * sr:          # never synthesize above Nyquist
                    break
                seg += np.sin(2.0 * np.pi * fh * t) / float(h)
            env = np.ones(n, dtype=np.float64)
            a, r = min(atk, n), min(rel, n)
            env[:a] = np.linspace(0.0, 1.0, a)
            env[n - r:] = np.linspace(1.0, 0.0, r)
            out[pos:end] = seg * env
        pos = end

    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * amp
    return out.astype(np.float32)


SYNTHESIZERS = {
    "f0": synthesize_melody,
    "energy": synthesize_energy,
    "chroma": synthesize_chroma,
    "rhythm": synthesize_rhythm,
}


# ============================================================
# THE PROBE SET
# ============================================================
class ConditionProbeSet:
    """Built probe set for ONE condition: synthesized stimuli + their extracted
    targets.

    `targets[i]` is (n_frames, dim) exactly like a dataset condition, so it can
    be fed to the sampler with no special-casing. The waveforms stay on disk and
    are read on demand -- only the handful that get logged are ever loaded."""

    def __init__(self, condition, directory, names, targets, sr, duration_s):
        self.condition = str(condition)
        self.dir = str(directory)
        self.names = list(names)
        self.targets = list(targets)
        self.sr = int(sr)
        self.duration_s = float(duration_s)

    def __len__(self):
        return len(self.targets)

    def wav_path(self, i: int) -> str:
        return os.path.join(self.dir, f"probe_{i:02d}_{self.names[i]}.wav")

    def wav(self, i: int) -> np.ndarray:
        import soundfile as sf
        y, _ = sf.read(self.wav_path(i), dtype="float32")
        return y


def _synth_fingerprint(condition: str) -> dict:
    """Identity of the SYNTHESIS: the hash of the generator's own source. Edit a
    synthesizer and every cached set for that condition rebuilds, instead of
    silently reusing targets produced by code that no longer exists."""
    src = inspect.getsource(SYNTHESIZERS[condition])
    helpers = "".join(inspect.getsource(f) for f in
                      (_harmonic_tone, _fade, _normalize, _midi_to_hz))
    return {"synth_sha1": hashlib.sha1((src + helpers).encode()).hexdigest()}


def _fingerprint(condition, n_probes, n_frames, sr, duration_s, extractor):
    """Everything that would change the targets. The extractor's parameters are
    read off the object with dir() -- NOT vars(), which sees only the instance
    __dict__ and would drop a parameter carried as a class attribute, leaving a
    stale cache reusable with no sign of it."""
    ex = {}
    for k in sorted(dir(extractor)):
        if k.startswith("_") or k == "device":   # device: speed, never values
            continue
        try:
            v = getattr(extractor, k)
        except Exception:
            continue
        if isinstance(v, (int, float, str, bool)):
            ex[k] = v
    if not ex:
        raise ValueError(
            f"{condition} probe: the extractor exposes no scalar parameters, so "
            f"the cache fingerprint would not react to a configuration change. "
            f"Refusing to build a probe set that could later be reused stale.")
    payload = {
        "condition": condition,
        "n_probes": int(n_probes),
        "n_frames": int(n_frames),
        "sr": int(sr),
        "duration_s": round(float(duration_s), 6),
        "stimuli": json.loads(json.dumps(PROBE_BANKS[condition][:n_probes])),
        "synth": _synth_fingerprint(condition),
        "extractor": ex,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def build_condition_probe_set(condition, probe_dir, n_frames, extractor,
                              n_probes: int = 16, duration_s: float = 5.0,
                              sr: int = DAC_SAMPLE_RATE, force: bool = False,
                              verbose: bool = True) -> ConditionProbeSet:
    """
    Build (or load from cache) the out-of-the-box probe set for `condition`
    ("f0" | "energy" | "chroma" | "rhythm").

    `extractor` is the RUN'S extractor for that condition
    (registry.frame_extractors[condition]), so the targets are produced by the
    exact configuration that produced the training targets. Nothing here
    re-implements one.
    """
    if condition not in PROBE_BANKS:
        raise ValueError(f"no probe bank for condition '{condition}'. "
                         f"Available: {sorted(PROBE_BANKS)}.")
    bank = PROBE_BANKS[condition]
    n_probes = max(1, min(int(n_probes), len(bank)))
    probe_dir = str(probe_dir)
    os.makedirs(probe_dir, exist_ok=True)
    meta_path = os.path.join(probe_dir, "meta.json")
    npz_path = os.path.join(probe_dir, "targets.npz")
    fp = _fingerprint(condition, n_probes, n_frames, sr, duration_s, extractor)
    names = [name for name, _ in bank[:n_probes]]
    tag = f"[{condition}-probe]"

    if not force and os.path.exists(meta_path) and os.path.exists(npz_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("fingerprint") == fp:
                data = np.load(npz_path)
                targets = [data[f"probe_{i:02d}"] for i in range(n_probes)]
                if all(os.path.exists(os.path.join(
                        probe_dir, f"probe_{i:02d}_{names[i]}.wav"))
                        for i in range(n_probes)):
                    if verbose:
                        print(f"{tag} cache hit: {n_probes} stimuli "
                              f"from {probe_dir}")
                    return ConditionProbeSet(condition, probe_dir, names,
                                             targets, sr, duration_s)
                if verbose:
                    print(f"{tag} cache metadata matches but wavs are "
                          f"missing -> rebuilding")
            elif verbose:
                print(f"{tag} configuration changed -> rebuilding")
        except Exception as e:
            print(f"{tag} unreadable cache ({e}) -> rebuilding")

    import soundfile as sf
    synth = SYNTHESIZERS[condition]
    if verbose:
        print(f"{tag} building {n_probes} elementary stimuli "
              f"({duration_s:.1f}s each) and extracting their targets...")

    targets = []
    for i, (name, spec) in enumerate(bank[:n_probes]):
        y = synth(spec, sr=sr, duration_s=duration_s)
        sf.write(os.path.join(probe_dir, f"probe_{i:02d}_{name}.wav"), y, sr)
        tgt = np.asarray(extractor.extract(y, sr, n_frames), dtype=np.float32)
        targets.append(tgt)
        if verbose:
            # A degenerate target means the probe would be scoring nothing.
            # Report it rather than let a meaningless row appear in the panel.
            # f0 is reported as VOICED COVERAGE, which is its real failure mode:
            # a contour extracted as almost entirely unvoiced would have the
            # probe scoring silence, and its std would look perfectly healthy.
            if condition == "f0":
                voiced = float((tgt[:, 0] > 0).mean())
                flag = ("  <-- almost all unvoiced, check fmin/fmax and "
                        "silence_db against the synthesized level"
                        if voiced < 0.10 else "")
                print(f"  [{i:02d}] {name:<20s} shape={tuple(tgt.shape)} "
                      f"voiced={voiced*100:5.1f}%{flag}")
            else:
                spread = float(np.std(tgt))
                flag = "  <-- FLAT, check the extractor" if spread < 1e-4 else ""
                print(f"  [{i:02d}] {name:<20s} shape={tuple(tgt.shape)} "
                      f"std={spread:.4f}{flag}")

    np.savez(npz_path, **{f"probe_{i:02d}": t for i, t in enumerate(targets)})
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": fp, "condition": condition, "names": names,
                   "n_frames": int(n_frames), "sr": int(sr),
                   "duration_s": float(duration_s)}, fh, indent=2)
    if verbose:
        print(f"{tag} saved to {probe_dir}")
    return ConditionProbeSet(condition, probe_dir, names, targets, sr,
                             duration_s)


# ============================================================
# PLOTS
# ============================================================
def _fig_to_tensor(fig) -> torch.Tensor:
    """matplotlib figure -> (3, H, W) float tensor in [0,1] for add_image.
    dpi is taken from the FIGURE, not hardcoded: a fixed dpi=100 here is what
    used to make these plots microscopic."""
    from PIL import Image
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=fig.dpi)
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def _title(condition, kind, label, step, prefix, guidance, score):
    bits = [f"{condition}_{kind} vs {condition}_gen"]
    if label:
        bits.append(str(label))
    sub = []
    if step is not None:
        sub.append(f"step {step}")
    if prefix:
        sub.append(str(prefix))
    if guidance is not None:
        sub.append(f"guidance={guidance}")
    if isinstance(score, (int, float)):
        sub.append(f"score={score:+.3f}")
    return "  ".join(bits) + ("\n" + " · ".join(sub) if sub else "")


def plot_f0_comparison(target, generated, kind="valid", label="",
                       step=None, prefix=None, guidance=None, corr=None,
                       fps: float = DAC_FRAMES_PER_S,
                       pad_octaves: float = 1.0) -> torch.Tensor:
    """
    "<target> vs f0_gen": the conditioning contour and the contour re-extracted
    from the generation it conditioned, OVERLAID on one log-Hz axis, with a thin
    voicing ribbon underneath.

    `kind` names the target in the title and in the legend: "probe" -> f0_probe
    (an elementary synthetic melody), "valid" -> f0_valid (a real validation
    recording). `label` says WHICH one ('scale up', 'sample #12').

    Design notes:
      * OVERLAID, not side by side: the quantity of interest is the DIFFERENCE
        between the two curves, and on shared axes that difference is a vertical
        distance you read directly instead of estimating across a gap.
      * y is Hz on a LOG scale. Pitch is perceived logarithmically, so on a
        linear axis the same musical interval looks bigger up high than down low.
      * the y window comes from the TARGET only, padded by `pad_octaves` on each
        side and clamped to the extractor's [fmin, fmax]. From the target only
        because the target never changes: the window is therefore identical at
        every step and the plots read as a time series, which an autoscaled axis
        would destroy by silently rescaling as the model improves.
      * unvoiced frames are BREAKS in the line (NaN), never a drop to 0 Hz. A
        line diving to the floor would draw a pitch glide that was never
        predicted; a gap says "nothing here", which is what 0 means.
      * voicing lives in its own RIBBON under the plot, NOT as shading across it.
        Shading every frame where the two disagree used to flood the whole figure
        when the generation was mostly unvoiced -- which is exactly what an
        untrained model produces, so the plot went unreadable precisely when it
        had the most to say.
      * the voiced-frame counts are printed on the figure. "The orange curve is
        missing" and "the orange curve is off-scale" are different failures and
        must not look the same.
    """
    kw = CONDITION_CONFIG.get("frame_level", {}).get("f0", {}).get("kwargs", {})
    fmin = float(kw.get("fmin", 50.0))
    fmax = float(kw.get("fmax", 1000.0))

    tgt_name = "f0_probe" if kind == "probe" else "f0_valid"
    C_TGT, C_GEN, C_WARN = "#1f77b4", "#e8710a", "#c0392b"

    t_hz = f0_norm_to_hz(target)
    g_hz = f0_norm_to_hz(generated)
    n = min(len(t_hz), len(g_hz))
    t_hz, g_hz = t_hz[:n], g_hz[:n]
    time = np.arange(n) / float(fps)

    tv, gv = t_hz > 0, g_hz > 0
    t_plot = np.where(tv, t_hz, np.nan)
    g_plot = np.where(gv, g_hz, np.nan)

    # Main axis + voicing ribbon, sharing the time axis.
    fig = plt.figure(figsize=(7.2, 3.9), dpi=120)
    gs = fig.add_gridspec(2, 1, height_ratios=[7, 1], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(time, t_plot, color=C_TGT, lw=2.6,
            label=f"{tgt_name}  (target / condition)")
    ax.plot(time, g_plot, color=C_GEN, lw=2.0, alpha=0.95,
            label="f0_gen  (re-extracted from the generation)")

    # ---- y window: from the target, so it never moves between steps ----
    if tv.any():
        lo = max(fmin, float(t_hz[tv].min()) / (2.0 ** pad_octaves))
        hi = min(fmax, float(t_hz[tv].max()) * (2.0 ** pad_octaves))
    else:
        lo, hi = fmin, fmax          # nothing voiced in the target: show it all
    if not (hi > lo):
        lo, hi = fmin, fmax

    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(0, time[-1] if n > 1 else 1.0)
    ax.set_ylabel("f0 (Hz, log)", fontsize=12.5)
    ax.grid(True, which="major", axis="both", alpha=0.30, lw=0.6)
    ax.grid(True, which="minor", axis="y", alpha=0.12, lw=0.4)
    ax.tick_params(axis="both", labelsize=11, length=3)
    ax.tick_params(axis="x", labelbottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- title: what is being compared, then the run coordinates ----
    head = f"{tgt_name}  vs  f0_gen"
    if label:
        head += f"   —   {label}"
    sub = []
    if prefix:
        sub.append(str(prefix))
    if step is not None:
        sub.append(f"step {step}")
    if guidance is not None:
        sub.append(f"guidance={guidance}")
    if corr is not None and np.isfinite(corr):
        sub.append(f"corr={corr:+.3f}")
    ax.set_title(head, fontsize=17, fontweight="bold",
                 pad=28 if sub else 10)
    if sub:
        ax.text(0.5, 1.012, "   ·   ".join(sub), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11, color="#555555")

    # ---- voiced coverage + off-scale count, spelled out ----
    n_tv, n_gv = int(tv.sum()), int(gv.sum())
    off = int((gv & ((g_hz < lo) | (g_hz > hi))).sum())
    info = (f"{tgt_name}: {n_tv}/{n} voiced      "
            f"f0_gen: {n_gv}/{n} voiced")
    if off:
        info += f"      {off} off-scale"
    ax.text(0.012, 0.03, info, transform=ax.transAxes, fontsize=10.5,
            va="bottom", ha="left", family="monospace",
            color=C_WARN if (n_gv == 0 or off) else "#444444",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none",
                      alpha=0.80))

    # ---- voicing ribbon: two rows, no flooding of the main plot ----
    axv.fill_between(time, 0.55, 1.45, where=tv, step="mid",
                     color=C_TGT, lw=0, alpha=0.85)
    axv.fill_between(time, -0.45, 0.45, where=gv, step="mid",
                     color=C_GEN, lw=0, alpha=0.85)
    axv.set_ylim(-0.7, 1.7)
    axv.set_yticks([0.0, 1.0])
    axv.set_yticklabels(["gen", "target"], fontsize=9.5)
    axv.set_xlabel("time (s)", fontsize=12.5)
    axv.tick_params(axis="x", labelsize=11, length=3)
    axv.tick_params(axis="y", length=0)
    axv.grid(True, axis="x", alpha=0.20, lw=0.5)
    for s in ("top", "right", "left"):
        axv.spines[s].set_visible(False)
    axv.set_ylabel("voiced", fontsize=10, color="#777777", labelpad=8)

    # Legend UNDER the figure: in any corner of the axes it would sooner or
    # later sit on top of the curves, and the y window is fixed by the target,
    # so there is no corner guaranteed to stay empty.
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.16))

    img = _fig_to_tensor(fig)
    plt.close(fig)
    return img


def plot_condition_comparison(condition, target, generated, kind="valid",
                              label="", step=None, prefix=None, guidance=None,
                              score=None, fps: float = DAC_FRAMES_PER_S,
                              dpi: int = 130) -> torch.Tensor:
    """
    Target vs re-extracted condition, as one image, in the form that suits the
    condition's shape:

      energy (T,1)  -> two curves on one axis. The question is whether the
                       generated envelope follows the target's SHAPE, so both
                       are drawn on the same axis and the eye compares them
                       directly.
      rhythm (T,2)  -> two stacked axes, beat and downbeat probability, each
                       with target and generated overlaid: a model can follow
                       the beat and miss the bar, and one axis would hide it.
      chroma (T,12) -> two stacked heatmaps, target above generated, sharing the
                       colour scale. Twelve overlaid curves are unreadable; the
                       question here is whether the same pitch classes light up
                       at the same times, which is a picture, not a plot.
    """
    if condition == "f0":
        # f0 has its own rendering and keeps it: a log-Hz axis (pitch is
        # perceived logarithmically), unvoiced frames as BREAKS rather than a
        # dive to 0 Hz, and voicing in a ribbon under the plot instead of
        # shading that floods the figure when the generation is mostly
        # unvoiced. None of that generalizes to a curve or a heatmap.
        return plot_f0_comparison(target, generated, kind=kind, label=label,
                                  step=step, prefix=prefix, guidance=guidance,
                                  corr=score, fps=fps)

    tgt = np.asarray(target, dtype=np.float32)
    gen = np.asarray(generated, dtype=np.float32)
    if tgt.ndim == 1:
        tgt = tgt[:, None]
    if gen.ndim == 1:
        gen = gen[:, None]
    n = min(len(tgt), len(gen))
    tgt, gen = tgt[:n], gen[:n]
    t = np.arange(n) / float(fps)
    head = _title(condition, kind, label, step, prefix, guidance, score)

    if condition == "chroma" or tgt.shape[1] >= 8:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), dpi=dpi, sharex=True)
        vmax = max(float(tgt.max()), float(gen.max()), 1e-6)
        for ax, arr, name in ((axes[0], tgt, "target"),
                              (axes[1], gen, "generated")):
            ax.imshow(arr.T, aspect="auto", origin="lower", vmin=0.0, vmax=vmax,
                      extent=[0, t[-1] if n else 0, -0.5, arr.shape[1] - 0.5],
                      cmap="magma", interpolation="nearest")
            ax.set_ylabel(f"{name}\npitch class")
            ax.set_yticks(range(0, arr.shape[1],
                                max(1, arr.shape[1] // 6)))
        axes[1].set_xlabel("time (s)")
        axes[0].set_title(head, fontsize=10)

    elif condition == "rhythm" or tgt.shape[1] == 2:
        chan = ["beat", "downbeat"]
        fig, axes = plt.subplots(tgt.shape[1], 1, figsize=(10, 4.6), dpi=dpi,
                                 sharex=True, squeeze=False)
        for c in range(tgt.shape[1]):
            ax = axes[c][0]
            ax.plot(t, tgt[:, c], lw=1.6, label="target", color="#1f77b4")
            ax.plot(t, gen[:, c], lw=1.2, label="generated", color="#d62728",
                    alpha=0.85)
            ax.set_ylabel(chan[c] if c < len(chan) else f"ch{c}")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            if c == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1][0].set_xlabel("time (s)")
        axes[0][0].set_title(head, fontsize=10)

    else:
        fig, ax = plt.subplots(figsize=(10, 3.4), dpi=dpi)
        ax.plot(t, tgt[:, 0], lw=1.8, label="target", color="#1f77b4")
        ax.plot(t, gen[:, 0], lw=1.3, label="generated", color="#d62728",
                alpha=0.85)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(condition)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(head, fontsize=10)

    fig.tight_layout()
    img = _fig_to_tensor(fig)
    plt.close(fig)
    return img


# ============================================================
# CLI
# ============================================================
def main():
    import argparse
    from conditions import ConditionRegistry

    ap = argparse.ArgumentParser(
        description="Build and inspect an out-of-the-box condition probe set.")
    ap.add_argument("condition", choices=sorted(PROBE_BANKS))
    ap.add_argument("probe_dir")
    ap.add_argument("--n_frames", type=int, required=True,
                    help="frames per chunk (dataset_meta.json: "
                         "latents_frames_per_chunk)")
    ap.add_argument("--n_probes", type=int, default=16)
    ap.add_argument("--duration_s", type=float, default=5.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    reg = ConditionRegistry(enabled_frame=[args.condition], enabled_global=[])
    extractor = reg.frame_extractors[args.condition]
    for attr in ("device", "_device"):
        if hasattr(extractor, attr):
            try:
                setattr(extractor, attr, args.device)
            except Exception:
                pass

    ps = build_condition_probe_set(
        args.condition, args.probe_dir, args.n_frames, extractor,
        n_probes=args.n_probes, duration_s=args.duration_s, force=args.force)
    print(f"\n{len(ps)} stimuli in {ps.dir}")
    for i, name in enumerate(ps.names):
        print(f"  [{i:02d}] {name:<20s} {ps.wav_path(i)}")


if __name__ == "__main__":
    main()
