# Probe stimuli banks for the listening panels.

import os

_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    _cache = os.path.join(_IRCAM_LOCAL, ".cache")
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", _cache)
    os.environ.setdefault("HF_HOME", os.path.join(_cache, "huggingface"))
    os.environ["TORCH_HOME"] = os.path.join(_cache, "torch")

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
from typing import Optional

import latent_codec as _lc
from condition_metrics import f0_norm_to_hz, render_midi_events


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

CHROMA_CHORDS = [
    ("C_major_triad",   [([60, 64, 67], 8)]),
    ("A_minor_cadence", [([57, 60, 64], 2), ([62, 65, 69], 2),
                         ([64, 68, 71], 2), ([57, 60, 64], 2)]),
    ("single_pc_C_G",   [([48, 60, 72], 4), ([43, 55, 67], 4)]),
    ("I_IV_V",          [([60, 64, 67], 3), ([65, 69, 72], 3),
                         ([67, 71, 74], 2)]),
    ("C_then_Fsharp",   [([60, 64, 67], 4), ([66, 70, 73], 4)]),
    ("fifths_staccato", [([60, 67], 1.5), ([], 0.5), ([67, 74], 1.5), ([], 0.5),
                         ([65, 72], 1.5), ([], 0.5), ([60, 67], 1.5), ([], 0.5)]),
    ("tritone_to_G",    [([60, 66], 4), ([59, 62, 67], 4)]),
    ("alternating_C_F", [([60, 64, 67], 2), ([65, 69, 72], 2),
                         ([60, 64, 67], 2), ([65, 69, 72], 2)]),
    ("cluster_shift",   [([60, 61, 62], 4), ([65, 66, 67], 4)]),
    ("whole_tone_alt",  [([60, 62, 64, 66], 2), ([61, 63, 65, 67], 2),
                         ([60, 62, 64, 66], 2), ([61, 63, 65, 67], 2)]),
    ("quartal_up_down", [([60, 65, 70], 3), ([62, 67, 72], 3),
                         ([60, 65, 70], 2)]),
    ("D_major_repeated", [([62, 66, 69], 1), ([], 0.4), ([62, 66, 69], 1),
                          ([], 0.4), ([62, 66, 69], 1), ([], 0.4),
                          ([62, 66, 69], 1)]),
    ("descending_5ths", [([60, 64, 67], 2), ([65, 69, 72], 2),
                         ([58, 62, 65], 2), ([63, 67, 70], 2)]),
    ("pc_sweep",        [([60], 1), ([62], 1), ([64], 1), ([65], 1),
                         ([67], 1), ([69], 1), ([71], 1), ([72], 1)]),
    ("Eb_two_phrases",  [([63, 67, 70], 1.5), ([56, 60, 63], 1.5), ([], 2),
                         ([58, 62, 65], 1.5), ([63, 67, 70], 1.5)]),
    ("cluster_then_triad", [([60, 61, 62, 63], 4), ([60, 64, 67], 4)]),
]

RHYTHM_GRIDS = [
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
    ("clicks_60_4",     (60, 4)),
    ("clicks_160_4",    (160, 4)),
    ("clicks_180_4",    (180, 4)),
    ("ritardando",      ((160, 80), 4)),
]

TEXT_PROMPTS = [
    ("pipe_organ",      "solo pipe organ in a large reverberant church"),
    ("choir",           "a cappella choir singing a slow hymn"),
    ("harpsichord",     "solo harpsichord playing a fast baroque piece"),
    ("string_quartet",  "string quartet playing a slow sustained chord"),
    ("distorted_guitar","distorted electric guitar power chords"),
    ("rock_drums",      "a loud rock drum kit with crash cymbals"),
    ("synth_bass",      "deep analog synthesizer bass drone"),
    ("edm_beat",        "fast electronic dance beat with a four on the floor kick"),
    ("solo_piano",      "solo piano playing a quiet melody"),
    ("trumpet",         "solo trumpet fanfare"),
    ("gregorian",       "gregorian chant in a stone cathedral"),
    ("acoustic_guitar", "bright plucked acoustic guitar"),
    ("metal_riff",      "heavy metal guitar riff with double bass drums"),
    ("ambient_pad",     "ambient synthesizer pad, no rhythm"),
    ("solo_cello",      "solo cello playing a low sustained note"),
    ("hand_percussion", "hand percussion with shakers and tambourine"),
]

IMAGE_SHAPES = [
    ("red_field",       {"kind": "solid",      "colors": [(200, 30, 30)]}),
    ("blue_field",      {"kind": "solid",      "colors": [(30, 60, 200)]}),
    ("black_white_split", {"kind": "halves",   "colors": [(15, 15, 15), (240, 240, 240)]}),
    ("vertical_stripes",{"kind": "stripes_v",  "colors": [(230, 200, 40), (30, 30, 60)], "n": 8}),
    ("horizontal_stripes", {"kind": "stripes_h", "colors": [(20, 140, 90), (245, 245, 235)], "n": 6}),
    ("checkerboard",    {"kind": "checker",    "colors": [(10, 10, 10), (250, 250, 250)], "n": 8}),
    ("concentric_rings",{"kind": "rings",      "colors": [(250, 250, 250), (190, 40, 120)], "n": 7}),
    ("white_disc",      {"kind": "disc",       "colors": [(20, 20, 40), (250, 240, 200)]}),
    ("triangle",        {"kind": "triangle",   "colors": [(245, 245, 245), (40, 40, 200)]}),
    ("diagonal_split",  {"kind": "diagonal",   "colors": [(250, 130, 20), (20, 60, 120)]}),
    ("cross",           {"kind": "cross",      "colors": [(250, 250, 250), (200, 20, 20)]}),
    ("dot_grid",        {"kind": "grid_dots",  "colors": [(245, 245, 245), (20, 20, 20)], "n": 6}),
    ("nested_squares",  {"kind": "nested",     "colors": [(250, 250, 250), (60, 120, 200)], "n": 6}),
    ("vertical_gradient", {"kind": "gradient_v", "colors": [(10, 10, 60), (250, 200, 60)]}),
    ("radial_burst",    {"kind": "burst",      "colors": [(250, 250, 250), (20, 20, 20)], "n": 16}),
    ("noise_field",     {"kind": "noise",      "colors": [(0, 0, 0), (255, 255, 255)], "seed": 0}),
]

CHORD_CHORDS = [
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

_B8 = [0, 1, 2, 3, 4, 5, 6, 7]
_BACKBEAT = ([(b, "kick") for b in (0, 2, 4, 6)]
             + [(b, "snare") for b in (1, 3, 5, 7)])
_EIGHTH_HATS = [(b / 2.0, "hihat_closed") for b in range(16)]
MIDI_PATTERNS = [
    ("C_major_scale",   {"beats": 8, "notes": [(b, 1, p) for b, p in zip(
                             _B8, [60, 62, 64, 65, 67, 69, 71, 72])],
                         "drums": []}),
    ("C_major_chord",   {"beats": 8, "notes": [(0, 8, 60), (0, 8, 64),
                                               (0, 8, 67)], "drums": []}),
    ("arpeggio_up_down", {"beats": 8, "notes": [(b, 1, p) for b, p in zip(
                             _B8, [60, 64, 67, 72, 67, 64, 60])],
                          "drums": []}),
    ("repeated_C",      {"beats": 8, "notes": [(b / 2.0, 0.4, 60)
                                               for b in range(16)],
                         "drums": []}),
    ("melody_and_bass", {"beats": 8, "notes": [
                             (0, 1, 64), (1, 1, 65), (2, 1, 67), (3, 1, 64),
                             (4, 1, 65), (5, 1, 69), (6, 2, 67),
                             (0, 2, 36), (2, 2, 43), (4, 2, 41), (6, 2, 43)],
                         "drums": []}),
    ("I_IV_V_I",        {"beats": 8, "notes": [
                             (b, 2, p) for b, chord in ((0, (60, 64, 67)),
                                                        (2, (65, 69, 72)),
                                                        (4, (67, 71, 74)),
                                                        (6, (60, 64, 67)))
                             for p in chord], "drums": []}),
    ("low_register",    {"beats": 8, "notes": [(b, 1, p) for b, p in zip(
                             _B8, [28, 31, 33, 35, 36, 38, 39, 40])],
                         "drums": []}),
    ("high_register",   {"beats": 8, "notes": [(b, 1, p) for b, p in zip(
                             _B8, [84, 86, 88, 89, 91, 93, 95, 96])],
                         "drums": []}),
    ("stabs_then_rest", {"beats": 8, "notes": [
                             (b, 0.5, p) for b in (0, 1, 2, 3)
                             for p in (62, 66, 69)], "drums": []}),
    ("backbeat",        {"beats": 8, "notes": [], "drums": _BACKBEAT}),
    ("rock_groove",     {"beats": 8, "notes": [],
                         "drums": _BACKBEAT + _EIGHTH_HATS}),
    ("open_hihat_offbeats", {"beats": 8, "notes": [],
                             "drums": [(b, "kick") for b in (0, 2, 4, 6)]
                             + [(b, "snare") for b in (1, 3, 5, 7)]
                             + [(b + 0.5, "hihat_open") for b in _B8]
                             + [(b, "hihat_closed") for b in _B8]}),
    ("tom_fill_crash",  {"beats": 8, "notes": [],
                         "drums": [(0, "crash"), (0, "kick"), (1, "snare"),
                                   (2, "kick"), (3, "snare"),
                                   (4, "tom_high"), (4.5, "tom_high"),
                                   (5, "tom_mid"), (5.5, "tom_mid"),
                                   (6, "tom_low"), (6.5, "tom_low"),
                                   (7, "crash"), (7, "kick")]}),
    ("groove_and_bass", {"beats": 8, "notes": [
                             (b / 2.0, 0.45, p) for b, p in zip(
                                 range(16), [36, 36, 48, 36, 40, 40, 52, 40,
                                             43, 43, 55, 43, 41, 41, 53, 43])],
                         "drums": _BACKBEAT + _EIGHTH_HATS}),
    ("ride_walking_bass", {"beats": 8, "notes": [(b, 1, p) for b, p in zip(
                               _B8, [36, 40, 43, 45, 47, 45, 43, 40])],
                           "drums": [(b, "ride") for b in _B8]
                           + [(b, "hihat_closed") for b in (1, 3, 5, 7)]}),
    ("band_I_vi_IV_V",  {"beats": 8, "notes": [
                             (b, 2, p) for b, chord in ((0, (60, 64, 67)),
                                                        (2, (57, 60, 64)),
                                                        (4, (53, 57, 60)),
                                                        (6, (55, 59, 62)))
                             for p in chord]
                         + [(b, 2, p) for b, p in ((0, 36), (2, 33),
                                                   (4, 29), (6, 31))],
                         "drums": _BACKBEAT + _EIGHTH_HATS}),
]

PROBE_BANKS = {
    "f0": PROBE_MELODIES,
    "energy": ENERGY_SHAPES,
    "chroma": CHROMA_CHORDS,
    "rhythm": RHYTHM_GRIDS,
    "text": TEXT_PROMPTS,
    "image": IMAGE_SHAPES,
    "chord": CHORD_CHORDS,
    "midi": MIDI_PATTERNS,
}

GLOBAL_PROBE_NAMES = set(CONDITION_CONFIG.get("global", {}))


def is_global_probe(condition: str) -> bool:
    return str(condition) in GLOBAL_PROBE_NAMES


def _slug(text) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in str(text).lower())
    return "_".join(p for p in s.split("_") if p)[:40] or "caption"


def text_probe_bank(caption_table=None, n_panels=None):
    table = caption_table or {}
    labels = [str(c) for c in (table.get("captions_text")
                               or table.get("captions") or [])]
    n_terms = table.get("n_terms")
    n = len(TEXT_PROMPTS)
    if not labels:
        return TEXT_PROMPTS, (f"the dataset has no caption file -> the {n} "
                              f"descriptions")
    if n_terms is None:
        return TEXT_PROMPTS, (f"the caption file does not say how many terms "
                              f"its captions have -> the {n} descriptions")
    if int(n_terms) > 1:
        return TEXT_PROMPTS, (f"the dataset's captions have {int(n_terms)} "
                              f"terms -> the {n} descriptions")
    if n_panels is not None:
        n = max(1, min(int(n_panels), n))
    words = [labels[i % len(labels)] for i in range(n)]
    more = f" (the first {n} of {len(labels)})" if len(labels) > n else ""
    return ([(_slug(w), w) for w in words],
            f"the dataset's captions are single labels -> those labels, in "
            f"turn: {', '.join(labels[:n])}{more}")


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


def synthesize_energy(breakpoints, sr: Optional[int] = None,
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
    sr = _lc.active_sr(sr)
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


def synthesize_chroma(segments, sr: Optional[int] = None,
                      duration_s: float = 5.0, amp: float = 0.25) -> np.ndarray:
    """Render a (midi list, beats) segment list as sustained additive chords."""
    sr = _lc.active_sr(sr)
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


def synthesize_rhythm(grid, sr: Optional[int] = None, duration_s: float = 5.0,
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
    sr = _lc.active_sr(sr)
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


def synthesize_melody(notes, sr: Optional[int] = None, duration_s: float = 5.0,
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
    sr = _lc.active_sr(sr)
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


def midi_probe_events(spec, duration_s: float = 5.0):
    """A MIDI_PATTERNS entry -> (pitched [(onset s, offset s, pitch)],
    drums [(onset s, class index)]), beats scaled to fill duration_s."""
    from conditions import MIDI_DRUM_CLASSES
    names = [n for n, _ in MIDI_DRUM_CLASSES]
    sec = float(duration_s) / (float(spec["beats"]) or 1.0)
    pitched = [(float(b) * sec, (float(b) + float(d)) * sec, int(p))
               for b, d, p in spec.get("notes", [])]
    drums = [(float(b) * sec, names.index(c)) for b, c in spec.get("drums", [])]
    return pitched, drums


def midi_probe_notes(spec, duration_s: float = 5.0):
    """The same events as (onset, offset, pitch, is_drum) notes -- the form
    MidiExtractor.from_notes rolls, as it rolls a MIDI file. A drum hit
    becomes its class's first General MIDI note (36 kick, 38 snare, ...),
    which the roll folds back into the same class."""
    from conditions import MIDI_DRUM_CLASSES
    pitched, drums = midi_probe_events(spec, duration_s)
    return ([(on, off, p, False) for on, off, p in pitched]
            + [(on, on + 0.05, MIDI_DRUM_CLASSES[k][1][0], True)
               for on, k in drums])


def synthesize_midi(spec, sr: Optional[int] = None, duration_s: float = 5.0,
                    amp: float = 0.25) -> np.ndarray:
    """A MIDI_PATTERNS entry rendered to listen to: notes as decaying harmonic
    tones, drum hits as class-coloured bursts (condition_metrics.
    render_midi_events, the renderer that sonifies a midi condition). Only the
    ear uses it: the probe's TARGET is the roll of the notes themselves
    (SYMBOLIC_TARGETS), and the generation is transcribed, not this waveform."""
    sr = _lc.active_sr(sr)
    pitched, drums = midi_probe_events(spec, duration_s)
    return render_midi_events(pitched, drums, sr,
                              int(round(duration_s * sr)), amp=amp)


SYMBOLIC_TARGETS = {"midi": midi_probe_notes}


def synthesize_text(spec, **_ignored) -> str:
    """The stimulus IS the prompt. This exists so the banks stay uniform (every
    condition has an entry in SYNTHESIZERS and the builder needs no special
    case); the `**_ignored` swallows the sr / duration_s the audio synthesizers
    take, which mean nothing to a string."""
    return str(spec)


def synthesize_image(spec, size: int = 512, **_ignored) -> np.ndarray:
    """
    Draw one abstract stimulus -> (size, size, 3) uint8 RGB.

    Purely deterministic: same spec, same pixels, on any machine and any run.
    That is what lets the cache fingerprint below stand for the image itself,
    and what makes panel 03 the same stimulus at every checkpoint. The one
    stochastic kind ("noise") carries its own seed in the spec for the same
    reason.

    Rendered at 512 and left there: CLIP resizes to its own 224 internally, and
    downscaling twice would only soften the edges that make these forms legible.
    """
    from PIL import Image, ImageDraw

    kind = spec.get("kind", "solid")
    colors = [tuple(int(v) for v in c) for c in spec.get("colors", [(0, 0, 0)])]
    bg = colors[0]
    fg = colors[1] if len(colors) > 1 else (255, 255, 255)
    n = int(spec.get("n", 8))
    s = int(size)

    if kind == "gradient_v":
        t = np.linspace(0.0, 1.0, s, dtype=np.float32)[:, None, None]
        arr = (np.asarray(bg, np.float32) * (1 - t)
               + np.asarray(fg, np.float32) * t)
        return np.repeat(arr, s, axis=1).astype(np.uint8)

    if kind == "noise":
        rng = np.random.default_rng(int(spec.get("seed", 0)))
        # Coarse blocks, not per-pixel snow: a 512x512 white-noise image is
        # nearly uniform grey once CLIP downsamples it to 224, which would make
        # this stimulus indistinguishable from a flat field.
        blocks = rng.integers(0, 2, size=(16, 16), dtype=np.int8)
        arr = np.where(blocks[..., None] > 0,
                       np.asarray(fg, np.uint8), np.asarray(bg, np.uint8))
        return np.asarray(Image.fromarray(arr.astype(np.uint8))
                          .resize((s, s), Image.NEAREST))

    img = Image.new("RGB", (s, s), bg)
    d = ImageDraw.Draw(img)

    if kind == "solid":
        pass
    elif kind == "halves":
        d.rectangle([0, 0, s // 2, s], fill=bg)
        d.rectangle([s // 2, 0, s, s], fill=fg)
    elif kind == "stripes_v":
        w = s / float(n)
        for i in range(0, n, 2):
            d.rectangle([i * w, 0, (i + 1) * w, s], fill=fg)
    elif kind == "stripes_h":
        h = s / float(n)
        for i in range(0, n, 2):
            d.rectangle([0, i * h, s, (i + 1) * h], fill=fg)
    elif kind == "checker":
        c = s / float(n)
        for i in range(n):
            for j in range(n):
                if (i + j) % 2 == 0:
                    d.rectangle([i * c, j * c, (i + 1) * c, (j + 1) * c], fill=fg)
    elif kind == "rings":
        step = s / float(2 * n)
        for i in range(n):
            o = i * step
            d.ellipse([o, o, s - o, s - o], fill=(fg if i % 2 == 0 else bg))
    elif kind == "disc":
        m = s * 0.12
        d.ellipse([m, m, s - m, s - m], fill=fg)
    elif kind == "triangle":
        d.polygon([(s // 2, int(s * 0.10)),
                   (int(s * 0.10), int(s * 0.90)),
                   (int(s * 0.90), int(s * 0.90))], fill=fg)
    elif kind == "diagonal":
        d.polygon([(0, 0), (s, 0), (0, s)], fill=fg)
    elif kind == "cross":
        a, b = int(s * 0.40), int(s * 0.60)
        d.rectangle([a, int(s * 0.08), b, int(s * 0.92)], fill=fg)
        d.rectangle([int(s * 0.08), a, int(s * 0.92), b], fill=fg)
    elif kind == "grid_dots":
        step = s / float(n)
        r = step * 0.28
        for i in range(n):
            for j in range(n):
                cx, cy = (i + 0.5) * step, (j + 0.5) * step
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg)
    elif kind == "nested":
        step = s / float(2 * n)
        for i in range(n):
            o = i * step
            d.rectangle([o, o, s - o, s - o],
                        outline=fg, width=max(2, int(step * 0.35)))
    elif kind == "burst":
        cx = cy = s / 2.0
        for i in range(n):
            a0 = 2.0 * np.pi * i / n
            a1 = a0 + np.pi / n
            d.polygon([(cx, cy),
                       (cx + s * np.cos(a0), cy + s * np.sin(a0)),
                       (cx + s * np.cos(a1), cy + s * np.sin(a1))], fill=fg)
    else:
        raise ValueError(f"unknown image probe kind {kind!r}")

    return np.asarray(img, dtype=np.uint8)


SYNTHESIZERS = {
    "f0": synthesize_melody,
    "energy": synthesize_energy,
    "chroma": synthesize_chroma,
    "rhythm": synthesize_rhythm,
    "text": synthesize_text,
    "image": synthesize_image,
    "chord": synthesize_chroma,
    "midi": synthesize_midi,
}


class ConditionProbeSet:
    def __init__(self, condition, directory, names, targets, sr, duration_s,
                 specs=None, tokens=None, tok_len=None):
        self.condition = str(condition)
        self.dir = str(directory)
        self.names = list(names)
        self.targets = list(targets)
        self.sr = int(sr)
        self.duration_s = float(duration_s)
        self.specs = list(specs) if specs is not None else []
        self.tokens = tokens
        self.tok_len = tok_len
        self.is_global = is_global_probe(condition)

    def __len__(self):
        return len(self.targets)

    def _stem(self, i: int) -> str:
        return os.path.join(self.dir, f"probe_{i:02d}_{self.names[i]}")

    def wav_path(self, i: int) -> str:
        if self.condition == "text":
            raise TypeError("a text probe has no waveform; use .text(i)")
        if self.condition == "image":
            raise TypeError("an image probe has no waveform; use .image_path(i)")
        return self._stem(i) + ".wav"

    def wav(self, i: int) -> np.ndarray:
        import soundfile as sf
        y, _ = sf.read(self.wav_path(i), dtype="float32")
        return y

    def image_path(self, i: int) -> str:
        if self.condition != "image":
            raise TypeError(f"{self.condition} probes are not images")
        return self._stem(i) + ".png"

    def image(self, i: int) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(self.image_path(i)).convert("RGB"))

    def text(self, i: int) -> str:
        if self.condition == "text" and i < len(self.specs):
            return str(self.specs[i])
        return self.names[i]

    def context(self, i: int):
        if self.tokens is None or i >= len(self.tokens):
            return None
        tok = torch.from_numpy(np.asarray(self.tokens[i], dtype=np.float32))
        n = int(self.tok_len[i])
        mask = torch.zeros(tok.shape[0], dtype=torch.bool)
        mask[:n] = True
        return {"tokens": tok.unsqueeze(0), "mask": mask.unsqueeze(0)}

    def label(self, i: int) -> str:
        return self.text(i)


def _synth_fingerprint(condition: str) -> dict:
    src = inspect.getsource(SYNTHESIZERS[condition])
    helpers = "" if is_global_probe(condition) else "".join(
        inspect.getsource(f) for f in
        (_harmonic_tone, _fade, _normalize, _midi_to_hz))
    if condition in SYMBOLIC_TARGETS:
        helpers += "".join(inspect.getsource(f) for f in
                           (midi_probe_events, SYMBOLIC_TARGETS[condition],
                            render_midi_events))
    return {"synth_sha1": hashlib.sha1((src + helpers).encode()).hexdigest()}


def _fingerprint(condition, n_probes, n_frames, sr, duration_s, extractor,
                 bank=None):
    ex = {}
    for k in sorted(dir(extractor)):
        if k.startswith("_") or k == "device":
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
    geometry = ({"n_frames": None, "sr": None, "duration_s": None}
                if is_global_probe(condition) else
                {"n_frames": int(n_frames), "sr": int(sr),
                 "duration_s": round(float(duration_s), 6)})
    bank = PROBE_BANKS[condition] if bank is None else bank
    payload = {
        "condition": condition,
        "n_probes": int(n_probes),
        "stimuli": json.loads(json.dumps(list(bank)[:n_probes])),
        "synth": _synth_fingerprint(condition),
        "extractor": ex,
        **geometry,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _build_global_probe_targets(condition, entries, synth, extractor,
                                stim_path, tag, verbose):
    targets = []
    if verbose:
        medium = "images" if condition == "image" else "prompts"
        print(f"{tag} building {len(entries)} {medium} and encoding "
              f"their targets...")
    for i, (name, spec) in enumerate(entries):
        stim = synth(spec)
        if condition == "image":
            from PIL import Image
            path = stim_path(i)
            Image.fromarray(np.asarray(stim, dtype=np.uint8)).save(path)
            emb = extractor.encode_image(path)
        else:
            emb = extractor.encode_text(str(stim))
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if not np.isfinite(emb).all():
            raise RuntimeError(
                f"{tag} stimulus {i:02d} ({name}) encoded to a non-finite "
                f"embedding. Refusing to build a probe set that would feed "
                f"NaN into the model.")
        targets.append(emb)

    if verbose and len(targets) > 1:
        M = np.stack(targets)
        S = M @ M.T
        keys = [json.dumps(spec, sort_keys=True) for _name, spec in entries]
        first = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        S[np.array([[a == b for b in keys] for a in keys])] = -np.inf
        uniq = sorted(first.values())
        U = S[np.ix_(uniq, uniq)]
        finite = U[np.isfinite(U)]
        for i, (name, _spec) in enumerate(entries):
            if first[keys[i]] != i:
                print(f"  [{i:02d}] {name:<20s} repeat of "
                      f"[{first[keys[i]]:02d}]")
                continue
            if not np.isfinite(S[i]).any():
                print(f"  [{i:02d}] {name:<20s} dim={targets[i].shape[0]} "
                      f"(the only distinct stimulus)")
                continue
            j = int(np.argmax(S[i]))
            flag = ("  <-- nearly identical to its neighbour: these two "
                    "stimuli drive the model with the same condition"
                    if S[i, j] > 0.95 else "")
            print(f"  [{i:02d}] {name:<20s} dim={targets[i].shape[0]} "
                  f"nearest={S[i, j]:+.3f} ({entries[j][0]}){flag}")
        if finite.size:
            print(f"{tag} bank spread: mean pairwise cosine "
                  f"{float(finite.mean()):+.3f}, max "
                  f"{float(finite.max()):+.3f} "
                  f"(lower = the stimuli are better separated)"
                  + (f", over {len(uniq)} distinct stimuli"
                     if len(uniq) < len(entries) else ""))
    return targets


PROBE_SUBSET_SEED = 0


def probe_subset_indices(bank_size: int, n: int) -> list:
    bank_size = int(bank_size)
    n = max(1, min(int(n), bank_size))
    if n >= bank_size:
        return list(range(bank_size))
    rng = np.random.default_rng(PROBE_SUBSET_SEED)
    return sorted(int(i) for i in rng.choice(bank_size, size=n, replace=False))


def build_condition_probe_set(condition, probe_dir, n_frames, extractor,
                              n_probes: int = 16, duration_s: float = 5.0,
                              sr: Optional[int] = None, force: bool = False,
                              verbose: bool = True,
                              bank=None) -> ConditionProbeSet:
    sr = _lc.active_sr(sr)
    if condition not in PROBE_BANKS:
        raise ValueError(f"no probe bank for condition '{condition}'. "
                         f"Available: {sorted(PROBE_BANKS)}.")
    bank = list(PROBE_BANKS[condition] if bank is None else bank)
    if not bank:
        raise ValueError(f"empty probe bank for condition '{condition}'")
    tag = f"[{condition}-probe]"
    if int(n_probes) > len(bank) and verbose:
        print(f"{tag} asked for {int(n_probes)} stimuli, the bank holds "
              f"{len(bank)} -> using all {len(bank)}")
    full_size = len(bank)
    bank = [bank[i] for i in probe_subset_indices(full_size, n_probes)]
    n_probes = len(bank)
    probe_dir = str(probe_dir)
    if n_probes < max(full_size, len(PROBE_BANKS[condition])):
        probe_dir = os.path.join(probe_dir, f"subset_{n_probes:02d}")
    os.makedirs(probe_dir, exist_ok=True)
    meta_path = os.path.join(probe_dir, "meta.json")
    npz_path = os.path.join(probe_dir, "targets.npz")
    fp = _fingerprint(condition, n_probes, n_frames, sr, duration_s, extractor,
                      bank)
    names = [name for name, _ in bank[:n_probes]]
    specs = [spec for _, spec in bank[:n_probes]]
    ext = {"image": ".png"}.get(condition, None if condition == "text" else ".wav")

    def _stim_path(i):
        return os.path.join(probe_dir, f"probe_{i:02d}_{names[i]}{ext}")

    if not force and os.path.exists(meta_path) and os.path.exists(npz_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("fingerprint") == fp:
                data = np.load(npz_path)
                targets = [data[f"probe_{i:02d}"] for i in range(n_probes)]
                tok = data["tok"] if "tok" in data.files else None
                tlen = data["tok_len"] if "tok_len" in data.files else None
                stale_text = (condition == "text" and tok is None)
                if stale_text and verbose:
                    print(f"{tag} cache predates the token sequences "
                          f"-> rebuilding")
                if not stale_text and (ext is None
                                       or all(os.path.exists(_stim_path(i))
                                              for i in range(n_probes))):
                    if verbose:
                        print(f"{tag} cache hit: {n_probes} stimuli "
                              f"from {probe_dir}")
                    return ConditionProbeSet(condition, probe_dir, names,
                                             targets, sr, duration_s,
                                             specs=specs, tokens=tok,
                                             tok_len=tlen)
                if verbose:
                    print(f"{tag} cache metadata matches but the stimulus "
                          f"files are missing -> rebuilding")
            elif verbose:
                print(f"{tag} configuration changed -> rebuilding")
        except Exception as e:
            print(f"{tag} unreadable cache ({e}) -> rebuilding")

    synth = SYNTHESIZERS[condition]

    if is_global_probe(condition):
        targets = _build_global_probe_targets(
            condition, bank[:n_probes], synth, extractor, _stim_path, tag,
            verbose)
        tok = tlen = None
        if condition == "text" and hasattr(extractor, "encode_tokens"):
            try:
                tok, tlen = extractor.encode_tokens([str(sp) for sp in specs])
                if verbose:
                    print(f"{tag} token sequences: {tok.shape[0]} x "
                          f"{tok.shape[1]} token(s) x {tok.shape[2]} "
                          f"(lengths {int(tlen.min())}..{int(tlen.max())})")
            except Exception as e:
                print(f"{tag} token sequences NOT built "
                      f"({type(e).__name__}: {e}); a cross-attention run "
                      f"will drive this probe with the null token.")
                tok = tlen = None
        extra = ({} if tok is None else {"tok": tok, "tok_len": tlen})
        np.savez(npz_path, **{f"probe_{i:02d}": t
                              for i, t in enumerate(targets)}, **extra)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"fingerprint": fp, "condition": condition,
                       "names": names,
                       "prompts": ([str(s) for s in specs]
                                   if condition == "text" else None),
                       "dim": int(targets[0].shape[0]) if targets else 0},
                      fh, indent=2)
        if verbose:
            print(f"{tag} saved to {probe_dir}")
        return ConditionProbeSet(condition, probe_dir, names, targets, sr,
                                 duration_s, specs=specs, tokens=tok,
                                 tok_len=tlen)

    import soundfile as sf
    if verbose:
        print(f"{tag} building {n_probes} elementary stimuli "
              f"({duration_s:.1f}s each) and extracting their targets...")

    targets = []
    symbolic = SYMBOLIC_TARGETS.get(condition)
    for i, (name, spec) in enumerate(bank[:n_probes]):
        y = synth(spec, sr=sr, duration_s=duration_s)
        sf.write(os.path.join(probe_dir, f"probe_{i:02d}_{name}.wav"), y, sr)
        if symbolic is not None:
            tgt = np.asarray(extractor.from_notes(symbolic(spec, duration_s),
                                                  n_frames), dtype=np.float32)
        else:
            tgt = np.asarray(extractor.extract(y, sr, n_frames),
                             dtype=np.float32)
        targets.append(tgt)
        if verbose:
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


def _fig_to_tensor(fig) -> torch.Tensor:
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
                       fps: Optional[float] = None,
                       pad_octaves: float = 1.0,
                       corr_name: str = "corr") -> torch.Tensor:
    fps = _lc.active_fps(fps)
    kw = CONDITION_CONFIG.get("frame_level", {}).get("f0", {}).get("kwargs", {})
    fmin = float(kw.get("fmin", 50.0))
    fmax = float(kw.get("fmax", 1000.0))

    tgt_name = f"f0_{kind}" if kind in ("probe", "test") else "f0_valid"
    C_TGT, C_GEN, C_WARN = "#1f77b4", "#e8710a", "#c0392b"

    t_hz = f0_norm_to_hz(target)
    g_hz = f0_norm_to_hz(generated)
    n = min(len(t_hz), len(g_hz))
    t_hz, g_hz = t_hz[:n], g_hz[:n]
    time = np.arange(n) / float(fps)

    tv, gv = t_hz > 0, g_hz > 0
    t_plot = np.where(tv, t_hz, np.nan)
    g_plot = np.where(gv, g_hz, np.nan)

    fig = plt.figure(figsize=(7.2, 3.9), dpi=120)
    gs = fig.add_gridspec(2, 1, height_ratios=[7, 1], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(time, t_plot, color=C_TGT, lw=2.6,
            label=f"{tgt_name}  (target / condition)")
    ax.plot(time, g_plot, color=C_GEN, lw=2.0, alpha=0.95,
            label="f0_gen  (re-extracted from the generation)")

    if tv.any():
        lo = max(fmin, float(t_hz[tv].min()) / (2.0 ** pad_octaves))
        hi = min(fmax, float(t_hz[tv].max()) * (2.0 ** pad_octaves))
    else:
        lo, hi = fmin, fmax
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
        sub.append(f"{corr_name}={corr:+.3f}")
    ax.set_title(head, fontsize=17, fontweight="bold",
                 pad=28 if sub else 10)
    if sub:
        ax.text(0.5, 1.012, "   ·   ".join(sub), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11, color="#555555")

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

    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2,
               frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, -0.16))

    img = _fig_to_tensor(fig)
    plt.close(fig)
    return img


def plot_midi_comparison(target, generated, kind="valid", label="", step=None,
                         prefix=None, guidance=None, score=None,
                         fps: Optional[float] = None, dpi: int = 130,
                         score_name: str = None) -> torch.Tensor:
    fps = _lc.active_fps(fps)
    from matplotlib.colors import ListedColormap
    from conditions import MIDI_N_KEYS, MIDI_LOWEST_KEY, MIDI_DRUM_CLASSES
    n, d0 = MIDI_N_KEYS, 2 * MIDI_N_KEYS
    nd = len(MIDI_DRUM_CLASSES)
    tgt = np.asarray(target, dtype=np.float32)
    gen = np.asarray(generated, dtype=np.float32)
    m = min(len(tgt), len(gen))
    tgt, gen = tgt[:m], gen[:m]

    used = np.flatnonzero((tgt[:, :n] > 0.5).any(0) | (gen[:, :n] > 0.5).any(0))
    lo = max(0, int(used.min()) - 3) if used.size else 39 - 12
    hi = min(n, int(used.max()) + 4) if used.size else 39 + 12

    cmap = ListedColormap(["#ffffff", "#9ecae1", "#08306b", "#d62728"])
    t_end = m / float(fps)
    ticks = [k for k in range(lo, hi) if (k + MIDI_LOWEST_KEY) % 12 == 0]
    names = ["C%d" % ((k + MIDI_LOWEST_KEY) // 12 - 1) for k in ticks]
    fig, axes = plt.subplots(
        4, 1, figsize=(10, 8.0), dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [4, 1.3, 4, 1.3], "hspace": 0.08})
    for (ax_k, ax_d), r, what in (((axes[0], axes[1]), tgt, "target (MIDI)"),
                                  ((axes[2], axes[3]), gen,
                                   "generated (YourMT3+)")):
        keys = np.where(r[:, n:d0] > 0.5, 2, np.where(r[:, :n] > 0.5, 1, 0))
        ax_k.imshow(keys[:, lo:hi].T, aspect="auto", origin="lower",
                    cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                    extent=[0, t_end, lo - 0.5, hi - 0.5])
        ax_k.set_yticks(ticks)
        ax_k.set_yticklabels(names, fontsize=7)
        ax_k.set_ylabel(what, fontsize=9)
        drums = np.where(r[:, d0:] > 0.5, 3, 0)
        ax_d.imshow(drums.T, aspect="auto", origin="lower", cmap=cmap,
                    vmin=0, vmax=3, interpolation="nearest",
                    extent=[0, t_end, -0.5, nd - 0.5])
        ax_d.set_yticks(range(nd))
        ax_d.set_yticklabels([c for c, _ in MIDI_DRUM_CLASSES], fontsize=6)
        ax_d.set_facecolor("#f4f4f4")
    axes[3].set_xlabel("time (s)")
    head = _title("midi", kind, label, step, prefix, guidance, score)
    if score_name:
        head = head.replace("score=", f"{score_name}=")
    axes[0].set_title(head, fontsize=10)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.07)
    img = _fig_to_tensor(fig)
    plt.close(fig)
    return img


def plot_condition_comparison(condition, target, generated, kind="valid",
                              label="", step=None, prefix=None, guidance=None,
                              score=None, fps: Optional[float] = None,
                              dpi: int = 130,
                              score_name: str = None) -> torch.Tensor:
    fps = _lc.active_fps(fps)
    if condition == "midi":
        return plot_midi_comparison(target, generated, kind=kind, label=label,
                                    step=step, prefix=prefix,
                                    guidance=guidance, score=score, fps=fps,
                                    dpi=dpi, score_name=score_name)
    if condition == "f0":
        return plot_f0_comparison(target, generated, kind=kind, label=label,
                                  step=step, prefix=prefix, guidance=guidance,
                                  corr=score, fps=fps,
                                  corr_name=score_name or "corr")

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


def main():
    import argparse
    from conditions import ConditionRegistry

    ap = argparse.ArgumentParser(
        description="Build and inspect an out-of-the-box condition probe set.")
    ap.add_argument("condition", choices=sorted(PROBE_BANKS))
    ap.add_argument("probe_dir")
    ap.add_argument("--n_frames", type=int, default=None,
                    help="frames per chunk (dataset_meta.json: "
                         "latents_frames_per_chunk). Required for the frame "
                         "conditions; ignored for text / image, whose target "
                         "is one embedding and does not depend on the geometry.")
    ap.add_argument("--n_probes", type=int, default=16)
    ap.add_argument("--duration_s", type=float, default=5.0)
    ap.add_argument("--codec", choices=["dac", "encodec"],
                    default="dac",
                    help="the dataset's codec (its dataset_meta.json): the "
                         "synthesis sample rate and the extractors' frame "
                         "grid follow it, as in the training")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    _lc.activate({"dac": "dac_44khz", "encodec": "encodec_32khz"}[args.codec])

    global_probe = is_global_probe(args.condition)
    if not global_probe and args.n_frames is None:
        ap.error(f"--n_frames is required for '{args.condition}' "
                 f"(its target is a per-frame curve)")

    if global_probe:
        reg = ConditionRegistry(enabled_frame=[],
                                enabled_global=[args.condition])
        extractor = reg.global_extractors[args.condition]
    else:
        reg = ConditionRegistry(enabled_frame=[args.condition],
                                enabled_global=[])
        extractor = reg.frame_extractors[args.condition]
    for attr in ("device", "_device"):
        if hasattr(extractor, attr):
            try:
                setattr(extractor, attr, args.device)
            except Exception:
                pass

    ps = build_condition_probe_set(
        args.condition, args.probe_dir, args.n_frames or 0, extractor,
        n_probes=args.n_probes, duration_s=args.duration_s, force=args.force)
    print(f"\n{len(ps)} stimuli in {ps.dir}")
    for i, name in enumerate(ps.names):
        if args.condition == "text":
            print(f"  [{i:02d}] {name:<20s} {ps.text(i)!r}")
        elif args.condition == "image":
            print(f"  [{i:02d}] {name:<20s} {ps.image_path(i)}")
        else:
            print(f"  [{i:02d}] {name:<20s} {ps.wav_path(i)}")


if __name__ == "__main__":
    main()
