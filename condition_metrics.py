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
    EnergyExtractor,
    CrepeF0Extractor,
    CONDITION_CONFIG,
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


# ============================================================
# ENERGY / DYNAMICS  -- Pearson correlation of the loudness curve
# ============================================================
def energy_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: float = DAC_FRAMES_PER_S) -> dict:
    """Pearson correlation between the target dynamics curve and the one
    re-extracted from the generation. This is exactly how Music ControlNet
    (Wu et al. 2024, arXiv:2311.07069) evaluates dynamics adherence. The curve
    is single-channel, shape (n_frames, 1)."""
    return {
        "corr": _pearson(target[:, 0], generated[:, 0]),
    }


# ============================================================
# F0  -- pitch-contour adherence + voicing agreement
# ============================================================
def f0_fidelity(target: np.ndarray, generated: np.ndarray,
                fps: float = DAC_FRAMES_PER_S) -> dict:
    """
    Adherence of the monophonic f0 contour (CrepeF0Extractor output). Channel 0
    is the normalized (log-)pitch with 0 == unvoiced; a possible channel 1 is
    the periodicity (ignored here). Two numbers, in the spirit of the melody
    metrics but for a continuous 1-D contour:
      - corr:        Pearson correlation of the pitch curve on the frames the
                     TARGET marks as voiced (where a pitch is actually expected);
      - voicing_acc: fraction of frames where target and generation agree on
                     voiced/unvoiced (both > 0 or both == 0).
    """
    t = target[:, 0]
    g = generated[:, 0]
    tv = t > 0
    gv = g > 0
    voiced_corr = _pearson(t[tv], g[tv]) if int(tv.sum()) >= 2 else float("nan")
    voicing_acc = float((tv == gv).mean()) if tv.size else float("nan")
    return {"corr": voiced_corr, "voicing_acc": voicing_acc}


FIDELITY_FNS = {
    "melody": melody_fidelity,
    "chroma": chroma_fidelity,
    "rhythm": rhythm_fidelity,
    "energy": energy_fidelity,
    "f0":     f0_fidelity,
}


# ============================================================
# INFLUENCE PANEL  -- consolidated TensorBoard text table
# ============================================================
def format_influence_panel(influence: dict, step: int, prefix: str = "EMA",
                           guidance: float = 1.0, n_samples: int = 0,
                           coverage: dict = None) -> str:
    """
    Render the per-condition influence dict as a single Markdown table for
    TensorBoard's add_text. The table contains ONLY the conditions active in the
    current run (it is built from whatever keys `influence` has), so it adapts
    automatically as conditions are added or removed.

    `influence` layout:
        { condition_name: { metric_name: {"cond": float|None,
                                          "null": float|None,
                                          "delta": float|None,
                                          "note": str (optional)} } }

    `coverage` (optional), from ConditionFidelityEvaluator.coverage():
        { "cond/metric": {"valid": int, "attempted": int,
                          "non_finite": int, "extract_errors": int} }
    Rendered as a "valid/attempted" column: the mean is only meaningful together
    with how many samples reached it. Failures (silence, degenerate curves,
    extraction errors) yield non-finite values that are excluded from the mean,
    so a score computed on few survivors can look BETTER than one computed on all.

    Δ = with-cond - null. All current metrics are higher-is-better, so Δ > 0
    means the condition pulled the generation toward its target.
    """
    def fmt(x):
        return f"{x:+.4f}" if isinstance(x, (int, float)) else "n/a"

    def fmt_plain(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"

    def fmt_cov(cname, metric):
        if not coverage:
            return "—"
        c = coverage.get(f"{cname}/{metric}")
        if not c:
            return "—"
        s = f"{c['valid']}/{c['attempted']}"
        bad = c.get("non_finite", 0) + c.get("extract_errors", 0)
        return s + (f" ⚠️{bad}" if bad else "")

    # Markdown (TensorBoard renders markdown but NOT raw HTML). No big '###'
    # heading -> lighter/smaller; the explanatory legend is a separate tag.
    lines = []
    lines.append(f"**Condition influence — step {step}** · "
                 f"_{prefix} · guidance={guidance} · {n_samples} samples_")
    lines.append("")
    lines.append("| Condition | Metric | with-cond | null | Δ influence | valid/used |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for cname in influence:
        for metric, vals in influence[cname].items():
            note = vals.get("note")
            if note:
                lines.append(f"| `{cname}` | {metric} | n/a | n/a | _{note}_ | "
                             f"{fmt_cov(cname, metric)} |")
            else:
                lines.append(
                    f"| `{cname}` | {metric} | {fmt_plain(vals.get('cond'))} | "
                    f"{fmt_plain(vals.get('null'))} | {fmt(vals.get('delta'))} | "
                    f"{fmt_cov(cname, metric)} |"
                )
    if coverage:
        incomplete = {k: c for k, c in coverage.items()
                      if c["valid"] < c["attempted"]}
        if incomplete:
            lines.append("")
            lines.append("⚠️ _Some samples did not contribute: the mean is over "
                         "the valid ones only, so it does NOT describe the "
                         "failures (degenerate/silent generations, extraction "
                         "errors). Read it together with valid/used._")
    return "\n".join(lines)


def format_influence_legend() -> str:
    """One-off legend for the Condition_influence panel, logged ONCE to a
    separate TensorBoard text tag (its own box), not in every per-step table.
    Markdown only (TensorBoard does not render raw HTML)."""
    return (
        "**How to read the Condition influence panel**\n\n"
        "- **with-cond** — adherence to the target when the condition is given "
        "to the model.\n"
        "- **null** — baseline adherence when the model generates freely "
        "(no condition); the chance level.\n"
        "- **Δ influence** = with-cond − null — the net effect of the condition. "
        "Δ > 0 means it pulls the generation toward its target; near 0 means it "
        "is being ignored. Watch Δ rise as training progresses.\n"
        "- Ranges: RPA / RCA / chroma cosine ∈ [0, 1]; rhythm / energy "
        "correlation and CLAP cosine ∈ [−1, 1]. Compare each row over time "
        "rather than across rows (different scales)."
    )


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
    "energy": EnergyExtractor,
    "f0":     CrepeF0Extractor,
}


# ============================================================
# CONDITION SONIFICATION  -- render each condition to an audible waveform
# ============================================================
def sonify_energy(curve, sr, fps: float = DAC_FRAMES_PER_S,
                  carrier_hz: float = 220.0, amp: float = 0.3) -> np.ndarray:
    """
    Render an energy/dynamics curve (T, 1) in [0, 1] to an audible waveform: a
    steady carrier tone whose AMPLITUDE follows the curve, so the loudness shape
    (forte/piano, crescendo/diminuendo) can be *heard* and compared with the
    conditioned generation. The envelope is linearly interpolated up to the
    sample rate to avoid zipper artifacts. Returns float32, same length as the
    audio of the same number of frames.
    """
    env = np.asarray(curve, dtype=np.float32).reshape(-1)     # (T,)
    T = len(env)
    spf = max(1, int(round(sr / float(fps))))
    n = T * spf
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, n, dtype=np.float64)
    env_up = np.interp(x_new, x_old, env).astype(np.float32)  # smooth envelope
    t = np.arange(n, dtype=np.float64) / float(sr)
    carrier = np.sin(2.0 * np.pi * carrier_hz * t).astype(np.float32)
    return (amp * env_up * carrier).astype(np.float32)


# name -> callable(array, sr, fps) -> waveform (float32). Conditions without a
# sonifier (e.g. chroma) are simply skipped. Easy to extend (e.g. rhythm clicks).
def sonify_f0(arr: np.ndarray, sr: int,
              fps: float = DAC_FRAMES_PER_S, amp: float = 0.2) -> np.ndarray:
    """
    Render the f0 condition to an audible sine contour, so the pitch curve the
    model is being conditioned on can be *listened to* (the qualitative check
    that catches octave jumps, phantom pitch in silence, wrong voicing).

    Input is the CrepeF0Extractor output: (T,1) or (T,2) with channel 0 = the
    normalized pitch and 0 == unvoiced (channel 1, periodicity, is ignored).
    The mapping is INVERTED using the extractor's ACTUAL parameters read from
    CONDITION_CONFIG, so it stays in sync if fmin/fmax/voiced_floor are changed:
        pitch_norm = voiced_floor + (1 - voiced_floor) * (log2(f/fmin) / log2(fmax/fmin))
    Phase is carried across consecutive voiced frames (and reset on unvoiced) to
    avoid clicks. Returns float32 in [-amp, amp].
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    pn = arr[:, 0].astype(np.float64)

    kw = CONDITION_CONFIG.get("frame_level", {}).get("f0", {}).get("kwargs", {})
    fmin = float(kw.get("fmin", 50.0))
    fmax = float(kw.get("fmax", 1000.0))
    floor = float(kw.get("voiced_floor", 0.05))

    voiced = pn > 0.0
    lo, hi = np.log2(fmin), np.log2(fmax)
    # undo the [voiced_floor, 1] remap, then the log-scale normalization
    p = np.clip((pn - floor) / max(1.0 - floor, 1e-8), 0.0, 1.0)
    freqs = np.where(voiced, np.exp2(p * (hi - lo) + lo), 0.0)

    T = arr.shape[0]
    spf = max(1, int(round(sr / float(fps))))
    out = np.zeros(T * spf, dtype=np.float32)
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


SONIFY_FNS = {
    "melody": sonify_melody,
    "energy": sonify_energy,
    "f0":     sonify_f0,
}


def sonify_condition(name: str, arr, sr: int,
                     fps: float = DAC_FRAMES_PER_S):
    """Dispatch to the right sonifier for a condition; returns a waveform
    (float32) or None if that condition has no audible rendering."""
    fn = SONIFY_FNS.get(name)
    if fn is None:
        return None
    try:
        return fn(arr, sr, fps)
    except Exception as e:
        print(f"  [sonify] skipped '{name}': {e}")
        return None


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
                 fps: float = DAC_FRAMES_PER_S, registry=None):
        self.fps = fps
        self.device = device
        self.extractors = {}

        # Report #15: the fidelity metric must re-extract conditions from the
        # generated audio with the SAME extractor configuration used to build the
        # targets at preprocessing time. If the run's ConditionRegistry is passed,
        # reuse its already-instantiated extractors (exact params: f0
        # with_periodicity / thresholds, energy weighting, etc.). Only fall back
        # to default-constructed extractors when no registry is available.
        run_extractors = getattr(registry, "frame_extractors", None) or {}

        for name in enabled_frame:
            if name in run_extractors:
                self.extractors[name] = run_extractors[name]
                continue
            if name not in _EXTRACTOR_FNS:
                continue
            # Fallback defaults. MelodyExtractor / ChromaExtractor / EnergyExtractor
            # take no args (defaults match extraction time). RhythmExtractor
            # (beat_this) and CrepeF0Extractor (torchcrepe) accept a device and
            # are much faster on GPU; the device does not change their values.
            if name in ("rhythm", "f0"):
                self.extractors[name] = _EXTRACTOR_FNS[name](device=device)
            else:
                self.extractors[name] = _EXTRACTOR_FNS[name]()
        self.reset()

    def reset(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)
        # Coverage accounting. Without it the mean silently describes only the
        # samples that SUCCEEDED: a model that degenerates on the hard cases
        # (silence, constant curves, extraction failures) produces NaN/Inf there,
        # those are dropped, and the average IMPROVES because the worst cases
        # vanished. The mean must always be read together with how many samples
        # actually contributed.
        self._attempted = defaultdict(int)     # per metric key
        self._non_finite = defaultdict(int)    # per metric key
        self._extract_errors = defaultdict(int)  # per CONDITION name
        self._cond_attempted = defaultdict(int)  # per CONDITION name

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
            self._cond_attempted[name] += 1
            target = np.asarray(target_cond[name], dtype=np.float32)
            try:
                generated = extractor.extract(gen_wav_np, sr, n_frames)
            except Exception as e:
                # COUNTED, not just printed: a re-extraction that keeps failing
                # is a property of the generations, and must be visible in the
                # coverage instead of quietly shrinking the denominator.
                self._extract_errors[name] += 1
                print(f"  [fidelity] re-extraction failed for '{name}': {e}")
                continue
            metrics = FIDELITY_FNS[name](target, generated, self.fps)
            for k, v in metrics.items():
                key = f"{name}/{k}"
                self._attempted[key] += 1
                # isfinite, NOT `v == v`: the latter lets +Inf/-Inf through and a
                # single infinity destroys the mean.
                if np.isfinite(v):
                    self._sums[key] += float(v)
                    self._counts[key] += 1
                else:
                    self._non_finite[key] += 1

    def coverage(self) -> dict:
        """Per metric key: how many samples actually contributed to the mean, and
        why the others did not. Report it NEXT TO the mean -- a high score on 20%
        of the samples is not a better model, it is a smaller denominator."""
        out = {}
        for key in self._attempted:
            name = key.split("/")[0]
            attempted = self._cond_attempted.get(name, 0)
            out[key] = {
                "valid": self._counts.get(key, 0),
                "attempted": attempted,
                "non_finite": self._non_finite.get(key, 0),
                "extract_errors": self._extract_errors.get(name, 0),
            }
        # conditions whose extraction NEVER succeeded produce no metric key at
        # all: surface them instead of letting them disappear from the report.
        for name, n_err in self._extract_errors.items():
            if not any(k.startswith(name + "/") for k in out):
                out[f"{name}/<no metric>"] = {
                    "valid": 0,
                    "attempted": self._cond_attempted.get(name, 0),
                    "non_finite": 0,
                    "extract_errors": n_err,
                }
        return out

    def results(self) -> dict:
        """Average over the samples that contributed a finite value."""
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }
