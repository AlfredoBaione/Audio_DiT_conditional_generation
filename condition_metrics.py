# condition_metrics.py
#
# Conditioning-fidelity metrics for the conditioned Audio DiT.
#
# Idea (paired, NOT distributional):
#   FD-DAC / FAD compare two *distributions* (real vs generated) and need
#   aggregate statistics (mean / covariance). Conditioning fidelity is a
#   *paired* comparison instead: each conditioned generation was produced from
#   ONE specific input condition (a validation f0 contour / chroma / rhythm),
#   so we
#   re-extract that same condition FROM the generated audio and compare it
#   one-to-one with the input condition, then average over the (fixed) metrics
#   subset.
#
# Re-extraction reuses the SAME extractor classes that produced the stored
# targets (conditions.py), so the generated-side representation is identical in
# nature to the target-side one and the comparison is fair.
#
# Per-condition distance (literature-grounded):
#   - chroma : mean per-frame cosine similarity between target and generated
#              chromagram (the melody/harmony adherence metric of MusicGen-Melody,
#              Copet et al. 2023, arXiv:2306.05284).
#   - chord  : the same per-frame cosine, on crema's 12-d chord-pitch vectors
#              (CremaChordExtractor) instead of the chromagram.
#   - rhythm : Pearson correlation of the beat and downbeat probability curves
#              (robust, hyper-parameter free). The stricter beat F-measure
#              (peak-picking + mir_eval.beat, in the spirit of Music ControlNet,
#              Wu et al. 2023, arXiv:2311.07069) is the mir family's rhythm
#              scorer, rhythm_mir_fidelity.
#
# Two FAMILIES can score the same (target, re-extracted) pairs, chosen per run
# by metrics.influence_family:
#   influence_metrics      -> the distances above (FIDELITY_FNS), the default;
#   mir_influence_metrics  -> mir_eval (Raffel et al., ISMIR 2014), the standard
#                             MIR metrics, for the conditions it covers
#                             (MIR_FIDELITY_FNS: f0 -> mir_eval.melody,
#                             chroma and chord -> mir_eval.multipitch,
#                             rhythm -> mir_eval.beat). energy, which mir_eval
#                             has nothing for, keeps ours.
#
# This module is evaluation-only: it never touches the trained weights.

import copy
import functools
import warnings
from collections import defaultdict

import numpy as np

from conditions import (
    ChromaExtractor,
    RhythmExtractor,
    EnergyExtractor,
    CrepeF0Extractor,
    CremaChordExtractor,
    CONDITION_CONFIG,
    DAC_FRAMES_PER_S,
)


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
    the periodicity (ignored here). ONE number, in the spirit of the
    pitch-transcription metrics of the literature but for a continuous
    1-D contour:
      - corr: Pearson correlation of the pitch curve on the frames the TARGET
              marks as voiced (where a pitch is actually expected). Fewer than
              two such frames -> NaN, excluded from the mean.

    A voicing-agreement score used to be reported alongside it and was dropped:
    on material that is mostly unvoiced it rewards agreeing about SILENCE, so a
    model that generates nothing at all scores near 1. The voiced/unvoiced
    picture is in the ribbon of the comparison plot, where it cannot be read as
    a quality number.
    """
    t = target[:, 0]
    g = generated[:, 0]
    tv = t > 0
    voiced_corr = _pearson(t[tv], g[tv]) if int(tv.sum()) >= 2 else float("nan")
    return {"corr": voiced_corr}


FIDELITY_FNS = {
    "chroma": chroma_fidelity,
    "rhythm": rhythm_fidelity,
    "energy": energy_fidelity,
    "f0":     f0_fidelity,
    "chord":  chroma_fidelity,   # same 12-d per-frame cosine, crema's vectors
}


# ============================================================
# F0 (mir_eval)  -- the melody-extraction metrics of the MIR literature
# ============================================================
def f0_mir_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: float = DAC_FRAMES_PER_S) -> dict:
    """
    The same pair as f0_fidelity -- the f0 GIVEN to the model and the one CREPE
    re-extracts from what it generated -- scored with mir_eval.melody (Raffel
    et al., ISMIR 2014), the melody-extraction metrics of MIREX (Salamon et al.,
    IEEE Signal Processing Magazine 2014). The target is the reference, the
    generation the estimate.

    Both curves go back to Hz through f0_norm_to_hz, which keeps 0 for
    unvoiced: that is mir_eval's own convention, so the voicing needs no
    translation. They share the DAC time base (frame i at i / fps, from 0), so
    mir_eval resamples nothing and every frame is scored as it is.

      raw_pitch_accuracy   share of the target's voiced frames where the
                           generation has a pitch within 50 cents of it
      raw_chroma_accuracy  the same, with octave errors forgiven
      voicing_recall       share of the target's voiced frames that the
                           generation also voices
      voicing_false_alarm  share of the target's UNVOICED frames that the
                           generation voices -- LOWER is better
      overall_accuracy     share of ALL frames right: voiced with the right
                           pitch, or unvoiced on both sides

    UNDEFINED IS NaN, NOT mir_eval's PLACEHOLDER. A target with no voiced frame
    has no denominator for the first three, one with no unvoiced frame none for
    the false alarm, and on 5-second chunks both happen. mir_eval then returns a
    fixed value instead of a measurement (recall 1, pitch accuracies 0, false
    alarm 0): it would enter the mean as a score no generation earned, and tie
    the with-cond and the null pass at the same number, flattening Δ. NaN takes
    that sample out of that row and counts it in valid/used -- the treatment
    f0/corr gives a target with fewer than two voiced frames. On every defined
    case the value IS mir_eval's. overall_accuracy is always defined.

    mir_eval is imported here, so a run on the default family never needs it;
    ConditionFidelityEvaluator imports it at startup when this family is asked
    for, so a missing package stops the run there and not at a metrics step.
    """
    import mir_eval

    keys = ("raw_pitch_accuracy", "raw_chroma_accuracy", "voicing_recall",
            "voicing_false_alarm", "overall_accuracy")
    nan = float("nan")
    ref_hz = f0_norm_to_hz(target)
    est_hz = f0_norm_to_hz(generated)
    n = min(len(ref_hz), len(est_hz))
    if n == 0:
        return {k: nan for k in keys}
    ref_hz, est_hz = ref_hz[:n], est_hz[:n]
    times = np.arange(n, dtype=np.float64) / float(fps)
    with warnings.catch_warnings():
        # mir_eval warns on every chunk with no voiced frame on either side.
        # Those are exactly the cases handled below, so the warning would only
        # repeat what the NaN and valid/used already say.
        warnings.simplefilter("ignore")
        s = mir_eval.melody.evaluate(times, ref_hz, times, est_hz)
    n_voiced = int((ref_hz > 0).sum())
    has_voiced, has_unvoiced = n_voiced > 0, n_voiced < n
    return {
        "raw_pitch_accuracy":
            float(s["Raw Pitch Accuracy"]) if has_voiced else nan,
        "raw_chroma_accuracy":
            float(s["Raw Chroma Accuracy"]) if has_voiced else nan,
        "voicing_recall":
            float(s["Voicing Recall"]) if has_voiced else nan,
        "voicing_false_alarm":
            float(s["Voicing False Alarm"]) if has_unvoiced else nan,
        "overall_accuracy":
            float(s["Overall Accuracy"]),
    }


# ============================================================
# CHROMA / CHORD (mir_eval)  -- the multiple-F0 metrics, on pitch classes
# ============================================================
# One octave of pitch-class frequencies, C4 .. B4 (Hz). A 12-d vector carries
# no octave: each ON class is placed here only to hand mir_eval a frequency,
# and the metrics read are its CHROMA ones, which fold the octave anyway.
_PITCH_CLASS_HZ = 261.6255653005986 * 2.0 ** (np.arange(12) / 12.0)


def pitch_class_mir_fidelity(target: np.ndarray, generated: np.ndarray,
                             fps: float = DAC_FRAMES_PER_S,
                             threshold: float = 0.5) -> dict:
    """
    chroma and chord scored with mir_eval.multipitch (Raffel et al., ISMIR
    2014), the MIREX multiple-F0 metrics (Poliner & Ellis 2007; Bay et al.,
    ISMIR 2009). The target is the reference, the generation the estimate.

    mir_eval compares, frame by frame, the LIST of notes that are ON. Both
    conditions are 12 continuous values per frame, so a pitch class is ON when
    its value is >= `threshold` (metrics.mir_threshold, 0.5): for chroma, at
    least that fraction of the frame's loudest class (ChromaExtractor scales
    every frame so its loudest class is 1); for chord, a crema probability of
    at least that. The same threshold on both sides, target and generation.

    Hits, extra and missing classes are summed over the frames of the chunk
    first, then divided (mir_eval's own averaging):
      chroma_precision          classes ON in both, per class ON in the
                                generation
      chroma_recall             classes ON in both, per class ON in the target
      chroma_accuracy           hits / (hits + extra + missing)
      chroma_miss_error         target classes left uncovered because the
                                generation has FEWER classes ON, per target
                                class -- LOWER is better
      chroma_false_alarm_error  classes ON in EXCESS of the target's count, per
                                target class -- LOWER is better; can exceed 1

    UNDEFINED IS NaN, as for f0: no class ON in the target anywhere (a chord
    target silent or chordless throughout) leaves recall and both errors
    without a denominator; none in the generation leaves precision without
    one; none on either side, accuracy. mir_eval would put a 0 there. On every
    defined case the value IS mir_eval's.
    """
    import mir_eval

    keys = ("chroma_precision", "chroma_recall", "chroma_accuracy",
            "chroma_miss_error", "chroma_false_alarm_error")
    nan = float("nan")
    t_on = np.asarray(target, dtype=np.float64) >= threshold
    g_on = np.asarray(generated, dtype=np.float64) >= threshold
    n = min(len(t_on), len(g_on))
    if n == 0:
        return {k: nan for k in keys}
    t_on, g_on = t_on[:n, :12], g_on[:n, :12]
    times = np.arange(n, dtype=np.float64) / float(fps)
    with warnings.catch_warnings():
        # "frequencies are all empty": the cases handled below.
        warnings.simplefilter("ignore")
        s = mir_eval.multipitch.evaluate(
            times, [_PITCH_CLASS_HZ[row] for row in t_on],
            times, [_PITCH_CLASS_HZ[row] for row in g_on])
    n_ref, n_est = int(t_on.sum()), int(g_on.sum())
    return {
        "chroma_precision":
            float(s["Chroma Precision"]) if n_est else nan,
        "chroma_recall":
            float(s["Chroma Recall"]) if n_ref else nan,
        "chroma_accuracy":
            float(s["Chroma Accuracy"]) if (n_ref or n_est) else nan,
        "chroma_miss_error":
            float(s["Chroma Miss Error"]) if n_ref else nan,
        "chroma_false_alarm_error":
            float(s["Chroma False Alarm Error"]) if n_ref else nan,
    }


# ============================================================
# RHYTHM (mir_eval)  -- the beat-tracking metrics, on beat INSTANTS
# ============================================================
# beat_this's own "minimal" postprocessing (beat_this/model/postprocessor.py),
# which is how beat_this itself turns its curves into beat times: a frame is a
# beat when it is the maximum of its window AND its probability is above 0.5.
# The window is max_pool1d(kernel 7) at beat_this's 50 frames/s: 7 frames of
# 20 ms, the "+-70 ms" its code comment names, i.e. +-3 frames = +-60 ms
# between frame centres. That time radius is what is reproduced here, on the
# curves as the project stores them (resampled to the DAC frames): +-5 DAC
# frames at 86.13 frames/s, and exactly beat_this's +-3 frames at 50.
_BEAT_THIS_FPS = 50.0
_BEAT_THIS_PEAK_HALF = 3              # frames at _BEAT_THIS_FPS
_BEAT_THIS_PEAK_PROB = 0.5            # logit > 0


def _peak_times(curve: np.ndarray, fps: float) -> np.ndarray:
    """(T,) beat or downbeat probability -> (n,) times in seconds, by
    beat_this's minimal rule: maximum of its +-60 ms window, probability > 0.5,
    and adjacent peak frames (a plateau) merged into their average."""
    c = np.asarray(curve, dtype=np.float64).reshape(-1)
    if c.size == 0:
        return np.zeros(0)
    half = max(1, int(round(_BEAT_THIS_PEAK_HALF * float(fps) / _BEAT_THIS_FPS)))
    padded = np.pad(c, half, mode="constant", constant_values=-np.inf)
    local_max = np.lib.stride_tricks.sliding_window_view(
        padded, 2 * half + 1).max(axis=1)
    frames = np.flatnonzero((c == local_max) & (c > _BEAT_THIS_PEAK_PROB))
    # beat_this's deduplicate_peaks(width=1), verbatim in behaviour: a frame
    # joins the current group when it is within 1 frame of the group's RUNNING
    # MEAN, and the group becomes that mean.
    merged = []                        # [running mean, count]
    for f in frames:
        if merged and f - merged[-1][0] <= 1:
            merged[-1][1] += 1
            merged[-1][0] += (f - merged[-1][0]) / merged[-1][1]
        else:
            merged.append([float(f), 1])
    return np.array([m for m, _ in merged], dtype=np.float64) / float(fps)


def rhythm_beat_times(arr: np.ndarray, fps: float = DAC_FRAMES_PER_S):
    """(T, 2) rhythm condition -> (beat times, downbeat times) in seconds.
    Each downbeat is moved onto the nearest beat, as beat_this does."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    beats = _peak_times(a[:, 0], fps)
    downs = _peak_times(a[:, 1], fps) if a.shape[1] > 1 else np.zeros(0)
    if beats.size and downs.size:
        downs = np.unique(beats[np.abs(beats[None, :] - downs[:, None])
                                .argmin(axis=1)])
    return beats, downs


def rhythm_mir_fidelity(target: np.ndarray, generated: np.ndarray,
                        fps: float = DAC_FRAMES_PER_S) -> dict:
    """
    The rhythm condition -- beat_this's beat and downbeat probability curves
    -- scored with mir_eval.beat (Raffel et al., ISMIR 2014), the MIREX
    beat-tracking metrics (Davies et al. 2009). The target is the reference,
    the generation the estimate.

    mir_eval compares beat INSTANTS, while the condition is two curves per
    frame: both are turned into instants with beat_this's own rule
    (rhythm_beat_times: maximum within +-60 ms, probability > 0.5, downbeats
    moved onto the nearest beat). Then, with mir_eval's default parameters:

      beat_f_measure      beats within +-70 ms of a target beat (F-measure)
      beat_cemgil         the same, weighted by closeness (Gaussian, 40 ms)
      beat_p_score        correlation of the two beat trains, within 20% of the
                          target's median beat interval
      beat_cmlt           share of beats right at the target's own metrical
                          level (phase and tempo within 17.5%)
      beat_amlt           the same, also accepting double / half tempo and the
                          off-beat
      downbeat_f_measure  the bar starts, F-measure within +-70 ms

    The individual metric functions are called, NOT mir_eval.beat.evaluate:
    evaluate() first drops every beat before 5 s (trim_beats), which on a 5 s
    chunk is all of them.

    UNDEFINED IS NaN, as for f0 and chroma: a target with no beat (or fewer than
    two, for P-score and the continuity scores, which need a beat interval) has
    nothing to be scored against, and mir_eval would put a 0 there. A
    GENERATION with no beat where the target has them keeps mir_eval's 0: that
    is a real failure, not a missing measurement. On every defined case the
    value IS mir_eval's.
    """
    import mir_eval

    ref_b, ref_d = rhythm_beat_times(target, fps)
    est_b, est_d = rhythm_beat_times(generated, fps)
    nan = float("nan")
    with warnings.catch_warnings():
        # "beats are empty" / "only one beat": the cases handled below.
        warnings.simplefilter("ignore")
        f = mir_eval.beat.f_measure(ref_b, est_b)
        cemgil, _ = mir_eval.beat.cemgil(ref_b, est_b)
        p = mir_eval.beat.p_score(ref_b, est_b)
        _cmlc, cmlt, _amlc, amlt = mir_eval.beat.continuity(ref_b, est_b)
        f_down = mir_eval.beat.f_measure(ref_d, est_d)
    one, two = ref_b.size >= 1, ref_b.size >= 2
    return {
        "beat_f_measure":     float(f) if one else nan,
        "beat_cemgil":        float(cemgil) if one else nan,
        "beat_p_score":       float(p) if two else nan,
        "beat_cmlt":          float(cmlt) if two else nan,
        "beat_amlt":          float(amlt) if two else nan,
        "downbeat_f_measure": float(f_down) if ref_d.size else nan,
    }


# The mir_eval family (metrics.influence_family: mir_influence_metrics): one
# entry per condition it scores. A condition WITHOUT an entry -- energy, which
# mir_eval has nothing for -- keeps its FIDELITY_FNS function under either
# family (see fidelity_fn).
MIR_FIDELITY_FNS = {
    "f0":     f0_mir_fidelity,
    "chroma": pitch_class_mir_fidelity,
    "chord":  pitch_class_mir_fidelity,   # same 12 pitch classes, crema's
    "rhythm": rhythm_mir_fidelity,
}

INFLUENCE_FAMILIES = ("influence_metrics", "mir_influence_metrics")


def fidelity_fn(name: str, family: str = "influence_metrics",
                mir_threshold: float = 0.5):
    """(function, family actually used) for frame condition `name` under the
    requested `family`. Only the mir family can fall back: a condition mir_eval
    does not cover is scored with ours, and the second element says so. The
    pitch-class scorer (chroma, chord) comes back with `mir_threshold`
    (metrics.mir_threshold) bound as its ON threshold."""
    if family == "mir_influence_metrics" and name in MIR_FIDELITY_FNS:
        fn = MIR_FIDELITY_FNS[name]
        if fn is pitch_class_mir_fidelity:
            fn = functools.partial(fn, threshold=float(mir_threshold))
        return fn, "mir_influence_metrics"
    return FIDELITY_FNS[name], "influence_metrics"


# Rows where a SMALLER value is the better one; every other row is
# higher-is-better. For these a NEGATIVE Δ means the condition helped, so the
# tables mark them with ↓ and the sign is never read backwards.
LOWER_IS_BETTER = {"voicing_false_alarm", "chroma_miss_error",
                   "chroma_false_alarm_error"}


def _metric_label(metric: str) -> str:
    return f"{metric} ↓" if metric in LOWER_IS_BETTER else metric


# ============================================================
# PAIRED INFLUENCE  -- with-cond vs null on the SAME samples
# ============================================================
def pair_influence(per_sample_cond: dict, per_sample_null: dict,
                   coverage_cond: dict = None, have_null: bool = True):
    """
    Build the influence table from PAIRED samples.

    Why pairing is not optional: a generation can be too degenerate to measure
    (silence, flat curve, extractor failure). Those samples yield NaN and are
    excluded from the mean -- correctly, but NOT symmetrically: the conditioned
    and the null pass do not fail on the same generations. Averaging each pass
    over "whatever survived in that pass" and subtracting gives a delta between
    two means computed on two DIFFERENT sample sets, and a change in that delta
    can come entirely from a change in the denominators. So, per metric, keep
    only the sample ids that produced a finite value on BOTH sides and average
    with-cond, null AND delta over exactly that set.

    Args:
        per_sample_cond / per_sample_null: {metric key: {sample id: value}},
            from ConditionFidelityEvaluator.per_sample().
        coverage_cond: the conditioned pass's coverage(), used for `attempted`
            and to carry the extraction-failure counts into the panel.
        have_null: False when no null pass was generated at all
            (sampling.metrics_uncond=false). There is then nothing to pair
            against, so the with-cond mean is reported unpaired and the delta is
            n/a -- never silently as if it had been paired.

    Returns:
        (influence, coverage) in the shape format_influence_panel expects, with
        `valid` = the PAIRED count and `unpaired` = how many samples were
        dropped because only one of the two sides could be measured.
    """
    coverage_cond = coverage_cond or {}
    influence, coverage = {}, {}

    for key in sorted(set(per_sample_cond) | set(per_sample_null)):
        c = per_sample_cond.get(key, {})
        n = per_sample_null.get(key, {})
        name, _, metric = key.partition("/")
        cov_src = coverage_cond.get(key, {})

        if have_null:
            common = sorted(set(c) & set(n))
            unpaired = len(set(c) ^ set(n))
            if common:
                cm = sum(c[i] for i in common) / len(common)
                nm = sum(n[i] for i in common) / len(common)
                vals = {"cond": cm, "null": nm, "delta": cm - nm}
            else:
                vals = {"cond": None, "null": None, "delta": None}
            valid = len(common)
        else:
            unpaired = 0
            vals = ({"cond": sum(c.values()) / len(c), "null": None, "delta": None}
                    if c else {"cond": None, "null": None, "delta": None})
            valid = len(c)

        influence.setdefault(name, {})[metric] = vals
        coverage[key] = {
            "valid": valid,
            "attempted": cov_src.get("attempted", max(len(c), len(n))),
            "unpaired": unpaired,
            "non_finite": cov_src.get("non_finite", 0),
            "extract_errors": cov_src.get("extract_errors", 0),
        }
    return influence, coverage


def pair_scalar(per_sample_cond: dict, per_sample_null: dict,
                have_null: bool = True):
    """Same pairing for a single scalar per sample (the CLAP audio<->text
    similarity). Returns (cond, null, delta, n_paired); values are None when
    nothing could be paired."""
    c, n = per_sample_cond or {}, per_sample_null or {}
    if not have_null:
        m = (sum(c.values()) / len(c)) if c else None
        return m, None, None, len(c)
    common = sorted(set(c) & set(n))
    if not common:
        return None, None, None, 0
    cm = sum(c[i] for i in common) / len(common)
    nm = sum(n[i] for i in common) / len(common)
    return cm, nm, cm - nm, len(common)


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

    `coverage` (optional), from pair_influence():
        { "cond/metric": {"valid": int, "attempted": int, "unpaired": int,
                          "non_finite": int, "extract_errors": int} }
    Rendered as a "valid/attempted" column: the mean is only meaningful together
    with how many samples reached it. Failures (silence, degenerate curves,
    extraction errors) yield non-finite values that are excluded from the mean,
    so a score computed on few survivors can look BETTER than one computed on all.
    `valid` is the PAIRED count and `unpaired` the samples measurable on only one
    of the two sides, which the pairing had to drop.

    Δ = with-cond - null. Every metric is higher-is-better except those in
    LOWER_IS_BETTER, marked ↓ in the Metric column: Δ > 0 means the condition
    pulled the generation toward its target, and on a ↓ row Δ < 0 does.
    with-cond, null and Δ are averaged over the SAME samples (see
    pair_influence), so the three columns always agree: Δ is exactly the
    difference of the two shown means.
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
        # `valid` is the PAIRED count: samples measurable on BOTH the with-cond
        # and the null generation, i.e. the ones all three columns are averaged
        # over. The warning counts everything that did not make it there.
        s = f"{c['valid']}/{c['attempted']}"
        bad = (c.get("non_finite", 0) + c.get("extract_errors", 0)
               + c.get("unpaired", 0))
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
            label = _metric_label(metric)
            if note:
                lines.append(f"| `{cname}` | {label} | n/a | n/a | _{note}_ | "
                             f"{fmt_cov(cname, metric)} |")
            else:
                lines.append(
                    f"| `{cname}` | {label} | {fmt_plain(vals.get('cond'))} | "
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



def format_influence_matrix(entries: list, step: int, prefix: str = "EMA",
                            guidance: float = 1.0, n_samples: int = 0,
                            extra: dict = None, extra_coverage: dict = None,
                            always_given: set = None) -> str:
    """
    Render SEVERAL condition subsets as one panel: a compact Δ matrix on top,
    the full per-subset tables underneath.

    This is the multi-condition counterpart of format_influence_panel. The
    question it answers is not "does the conditioning work" but "what does each
    COMBINATION of conditions actually buy", which needs the subsets side by
    side: a single table per subset, stacked, cannot be read that way -- the
    eye has to compare numbers pages apart.

    `extra` / `extra_coverage`: rows that are NOT a subset of the validation
    conditions and so cannot be a row of the matrix -- the probe rows. They are
    rendered as their own table under it. Without this they would be silently
    dropped whenever subsets and probes are enabled together, since the matrix
    is built from `entries` alone.

    `entries`: [(label, influence, coverage)], in the order they should appear.
        `label` names the subset ("all", "no_chroma", "only_f0", "f0+energy");
        `influence` and `coverage` are exactly what pair_influence returns for
        the generations of THAT subset, scored against the SAME null pass.

    The matrix holds Δ only (with-cond minus null). Δ is the whole point here:
    the absolute with-cond score of a subset says little on its own, while Δ
    says how far that combination pulled the generation toward the target,
    against a baseline shared by every row -- which is what makes the rows
    comparable at all.

    COLUMNS ARE EVERY ACTIVE CONDITION, NOT ONLY THE ONES IN THE SUBSET. The
    interesting numbers are exactly the off-subset cells: giving f0 alone and
    watching what happens to chroma is how a condition's side effects show up.
    A cell whose condition was not part of that row's subset is marked with a
    degree sign, so "conditioned on it" and "measured anyway" never get
    confused when reading the table.

    `always_given`: conditions handed to the model in EVERY row, which the
    subset labels therefore say nothing about -- the GLOBAL ones (text, image).
    The subsets vary the frame conditions only, so without this a row labelled
    "only_f0" would mark text and image as side effects when in fact both
    conditioned that generation.
    """
    def fmt_delta(x):
        return f"{x:+.4f}" if isinstance(x, (int, float)) else "n/a"

    # Column order: every (condition, metric) seen in any entry, condition
    # first-seen order preserved so f0 does not jump around between steps.
    cols = []
    for _lab, infl, _cov in entries:
        for cname in infl:
            for metric in infl[cname]:
                if (cname, metric) not in cols:
                    cols.append((cname, metric))

    lines = []
    lines.append(f"**Condition influence by subset — step {step}** · "
                 f"_{prefix} · guidance={guidance} · {n_samples} samples_")
    lines.append("")
    if cols:
        lines.append("| Conditions given | "
                     + " | ".join(f"`{c}`/{_metric_label(m)}" for c, m in cols)
                     + " |")
        lines.append("|---|" + "---:|" * len(cols))
        pinned = set(always_given or ())
        for label, infl, _cov in entries:
            given = _subset_names_of(label, infl, always_given=pinned)
            cells = []
            for cname, metric in cols:
                vals = infl.get(cname, {}).get(metric)
                txt = fmt_delta(vals.get("delta")) if vals else "n/a"
                # ° marks a condition that was NOT given for this row but was
                # measured anyway -- a side effect, not an adherence.
                if (given is not None and cname not in given
                        and cname not in pinned and txt != "n/a"):
                    txt += "°"
                cells.append(txt)
            lines.append(f"| **{label}** | " + " | ".join(cells) + " |")
        lines.append("")
        # The ↓ clause only when a lower-is-better column is actually there, so
        # a table without one reads exactly as it always did.
        _lower = any(m in LOWER_IS_BETTER for _c, m in cols)
        lines.append("_Δ = with-cond − null, higher is better"
                     + (" (lower for the ↓ metrics)" if _lower else "")
                     + ". ° = this condition was NOT given to the model in that "
                     "row; the number is a side effect of the others._")

    # ---- probe rows: their own table, not a row of the matrix ----
    # A probe is not a subset of the validation conditions -- it is a different
    # STIMULUS -- so putting it in the matrix would invite reading its delta
    # against the subset rows, which are measured on other material entirely.
    if extra:
        lines.append("")
        lines.append("---")
        lines.append("**out-of-the-box probes** "
                     "_(synthetic stimuli, not the validation set)_")
        lines.append("")
        lines.extend(format_influence_panel(
            extra, step, prefix=prefix, guidance=guidance,
            n_samples=n_samples, coverage=extra_coverage).split('\n')[1:])

    # ---- the detailed tables, one per subset, under the matrix ----
    for label, infl, cov in entries:
        lines.append("")
        lines.append(f"---")
        lines.append(f"**{label}**")
        lines.append("")
        body = format_influence_panel(infl, step, prefix=prefix,
                                      guidance=guidance, n_samples=n_samples,
                                      coverage=cov)
        # drop the per-table header line: the matrix header above already says
        # step/prefix/guidance, and repeating it once per subset is noise.
        lines.extend(body.split("\n")[1:])
    return "\n".join(lines)


def _subset_names_of(label: str, influence: dict, always_given: set = None):
    """
    The condition names a subset label says were GIVEN, or None when the label
    does not encode them (then no cell is marked as a side effect).

    Labels are produced by resolve_influence_subsets in training_cond.py:
        "all"          -> every active condition   (nothing is a side effect)
        "only_<name>"  -> that one
        "no_<name>"    -> everything except that one
        "a+b+c"        -> exactly those

    `always_given` names the conditions handed to the model in every row (the
    global ones): they are added back because the labels describe the FRAME
    subsets only, and a "no_<name>" label must not subtract them from the set of
    keys either.
    """
    pinned = set(always_given or ())
    if label == "all":
        return None
    if label.startswith("only_"):
        return {label[len("only_"):]} | pinned
    if label.startswith("no_"):
        return (set(influence.keys()) | pinned) - {label[len("no_"):]}
    if "+" in label:
        return set(label.split("+")) | pinned
    return {label} | pinned

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
        "is being ignored. Watch Δ rise as training progresses. On a row marked "
        "↓ (lower is better) it is the other way round: there Δ < 0 is the "
        "pull.\n"
        "- **Metric** — WHAT is being correlated, per condition. Every row is a "
        "similarity between the condition GIVEN to the model and the same "
        "descriptor RE-EXTRACTED from the audio it generated, then averaged over "
        "the samples counted in valid/used.\n"
        "  - `f0/corr` and `f0_probe/corr` — Pearson correlation between the "
        "target pitch curve and the one CREPE re-extracts from the generation, "
        "computed ONLY on the frames the TARGET marks as voiced (the frames where "
        "a pitch is actually expected). Fewer than two such frames -> that sample "
        "is NaN and drops out of the mean. It scores the SHAPE of the contour, not "
        "its absolute height: a generation an octave off but following the same "
        "contour still correlates high.\n"
        "  - With `metrics.influence_family: mir_influence_metrics` f0 is scored "
        "instead with mir_eval.melody, the standard melody-extraction metrics "
        "(MIREX), on the same two curves: the target is the reference, the "
        "generation the estimate. `raw_pitch_accuracy` — share of the target's "
        "voiced frames where the generation's pitch is within 50 cents (a "
        "quarter tone) of it; `raw_chroma_accuracy` — the same, octave errors "
        "forgiven; `voicing_recall` — share of the target's voiced frames the "
        "generation also voices; `voicing_false_alarm` ↓ — share of the "
        "target's UNVOICED frames the generation voices, lower is better; "
        "`overall_accuracy` — share of ALL frames right (right pitch, or "
        "unvoiced on both sides). Overall accuracy rewards agreeing on silence "
        "as much as on pitch: on a mostly unvoiced target it is high even for a "
        "generation that plays nothing, so read it next to raw pitch. A sample "
        "whose target has no voiced frame has no value in the first three (one "
        "with no unvoiced frame none in the false alarm): it drops out of those "
        "rows and is counted in valid/used.\n"
        "  - `energy/corr` and `rhythm/corr` — the same Pearson correlation, over "
        "all frames of the envelope / onset curve.\n"
        "  - `chroma/cosine` — mean cosine between the 12-d chroma vectors; "
        "`chord/cosine` — the same, on crema's 12-d chord-pitch vectors; "
        "`text/clap_cos` — CLAP audio-text cosine.\n"
        "  - With `metrics.influence_family: mir_influence_metrics` chroma and "
        "chord are scored instead with mir_eval.multipitch, the MIREX "
        "multiple-F0 metrics, on the pitch classes that are ON in each frame: "
        "ON means a value ≥ `metrics.mir_threshold` (0.5) — for chroma at "
        "least half the energy of the frame's loudest class, for chord a "
        "crema probability of at least one half. Hits, extra and missing "
        "classes are summed over the chunk. `chroma_precision` — share of the "
        "classes ON in the generation that are ON in the target too; "
        "`chroma_recall` — share of the target's ON classes the generation "
        "also has ON; `chroma_accuracy` — the two at once, hits / (hits + "
        "extra + missing); `chroma_miss_error` ↓ — target classes left "
        "uncovered because the generation has fewer classes ON, per target "
        "class; `chroma_false_alarm_error` ↓ — classes ON in excess of the "
        "target's count, per target class (it can exceed 1). On material "
        "without notes (drums, noise) chroma turns 6–7 classes of 12 ON, so "
        "there these rows measure agreement on noise. A sample whose target "
        "has no class ON at all (a chord target silent throughout) has no "
        "recall and no errors; one whose generation has none, no precision.\n"
        "  - With `metrics.influence_family: mir_influence_metrics` rhythm is "
        "scored instead with mir_eval.beat, the MIREX beat-tracking metrics, on "
        "beat INSTANTS: both curves are turned into instants with beat_this's "
        "own rule (a frame is a beat when it is the maximum within ±60 ms -- "
        "the window beat_this calls ±70 ms -- and its probability is above "
        "0.5; each downbeat is moved onto the nearest beat). `beat_f_measure` — beats within ±70 ms of a target beat; "
        "`beat_cemgil` — the same, weighted by how close (Gaussian, 40 ms); "
        "`beat_p_score` — correlation of the two beat trains within 20% of the "
        "target's beat interval; `beat_cmlt` — share of beats right at the "
        "target's own tempo and phase (17.5% tolerance); `beat_amlt` — the "
        "same, also accepting double / half tempo and the off-beat; "
        "`downbeat_f_measure` — the bar starts, within ±70 ms. A 5 s chunk "
        "holds a handful of beats and one or two bar starts, so the continuity "
        "scores and the downbeat F-measure are coarse per sample. A sample "
        "whose target has no beat (fewer than two, for P-score and the "
        "continuity scores; no bar start, for the downbeat row) has no value "
        "there; a generation with no beat where the target has them scores "
        "0.\n"
        "- Ranges: chroma / chord cosine and the mir_eval rows ∈ [0, 1], "
        "except `chroma_false_alarm_error` (≥ 0, no upper bound); rhythm / "
        "energy correlation and CLAP cosine ∈ [−1, 1]. Compare each row over "
        "time rather than across rows (different scales).\n"
        "- **valid/used** — how many generations the row is actually averaged "
        "over, out of how many were attempted. A generation too degenerate to "
        "measure (silence, flat curve, extractor failure) is excluded. The three "
        "value columns are averaged over the SAME samples: only those measurable "
        "on both the with-cond and the null generation count, so Δ can never be "
        "an artefact of the two sides having different denominators. ⚠️ is how "
        "many samples did not make it into that set. Read a rising Δ together "
        "with valid/used: if the denominator is collapsing at the same time, the "
        "row is describing fewer and fewer generations."
    )


_EXTRACTOR_FNS = {
    "chroma": ChromaExtractor,
    "rhythm": RhythmExtractor,
    "energy": EnergyExtractor,
    "f0":     CrepeF0Extractor,
    "chord":  CremaChordExtractor,
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


# SONIFY_FNS below maps name -> callable(array, sr, fps) -> waveform (float32).
# Every frame condition has one (chord reuses chroma's pad: the same 12 pitch
# classes), so every condition a run is trained on can be heard beside the
# generation it produced. A condition added later without a sonifier is simply
# skipped by sonify_condition rather than failing.
def f0_norm_to_hz(arr: np.ndarray) -> np.ndarray:
    """
    INVERSE of the CrepeF0Extractor normalization: (T,) or (T,1) or (T,2) of
    normalized pitch -> (T,) of Hz, with 0 kept as "unvoiced".

    The extractor maps a voiced frame as
        pitch_norm = voiced_floor + (1 - voiced_floor) * log2(f/fmin)/log2(fmax/fmin)
    and reserves 0 exclusively for unvoiced. This undoes both steps, reading
    fmin / fmax / voiced_floor from CONDITION_CONFIG rather than hardcoding them,
    so it cannot drift if those are retuned. Channel 1 (periodicity), when
    present, is ignored.

    Single source of truth for every place that has to turn the stored condition
    back into physical pitch: the sonifier below and the f0 comparison plots
    (probe_conditions.py). Two independent inversions would be two chances
    to disagree.
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
    return np.where(voiced, np.exp2(p * (hi - lo) + lo), 0.0)


def sonify_f0(arr: np.ndarray, sr: int,
              fps: float = DAC_FRAMES_PER_S, amp: float = 0.2) -> np.ndarray:
    """
    Render the f0 condition to an audible sine contour, so the pitch curve the
    model is being conditioned on can be *listened to* (the qualitative check
    that catches octave jumps, phantom pitch in silence, wrong voicing).

    Input is the CrepeF0Extractor output: (T,1) or (T,2) with channel 0 = the
    normalized pitch and 0 == unvoiced (channel 1, periodicity, is ignored).
    The normalized->Hz inversion is f0_norm_to_hz (shared with the comparison
    plots). Phase is carried across consecutive voiced frames (and reset on
    unvoiced) to avoid clicks. Returns float32 in [-amp, amp].
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    freqs = f0_norm_to_hz(arr)

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


def sonify_chroma(arr: np.ndarray, sr: int,
                  fps: float = DAC_FRAMES_PER_S, amp: float = 0.25,
                  base_hz: float = 261.625565) -> np.ndarray:
    """
    Render a chromagram (T, 12) to an audible pad, so the harmony the model is
    being conditioned on can be *listened to* next to the generation it
    produced -- the qualitative check that catches a chroma target that is
    smeared, transposed, or stuck on one class.

    Twelve fixed sine partials, one per pitch class starting at `base_hz` (C4),
    each scaled by that class's weight interpolated up to the sample rate. The
    frequencies are constant, so phase is continuous by construction and there
    are no clicks at frame boundaries; the interpolation is what keeps the
    weights from stepping audibly.

    Octave information is deliberately absent -- a chromagram does not carry it,
    and inventing a voicing would make the rendering say more than the condition
    does. The sum is normalized by the per-frame total weight so a dense frame
    does not simply come out louder than a sparse one: what should be audible is
    WHICH classes are present, not how many.
    """
    ch = np.asarray(arr, dtype=np.float32)
    if ch.ndim == 1:
        ch = ch[:, None]
    T, K = ch.shape
    spf = max(1, int(round(sr / float(fps))))
    n = T * spf
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, n, dtype=np.float64)
    t = np.arange(n, dtype=np.float64) / float(sr)

    out = np.zeros(n, dtype=np.float64)
    for k in range(K):
        w = np.interp(x_new, x_old, ch[:, k].astype(np.float64))
        if not np.any(w > 1e-6):
            continue
        out += w * np.sin(2.0 * np.pi * (base_hz * (2.0 ** (k / 12.0))) * t)
    # Per-frame normalization: divide by the summed weight, not by the peak, so
    # a triad and a single note come out at comparable loudness.
    norm = np.interp(x_new, x_old,
                     np.maximum(ch.sum(axis=1).astype(np.float64), 1e-6))
    return (amp * out / norm).astype(np.float32)


def sonify_rhythm(arr: np.ndarray, sr: int,
                  fps: float = DAC_FRAMES_PER_S, amp: float = 0.35,
                  beat_hz: float = 1000.0,
                  downbeat_hz: float = 2000.0) -> np.ndarray:
    """
    Render a beat grid (T, 2) -- channel 0 beat probability, channel 1 downbeat
    -- to an audible click track, so the metre the model is being conditioned on
    can be *heard* against the generation.

    Peaks are picked rather than the curves being used as an envelope: a beat is
    an EVENT, and amplitude-modulating a tone by a smooth probability curve
    renders a pulsation that is much harder to compare with a real attack. Each
    peak becomes a short exponentially decaying sine burst; downbeats are higher
    and louder so the bar line is audible as such.

    The threshold is relative to the curve's own maximum, so a confident grid and
    a hesitant one both sonify: the point is to hear WHERE the beats are, and an
    absolute threshold would render silence for exactly the ambiguous material
    the probe exists to expose. A flat curve yields no peak and therefore no
    clicks, which is the honest rendering of "no grid here".
    """
    r = np.asarray(arr, dtype=np.float32)
    if r.ndim == 1:
        r = r[:, None]
    T = r.shape[0]
    spf = max(1, int(round(sr / float(fps))))
    out = np.zeros(T * spf, dtype=np.float32)

    click_len = min(int(0.05 * sr), T * spf)
    if click_len <= 0:
        return out
    tt = np.arange(click_len, dtype=np.float64) / float(sr)
    decay = np.exp(-tt * 60.0)

    for ch in range(min(2, r.shape[1])):
        curve = r[:, ch].astype(np.float64)
        peak = float(curve.max())
        if peak <= 1e-6:
            continue
        thr = 0.5 * peak
        hz = downbeat_hz if ch == 1 else beat_hz
        gain = amp if ch == 1 else amp * 0.6
        click = (gain * decay * np.sin(2.0 * np.pi * hz * tt)).astype(np.float32)
        for i in range(T):
            if curve[i] < thr:
                continue
            # strict local maximum, so a plateau fires once on its leading edge
            if i > 0 and curve[i - 1] > curve[i]:
                continue
            if i + 1 < T and curve[i + 1] >= curve[i]:
                continue
            s = i * spf
            e = min(s + click_len, out.size)
            out[s:e] += click[:e - s]
    return out


SONIFY_FNS = {
    "energy": sonify_energy,
    "f0":     sonify_f0,
    "chroma": sonify_chroma,
    "rhythm": sonify_rhythm,
    "chord":  sonify_chroma,
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
        results = evaluator.results()   # {"f0/corr": ..., "chroma/cosine": ...}

    `target_cond` is the dict of input conditions for that sample, as numpy
    arrays of shape (n_frames, dim) keyed by condition name (e.g. "f0").
    Only the conditions in `enabled_frame` are evaluated.

    `family` (metrics.influence_family) picks what scores each condition:
    "influence_metrics" (ours, FIDELITY_FNS) or "mir_influence_metrics"
    (mir_eval, for the conditions in MIR_FIDELITY_FNS; the others keep ours).
    `mir_threshold` (metrics.mir_threshold) is the value at which a chroma or
    chord pitch class counts as ON for the mir family; ignored otherwise.
    """

    def __init__(self, enabled_frame, device: str = "cpu",
                 fps: float = DAC_FRAMES_PER_S, registry=None,
                 family: str = "influence_metrics",
                 mir_threshold: float = 0.5):
        if family not in INFLUENCE_FAMILIES:
            raise ValueError(f"influence family {family!r} is not one of "
                             f"{list(INFLUENCE_FAMILIES)}")
        if not 0.0 < float(mir_threshold) <= 1.0:
            raise ValueError(f"mir_threshold {mir_threshold!r} is not in "
                             f"(0, 1]")
        self.fps = fps
        self.device = device
        self.extractors = {}
        # Sample ids whose (target, generated) CURVES are to be kept, not just
        # the scalar they collapse into. The scalar answers "how close?", the
        # curves answer "close HOW?" -- which is what the TensorBoard comparison
        # plots show. Kept for a handful of ids only: the re-extraction already
        # happens for every scored sample, so this costs no extra work, but
        # holding every contour would grow with the generations measured.
        # Set with keep_contours_for(); empty = keep nothing (the default, so
        # every existing caller is unaffected).
        self._keep_ids = set()

        # Report #15: the fidelity metric must re-extract conditions from the
        # generated audio with the SAME extractor configuration used to build the
        # targets at preprocessing time. If the run's ConditionRegistry is passed,
        # reuse its already-instantiated extractors (exact params: f0
        # with_periodicity / thresholds, energy weighting, etc.). Only fall back
        # to default-constructed extractors when no registry is available.
        run_extractors = getattr(registry, "frame_extractors", None) or {}

        for name in enabled_frame:
            if name in run_extractors:
                # Keep the exact extractor parameters used to build the stored
                # targets, but use a shallow copy so the metrics-only runtime
                # device does not mutate the registry shared with the datasets.
                extractor = copy.copy(run_extractors[name])
                if name == "f0":
                    extractor.device = device
                elif name == "rhythm":
                    extractor._device = device
                self.extractors[name] = extractor
                continue
            if name not in _EXTRACTOR_FNS:
                continue
            # Fallback defaults. ChromaExtractor / EnergyExtractor
            # take no args (defaults match extraction time). RhythmExtractor
            # (beat_this) and CrepeF0Extractor (torchcrepe) accept a device and
            # are much faster on GPU; the device does not change their values.
            if name in ("rhythm", "f0"):
                self.extractors[name] = _EXTRACTOR_FNS[name](device=device)
            else:
                self.extractors[name] = _EXTRACTOR_FNS[name]()

        # Which function scores each condition, resolved ONCE from `family`
        # (see fidelity_fn); add_sample only looks it up. `families` records
        # what each condition actually got: under the mir family a condition
        # mir_eval does not cover keeps ours, and the startup log prints it so
        # a row is never read as the wrong kind of number.
        self.family = family
        self.fidelity_fns, self.families = {}, {}
        for name in self.extractors:
            self.fidelity_fns[name], self.families[name] = fidelity_fn(
                name, family, mir_threshold)
        if "mir_influence_metrics" in self.families.values():
            import mir_eval  # noqa: F401 -- fail at startup, not at a metrics step
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
        # PER-SAMPLE values ({metric key: {sample_id: finite value}}), kept so the
        # conditioned and the null pass can be compared on the SAME samples (see
        # pair_influence). The running sums above cannot do that: they collapse
        # each pass into a mean over whatever survived in THAT pass, and the two
        # passes do not survive on the same samples.
        self._per_sample = defaultdict(dict)
        self._auto_id = 0
        # {condition name: {sample_id: (target curve, generated curve)}}
        self._contours = defaultdict(dict)

    @property
    def active(self) -> bool:
        return len(self.extractors) > 0

    def keep_contours_for(self, sample_ids):
        """Ask for the raw (target, generated) curves of these sample ids to be
        retained by add_sample, on top of the usual scalar metrics. Survives
        reset() (it is a request, not accumulated state), so it can be set once
        and honoured by every subsequent pass -- which is what lets the
        conditioned and the null pass hand back curves for the SAME samples."""
        self._keep_ids = set(sample_ids or ())

    def contours(self, name: str = None):
        """The curves retained by keep_contours_for. With `name`, the dict for
        that condition ({sample_id: (target, generated)}); without, the whole
        {condition: {sample_id: ...}} mapping. Copies of the internal dicts, so
        a caller can hold them across the reset() of the next pass."""
        if name is not None:
            return dict(self._contours.get(name, {}))
        return {k: dict(v) for k, v in self._contours.items()}

    def add_sample(self, gen_wav_np: np.ndarray, sr: int, n_frames: int,
                   target_cond: dict, sample_id=None):
        """Re-extract every enabled condition from one generated waveform and
        accumulate its fidelity against the paired target condition.

        `sample_id` identifies WHICH generation this is, so the conditioned and
        the null pass can later be paired sample by sample (pair_influence).
        Pass the same id for the two generations that share a target; omit it to
        fall back to the call order."""
        if gen_wav_np.ndim > 1:
            gen_wav_np = gen_wav_np.squeeze()
        gen_wav_np = np.ascontiguousarray(gen_wav_np, dtype=np.float32)
        sid = self._auto_id if sample_id is None else sample_id
        self._auto_id += 1

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
            if sid in self._keep_ids:
                self._contours[name][sid] = (target.copy(),
                                             np.asarray(generated).copy())
            metrics = self.fidelity_fns[name](target, generated, self.fps)
            for k, v in metrics.items():
                key = f"{name}/{k}"
                self._attempted[key] += 1
                # isfinite, NOT `v == v`: the latter lets +Inf/-Inf through and a
                # single infinity destroys the mean.
                if np.isfinite(v):
                    self._sums[key] += float(v)
                    self._counts[key] += 1
                    self._per_sample[key][sid] = float(v)
                else:
                    self._non_finite[key] += 1

    def export_state(self) -> dict:
        """Everything add_sample has accumulated since the last reset(), as
        plain dicts. Multi-GPU: what a rank sends to rank 0, whose evaluator
        then merge_state()s it and ends up as if it had measured every sample
        itself. The curves kept for the plots are not included."""
        return {
            "sums": dict(self._sums),
            "counts": dict(self._counts),
            "attempted": dict(self._attempted),
            "non_finite": dict(self._non_finite),
            "extract_errors": dict(self._extract_errors),
            "cond_attempted": dict(self._cond_attempted),
            "per_sample": {k: dict(v) for k, v in self._per_sample.items()},
        }

    def merge_state(self, state: dict):
        """Add another evaluator's export_state() to this one. Their samples
        must be different ones (the ids are positions in one list)."""
        for name in ("sums", "counts", "attempted", "non_finite",
                     "extract_errors", "cond_attempted"):
            mine = getattr(self, "_" + name)
            for k, v in (state.get(name) or {}).items():
                mine[k] += v
        for k, d in (state.get("per_sample") or {}).items():
            self._per_sample[k].update(d)

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

    def per_sample(self) -> dict:
        """{metric key: {sample_id: value}} for the samples that produced a
        FINITE value. This is what pair_influence needs: `results()` has already
        thrown the sample identities away."""
        return {k: dict(v) for k, v in self._per_sample.items()}

    def results(self) -> dict:
        """Average over the samples that contributed a finite value.

        NB: unpaired -- the mean is over whatever survived in THIS pass. For the
        with-cond vs null comparison use pair_influence() instead."""
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }
