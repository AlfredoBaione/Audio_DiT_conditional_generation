# Condition influence: re-extraction, metrics (ours / mir_eval), the table.

import copy
import functools
import warnings
from collections import defaultdict

import numpy as np
from typing import Optional

import latent_codec as _lc

from conditions import (
    ChromaExtractor,
    RhythmExtractor,
    EnergyExtractor,
    CrepeF0Extractor,
    CremaChordExtractor,
    MidiExtractor,
    MIDI_N_KEYS,
    MIDI_LOWEST_KEY,
    MIDI_DRUM_CLASSES,
    midi_roll_to_events,
    CONDITION_CONFIG,
)


def chroma_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: Optional[float] = None) -> dict:
    fps = _lc.active_fps(fps)
    num = (target * generated).sum(axis=1)
    den = (np.linalg.norm(target, axis=1) * np.linalg.norm(generated, axis=1))
    valid = den > 1e-8
    if not np.any(valid):
        return {"cosine": float("nan")}
    cos = num[valid] / den[valid]
    return {"cosine": float(cos.mean())}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-8 else float("nan")


def rhythm_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: Optional[float] = None) -> dict:
    fps = _lc.active_fps(fps)
    return {
        "beat_corr":     _pearson(target[:, 0], generated[:, 0]),
        "downbeat_corr": _pearson(target[:, 1], generated[:, 1]),
    }


def energy_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: Optional[float] = None) -> dict:
    fps = _lc.active_fps(fps)
    return {
        "corr": _pearson(target[:, 0], generated[:, 0]),
    }


def f0_fidelity(target: np.ndarray, generated: np.ndarray,
                fps: Optional[float] = None) -> dict:
    fps = _lc.active_fps(fps)
    t = target[:, 0]
    g = generated[:, 0]
    tv = t > 0
    voiced_corr = _pearson(t[tv], g[tv]) if int(tv.sum()) >= 2 else float("nan")
    return {"corr": voiced_corr}


_MIDI_ONSET_SIGMA = 2.0


def _onset_corr(target: np.ndarray, generated: np.ndarray) -> float:
    from scipy.ndimage import gaussian_filter1d
    t = np.asarray(target, dtype=np.float64)
    g = np.asarray(generated, dtype=np.float64)
    if not np.any(t > 0.5):
        return float("nan")
    if not np.any(g > 0.5):
        return 0.0
    ts = gaussian_filter1d(t, _MIDI_ONSET_SIGMA, axis=0, mode="constant")
    gs = gaussian_filter1d(g, _MIDI_ONSET_SIGMA, axis=0, mode="constant")
    return _pearson(ts.ravel(), gs.ravel())


def midi_fidelity(target: np.ndarray, generated: np.ndarray,
                  fps: Optional[float] = None) -> dict:
    fps = _lc.active_fps(fps)
    n, d0 = MIDI_N_KEYS, 2 * MIDI_N_KEYS
    t = np.asarray(target, dtype=np.float64)
    g = np.asarray(generated, dtype=np.float64)
    m = min(len(t), len(g))
    t, g = t[:m], g[:m]
    ts, gs = t[:, :n], g[:, :n]
    nt, ng = np.linalg.norm(ts, axis=1), np.linalg.norm(gs, axis=1)
    active = (nt > 0) | (ng > 0)
    if np.any(active):
        den = nt[active] * ng[active]
        num = (ts[active] * gs[active]).sum(axis=1)
        cos = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        cosine = float(cos.mean())
    else:
        cosine = float("nan")
    return {"cosine": cosine,
            "onset_corr": _onset_corr(t[:, n:d0], g[:, n:d0]),
            "drum_corr": _onset_corr(t[:, d0:], g[:, d0:])}


FIDELITY_FNS = {
    "chroma": chroma_fidelity,
    "rhythm": rhythm_fidelity,
    "energy": energy_fidelity,
    "f0":     f0_fidelity,
    "chord":  chroma_fidelity,
    "midi":   midi_fidelity,
}


def f0_mir_fidelity(target: np.ndarray, generated: np.ndarray,
                    fps: Optional[float] = None) -> dict:
    fps = _lc.active_fps(fps)
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


_PITCH_CLASS_HZ = 261.6255653005986 * 2.0 ** (np.arange(12) / 12.0)


def pitch_class_mir_fidelity(target: np.ndarray, generated: np.ndarray,
                             fps: Optional[float] = None,
                             threshold: float = 0.5) -> dict:
    fps = _lc.active_fps(fps)
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


_BEAT_THIS_FPS = 50.0
_BEAT_THIS_PEAK_HALF = 3
_BEAT_THIS_PEAK_PROB = 0.5


def _peak_times(curve: np.ndarray, fps: float) -> np.ndarray:
    c = np.asarray(curve, dtype=np.float64).reshape(-1)
    if c.size == 0:
        return np.zeros(0)
    half = max(1, int(round(_BEAT_THIS_PEAK_HALF * float(fps) / _BEAT_THIS_FPS)))
    padded = np.pad(c, half, mode="constant", constant_values=-np.inf)
    local_max = np.lib.stride_tricks.sliding_window_view(
        padded, 2 * half + 1).max(axis=1)
    frames = np.flatnonzero((c == local_max) & (c > _BEAT_THIS_PEAK_PROB))
    merged = []
    for f in frames:
        if merged and f - merged[-1][0] <= 1:
            merged[-1][1] += 1
            merged[-1][0] += (f - merged[-1][0]) / merged[-1][1]
        else:
            merged.append([float(f), 1])
    return np.array([m for m, _ in merged], dtype=np.float64) / float(fps)


def rhythm_beat_times(arr: np.ndarray, fps: Optional[float] = None):
    fps = _lc.active_fps(fps)
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
                        fps: Optional[float] = None) -> dict:
    fps = _lc.active_fps(fps)
    import mir_eval

    ref_b, ref_d = rhythm_beat_times(target, fps)
    est_b, est_d = rhythm_beat_times(generated, fps)
    nan = float("nan")
    with warnings.catch_warnings():
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


def _transcription_prf(ref, est, pitch_of) -> tuple:
    import mir_eval
    nan = float("nan")
    if not ref and not est:
        return nan, nan, nan
    if not est:
        return nan, 0.0, 0.0
    if not ref:
        return 0.0, nan, nan

    def arrays(ev):
        on = np.array([e[0] for e in ev], dtype=np.float64)
        off = np.array([e[1] for e in ev], dtype=np.float64)
        iv = np.stack([on, np.maximum(off, on + 1e-3)], axis=1)
        hz = np.array([pitch_of(e) for e in ev], dtype=np.float64)
        return iv, hz

    ri, rp = arrays(ref)
    ei, ep = arrays(est)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, ei, ep, onset_tolerance=0.05, offset_ratio=None)
    return float(p), float(r), float(f)


def midi_mir_fidelity(target: np.ndarray, generated: np.ndarray,
                      fps: Optional[float] = None) -> dict:
    fps = _lc.active_fps(fps)
    from mir_eval.util import midi_to_hz
    t_notes, t_drums = midi_roll_to_events(target, fps)
    g_notes, g_drums = midi_roll_to_events(generated, fps)
    np_, nr, nf = _transcription_prf(t_notes, g_notes,
                                     lambda e: midi_to_hz(e[2]))
    as_note = [(o, o + 0.05, k) for o, k in t_drums]
    as_est = [(o, o + 0.05, k) for o, k in g_drums]
    dp, dr, df = _transcription_prf(as_note, as_est,
                                    lambda e: midi_to_hz(60 + e[2]))
    return {"note_precision": np_, "note_recall": nr, "note_f1": nf,
            "drum_precision": dp, "drum_recall": dr, "drum_f1": df}


MIR_FIDELITY_FNS = {
    "f0":     f0_mir_fidelity,
    "chroma": pitch_class_mir_fidelity,
    "chord":  pitch_class_mir_fidelity,
    "rhythm": rhythm_mir_fidelity,
    "midi":   midi_mir_fidelity,
}

INFLUENCE_FAMILIES = ("influence_metrics", "mir_influence_metrics")


def fidelity_fn(name: str, family: str = "influence_metrics",
                mir_threshold: float = 0.5):
    if family == "mir_influence_metrics" and name in MIR_FIDELITY_FNS:
        fn = MIR_FIDELITY_FNS[name]
        if fn is pitch_class_mir_fidelity:
            fn = functools.partial(fn, threshold=float(mir_threshold))
        return fn, "mir_influence_metrics"
    return FIDELITY_FNS[name], "influence_metrics"


LOWER_IS_BETTER = {"voicing_false_alarm", "chroma_miss_error",
                   "chroma_false_alarm_error"}


def _metric_label(metric: str) -> str:
    return f"{metric} ↓" if metric in LOWER_IS_BETTER else metric


def pair_influence(per_sample_cond: dict, per_sample_null: dict,
                   coverage_cond: dict = None, have_null: bool = True):
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


def format_influence_panel(influence: dict, step: int, prefix: str = "EMA",
                           guidance: float = 1.0, n_samples: int = 0,
                           coverage: dict = None) -> str:
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
        bad = (c.get("non_finite", 0) + c.get("extract_errors", 0)
               + c.get("unpaired", 0))
        return s + (f" ⚠️{bad}" if bad else "")

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
    def fmt_delta(x):
        return f"{x:+.4f}" if isinstance(x, (int, float)) else "n/a"

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
                if (given is not None and cname not in given
                        and cname not in pinned and txt != "n/a"):
                    txt += "°"
                cells.append(txt)
            lines.append(f"| **{label}** | " + " | ".join(cells) + " |")
        lines.append("")
        _lower = any(m in LOWER_IS_BETTER for _c, m in cols)
        lines.append("_Δ = with-cond − null, higher is better"
                     + (" (lower for the ↓ metrics)" if _lower else "")
                     + ". ° = this condition was NOT given to the model in that "
                     "row; the number is a side effect of the others._")

    if extra:
        lines.append("")
        lines.append("---")
        lines.append("**out-of-the-box probes** "
                     "_(synthetic stimuli, not the validation set)_")
        lines.append("")
        lines.extend(format_influence_panel(
            extra, step, prefix=prefix, guidance=guidance,
            n_samples=n_samples, coverage=extra_coverage).split('\n')[1:])

    for label, infl, cov in entries:
        lines.append("")
        lines.append(f"---")
        lines.append(f"**{label}**")
        lines.append("")
        body = format_influence_panel(infl, step, prefix=prefix,
                                      guidance=guidance, n_samples=n_samples,
                                      coverage=cov)
        lines.extend(body.split("\n")[1:])
    return "\n".join(lines)


def _subset_names_of(label: str, influence: dict, always_given: set = None):
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
    )


_EXTRACTOR_FNS = {
    "chroma": ChromaExtractor,
    "rhythm": RhythmExtractor,
    "energy": EnergyExtractor,
    "f0":     CrepeF0Extractor,
    "chord":  CremaChordExtractor,
    "midi":   MidiExtractor,
}


def sonify_energy(curve, sr, fps: Optional[float] = None,
                  carrier_hz: float = 220.0, amp: float = 0.3) -> np.ndarray:
    fps = _lc.active_fps(fps)
    env = np.asarray(curve, dtype=np.float32).reshape(-1)
    T = len(env)
    spf = max(1, int(round(sr / float(fps))))
    n = T * spf
    x_old = np.linspace(0.0, 1.0, T, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, n, dtype=np.float64)
    env_up = np.interp(x_new, x_old, env).astype(np.float32)
    t = np.arange(n, dtype=np.float64) / float(sr)
    carrier = np.sin(2.0 * np.pi * carrier_hz * t).astype(np.float32)
    return (amp * env_up * carrier).astype(np.float32)


def f0_norm_to_hz(arr: np.ndarray) -> np.ndarray:
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
    p = np.clip((pn - floor) / max(1.0 - floor, 1e-8), 0.0, 1.0)
    return np.where(voiced, np.exp2(p * (hi - lo) + lo), 0.0)


def sonify_f0(arr: np.ndarray, sr: int,
              fps: Optional[float] = None, amp: float = 0.2) -> np.ndarray:
    fps = _lc.active_fps(fps)
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
                  fps: Optional[float] = None, amp: float = 0.25,
                  base_hz: float = 261.625565) -> np.ndarray:
    fps = _lc.active_fps(fps)
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
    norm = np.interp(x_new, x_old,
                     np.maximum(ch.sum(axis=1).astype(np.float64), 1e-6))
    return (amp * out / norm).astype(np.float32)


def sonify_rhythm(arr: np.ndarray, sr: int,
                  fps: Optional[float] = None, amp: float = 0.35,
                  beat_hz: float = 1000.0,
                  downbeat_hz: float = 2000.0) -> np.ndarray:
    fps = _lc.active_fps(fps)
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
            if i > 0 and curve[i - 1] > curve[i]:
                continue
            if i + 1 < T and curve[i + 1] >= curve[i]:
                continue
            s = i * spf
            e = min(s + click_len, out.size)
            out[s:e] += click[:e - s]
    return out


def render_midi_events(pitched, drums, sr: int, n_samples: int,
                       amp: float = 0.3, seed: int = 4321) -> np.ndarray:
    """Notes and drum hits -> a waveform of exactly n_samples, to LISTEN to a
    midi condition (sonify_midi) and to a midi probe stimulus
    (probe_conditions.synthesize_midi).

    pitched: [(onset s, offset s, MIDI pitch)]; drums: [(onset s, class)] with
    the classes of conditions.MIDI_DRUM_CLASSES. A note is four harmonics at
    1/k with a 5 ms attack and an exponential decay, cut with a 20 ms release
    at its offset; a drum hit is a short burst coloured by its class -- a
    falling sine for kick and toms, noise for snare and cymbals, short for a
    closed hi-hat, long for crash and ride. Fixed-seed noise, so the same
    events always sound the same.

    No timbre is implied: the roll does not say which instrument plays, and
    nobody should hear a piano in it."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(n_samples), dtype=np.float64)
    atk, rel = max(1, int(0.005 * sr)), max(1, int(0.02 * sr))
    for on, off, pitch in pitched:
        s = int(round(on * sr))
        n = min(max(int(round((off - on) * sr)), rel + atk), out.size - s)
        if s < 0 or n <= 0:
            continue
        f = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        t = np.arange(n) / float(sr)
        tone = sum(np.sin(2.0 * np.pi * f * h * t) / h
                   for h in range(1, 5) if f * h < 0.45 * sr)
        env = np.exp(-t * 2.5)
        env[:atk] *= np.linspace(0.0, 1.0, atk)
        env[-rel:] *= np.linspace(1.0, 0.0, rel)
        out[s:s + n] += 0.5 * tone * env
    # (seconds of decay, falling-sine start Hz or None for noise, high-passed)
    drum_voice = {"kick": (0.15, 120.0, False), "snare": (0.12, None, False),
                  "hihat_closed": (0.04, None, True),
                  "hihat_open": (0.25, None, True),
                  "tom_low": (0.2, 100.0, False), "tom_mid": (0.2, 150.0, False),
                  "tom_high": (0.2, 220.0, False), "crash": (0.6, None, True),
                  "ride": (0.3, None, True)}
    for on, k in drums:
        s = int(round(on * sr))
        name = MIDI_DRUM_CLASSES[int(k)][0]
        dec, hz, bright = drum_voice[name]
        n = min(int(4 * dec * sr), out.size - s)
        if s < 0 or n <= 0:
            continue
        t = np.arange(n) / float(sr)
        env = np.exp(-t / dec)
        if hz is not None:
            # pitch falls by half over the hit (f = hz * (1 - t / 2T)): the
            # thump of a membrane
            T = n / float(sr)
            burst = np.sin(2.0 * np.pi * hz * (t - t * t / (4.0 * T)))
        else:
            burst = rng.standard_normal(n)
            if bright:
                burst = np.diff(burst, prepend=0.0)
        out[s:s + n] += burst * env
    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * amp
    return out.astype(np.float32)


def sonify_midi(arr: np.ndarray, sr: int,
                fps: Optional[float] = None) -> np.ndarray:
    fps = _lc.active_fps(fps)
    r = np.asarray(arr, dtype=np.float32)
    spf = max(1, int(round(sr / float(fps))))
    pitched, drums = midi_roll_to_events(r, fps)
    return render_midi_events(pitched, drums, sr, r.shape[0] * spf)


SONIFY_FNS = {
    "energy": sonify_energy,
    "f0":     sonify_f0,
    "chroma": sonify_chroma,
    "rhythm": sonify_rhythm,
    "chord":  sonify_chroma,
    "midi":   sonify_midi,
}


def sonify_condition(name: str, arr, sr: int,
                     fps: Optional[float] = None):
    fps = _lc.active_fps(fps)
    fn = SONIFY_FNS.get(name)
    if fn is None:
        return None
    try:
        return fn(arr, sr, fps)
    except Exception as e:
        print(f"  [sonify] skipped '{name}': {e}")
        return None


class ConditionFidelityEvaluator:
    def __init__(self, enabled_frame, device: str = "cpu",
                 fps: Optional[float] = None, registry=None,
                 family: str = "influence_metrics",
                 mir_threshold: float = 0.5):
        fps = _lc.active_fps(fps)
        if family not in INFLUENCE_FAMILIES:
            raise ValueError(f"influence family {family!r} is not one of "
                             f"{list(INFLUENCE_FAMILIES)}")
        if not 0.0 < float(mir_threshold) <= 1.0:
            raise ValueError(f"mir_threshold {mir_threshold!r} is not in "
                             f"(0, 1]")
        self.fps = fps
        self.device = device
        self.extractors = {}
        self._keep_ids = set()

        run_extractors = getattr(registry, "frame_extractors", None) or {}

        for name in enabled_frame:
            if name in run_extractors:
                extractor = copy.copy(run_extractors[name])
                if name in ("f0", "midi"):
                    extractor.device = device
                elif name == "rhythm":
                    extractor._device = device
                self.extractors[name] = extractor
                continue
            if name not in _EXTRACTOR_FNS:
                continue
            if name in ("rhythm", "f0", "midi"):
                self.extractors[name] = _EXTRACTOR_FNS[name](device=device)
            else:
                self.extractors[name] = _EXTRACTOR_FNS[name]()
        for extractor in self.extractors.values():
            if callable(getattr(extractor, "prepare", None)):
                extractor.prepare()

        self.family = family
        self.fidelity_fns, self.families = {}, {}
        for name in self.extractors:
            self.fidelity_fns[name], self.families[name] = fidelity_fn(
                name, family, mir_threshold)
        if "mir_influence_metrics" in self.families.values():
            import mir_eval
        self.reset()

    def reset(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)
        self._attempted = defaultdict(int)
        self._non_finite = defaultdict(int)
        self._extract_errors = defaultdict(int)
        self._cond_attempted = defaultdict(int)
        self._per_sample = defaultdict(dict)
        self._auto_id = 0
        self._contours = defaultdict(dict)

    @property
    def active(self) -> bool:
        return len(self.extractors) > 0

    def keep_contours_for(self, sample_ids):
        self._keep_ids = set(sample_ids or ())

    def contours(self, name: str = None):
        if name is not None:
            return dict(self._contours.get(name, {}))
        return {k: dict(v) for k, v in self._contours.items()}

    def add_sample(self, gen_wav_np: np.ndarray, sr: int, n_frames: int,
                   target_cond: dict, sample_id=None):
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
                if np.isfinite(v):
                    self._sums[key] += float(v)
                    self._counts[key] += 1
                    self._per_sample[key][sid] = float(v)
                else:
                    self._non_finite[key] += 1

    def export_state(self) -> dict:
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
        for name in ("sums", "counts", "attempted", "non_finite",
                     "extract_errors", "cond_attempted"):
            mine = getattr(self, "_" + name)
            for k, v in (state.get(name) or {}).items():
                mine[k] += v
        for k, d in (state.get("per_sample") or {}).items():
            self._per_sample[k].update(d)

    def coverage(self) -> dict:
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
        return {k: dict(v) for k, v in self._per_sample.items()}

    def results(self) -> dict:
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }
