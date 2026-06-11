# Conditioned Audio Diffusion Transformer for Controllable Music Generation

Controllable music generation in the latent space of a neural audio codec, trained with the Rectified Flow objective. This is the **conditioned** extension of the unconditional project: the same Diffusion Transformer (DiT) operating on per-frame [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) (DAC) latents is now driven by **frame-level** controls (melody, harmony, rhythm) and **global** controls (text, image). Distributional evaluation uses the same two complementary latent-space metrics as the unconditional project: the Fréchet distance (FD-DAC) and the Kullback–Leibler divergence (KL), both under a shared multivariate-Gaussian model of the DAC latent distributions.

This repository was developed at IRCAM (UMR STMS, Sound Analysis-Synthesis team) within the context of a doctoral research project. It shares its backbone, training objective and evaluation with the unconditional repository; this document covers only what conditioning adds.


## Overview

The pipeline has four stages:

1. **Tokenisation.** Raw audio is encoded into a sequence of 1024-dimensional continuous latent frames at ≈86 Hz with DAC at 44.1 kHz; a 5-second segment yields ≈431 frames.
2. **Condition extraction.** From the same audio, time-aligned control signals are pre-computed once and stored next to the latents: a per-frame melody, a per-frame chromagram (harmony) and per-frame beat/downbeat curves (rhythm). Optional global controls (a text and/or an image embedding per sample) are produced on the fly.
3. **Conditioned generative model.** A Diffusion Transformer is trained with a [Rectified Flow](https://arxiv.org/abs/2209.03003) objective to predict the velocity field of the noise→data interpolation, conditioned on the controls. Frame controls are injected by concatenation on the feature dimension at the input; global controls modulate every block through AdaLN.
4. **Decoding.** At inference, latents are drawn by Euler integration of the learned velocity field (with classifier-free guidance) and decoded to waveform by the frozen DAC decoder.


## Architectural choices

### Generator: ConditionedAudioDiT

The backbone is identical to the unconditional `AudioDiT` (`network.py`) and is extended **without modifying the transformer block**: RoPE self-attention ([Su et al., 2021](https://arxiv.org/abs/2104.09864)), SwiGLU feed-forward ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)), AdaLN-Zero conditioning of the timestep ([Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748)), token-level inputs (one DAC frame = one token, no patching), no additive positional embedding. The four size variants S / B / G / L are unchanged (the conditioning adds only a thin input projection and the per-condition adapters).

### Conditioning mechanism

Let the DAC latent of a chunk be a token sequence `x ∈ R^{T×1024}` (one token per frame). Conditioning is added in two places, following two different references:

**Frame-level controls — concatenation on the feature dimension (JASCO).** Each frame control `k` is a time-aligned matrix `c_k ∈ R^{T×d_k^raw}` resampled to the *same* length `T` as the latent. Each is projected by a single linear layer `p_k = c_k W_k ∈ R^{T×d_k^out}` and concatenated to the latent on the feature axis before a single input projection:

```
x̃ = [ x ‖ p_melody ‖ p_chroma ‖ p_rhythm ] ∈ R^{T × (1024 + Σ d_k^out)}
h  = x̃ W_in + b_in                         ∈ R^{T × D_hidden}
```

This is exactly the mechanism of JASCO ([Tal et al., 2024](https://arxiv.org/abs/2406.10970)): per-control projection followed by feature-dimension concatenation, then one input linear. The transformer block is untouched.

**Global controls — AdaLN (official DiT class-label mechanism).** Each global control is encoded to `D_hidden` and **added** to the timestep embedding to form the conditioning vector `c = t_emb + Σ g_global`, which modulates every block and the final layer. This is the class-label conditioning of [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748).

### Frame controls

| Control | Backbone | Raw repr. `d_raw` | Reduction | Proj. `d_out` |
|---|---|---|---|---|
| **Melody** | basic-pitch (`note` posteriorgram) | 88 (piano keys) | per-frame argmax → one-hot of the dominant pitch, silent frames zeroed (threshold 0.3) | 64 |
| **Harmony** | librosa chroma CQT | 12 (pitch classes) | continuous, per-frame normalised | 64 |
| **Rhythm** | beat_this (`Audio2Frames`) | 2 (beat, downbeat) | sigmoid of the framewise logits → two probability curves | 32 |

All three are resampled on the time axis to the DAC frame grid, so frame `t` of every control aligns 1:1 with token `t`. basic-pitch and the melody backbone run at ≈86 fps (the DAC frame rate), beat_this at 50 fps (upsampled to ≈86).

### Global controls

A text embedding (CLAP) and/or an image embedding (CLIP) per sample, each projected to `D_hidden` and summed into the AdaLN vector. In the current default these are disabled; the slots exist so free-form prompts can be added later without an architecture change.

### Classifier-free guidance

A single model is trained with per-sample **CFG dropout**: with probability `p_drop_all` all controls are replaced by their null (zero) value, with `p_drop_frame` only the frame controls, with `p_drop_global` only the global ones. At sampling time the velocity is extrapolated between the conditional and unconditional predictions with a `guidance_scale`. Null controls are exactly the all-zero tensors, so any subset of controls can be activated at inference.

### Training objective and sampling

Unchanged from the unconditional project. Rectified Flow with `t ∼ LogitNormal(0,1)`, target velocity `v = x₁ − x₀`, MSE loss; first-order Euler integration from `t = 0.001` to `t = 0.999` over 50 steps by default, now with CFG.

### Evaluation metrics

Computed every `intervals.metrics` steps on a **fixed** subset of the validation set (deterministic indices, so the curves are comparable across steps), along three independent axes:

- **Unconditional generation** — `Fd_dac_uncond` / `Kl_uncond/real_gen` / `Kl_uncond/gen_real`. Samples generated with null conditions (no guidance). This is the only metric that is apples-to-apples comparable with the unconditional model: it measures free-generation quality of the conditioned-trained model. Toggle with `sampling.metrics_uncond`.
- **Conditional generation** — `Fd_dac_cond` / `Kl_cond/real_gen` / `Kl_cond/gen_real`. Each sample generated from one specific validation condition (with CFG guidance). Distributional fidelity of the conditioned generations to the real data. Not comparable with the unconditional model, because conditioning restricts the distribution.
- **Conditioning fidelity** — `Cond_fidelity/<name>/<metric>`. A *paired* measure (not distributional): the condition is re-extracted from each conditioned generation and compared one-to-one with the input condition. Per condition: melody → Raw Pitch Accuracy + Raw Chroma Accuracy (`mir_eval.melody`); harmony → mean per-frame chroma cosine (MusicGen-Melody style); rhythm → beat/downbeat curve correlation (Music ControlNet-style adherence). Re-extraction uses the same extractor configuration as `extract_conditions.py`, so the two sides are directly comparable.

Both distributional metrics are computed **entirely in the (normalized) DAC latent space**, modelling the real and generated latent frames as multivariate Gaussians `N(μ, Σ)` with **full** 1024×1024 covariance. **FD-DAC** is the [Heusel et al.](https://arxiv.org/abs/1706.08500)-style Fréchet (Wasserstein-2) distance between the two Gaussians; it is symmetric. **KL divergence** is the closed-form Kullback–Leibler between the same two Gaussians and, being asymmetric, is reported in **both directions** (`real‖gen` and `gen‖real`), computed via a numerically-stable Cholesky factorization with covariance regularization. A single set of real-validation reference statistics (`μ`, `Σ`) is pre-computed once over the entire validation split, cached, and **shared by both metrics** — no audio decoding or external embedding (e.g. Encodec) is involved. The conditioning-fidelity metrics computed are driven by `conditioning.enabled_frame`/`enabled_global`.

> **Note.** A previous version also computed the Fréchet Audio Distance (FAD) on Encodec embeddings of decoded waveforms. FAD has been removed (in both the conditional and unconditional projects): the distributional evaluation is now latent-only, which removes the audio-decoding cost and the Encodec dependency. Audio is still decoded inside the metrics step, but only for the conditioning-fidelity re-extraction and for the TensorBoard audio/spectrogram previews — not for any distributional metric.


## Faithfulness to the literature

This project is faithful to JASCO in the **injection mechanism and temporal alignment**, and to JASCO's **melody method** (with a modern extractor); it **deliberately deviates** from JASCO on the harmony and rhythm *representations*, adopting Music ControlNet's rhythm representation instead. The table makes this explicit.

| Aspect | JASCO | This project | Faithful? |
|---|---|---|---|
| Frame injection (concat on feature dim + per-control linear + single input linear) | yes | identical | ✅ faithful |
| Temporal alignment (resample to latent fps, 1:1 token) | yes | identical | ✅ faithful |
| Melody reduction (interpolate → argmax → one-hot → silence mask) | yes | identical | ✅ faithful |
| Melody backbone | Deep Salience ([Bittner et al., 2017](https://github.com/rabitt/ismir2017-deepsalience)) | basic-pitch ([Bittner et al., 2022](https://arxiv.org/abs/2203.09893), same author) | ≈ same family, tool substituted |
| Melody bins | 53 | 88 (piano keys) | ⚠️ dimensionality differs |
| Harmony | discrete chords (194-vocab embedding) | continuous chroma-12 (linear) | ⚠️ representation differs |
| Rhythm | separated drums stem (waveform) | beat + downbeat curves ([Music ControlNet](https://arxiv.org/abs/2311.07069)) via beat_this | ⚠️ representation differs |
| Global text conditioning | AdaLN-style global | AdaLN (official DiT) | ✅ faithful |

The deviations are intentional and grounded: chroma is a standard, lightweight harmony proxy that does not require a chord-recognition model, and beat/downbeat curves are Music ControlNet's rhythm control, which does not require source separation and applies to music without drums. If exact JASCO parity on harmony and rhythm becomes a goal, the routes are a chord-recognition vocabulary for harmony and a Demucs drums stem for rhythm.


## Repository layout

```
.
├── network.py                   # AudioDiT backbone (S / B / G / L) — unconditional
├── network_cond.py              # ConditionedAudioDiT: extends AudioDiT with frame concat + global AdaLN
├── conditions.py                # Extractors (melody / chroma / rhythm, text / image) + ConditionRegistry + encoders
├── extract_conditions.py        # Pre-computes frame conditions (.npz) next to the DAC latents
├── audio_dataset_cond.py        # Conditioned dataset + collate (loads latents + per-frame conditions)
├── training_cond.py             # Conditioned training (Rectified Flow + CFG dropout, EMA, metrics)
├── sampling_cond.py             # Conditioned Euler sampling + CFG (generation / editing)
├── test_cond.py                 # Conditioned generation + comparison with real samples (TensorBoard)
├── sanity_check.py              # End-to-end pipeline check (shapes, gradient, single-batch overfit)
├── audio_dataset_npy.py         # Dataset, normaliser, DAC loader (shared)
├── preprocess_dataset.py        # Audio → DAC latents (shared)
├── metrics.py                   # FD-DAC and KL divergence (latent-space, full covariance, shared)
├── condition_metrics.py         # Conditioning-fidelity metrics (melody/chroma/rhythm)
├── configs/
│   └── cond_default.yaml        # Conditioned OmegaConf configuration
└── README.md
```

`runs/<run_name>/` and `cache/` are produced at runtime and excluded from version control.


## Configuration

`configs/cond_default.yaml` mirrors the unconditional config and adds a `conditioning` section. The per-run selection of active controls is made there (a subset of the pool declared in `CONDITION_CONFIG` in `conditions.py`):

```yaml
model:
  kind: 'S'                 # S | B | G | L
  duration_s: 5.0
  drop: 0.0
data:
  train_batch_size: 8
  val_batch_size: 8
  grad_accum: 1
conditioning:
  p_drop_all:    0.10       # CFG dropout: all controls -> unconditional
  p_drop_frame:  0.05       # only frame controls
  p_drop_global: 0.05       # only global controls
  guidance_scale: 3.0       # CFG scale at sampling
  enabled_frame:  ["melody"]    # null = all enabled; [] = none; list = subset
  enabled_global: []            # e.g. ["text", "image"]
paths:
  dataset_root:   "./dataset_ready_cond/latents"
  wav_root:       "./dataset_ready_cond/wav"
  condition_root: "./dataset_ready_cond/conditions"
  image_root:     "./image_dataset_cond"
```

The active set is persisted into every checkpoint (`frame_cond_dims`, `frame_cond_out_dims`, `global_configs`), so sampling and evaluation rebuild the exact model without the YAML.


## Usage

### 1. Pre-processing (DAC latents)

Same as the unconditional project:

```bash
python preprocess_dataset.py raw_audio/ dataset_ready_cond/ --chunk_length 5 --device cuda
```

### 2. Condition extraction

Pre-compute the frame conditions (one `.npz` per chunk, keyed by control name) next to the latents:

```bash
python extract_conditions.py dataset_ready_cond/ --device cuda
```

### 3. Sanity check (recommended before any long run)

Validates the whole chain — registry → dataset → batch shapes → model → gradient → single-batch overfit — in a few minutes:

```bash
python sanity_check.py \
    --latent_root    ./dataset_ready_cond/latents \
    --condition_root ./dataset_ready_cond/conditions \
    --enabled_frame  melody \
    --kind S --batch_size 4 --overfit_steps 300
```

### 4. Training

```bash
# Default conditioned run (melody only, as in cond_default.yaml)
python training_cond.py --run_name "cond_S_melody"

# CLI overrides (dotlist syntax)
python training_cond.py --run_name "cond_G_full" \
    model.kind=G conditioning.enabled_frame="[melody,chroma,rhythm]"

# Resume
python training_cond.py --resume runs/<run>/checkpoints/checkpoint_step50000.pt
```

### 5. Inference / evaluation

```bash
python test_cond.py --ckpt runs/<run>/checkpoints/best_model.pt --n_samples 16 --steps 100
```


## Dependencies

In addition to the unconditional requirements (`torch >= 2.0`, `torchaudio`, `numpy < 2`, `descript-audio-codec`, `soundfile`, `omegaconf`, `matplotlib`, `tensorboard`, `tqdm`, `scipy`):

```
basic-pitch          # melody (polyphonic note posteriorgram; ONNX on Windows)
beat_this            # rhythm (beat + downbeat; PyTorch, no madmom/DBN)
librosa              # harmony (chroma CQT) + audio I/O for extraction
transformers         # CLAP text encoder (global, optional)
Pillow               # CLIP image conditioning (global, optional)
```


## References

- O. Tal, A. Ziv, I. Gat, F. Kreuk, Y. Adi. *Joint Audio and Symbolic Conditioning for Temporally Controlled Text-to-Music Generation* (JASCO). ISMIR 2024. [arXiv:2406.10970](https://arxiv.org/abs/2406.10970)
- R. M. Bittner, J. J. Bosch, D. Rubinstein, G. Meseguer-Brocal, S. Ewert. *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription* (basic-pitch). ICASSP 2022. [arXiv:2203.09893](https://arxiv.org/abs/2203.09893)
- F. Foscarin, J. Schlüter, G. Widmer. *Beat This! Accurate Beat Tracking Without DBN Postprocessing*. ISMIR 2024. [arXiv:2407.21658](https://arxiv.org/abs/2407.21658)
- S.-L. Wu, C. Donahue, S. Watanabe, N. J. Bryan. *Music ControlNet: Multiple Time-varying Controls for Music Generation*. 2023. [arXiv:2311.07069](https://arxiv.org/abs/2311.07069)
- W. Peebles, S. Xie. *Scalable Diffusion Models with Transformers* (DiT). ICCV 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- N. Shazeer. *GLU Variants Improve Transformer*. 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- R. Kumar et al. *High-Fidelity Audio Compression with Improved RVQGAN* (DAC). NeurIPS 2023. [arXiv:2306.06546](https://arxiv.org/abs/2306.06546)
- Y. Wu et al. *Large-scale Contrastive Language-Audio Pretraining* (CLAP). ICASSP 2023. [arXiv:2211.06687](https://arxiv.org/abs/2211.06687)
- A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)


## Acknowledgements

This work was carried out at IRCAM as part of a doctoral research project on neural audio generation. It extends the unconditional Audio DiT repository with controllable, temporally-aligned conditioning.
