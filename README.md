# Conditioned Audio DiT (DAC latent space)

Conditioned audio generation with **Rectified Flow** and a **Diffusion Transformer
(DiT)** operating in **DAC (44.1 kHz) pre-quantizer latent space**, with
classifier-free guidance. Frame-level conditions (melody, chroma, rhythm, energy,
f0) are concatenated on the feature dimension (JASCO-style); global conditions
(CLAP-text, CLIP-image) are injected via AdaLN.

The pipeline is: **stream-encode audio → latents (+ conditions) → train → sample/edit**.
Preprocessing is a streaming encoder that never materialises full WAVs to disk;
the train/val/test split is decided **in code at training time** (not on disk).

---

## Repository layout

| File | Role |
|------|------|
| `preprocess_stream.py` | Streaming preprocessing: chunk → DAC-encode on the fly → save latents (+ optional WAV / conditions). Incremental, acoustic rules, parallel workers, batched DAC. |
| `conditions.py` | Condition registry + extractors (melody/chroma/rhythm/energy/f0, CLAP-text, CLIP-image) and the `FrameConditionEncoder`. |
| `audio_dataset_npy.py` | Unconditional latent dataset **and** the shared split machinery (`compute_split`, stratified + source-grouped + seeded, persisted test manifest). |
| `audio_dataset_cond.py` | Conditioned dataset + `build_conditioned_datasets` (re-exports `compute_split`). |
| `network_cond.py` | The `ConditionedAudioDiT` model. |
| `training_cond.py` | Training loop, split construction, cache validation, TensorBoard logging. |
| `test_cond.py` | Evaluate a checkpoint on the **persisted test set** (conditioned generation vs. real). |
| `sampling_cond.py` | Generate / edit audio from a checkpoint with CFG. |
| `extract_conditions.py` | Standalone tool to add a frame condition to an existing latents dataset. |
| `metrics.py`, `condition_metrics.py` | FD-DAC/KL/FAD + per-condition fidelity (melody RPA/RCA, energy/f0 correlation, …). |
| `launch_training_cond.py` | IRCAM-only GPU-lock wrapper around `training_cond.py`. |
| `verify_paired_sampler.py` | Standalone A/B test: checks the fused paired metric sampler matches the reference two-pass one (weight-agnostic; no checkpoint/dataset needed). |
| `cond_default.yaml` | Default configuration. |

---

## Installation

### 1. PyTorch (match your CUDA first)

Install the Torch stack for **your** CUDA before anything else, e.g.:

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. The rest

```bash
pip install -r requirements.txt
```

`beat_this` (the `rhythm` condition) is not on PyPI — install it only if you use rhythm:

```bash
pip install "beat_this @ git+https://github.com/CPJKU/beat_this.git"
```

### 3. ffmpeg (only for `--acoustic_rules`)

The acoustic treatment (silence trim + loudness normalization + stereo split) shells
out to `ffmpeg`/`ffprobe`, which must be on `PATH`:

```bash
ffmpeg -version && ffprobe -version
# Windows: winget install ffmpeg   (then reopen the terminal)
```

### 4. Config location

`training_cond.py` / `test_cond.py` default to `configs/cond_default.yaml`. Either
place the config there once:

```bash
mkdir -p configs && cp cond_default.yaml configs/cond_default.yaml
```

or pass `--config cond_default.yaml` on every call.

> **TensorFlow note:** the scripts set `USE_TF=0` so `transformers` (CLAP/CLIP) uses
> the PyTorch backend. Keep it that way to avoid protobuf/TF import clashes.

---

## 1) Preprocessing — `preprocess_stream.py`

For each source file: load → mono → resample (44.1 kHz) → (optional acoustic
rules) → fixed-length chunking → **DAC-encode each chunk immediately** → save the
latent `(72, T)`. Full WAVs are never written to disk.

```bash
python preprocess_stream.py <SRC> <OUT> --device cuda \
    --conditions melody,energy,f0 \
    --acoustic_rules \
    --num_workers 0 --batch_size 8
```

Output (split-less, mirrors the source class tree):

```
OUT/
  latents/<class...>/*.npy            # (72, T) float32, pre-quant DAC
  conditions/<class...>/*.npz         # only with --conditions
  wav/<class...>/*.wav                # only with --save_wav
  global_conditions/<text|image>/...  # only with --global
  dataset_meta.json                   # chunk + acoustic params (re-run safety)
  source_manifest.json                # what each source was (size/mtime) and which
                                      #   chunks it produced -> detects sources
                                      #   edited in place (stale latents) and
                                      #   deleted sources (orphan outputs)
```

Useful flags: `--sr 44100` (required for the 44 kHz DAC), `--chunk_duration`,
`--chunk_overlap`, `--save_wav`, `--global text,image` (+ `--image_root`),
`--num_workers` (parallel CPU work; keep **0 on Windows**), `--batch_size` (DAC
batch), `--force`.

**Incremental:** re-run later with a new `--conditions` to add a condition. Latents
already on disk are **not** re-encoded; only the missing condition is extracted and
merged into the `.npz`. Keep the **same** chunk/acoustic parameters — `dataset_meta.json`
hard-fails on a mismatch (with `--force` too: that flag recomputes what it finds, it
does **not** authorise changing parameters).

On every re-run `source_manifest.json` is checked and reports:

* **sources edited in place** — same name, different bytes: their latents are stale
  and would be silently kept. Re-run with `--force` to re-encode them.
* **sources deleted/renamed** — their outputs are orphans that still feed the split,
  the normalizer and the training. `--prune_orphans` removes exactly those files.

```bash
# first: latents + f0
python preprocess_stream.py <SRC> <OUT> --device cuda --acoustic_rules --conditions f0
# later: add energy without recomputing latents/f0
python preprocess_stream.py <SRC> <OUT> --device cuda --acoustic_rules --conditions energy
```

> On a GPU box, put f0/CREPE on CUDA by setting `"device": "cuda"` in the `"f0"`
> entry of `CONDITION_CONFIG` (conditions.py). With `--num_workers > 0` the
> extractors are forced to CPU (CUDA in forked workers is unstable).

`extract_conditions.py` is a fallback for adding a condition when only the latents
(and optionally WAVs) remain — it decodes the latent back to audio if no WAV is present.

---

## 2) Train/val/test split (in code)

There are **no split folders on disk**. `compute_split` (in `audio_dataset_npy.py`)
splits the dataset:

- **stratified by class** (each class contributes to every split by ratio),
- **grouped by source file** — all chunks *and both stereo channels* of one source
  stay in the same split (no chunk-level leakage),
- **deterministic** from a seed,
- the **test set is persisted** to `OUT/../splits/test_split_<hash>.json`, so every
  run on the same dataset+params reuses the exact same test set. Train/val are
  re-derived deterministically.

Configured under `data.split` in the YAML:

```yaml
data:
  split:
    ratios: [0.8, 0.1, 0.1]   # train / val / test (per class, over source groups)
    seed: 42
    group_by_source: true
    stratify_by_class: true
    save_test_manifest: true
```

---

## 3) Training — `training_cond.py`

```bash
python training_cond.py --config configs/cond_default.yaml \
    --run_name "cond_S_f0" \
    conditioning.enabled_frame='[f0]'
```

- Selects conditions via `conditioning.enabled_frame` / `enabled_global` (YAML or CLI).
- Builds the split, fits/loads the normalizer, and logs the split composition
  (train/val/test file & chunk counts) **inside the `config` panel** on the
  TensorBoard **Text** tab, nested under `data.split.composition`. Parameter counts
  are nested under `model.n_params_*` and also logged as `Model/n_params_M`.
- **Cache safety:** the normalizer and FD-DAC reference are tied to the dataset via
  `cache_dir/cache_meta.json`. A stale or unverifiable cache **hard-fails**.
- CLI overrides use dotlist syntax (e.g. `model.kind=B training.lr=5e-5`).
- Resume: `--resume runs/<prev>/checkpoints/checkpoint_step50000.pt`.

### Which metrics are computed

`metrics.enabled` selects the distributional metrics, mirroring the unconditional
project's registry:

```yaml
metrics:
  enabled: ["fd_dac", "kl_dac"]   # [] turns them off (and skips their reference)
  seed: 0
```

Listing a metric is an explicit request: an unsupported name is a **hard error at
startup**, not a silent skip. Both metrics share the generated mean/covariance, so
asking for both costs essentially the same as asking for one.

### Unconditional training with the same code

Disabling every condition turns this into a plain **unconditional** run: the model
builds no conditioning modules (`input_proj`/AdaLN collapse to the unconditional
DiT), no conditions are read from disk, and CFG never engages.

```bash
python training_cond.py --config configs/cond_default.yaml \
    --run_name "uncond_L" model.kind=L \
    conditioning.enabled_frame='[]' conditioning.enabled_global='[]'
```

The **TensorBoard logging follows the mode**: a conditioned run logs the two-axis
scheme (`Fd_dac_cond` vs `Fd_dac_uncond`, `Kl_cond/*` vs `Kl_uncond/*`, plus the
`Condition_influence` panel and real/with-cond/without-cond audio); an
unconditional run logs a single axis under the unconditional project's own tags
(`Fd_dac`, `Kl_real_gen`, `Kl_gen_real`, `Audio_generated_{prefix}_{i}`), with no
influence panel. `Train/*` and `Validation/Loss*` are identical in both.

> **After changing preprocessing or split params, use a FRESH `paths.cache_dir`.**
> The new latent statistics make any previous normalizer / FD reference stale, and
> the cache guard will stop the run until you point to a clean cache directory.

Monitor: `tensorboard --logdir <runs_dir>`.

### Verifying the fused metric sampler (optional, one-off)

At metrics time the conditioned + unconditional samples are generated by a **fused
paired sampler**: each sample needs 3 rows per Euler step (`conditioned`,
`cfg-null`, `unconditional` — all three are required by the CFG math), and
`sampling.metrics_samples_per_forward` decides how many samples share one forward:

```yaml
sampling:
  metrics_samples_per_forward: 1   # 0 = serial reference (lowest VRAM, slowest)
                                   # 1 = batch 3  (default)
                                   # 2 = batch 6, N = batch 3N (faster, higher peak)
```

It maps directly onto the activation peak, so **if the metrics step OOMs, lower it**
(0 restores the reference sampler — same numbers, no code change); raise it only
after checking `nvidia-smi` at a metrics step. Fusing needs a CFG to fuse, so the
serial path runs anyway when guidance ≤ 1, `metrics_uncond=false`, or no condition
is active.

The value does **not** change which samples are generated (the noise is drawn per
sample), so the metric values are the same for any setting. Batching does change
CUDA op ordering, so results match the serial path only up to floating-point
tolerance — check that once on your hardware:

```bash
python verify_paired_sampler.py --kind L --steps 100 --spf 1   # add --use_amp if you train with AMP
```

It builds a random-initialised model (no checkpoint or dataset needed), runs both
samplers from the same seed and prints `PASS`/`FAIL`.

---

## 4) Evaluate & sample

**Test set (same one training persisted):**

```bash
python test_cond.py --ckpt runs/<run>/checkpoints/best_model_step<N>.pt
```

The split parameters are restored from the checkpoint config, so the persisted test
manifest is reused automatically. (Checkpoints are named `best_model_step<N>.pt`,
`checkpoint_step<N>.pt`, and `checkpoint_last_step<N>.pt` — pick the one you want.)

**Generate / edit:** `sampling_cond.py` takes the checkpoint and the mode as
**positional** arguments (`checkpoint` then `generate`|`edit`):

```bash
# generate (length defaults to the checkpoint's n_frames; pass --duration to override)
python sampling_cond.py <ckpt> generate \
    --condition_npz cond.npz --label piano --guidance 3.0 --output out/

# edit an existing file (conditions aligned to the SOURCE length)
python sampling_cond.py <ckpt> edit --source in.wav \
    --condition_wav melody_ref.wav --strength 0.4 --output out/
```

If a checkpoint requires conditions and you omit them, the script stops unless you
pass `--allow_null_frame_conditions` / `--allow_null_global_conditions`.

---

## Running on IRCAM servers

- Launch training through the GPU-lock wrapper (needs the internal `manage_gpus`):
  ```bash
  python launch_training_cond.py --num-gpus 1 --config configs/cond_default.yaml [overrides]
  ```
  Elsewhere (e.g. a local Windows box) run `python training_cond.py` directly.
- Model caches (DAC / basic-pitch / HuggingFace / CREPE) are auto-redirected to the
  machine-local disk when `/data/anasynth_nonbp/baione` exists, to avoid the NFS HOME quota.
- The default `paths.runs_dir` / `paths.cache_dir` are **relative** (`./runs`,
  `./cache`). On IRCAM, override them with the shared absolute paths:
  ```bash
  python training_cond.py --config cond_default.yaml \
      paths.runs_dir=/data2/anasynth_nonbp/baione/runs \
      paths.cache_dir=/data2/anasynth_nonbp/baione/cache
  ```
- The single-GPU setup is assumed (no DDP). Multi-GPU would only need a
  `DistributedSampler` on the map-style datasets; the split logic is unaffected.

---

## End-to-end quickstart (small dataset)

```bash
# 0) sanity check on a handful of files (Windows: keep --num_workers 0)
python preprocess_stream.py <SRC_mini> out_mini --device cuda \
    --acoustic_rules --conditions f0 --num_workers 0 --batch_size 4

# 1) full preprocessing
python preprocess_stream.py <SRC> dataset_ready_cond --device cuda \
    --acoustic_rules --conditions f0 --num_workers 0 --batch_size 8

# 2) short training run (writes the test manifest + normalizer into a FRESH cache)
python training_cond.py --config cond_default.yaml --run_name smoke \
    conditioning.enabled_frame='[f0]' training.num_steps=200 \
    paths.dataset_root=./dataset_ready_cond/latents \
    paths.condition_root=./dataset_ready_cond/conditions \
    paths.runs_dir=./runs paths.cache_dir=./cache_smoke

# 3) evaluate on the persisted test set (use the actual best_model_step<N>.pt written)
python test_cond.py --ckpt runs/smoke/checkpoints/best_model_step<N>.pt
```
