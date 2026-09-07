# Conditioned Audio DiT (DAC latent space)

Conditioned audio generation with **Rectified Flow** and a **Diffusion Transformer
(DiT)** operating in **DAC (44.1 kHz) pre-quantizer latent space**, with
classifier-free guidance. Frame-level conditions (f0, chroma, rhythm, energy)
are concatenated on the feature dimension (JASCO-style); global conditions
(CLAP-text, CLIP-image) are injected via AdaLN.

The pipeline is: **stream-encode audio → latents (+ conditions) → train → sample/edit**.
Preprocessing is a streaming encoder that never materialises full WAVs to disk;
the train/val/test split is decided **at preprocessing time**, over the source
files, and recorded in `OUT/splits.json` — the training reads it, never recomputes
it (there are still no split folders on disk).

---

## Repository layout

| File | Role |
|------|------|
| `preprocess_stream.py` | Streaming preprocessing: chunk → DAC-encode on the fly → save latents (+ optional per-split WAV / conditions). **Decides the train/val/test split** (`splits.json`). Incremental, acoustic rules, parallel workers, batched DAC. Driven by flags or `--config`. |
| `configs/preprocess_default.yaml` | Preprocessing config: every long flag as a key. Precedence: defaults < file < CLI. |
| `conditions.py` | Condition registry + extractors (f0/chroma/rhythm/energy, CLAP-text, CLIP-image) and the `FrameConditionEncoder`. |
| `audio_dataset_npy.py` | Unconditional latent dataset **and** the split reader (`load_source_split`). `compute_split` is the old in-code split, kept for `--import_legacy_split` and the unconditional builder. |
| `audio_dataset_cond.py` | Conditioned dataset + `build_conditioned_datasets` (reads the recorded split). |
| `network_cond.py` | The `ConditionedAudioDiT` model. |
| `training_cond.py` | Training loop, cache validation, TensorBoard logging. |
| `test_cond.py` | Evaluate a checkpoint on the **recorded test set** (conditioned generation vs. real). |
| `sampling_cond.py` | Generate / edit audio from a checkpoint with CFG. |
| `extract_conditions.py` | Standalone tool to add a frame condition to an existing latents dataset. |
| `metrics.py`, `condition_metrics.py` | FD-DAC/KL/FAD + per-condition fidelity (f0/energy correlation, chroma cosine, …). |
| `probe_conditions.py` | Out-of-the-box probe sets for **every condition** — frame (f0, energy, chroma, rhythm) *and* global (text, image): elementary synthetic stimuli, targets produced by the run's own extractor/encoder, cached behind a fingerprint, plus the comparison plots. One bank + one synthesizer per condition. |
| `launch_training_cond.py` | IRCAM-only GPU-lock wrapper around `training_cond.py`. |
| `configs/cond_default.yaml` | Default training configuration. |

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

`wav2clip` is what makes the **image** condition's influence row a number: CLIP
embeds images, CLAP embeds audio, and the two spaces are unrelated, so an
audio-vs-image cosine needs a model that puts audio *in CLIP's own space*.
It is in `requirements.txt`; without it the run still trains on the image
condition and the panel says `no audio->CLIP embedder` instead of inventing a
zero.

```bash
pip install wav2clip
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
    --conditions f0,energy,chroma \
    --acoustic_rules \
    --num_workers 0 --batch_size 8
```

Output (no split folders, mirrors the source class tree):

```
OUT/
  latents/<class...>/*.npy            # (72, T) float32, pre-quant DAC
  conditions/<class...>/*.npz         # per-CHUNK conditions: --conditions
                                      #   and/or the per-chunk half of --global
  wav/<class...>/*.wav                # only with --save_wav (per split)
  global_conditions/image/<class>.npy # per-CLASS conditions: --global image
  global_conditions/image/<class>.json#   the image file names, in row order
  global_conditions/text_vocab.npy    # --global text: the label vocabulary,
  global_conditions/text_vocab.json   #   CLAP-encoded once (see §3, panels)
  dataset_meta.json                   # chunk + acoustic params (re-run safety)
  splits.json                         # source -> train/val/test, plus the
                                      #   per-split counts in BOTH units (see §2)
  source_manifest.json                # what each source was (size/mtime) and which
                                      #   chunks it produced -> detects sources
                                      #   edited in place (stale latents) and
                                      #   deleted sources (orphan outputs)
```

Useful flags: `--sr 44100` (required for the 44 kHz DAC), `--chunk_duration`,
`--chunk_overlap`, `--global text,image` (+ `--image_root`), `--num_workers`
(parallel CPU work; keep **0 on Windows**), `--batch_size` (DAC batch), `--force`,
and `--config` to keep all of it in a YAML instead.

**Conditions: `--conditions` and `--global` are independent and compose.** Either
can be used alone, both together, and each takes any subset — `--conditions f0`,
`--global text`, `--conditions f0,energy --global text,image`. Adding one later
re-reads the audio but re-encodes no latent and drops no condition already on
disk, so a dataset grows a condition at a time.

What differs between the two is not how the model consumes them but **where the
value comes from**, which is what decides where it is stored:

| | computed from | written to | when |
|---|---|---|---|
| every `--conditions` name | that chunk's audio | the chunk's `.npz`, `(T, dim)` | in the stream |
| `--global text` | that chunk's audio | the **same** `.npz`, `(dim,)` | in the stream |
| `--global image` | `--image_root/<class>/*` | `global_conditions/image/<class>.npy` | after the stream |

So `text` is extracted and resumed exactly like `f0`: it is the CLAP embedding
of the chunk's own audio, which at inference is swapped for the CLAP embedding
of a written prompt (the two towers share one space). It is *not* the embedding
of the class name — that would make it carry the same single bit as `image`,
and no ablation could then separate the two.

`--global image` encodes **every** picture of a class, not a sample of them: the
training draws one at random per epoch, so a cap here would silently cap that
augmentation for every run that ever reads the dataset. The `.json` beside each
bank lists the file names in row order, which is what lets a validation panel
show the picture a generation was conditioned on.

A bank is reused only while it still **describes its folder**: a re-run compares
the recorded file list with what is on disk and re-encodes the classes whose
pictures changed, saying which. So "drop ten new pictures into a class and
re-run the preprocessing" works — you do not need `--force` (which would also
recompute every chunk condition in the dataset). A class whose folder is
unchanged is never re-encoded, and CLIP is not even loaded when none changed.
Growing a bank changes the image condition of that class, so a checkpoint
trained before the change was trained on a different bank.

`--global text` also writes `global_conditions/text_vocab.{npy,json}`: the
phrase list in `conditions.TEXT_LABEL_VOCAB`, CLAP-encoded once and stored WITH
the dataset. The stored `text` condition is the embedding of a chunk's own
audio, and CLAP has no decoder, so a panel has no sentence to show; the
vocabulary lets it name the **nearest phrase** instead, always printed with its
cosine because it is a retrieval, not a translation. Editing the list and
re-running the preprocessing rewrites the labels without touching a single
chunk of audio.

`--save_wav` takes **which splits** to write: `none` (default), `all`, or a subset
such as `val` / `val,test`. A bare `--save_wav` still means `all`. These WAVs are
the **real source audio** — they never pass through the DAC — which is what makes
them a standard FAD reference (`metrics.fad_reference: "wav"`); `val` alone costs
roughly a tenth of the disk of `all`.

```bash
# latents + conditions + the wavs of the validation split only
python preprocess_stream.py <SRC> <OUT> --device cuda \
    --acoustic_rules --conditions f0,energy,chroma \
    --save_wav val --num_workers 8 --batch_size 16
```

**Incremental:** re-run later with a new `--conditions` to add a condition. Latents
already on disk are **not** re-encoded; only the missing condition is extracted and
merged into the `.npz`. Keep the **same** chunk/acoustic parameters — `dataset_meta.json`
hard-fails on a mismatch (with `--force` too: that flag recomputes what it finds, it
does **not** authorise changing parameters).

**Re-runs are cheap:** a source whose outputs this run would write are *all*
already on disk is **not decoded at all** — the answer comes from
`source_manifest.json` plus file headers, never from the audio. So a no-op re-run
costs a scan, and "add the wavs of the validation split" touches ~10% of the
corpus instead of re-decoding everything.

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

## 2) Train/val/test split (decided at preprocessing time)

There are **no split folders on disk**: the split is a lookup table,
`OUT/splits.json`, written by `preprocess_stream.py` and **read** by the training
(`load_source_split` in `audio_dataset_npy.py`). It is assigned over the **source
files**, before anything is decoded:

- **stratified by class** (each class contributes to every split by ratio),
- **grouped by source file** — all chunks *and both stereo channels* of one source
  stay in the same split (no chunk-level leakage),
- **deterministic** from a seed,
- **written once and never reshuffled**: a re-run assigns only the *new* sources
  into the existing split. Re-deciding it requires `--resplit`, because promoting
  yesterday's training material to today's test set silently invalidates every
  evaluation of an existing checkpoint.

`splits.json` records the size of the split in **both units**: `counts` is source
files (the unit it is assigned over) and `chunk_counts` is latent chunks — the
samples the training actually sees. The two are not proportional: a long source
yields more chunks than a short one, so 80/10/10 over sources is only roughly
80/10/10 over samples. `chunk_counts` is written at the END of a run, because it
needs `source_manifest.json` (which chunks each source owns) and because the split
itself is decided before anything is encoded; a run that changes the assignment
rewrites it rather than leaving a stale number behind. `--split_only` refreshes it
without decoding anything.

Configured in `configs/preprocess_default.yaml` (or on the CLI), once, with the
dataset — **not** per training run:

```yaml
split_ratios: "0.8,0.1,0.1"   # train / val / test (per class, over sources)
split_seed:   42
no_stratify:  false
```

The training has no split knobs left. `paths.splits_path: null` in
`cond_default.yaml` means "the dataset's own `splits.json`"; set it only to point
a run at a split file kept elsewhere. The recorded **assignment** (not just its
parameters) is part of the cache fingerprint, so growing the dataset correctly
invalidates the normalizer.

Datasets built before the split moved here have no `splits.json` and the training
stops with an actionable message. Give them one, once:

```bash
# reproduces the split the training used to compute in-code (bit for bit),
# so a run already in flight keeps exactly the same val/test sets:
python preprocess_stream.py <SRC> <OUT> --import_legacy_split

# or a fresh split, no decoding, no models loaded:
python preprocess_stream.py <SRC> <OUT> --split_only
```

---

## 3) Training — `training_cond.py`

```bash
python training_cond.py --config configs/cond_default.yaml \
    --run_name "cond_S_f0" \
    conditioning.enabled_frame='[f0]'
```

- Selects conditions via `conditioning.enabled_frame` / `enabled_global` (YAML or CLI).
- Reads the split from `splits.json`, fits/loads the normalizer on the **train**
  split only, and logs the split composition and the parameters it was created with
  (train/val/test file & chunk counts) **inside the `config` panel** on the
  TensorBoard **Text** tab, nested under `data.split.composition`. Parameter counts
  are nested under `model.n_params_*` (and printed at startup). They are
  deliberately NOT logged as scalars: a constant is a flat point at step 0 that
  only clutters the Scalars dashboard.
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
                                  # add "fad_vggish" for the audio-domain FAD
  seed: 0
  fidelity_device: "cuda"         # "cuda" | "cpu" — device for the re-extraction
                                  # (CREPE / beat_this / CLAP-audio) that feeds the
                                  # condition-influence panel. FD-DAC/KL always run
                                  # on the GPU regardless. The device does not
                                  # change the values, only speed: "cpu" is an
                                  # escape hatch if the metrics step runs out of
                                  # VRAM (there the model AND the DAC decoder are
                                  # already resident).
```

Listing a metric is an explicit request: an unsupported name is a **hard error at
startup**, not a silent skip. `fd_dac` and `kl_dac` share the generated
mean/covariance, so asking for both costs essentially the same as asking for one.

### Global-condition similarity (automatic, when a global is active)

Two more scalar curves appear beside FD-DAC/KL/FAD whenever the run is
conditioned on a global and that global can be scored — **how close the
generation lands to the condition it was given**:

| tag | space | how the generation gets there |
|---|---|---|
| `Validation/Metrics/Audio_text_similarity` | CLAP | CLAP's **audio** tower embeds the generation; the stored condition is already a CLAP vector |
| `Validation/Metrics/Audio_image_similarity` | CLIP | **Wav2CLIP** embeds the generation into CLIP's space; the stored condition is the CLIP vector of the picture |

They are cosines, and they are **scalars, not panel rows**, because they are
what you watch across training the way you watch FD-DAC. Their sample count is
its own knob:

```yaml
sampling:
  n_similarity_samples: 128   # generations decoded + embedded for the two curves
```

It is deliberately larger than `n_influence_samples` (a mean of one scalar per
sample converges fast, but each sample still costs a DAC decode plus an embedder
forward) and smaller than `n_metrics_samples` (which estimates a covariance from
latents alone and is far cheaper per sample). A global with no embedder installed
logs **no curve at all**: an absent curve says "not measured", a curve pinned at
zero would say "no similarity".

The same two numbers also appear in the influence panel, but there as a **paired
delta** against the null generation, on the influence set — which is the honest
way to read them, since the absolute cosine of an audio-image pair is small even
when the conditioning works (see the probes section).

### FAD-VGGish (optional, off by default)

`fd_dac` and `kl_dac` score the **DAC latent space**. `fad_vggish` scores the
**audio**, through embeddings of a model trained on real recordings — it is the
number the controllable-music literature reports, so it is what makes your
results comparable with published ones. It is not needed to follow a training:
its use is the final, offline evaluation of a checkpoint.

```yaml
metrics:
  enabled: ["fd_dac", "kl_dac", "fad_vggish"]
  fad_device: "cuda"       # VGGish embedder device (speed only)
  fad_reference: "wav"     # "wav" | "decoded" — see below
sampling:
  n_fad_samples: 512       # generations scored; each costs a DAC decode + VGGish
```

It logs `Fad_vggish_cond` / `Fad_vggish_uncond` next to `Fd_dac_cond` /
`Fd_dac_uncond` (a single `Fad_vggish` on an unconditional run), on the same two
axes as every other distributional metric.

**What it is compared against** (`metrics.fad_reference`):

| | reference | needs | comparable with the literature |
|---|---|---|---|
| `wav` (default) | the **real** validation wavs | preprocessing run with `--save_wav val` (or `all`) | **yes** |
| `decoded` | the validation latents decoded through DAC | nothing | no |

`decoded` puts both sides through the same codec, which isolates the model from
the codec's own artifacts — arguably a fairer measure of the *model* — but the
absolute value is not the FAD other papers report. If `wav` is selected and the
wavs are missing, the run **stops at startup** and says which file it looked
for: there is no silent fallback between the two.

In both cases the reference file list comes from the **val split**, never from
globbing the wav directory: `wav/` mirrors the source tree and (with
`--save_wav all`) holds train, val and test together, so a glob would build the
"real" distribution on the test set as well. The mode is part of the
cache file name, so switching it never reuses the other one's statistics, and
the cache is guarded by the same fingerprint as the normalizer and the FD-DAC
reference.

**Cost.** Every scored generation is decoded (DAC) and embedded (VGGish) *on
top* of the existing metrics step — `sampling.n_fad_samples` is the knob that
bounds it. The statistics are accumulated as running sums, so raising it costs
time, never memory.

**One-off environment check.** The VGGish weights are fetched via `torch.hub`
(`harritaylor/torchvggish`), which needs network access the first time. On a
compute node without internet, pre-fetch them once from a login node with
`TORCH_HOME` pointed at shared storage:

```bash
TORCH_HOME=/data/anasynth_nonbp/baione/.cache/torch \
python -c "import torch; torch.hub.load('harritaylor/torchvggish','vggish'); print('vggish ok')"
```

`fad_encodec` exists in `metrics.py` but is **not** wired into this pipeline:
listing it is a hard error at startup, not a silent skip.

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
`Condition_influence` panel and the per-sample audio panels); an unconditional
run logs a single axis under the unconditional project's own tags (`Fd_dac`,
`Kl_real_gen`, `Kl_gen_real`), with no influence panel. `Train/*` and
`Validation/Loss*` are identical in both.

AUDIO is organised as ONE BLOCK PER CONDITIONED SAMPLE plus TWO COLLECTED
GROUPS. The dashboard groups cards by the text before the first `/` and lays
each group out as a grid that wraps every 2-3 cards, so the SAMPLE is the group
and the f0 target and the generation it produced take slots 1 and 2 -- the only
two positions that stay side by side at every window width. The unconditional
generations and the real recordings are NOT in those blocks: each is collected
into a group of its own, one card per sample, so they can be heard as a grid of
peers. `audio_panel_tags()` in `training_cond.py` builds the names and is the
single source of truth.

```
validation_XX/1_f0_validation_XX                  the sonified f0 target
validation_XX/2_generation_with_f0_validation_XX  the generation it conditioned
validation_XX/3+_<cond>_validation_XX             energy, chroma, ...
probe_XX_<melody>/1_f0_probe_XX , /2_generation_with_f0_probe_XX

uncond generation/uncond_validation_XX            null generation, same noise
uncond generation/uncond_probe_XX_<melody>          "     "        (probe)
ground truth/real_validation_XX                   the real recording
```

The index in a collected card's name is the cross-reference: `uncond_validation_03`
is the null twin of `validation_03/2_generation_with_f0_validation_03`, drawn from
the same noise, and `real_validation_03` is the recording that block's conditions
were extracted from. In a PURE-UNCONDITIONAL run nothing is dropped to obtain the
generation, so it goes straight to `uncond generation/` and no per-sample block is
created: the audio window is then exactly the two collected groups.

The generation card is named after the frame condition that leads the block, so
a run conditioned ONLY on globals — which has no frame condition to name it
after, and no waveform to put beside it — names it plainly:
`validation_XX/2_generation_validation_XX`. It is still a per-sample block with
its own null twin in `uncond generation/`; only a run with **no** conditioning at
all collapses the two into one card. When `text` is active, the block header
carries the label of the sample (see the probe section), and every card of that
sample — the conditions, the generation, the subset generations, the comparison
images, the image card — sits under that one header.

### Condition subsets — the delta matrix

`sampling.influence_subsets` turns the Condition_influence panel into a MATRIX:
one row per combination of conditions given to the model, one column per
(condition, metric), each cell the Δ against the SAME null pass — a shared
baseline is what makes the rows comparable with one another.

```yaml
influence_subsets: []                            # off (default): one table
influence_subsets: ["all", "loo"]                # the standard ablation
influence_subsets: ["all", "loo", "singletons"]
influence_subsets: [["f0"], ["f0", "energy"]]    # hand-picked
```

`all` = every active condition (free — the conditioned pass the step already
runs IS it). `loo` = leave-one-out, the marginal contribution of each condition
at the point where the model is actually used. `singletons` = each condition
alone. Columns cover EVERY active condition, not only the ones given in that
row: the off-subset cells (marked `°`) are the side effects — give f0 alone and
watch what happens to chroma. Each row other than `all` is a full extra
generation pass, so **the metrics step grows linearly with the number of rows**.

**The subsets vary the FRAME conditions only.** The globals are handed to the
model in every row, so `text` and `image` appear as ordinary columns with a
value per row — never marked `°` — and what they show is the *side effect*: what
dropping chroma does to how well the generation still matches its prompt. A run
with no frame condition therefore has no matrix (there is nothing to subset) and
falls back to the single table.

Each subset also gets its own audio card inside the sample's block
(`validation_XX/2_gen_no_chroma_validation_XX`), so the whole combination
ladder of one validation sample plays side by side.

### Probe sets — the ablation instrument

Elementary synthetic stimuli, unambiguous by construction, with targets
extracted by the run's OWN extractor. They answer "does this conditioning work
at all", which the validation rows cannot: a real f0 contour is ornamented, a
real energy envelope jittery, a real beat grid may not exist on this material,
so a middling score there does not separate a weak conditioning from an
ambiguous target.

| condition | stimuli | the stimulus IS |
|---|---|---|
| f0 | scale, arpeggio, octave leap, sustained note, rests | a waveform |
| energy | crescendo, diminuendo, four stabs, swell, plateau, staircases | a waveform |
| chroma | sustained triads, I-IV-V, single pitch class, clusters | a waveform |
| rhythm | click grids at fixed tempi, downbeat every N, accelerando | a waveform |
| **text** | 16 instrument/style prompts ("solo pipe organ in a large reverberant church", "fast electronic dance beat…") | a **string**, CLAP-encoded |
| **image** | 16 abstract figures — colour fields, stripes, checkerboard, rings, gradient, noise | a **.png**, CLIP-encoded |

The two families differ only in medium. A frame target is a `(n_frames, dim)`
curve **extracted** from a waveform; a global target is a `(dim,)` vector
**encoded** from a string or a picture, by the run's own CLAP/CLIP — so the
probe drives the model in exactly the space it was conditioned in. Nothing
re-extracts a picture from audio, so a global probe has no "target vs
re-extracted" plot: the image is shown as-is, the prompt names the panel, and
the adherence is a cosine in the influence table.

> **Read the abstract IMAGE bank knowing what it is.** What reaches the model is
> not the picture but CLIP's *reading* of it, and a flat colour field has no
> musical meaning for CLIP to read: its embedding lands far from the paintings
> and album covers the training conditioned on. A weak image-probe row does
> **not** by itself prove the image conditioning is broken — unlike the f0 bank,
> where a rising scale is unambiguous and a failure is a failure. The validation
> rows, on in-corpus images, carry that verdict. What the bank *does* establish
> — and what the build prints as "bank spread" — is whether 16 distinct stimuli
> produce 16 distinct embeddings, i.e. whether the slot can carry information at
> all. The same report is printed for the text bank.

A bank is built for **every condition active in the run** — there is nothing to
turn on per condition. The single knob is **`sampling.n_influence_samples`**
(default 16), the size of the INFLUENCE SET: N probe stimuli *and* N validation
samples. Each probe panel drives all the active conditions at once with the i-th
stimulus of their own bank. Cost is N **paired** generations per family per
metrics step, whatever the number of conditions.

Every sample of the set is scored **and** plotted **and** played — the table, the
Images window and the Audio window describe the same N samples, so the number you
read is about the curve you are looking at.

> Removed knobs, no longer read at all: `n_probes` (→ `n_influence_samples`),
> `n_cond_plot` and its alias `n_f0_plot` (every sample is plotted now), and the
> per-condition `n_f0_probe` / `n_energy_probe` / `n_chroma_probe` /
> `n_rhythm_probe` (already inert before). A leftover in your config does nothing.

All six banks live in ONE module, `probe_conditions.py`: one bank, one
synthesizer and one plot branch per condition, so every condition is set up,
built, drawn and scored the same way. Adding a seventh means adding a bank and a
synthesizer, nothing else.

Each FRAME condition gets the SAME two images, N of each, grouped **per sample**
so the Images tab collapses into the same sections as the Audio tab instead of
one flat wall:

```
validation_XX/<cond>_target_vs_gen   target vs generated, on a real val sample
probe_XX/<cond>_target_vs_gen        the same, on the unambiguous probe stimulus
validation_XX/image_condition        the picture that sample was conditioned on
probe_XX/image_condition             the probe figure, in the same block
```

The two global conditions occupy the same blocks by other means: the **image**
is shown as a card (there is nothing to compare it against), and the **text**
prompt names the block itself — `probe_03 [string quartet playing a slow
sustained chord]`, and for a validation sample the category plus the nearest
phrase to its stored CLAP vector, `validation_00 [Baroque sacred · "solo pipe
organ" (+0.58)]`. That label is a retrieval over `text_vocab`, which is why the
cosine is always beside it.

drawn in the form that suits the shape — f0 on a log-Hz axis with a voicing
ribbon, energy and rhythm as overlaid curves, chroma as paired heatmaps. Plus an
audio block (`<cond>probe_XX_<name>/`) and a `<cond>_probe` row in the influence
table. The `_valid_` images are FREE: the curves come from the re-extraction the
influence table already runs.

Build any probe set standalone to look at it before training:

```bash
python probe_conditions.py f0     ./cache/f0_probe     --n_frames 431
python probe_conditions.py energy ./cache/probe_energy --n_frames 431
python probe_conditions.py chroma ./cache/probe_chroma --n_frames 431
python probe_conditions.py rhythm ./cache/probe_rhythm --n_frames 431
# the global banks: --n_frames does not apply (one embedding, no chunk geometry)
python probe_conditions.py text   ./cache/probe_text
python probe_conditions.py image  ./cache/probe_image
```

Each prints its bank and, for the global ones, the **spread** — every stimulus's
cosine to its nearest neighbour and the bank's mean pairwise cosine, with a
warning on any pair above 0.95 ("these two stimuli drive the model with the same
condition"). Read it before trusting a global probe row.

> The rhythm bank is ordered by how reliably `beat_this` recovers the intended
> tempo: 12 of the 16 grids come back at the right metrical level, the last four
> (60/160/180 bpm and the ritardando) fall into the tempo-octave ambiguity of
> beat tracking. They still score correctly — the target is whatever the run's
> extractor produced — but they are confusing to look at, so they sit past the
> plotted head.

IMAGES carry the matching overlays for every active condition, in the SAME
per-sample blocks as the audio: `validation_XX/<cond>_target_vs_gen` and
`probe_XX/<cond>_target_vs_gen`. See the probe section above for the forms each
takes.

The two COLLECTED audio groups — `ground truth/` (the recordings) and
`uncond generation/` (the same model with no conditions) — are listening
material, sized by `sampling.n_audio_samples` per family, and are a prefix of the
influence set: `real_validation_03` is the recording block `validation_03` was
conditioned from.

> **After changing preprocessing, or after the split changes, use a FRESH
> `paths.cache_dir`.** The fingerprint covers the recorded split *assignment*, not
> just its parameters, so adding sources to a dataset also invalidates it: the
> normalizer is fitted on the train split. The cache guard stops the run until you
> point to a clean cache directory.
>
> **Moving a dataset to another machine also invalidates its cache.** The
> fingerprint hashes each latent's size *and mtime*, and a copy rewrites the
> mtimes, so the guard fires with `[cache] STALE CACHE` even though the bytes are
> identical. That is deliberate — mtime is what catches a source edited in place
> — so on the destination machine point `paths.cache_dir` at a fresh directory
> and let the normalizer refit (a few minutes; the FD-DAC reference rebuilds with
> it). Copying the cache across is never worth it.

Monitor: `tensorboard --logdir <runs_dir>`.

### The fused metric sampler (`sampling.metrics_samples_per_forward`)

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
serial path runs anyway when guidance ≤ 1 or no condition is active.

"Same numbers" includes the influence table: the fused sampler gets the null
pass for free out of the CFG math, the serial one has to generate it, and it
does so whenever anything can be scored against it — `metrics_uncond: false`
switches off the uncond *distributional* metrics, never the Δ column. Setting
`0` therefore costs one extra generation pass per metrics step (that is the
price of not fusing), not a column of the panel.

The value does **not** change which samples are generated (the noise is drawn per
sample), so the metric values are the same for any setting. Batching does change
CUDA op ordering, so the fused and the serial path agree to floating-point
tolerance, not bit for bit.

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
    --condition_wav reference.wav --strength 0.4 --output out/
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
- Model caches (DAC / CREPE / beat_this / HuggingFace) are auto-redirected to the
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

# 2) short training run (reads splits.json, writes the normalizer into a FRESH cache)
python training_cond.py --config cond_default.yaml --run_name smoke \
    conditioning.enabled_frame='[f0]' training.num_steps=200 \
    paths.dataset_root=./dataset_ready_cond/latents \
    paths.condition_root=./dataset_ready_cond/conditions \
    paths.runs_dir=./runs paths.cache_dir=./cache_smoke

# 3) evaluate on the recorded test set (use the actual best_model_step<N>.pt written)
python test_cond.py --ckpt runs/smoke/checkpoints/best_model_step<N>.pt
```
