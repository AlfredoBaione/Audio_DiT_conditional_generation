# Conditioned Audio DiT (DAC/ENCODEC latent space)

Conditioned audio generation with **Rectified Flow** and a **Diffusion Transformer
(DiT)** operating in **DAC (44.1 kHz) pre-quantizer latent space** or **ENCODEC (32 kHz) latent space**, with
classifier-free guidance. Frame-level conditions (f0, chroma, rhythm, energy,
chord, midi) are concatenated on the feature dimension (JASCO-style) and can optionally be
**re-injected in depth** (`model.frame_reinject_every`), scaled by a
per-condition gate that depends on the denoising step; global conditions
(CLAP-text, CLIP-image) are injected via AdaLN, and the text can
additionally be **cross-attended as a token sequence**
(`model.text_cross_every`, PixArt-alpha order). The self-attention can be the
DIFF Transformer's **differential attention** (`model.attention`), the
training timesteps can follow **Stable Audio 3's recipe** (`training.t_sampler`),
and so can the Euler steps of every generation (`sampling.t_schedule`).

The pipeline is: **stream-encode audio → latents (+ conditions) → train → sample/edit**.
Preprocessing is a streaming encoder that never materialises full WAVs to disk;
the train/val/test split is decided **at preprocessing time**, over the source
files, and recorded in `OUT/splits.json` — the training reads it, never recomputes
it (there are still no split folders on disk).

---

## Repository layout

| File | Role |
|------|------|
| `preprocess_stream.py` | Streaming preprocessing: chunk → DAC/ENCODEC-encode on the fly → save latents (+ optional per-split WAV / conditions). **Decides the train/val/test split** (`splits.json`). Incremental, acoustic rules, parallel workers, batched DAC/ENCODEC. Driven by flags or `--config`. |
| `configs/preprocess_default.yaml` | Preprocessing config: every long flag as a key. Precedence: defaults < file < CLI. |
| `conditions.py` | Condition registry + extractors (f0/chroma/rhythm/energy/chord/midi, CLAP-text, CLIP-image) and the `FrameConditionEncoder`. |
| `crema_chord.py` + `crema_chord_weights.npz` | PyTorch port of crema's chord model, with its original weights: the backbone of the `chord` condition (see §1, *The `chord` condition*). |
| `midi_transcriber.py` | YourMT3+ (audio → MIDI notes), the re-extraction of the `midi` condition from generated audio; downloads its code and checkpoint on first use (see §1, *The `midi` condition*). `--selftest` checks a machine. |
| `audio_dataset_npy.py` | Unconditional latent dataset **and** the split reader (`load_source_split`). `compute_split` is the old in-code split, kept for `--import_legacy_split` and the unconditional builder. |
| `audio_dataset_cond.py` | Conditioned dataset + `build_conditioned_datasets` (reads the recorded split). |
| `network_cond.py` | The `ConditionedAudioDiT` model. |
| `training_cond.py` | Training loop, cache validation, TensorBoard logging. |
| `test_cond.py` | Test a checkpoint on the **recorded test set**: the training's metrics step and panels, on the test split (§4). |
| `sampling_cond.py` | Generate / edit audio from a checkpoint with CFG. |
| `extract_conditions.py` | Standalone tool to add a frame condition to an existing latents dataset. |
| `metrics.py`, `condition_metrics.py` | FD/KL/FAD + per-condition fidelity (f0/energy correlation, chroma/chord cosine, …). |
| `probe_conditions.py` | Out-of-the-box probe sets for **every condition** — frame (f0, energy, chroma, rhythm) *and* global (text, image): elementary synthetic stimuli, targets produced by the run's own extractor/encoder, cached behind a fingerprint, plus the comparison plots. One bank + one synthesizer per condition. |
| `launch_training_cond.py` | IRCAM-only GPU-lock wrapper around `training_cond.py`; with `--num-gpus N` > 1 it runs one training process per GPU (see "Several GPUs"). |
| `launch_test_cond.py` | IRCAM-only GPU-lock wrapper around `test_cond.py`, the twin of `launch_training_cond.py`; with `--num-gpus N` > 1 it runs one test process per GPU (see §4). |
| `configs/training_cond_default.yaml` | Default training configuration. |
| `configs/test_cond_default.yaml` | Test configuration: how many test samples are measured, how many panels and uncond generations are shown. |

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

`wav2clip` is what makes the **image** condition's column of the
Condition_influence table a number: CLIP embeds images, CLAP embeds audio, and
the two spaces are unrelated, so an audio-vs-image cosine needs a model that
puts audio *in CLIP's own space*. It is in `requirements.txt`; without it the
run still trains on the image condition, and the `image/clip_sim` column reads
`n/a` (the log says `no audio->CLIP embedder`) instead of inventing a zero.

```bash
pip install wav2clip
```

`transformers` is pinned to **4.45.1**: YourMT3+, the transcriber of the `midi`
condition, subclasses transformers' internal T5 classes, which 4.46+ changed
(under 4.57.6 it dies with a TypeError in its decoder). The pin changes nothing
else: CLAP text vectors, CLAP token sequences and CLAP audio vectors are
bit-identical under 4.45.1 and 4.57.6, CLIP image vectors within 1.8e-7. On an
existing environment: `pip install transformers==4.45.1 lightning deprecated mido`,
then `python midi_transcriber.py --selftest`.

### 3. ffmpeg (only for `--acoustic_rules`)

The acoustic treatment (silence trim + loudness normalization + stereo split) shells
out to `ffmpeg`/`ffprobe`, which must be on `PATH`:

```bash
ffmpeg -version && ffprobe -version
# Windows: winget install ffmpeg   (then reopen the terminal)
```

### 4. Config location

`training_cond.py` defaults to `configs/training_cond_default.yaml`. Either
place the config there once:

```bash
mkdir -p configs && cp training_cond_default.yaml configs/training_cond_default.yaml
```

or pass `--config training_cond_default.yaml` on every call. `test_cond.py` starts from
the config stored in the checkpoint and layers `configs/test_cond_default.yaml` on top
(§4).

> **TensorFlow note:** the scripts set `USE_TF=0` so `transformers` (CLAP/CLIP) uses
> the PyTorch backend. Keep it that way to avoid protobuf/TF import clashes.

---

## 1) Preprocessing — `preprocess_stream.py`

For each source file: load → mono → resample (the codec's rate: 44.1 kHz for
DAC, 32 kHz for EnCodec) → (optional acoustic rules) → fixed-length chunking →
**encode each chunk immediately** → save the latent `(D, T)` (`D` = 72 for DAC,
128 for EnCodec). Full WAVs are never written to disk.

```bash
python preprocess_stream.py <SRC> <OUT> --device cuda \
    --conditions f0,energy,chroma \
    --acoustic_rules \
    --num_workers 0 --batch_size 8
```

Output (no split folders, mirrors the source class tree):

```
OUT/
  latents/<class...>/*.npy            # (D, T) float32: 72-d pre-quant DAC,
                                      #   or 128-d EnCodec (--codec)
  conditions/<class...>/*.npz         # per-CHUNK conditions: --conditions
                                      #   and/or the per-chunk half of --global
  wav/<class...>/*.wav                # only with --save_wav (per split)
  global_conditions/image/<class>.npy # per-CLASS conditions: --global_conds image
  global_conditions/image/<class>.json#   the image file names, in row order
  global_conditions/text_vocab.npy    # --global_conds text: the label vocabulary,
  global_conditions/text_vocab.json   #   CLAP-encoded once (see §3, panels)
  global_conditions/text_labels.jsonl # --global_conds text: per-CHUNK description,
  global_conditions/text_labels_cos.npy #  class + nearest phrases, plus the
  global_conditions/text_labels.json  #   full cosine table (see below)
  dataset_meta.json                   # chunk + acoustic params (re-run safety)
  splits.json                         # source -> train/val/test, plus the
                                      #   per-split counts in BOTH units (see §2)
  source_manifest.json                # what each source was (size/mtime) and which
                                      #   chunks it produced -> detects sources
                                      #   edited in place (stale latents) and
                                      #   deleted sources (orphan outputs);
                                      #   also HOW the classes were decided
                                      #   (--label_source, see below)
```

Useful flags: `--codec` (see below), `--chunk_duration`, `--chunk_overlap`,
`--global_conds text,image` (+ `--image_root`), `--label_source` (where the class
comes from, see below), `--num_workers` (parallel CPU work; keep **0 on
Windows**), `--batch_size` (encoder batch), `--force`, and `--config` to keep all
of it in a YAML instead. `--sr` follows the codec and is best left unset.

### Which autoencoder makes the latents — `--codec`

| `--codec` | sample rate | latent | frames/s | 5 s | 10 s | 30 s |
|---|---|---|---|---|---|---|
| `dac` (default) | 44.1 kHz, mono | 72-d, the DAC's pre-quantizer latents, decoded by re-quantizing them | 86.13 | 431 | 862 | ~2584 |
| `encodec` | 32 kHz, mono | 128-d, EnCodec's **continuous** encoder output (MusicGen's model, `facebook/encodec_32khz`), decoded by its decoder without the quantizer | 50 | 250 | 500 | 1500 |

`dataset_meta.json` and the checkpoints record the codec as `dac_44khz` or
`encodec_32khz`.

```bash
python preprocess_stream.py <SRC> <OUT_ENCODEC> --device cuda --codec encodec     --conditions f0,chroma,energy,rhythm --chunk_duration 10
```

- **The codec is part of the dataset**: recorded in `dataset_meta.json`
  (`codec`, `latent_dim`, `sr`), one OUT dir holds one codec, and a re-run with
  another codec into it is refused. **An incremental run on an EnCodec dataset (adding a condition, with
  or without `--skip_encoder`) must repeat `--codec encodec`**, like every
  other parameter the dataset was built with; forgetting it stops the run with
  that instruction, before anything is loaded.
- **Everything after reads it back, nothing assumes it.** The conditions are
  extracted on the codec's frame grid (chroma / energy hop = the codec's hop,
  chord / midi at its frame rate); the training reads the codec from the
  dataset, sizes the model's tokens from it (72 or 128), decodes with it, and
  writes it into the checkpoint; `test_cond.py`, `sampling_cond.py` and the
  `compare_*.py` scripts read it from the checkpoint, and `test_cond.py` refuses
  a test dataset of another codec. A resume or a warm start across codecs is
  refused.
- **Latent metrics are per codec.** FD and KL are measured in the codec's
  latent space: the TensorBoard tags read `Fd_dac_*` on DAC runs (unchanged)
  and `Fd_encodec_*` on EnCodec runs, and the two are not comparable. FAD and
  the condition metrics are measured on the audio and compare across codecs.
- **Reconstruction** (6 Oct 2026, 24 instruments_v2 clips of 5 s, encode →
  decode, metrics at 32 kHz): DAC SI-SDR 13.2 dB, log-mel L1 0.182; EnCodec
  continuous 10.1 dB / 0.272; EnCodec through its 4 codebooks 7.0 dB / 0.291 —
  hence the continuous latents.
- The EnCodec weights are downloaded from HuggingFace the first time
  (`transformers`, already a requirement), like the DAC's.

### Where the class of a source comes from — `--label_source`

The class of a source is **the output folder it is encoded into**, and everything
downstream reads it back from there: the stratified split (§2), `_class_of_file`
in the training dataset, the per-class image bank, the panel captions. Nothing
after the preprocessing knows how that folder name was decided.

| `--label_source` | the class of a file is | for |
|---|---|---|
| `dir` (default) | its own subdirectory under `<SRC>` — `SRC/rock/x.mp3` → `latents/rock/` | corpora already laid out by class |
| `csv` | what `--label_csv` says it is, and *that* becomes the output folder | corpora that ship one flat audio folder plus a metadata file |

```bash
python preprocess_stream.py <SRC> <OUT> --device cuda \
    --label_source csv --label_csv metadata.csv
```

**Two flags, and a fixed contract for the CSV** — no options around it:

```
file,label
violin_01.wav,violin
drum_07.wav,drum
```

* one **file** column, named `file` or `filename`; one **label** column, named
  `label` or `class`. Case does not matter, so `FileName,Class` reads as is;
* the file column holds the name, or the path relative to `<SRC>`;
* the label is the class, used **verbatim** as the folder name;
* **every source file must have a row.**

Two accepted spellings per column, not a flag: they are the two the world
actually uses and they are mutually exclusive in practice. A CSV carrying *both*
`label` and `class` is one whose author meant two different things, so that is
an error too — rename or drop one column.

Everything else is fixed *in the CSV*: a third column name, a typo in a label, a
file with no row. A CSV is a text file under your control, and every flag that
"handles" one of those cases is a rule that will be silently wrong on the next
corpus.

The raw audio is never moved or copied — only the encoded output is grouped by
class. And **switching modes renames nothing**: `src_hash` and every chunk file
name are computed from the *source* path, so the manifest, a resumed run and the
split's source groups are unaffected by which mode is in use.

Rows are matched to files by **path relative to `<SRC>`**, then by **bare file
name**, then by either of those **case-insensitively** (a CSV and a filesystem
routinely disagree on case, and on Windows the disagreement is invisible).

Every run prints what the CSV covered, plus the class histogram — the cheapest
check that the labels are the ones you expect:

```
[labels] metadata.csv: 2628 row(s), 4 label(s)
[labels] 2628/2628 source file(s) labelled from metadata.csv (2628 row(s))
  2628 files, 4 classes
  classes: drum 700, guitar 700, violin 700, piano 528
```

Five things stop the run rather than being guessed:

* **a column that cannot be identified**, or two candidates for the same role.
* **a source file with no row.** The CSV must cover the corpus.
* **an ambiguous file** — a bare name claimed by two rows with *different*
  labels. Put the relative path in the `file` column, or drop the duplicate row.
* **a label that cannot be a directory name** (`gui/tar`, `A:B`). The class name
  *is* the folder name, so sanitizing it silently would break the
  `image_root/<class>` match for exactly those classes.
* **a change of mode on an existing `<OUT>`.** The mode is recorded in
  `source_manifest.json`; re-running the other way does not overwrite the
  dataset, it writes a *second* copy under different class folders. Use a fresh
  output dir.

> **Name classes as plain words.** A label is a folder name *and* a phrase handed
> to a text encoder when the `text` condition is active, and the second job is
> unforgiving. Measured 9 Sept 2026 with `laion/clap-htsat-unfused`, mean cosine
> between four instrument classes: **0.883** named `Sound_Drum` … `Sound_Violin`
> (max 0.975 — `Sound_Guitar` and `Sound_Piano` differ by 0.025), **0.375** as
> hand-written phrases, **0.103** as the bare nouns `drum`, `guitar`, `piano`,
> `violin`. The shared prefix is most of what the tokenizer sees. Nothing in the
> code rewrites a name to fix this: a name rewritten at encode time would stop
> being the name the folder, the split and the panels talk about.

**Conditions: `--conditions` and `--global` are independent and compose.** Either
can be used alone, both together, and each takes any subset — `--conditions f0`,
`--global_conds text`, `--conditions f0,energy --global_conds text,image`. Adding one later
re-reads the audio but re-encodes no latent and drops no condition already on
disk, so a dataset grows a condition at a time.

What differs between the two is not how the model consumes them but **where the
value comes from**, which is what decides where it is stored:

| | computed from | written to | when |
|---|---|---|---|
| every `--conditions` name | that chunk's audio | the chunk's `.npz`, `(T, dim)` | in the stream |
| `--global_conds text` | that chunk's audio | the **same** `.npz`, `(dim,)` | in the stream |
| `--global_conds image` | `--image_root/<class>/*` | `global_conditions/image/<class>.npy` | after the stream |

So `text` is extracted and resumed exactly like `f0`: it is the CLAP embedding
of the chunk's own audio, which at inference is swapped for the CLAP embedding
of a written prompt (the two towers share one space). It is *not* the embedding
of the class name — that would make it carry the same single bit as `image`,
and no ablation could then separate the two.

`--global_conds image` encodes **every** picture of a class, not a sample of them: the
training draws one at random per epoch, so a cap here would silently cap that
augmentation for every run that ever reads the dataset. The `.json` beside each
bank lists the file names in row order, which is what lets a test panel show
the picture a generation was conditioned on.

A bank is reused only while it still **describes its folder**: a re-run compares
the recorded file list with what is on disk and re-encodes the classes whose
pictures changed, saying which. So "drop ten new pictures into a class and
re-run the preprocessing" works — you do not need `--force` (which would also
recompute every chunk condition in the dataset). A class whose folder is
unchanged is never re-encoded, and CLIP is not even loaded when none changed.
Growing a bank changes the image condition of that class, so a checkpoint
trained before the change was trained on a different bank.

`--global_conds text` also writes `global_conditions/text_vocab.{npy,json}`: the
phrase list in `conditions.TEXT_LABEL_VOCAB`, CLAP-encoded once and stored WITH
the dataset. The stored `text` condition is the embedding of a chunk's own
audio, and CLAP has no decoder, so a panel has no sentence to show; the
vocabulary lets it name the **nearest phrase** instead, always printed with its
cosine because it is a retrieval, not a translation. Editing the list and
re-running the preprocessing rewrites the labels without touching a single
chunk of audio.

### The per-chunk text description — `--text_labels_n`

The stored `text` condition is a 512-d CLAP vector and CLAP has no decoder, so
nothing about a chunk is readable from it directly. `--global_conds text` therefore
also writes, **for every chunk of the dataset**, the text that describes it:

```
global_conditions/text_labels.jsonl
  {"chunk": "Sound_Violin/xxx__c0003", "class": "Sound_Violin",
   "phrases": [["a bowed tremolo", 0.3112]],
   "caption": "Sound_Violin · \"a bowed tremolo\" (+0.31)"}
global_conditions/text_labels_cos.npy    (n_chunks, n_phrases) float16
global_conditions/text_labels.json       fingerprint, phrase list, counts
```

`--text_labels_n` counts the caption's terms **including the class**: 1 = the
class alone, 2 = class + the nearest phrase, 3 = class + the two nearest.

Two properties are worth stating explicitly, because they are the reason the
file exists at all rather than being computed on the fly:

* **It covers the whole dataset, not the panels.** This description used to be
  computed inside the training process, at panel-drawing time, for the ~16
  validation samples that get a panel, and it existed nowhere else. How many
  panels a machine can afford must not decide how much of the corpus is
  described. It is a property of the dataset, so it lives with the dataset.
* **The training does not decide how much of it to show.** It reads the caption
  the dataset carries (`load_text_captions`), so the words under a panel are the
  words on disk by construction. A dataset without the sidecar (built before it
  existed) falls back to the old in-process retrieval, so older runs keep their
  captions.

The full cosine table is stored alongside — (n_chunks × n_phrases) float16, about
20 MB for 50k chunks and 200 phrases — so `--text_labels_n` is not a commitment:
a different N, a threshold, "every chunk scoring above 0.3 on *a dry close
recording*", are all re-derived from the `.npy` without re-reading one `.npz`.
Row *i* of the table is line *i* of the `.jsonl`.

Alongside the descriptions, the CLAP **text** embedding of each DISTINCT caption
is stored (`text_labels_emb.npy`) — four vectors for a four-class corpus at
`--text_labels_n 1`, not one per chunk. That is what lets the validation
generate FROM the description (`sampling.validation_text_from_caption`, §3)
without CLAP ever entering the training process.

The caption is encoded **verbatim**: the only difference between what you read
and what CLAP is given is punctuation (`violin · "a bowed tremolo" (+0.31)` is
encoded as `violin, a bowed tremolo` — quotes and cosines are bookkeeping for a
reader). Both strings are stored, as `captions` and `captions_text`.

Name classes as plain words — see the box under `--label_source`: the label is
what the text encoder is handed, and `Sound_Guitar` vs `Sound_Piano` scores 0.975
where `guitar` vs `piano` scores far lower.

The mean cosine between the caption vectors is printed at the end of every run
that writes them — a flat text curve three hours into a training should never be
the first sign that the captions were inseparable.

Rebuilt automatically whenever the vocabulary changes (the fingerprint covers
the phrases, the CLAP checkpoint and N). **Changing the vocabulary therefore
costs one cheap pass and no audio is touched**: the vectors are already on disk
and no model is loaded for this step.

Read the phrases for what they are: a **closed-vocabulary retrieval**, never a
description. The class is exact — it is the chunk's folder — while a phrase at
+0.08 is merely the least bad match in the list, which is why the cosine is
stored and printed next to every one of them.

`--save_wav` takes **which splits** to write: `none` (default), `all`, or a subset
such as `val` / `val,test`. A bare `--save_wav` still means `all`. These WAVs are
the **real source audio** — they never pass through the encoder — which is what makes
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
(and optionally WAVs) remain — it decodes the latent back to audio, with the
dataset's codec (DAC or EnCodec), if no WAV is present.

### The `chord` condition — crema, ported to PyTorch

`chord` is harmony as a chord recognizer hears it: for every frame, the
probability that each of the 12 pitch classes belongs to the chord being played
— the `chord_pitch` output of [crema](https://github.com/bmcfee/crema) (McFee &
Bello, ISMIR 2017). Same shape as `chroma`, `(n_frames, 12)` in `[0, 1]`, but
where chroma measures how much *energy* each pitch class has (melody, overtones
and drums included), `chord` says which notes make up the *chord*.

crema's released model is a Keras 2.2.2 / TensorFlow file, and crema no longer
runs as it is: it does not import under Keras 3 — every TensorFlow ≥ 2.16, so both
the local env and IRCAM's tf2.18 (crema issue #41) — and its feature library,
pumpp, fails with scikit-learn ≥ 1.6. `crema_chord.py` re-implements the network
in PyTorch and loads the **original weights unchanged** from
`crema_chord_weights.npz` (2.1 MB, next to it). Nothing is retrained.

- **Verified** against the original (crema 0.2.0 under Keras 2), same process, on
  real chunks of four instrument classes, whole tracks, generations of this
  model and edge cases: features bit-identical, outputs within 3e-6, the same
  arg-max chord on 100% of the frames. Check a new machine with
  `python crema_chord.py --selftest`.
- **Add it** like any frame condition, incrementally (same chunk/acoustic flags
  as the dataset was built with):
  ```bash
  python preprocess_stream.py <SRC> <OUT> --device cuda --acoustic_rules --conditions chord
  ```
  then list `"chord"` in `conditioning.enabled_frame`. It runs on CPU in the
  preprocessing workers, like chroma (~0.1 s per 5 s chunk).
- **The weights file travels with the code.** Copy `crema_chord_weights.npz` to
  IRCAM together with `crema_chord.py` (`.gitignore` keeps it despite `*.npz`).
- **Read it knowing the context.** crema was built to read whole tracks. On 5 s
  chunks it names the same chord as on the whole track in 58% of the frames
  (cosine 0.91 between the 12-d vectors, measured on 3 tracks); the cause is the
  network's context, not the cut of the audio. The training target and the
  re-extraction from a 5 s generation both see 5 s, so `chord/cosine` compares
  like with like.
- **Silence is zeros.** crema has no notion of absolute level (a triad reads the
  same at −20 and at −90 dBFS) and hears chords in silence too, so the extractor
  zeroes the frames below `silence_db` (−60 dBFS, in `CONDITION_CONFIG`, as for
  f0): silence means "no chord", as zeros mean absence for every condition.
- `python crema_chord.py song.wav` prints a chord timeline of any file — a quick
  look at what the condition is made of (frame-wise: crema's own HMM smoothing
  is not applied).

### The `midi` condition — the dataset's own MIDI

`midi` is the one frame condition that is **not computed from the audio**. For a
dataset that carries its MIDI — renders of MIDI files (Lakh through
`render_midi.py`, Slakh) or any audio with a MIDI aligned to it — it gives the
model the notes being played, per frame, as 185 values 0/1: 88 keys **sounding**
(A0–C8), 88 **note starts** (24% of Lakh's notes restart on the same key within
one frame, invisible to "sounding" alone) and 9 **drum classes** (Magenta's:
kick, snare, closed / open hi-hat, low / mid / high tom, crash, ride). The
instruments are merged and velocity is not kept.

- **Where the MIDI goes.** Next to its audio, with the same name:
  `rock/song.mp3` + `rock/song.mid` (or `.midi`, any case). Its time 0 must be
  the start of the audio, as it is for a render. No flag:
  ```bash
  python preprocess_stream.py <SRC> <OUT> --device cuda --conditions midi
  ```
  Every source is checked before anything is written: an audio file without its
  MIDI, or with two, stops the run with the list. Each chunk gets the MIDI
  window that starts where the chunk starts in the original file — the
  `--acoustic_rules` trim included — never inferred from the chunk index (which
  counts the chunks that survived the gates).
- **Adding it to a dataset that already exists** (Lakh on the server): put each
  `.mid` next to the audio file the dataset was built from — `render_midi.py`
  mirrored the tree, so the relative paths already match —
  ```bash
  robocopy <Lakh_midi> <Lakh_mp3> *.mid /S                                 # Windows
  rsync -a --include='*/' --include='*.mid' --exclude='*' <Lakh_midi>/ <Lakh_mp3>/   # Linux
  ```
  then re-run on the **same SRC** with the **same parameters** the dataset was
  built with (a different one stops the run, see `dataset_meta.json`):
  ```bash
  python preprocess_stream.py <SRC> <OUT> --conditions midi --skip_encoder
  ```
  The latents are not touched (`--skip_encoder` reads T from them), the other
  conditions stay, only the `midi` key is added. On a 1551-chunk Lakh subset:
  15 s; latents, wavs and chroma bit-identical before and after; the stored roll
  equal to the MIDI's at the chunk's start in 1551/1551 chunks (1004/1004 with
  `--acoustic_rules`, trims up to 3 s). Disk: ~1.7 KB per chunk (~1.2 GB for
  757k chunks).
- **Timing.** The MIDI is read with mido, which applies the tempo changes of
  every track as a player does (pretty_midi reads track 0 only and drifts on
  2.5% of Lakh's files). Against the renders, the onsets a transcriber finds in
  the audio sit 0.0 ms (median) from the stored roll.
- **Re-extraction: YourMT3+** (`midi_transcriber.py`). A generated chunk has no
  MIDI: its notes are transcribed back with YourMT3+ (Chang et al., IEEE MLSP
  2024; checkpoint *YPTF.MoE+Multi (noPS)*), polyphonic and with drums, on
  `metrics.fidelity_device`, and rolled with the same function as the dataset's
  MIDI. Code and checkpoint (~0.6 GB) are downloaded on first use from the
  authors' HuggingFace Space at a pinned revision into the HuggingFace cache —
  nothing of it is in this repository (its GitHub LICENSE is GPL-3.0). It needs
  **transformers 4.45.x** (see Installation); check a machine with
  `python midi_transcriber.py --selftest`. It is loaded at startup, so a run
  that cannot transcribe stops before training.
- **Its ceiling.** On the real Lakh renders against their own MIDI (200 chunks
  of 50 pieces): note F1 0.53, drum F1 0.59 (kick 0.80, snare 0.70, closed
  hi-hat 0.57, crash 0.53, ride 0.40, toms 0.26–0.30, open hi-hat 0.21). The
  midi rows of a run cannot go above that; read them against these numbers, not
  against 1. YourMT3+ reads audio in 2.048 s segments, so the last 0.9 s of a 5 s
  chunk sits in a mostly zero-padded one: recall there 0.39 instead of ~0.45
  (drums 0.45 instead of ~0.57).
- **Cost.** 1.2 s per 5 s chunk on a laptop RTX 5050 (0.8 GB), 4.3 s on CPU;
  every metrics step transcribes 2 × `n_metrics_samples` generations (with and
  without the condition): ~20 min for 512 on that GPU. Several GPUs split it.
- **Table rows** (see *influence_family* below): ours `cosine` (sounding keys,
  per frame), `onset_corr`, `drum_corr`; mir_eval `note_precision`,
  `note_recall`, `note_f1`, `drum_precision`, `drum_recall`, `drum_f1`.
- `extract_conditions.py` refuses `midi` (it never sees the sources and their
  MIDI); `sampling_cond.py` with a reference audio gets the `midi` condition by
  transcribing it (it does not read a `.mid`).

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
`training_cond_default.yaml` means "the dataset's own `splits.json`"; set it only to point
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
python training_cond.py --config configs/training_cond_default.yaml \
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
- **Cache safety:** the normalizer and the FD reference (in the latent space of
  the dataset's codec, DAC or EnCodec) are tied to the dataset via
  `cache_dir/cache_meta.json`. A stale or unverifiable cache **hard-fails**.
- CLI overrides use dotlist syntax (e.g. `model.kind=B training.lr=5e-5`).
- Resume: `--resume runs/<prev>/checkpoints/checkpoint_step50000.pt`.

### Several GPUs — `launch_training_cond.py --num-gpus N`

```bash
python launch_training_cond.py --num-gpus 2 --config configs/training_cond_default.yaml [overrides]
```

- **One GPU does not change.** `python training_cond.py ...` and
  `--num-gpus 1` run the training in a single process, as before: none of the
  multi-GPU code runs (checked on 3 Oct 2026: with `training.lr=0` the old and
  the new code give the same losses, metrics, panels, checkpoints and dumps,
  byte for byte).
- **N GPUs** (PyTorch DistributedDataParallel, NCCL): the launcher locks the
  N GPUs, starts one training process per GPU and stays alive as their parent —
  the locks are its. Each process takes the locks over before importing torch
  (LAB rules, "Sharing GPU locks with sub-processes"). If one process dies, the
  launcher stops the others and exits with its code.
- **The batch.** `data.train_batch_size` is PER GPU: one optimizer step sees
  `train_batch_size × grad_accum × N` samples, printed at startup
  (`[multi-GPU] ... batch per optimizer step`). The learning rate is NOT
  rescaled: with N the batch changes, and the LR is yours to adapt. With
  `grad_accum` > 1 the gradients are averaged over the GPUs once per optimizer
  step.
- **Split over the GPUs:** the training batches (each GPU its own share of
  every epoch); the validation loss (batch b on GPU b mod N, the sums added
  up); the metrics step — generation j of the `n_metrics_samples` is
  generated, decoded and measured on GPU j mod N, from the SAME starting noise
  it has on one GPU, and GPU 0 adds up what the others measured and computes
  FD, KL, FAD and the Condition_influence panel.
- **GPU 0 alone:** TensorBoard, the checkpoints, the console; the references
  at startup (normalizer, FD / FAD reference, probe banks), while the others
  wait; the listening panels (test, probe, uncond cards). The other processes
  write their console to `runs/<run>/rank<R>.log`; their errors still reach
  the launcher's console.
- **Noise:** each GPU draws its own x0 / t / CFG dropout (the seed is offset by
  the rank); the weights, the EMA and the data permutation are the same
  everywhere.
- **Resume** works as on one GPU (statistically equivalent, not bit-identical),
  and a checkpoint written with N GPUs can be resumed with any number.
- **Waiting.** The other GPUs wait for GPU 0 inside a collective while it works
  alone; the time-out of that wait is 180 min (`DIST_TIMEOUT_MIN` in
  `training_cond.py`).
- **Stopping.** Ctrl+C: GPU 0 leaves the training loop and writes
  `checkpoint_last`, as a single-GPU run does, then the launcher stops the
  others; a second Ctrl+C stops everything at once (the last checkpoint can
  then be lost). A kill (SIGTERM) stops every process at once, without
  `checkpoint_last` — as it does a single-GPU run.
- **On the servers** the locks are SOFT from the second GPU on (LAB rules,
  "GPU lock levels"): a user with hard-lock authority can take one, and then
  the whole run stops — resume from the last periodic checkpoint.
- The test too: `launch_test_cond.py --num-gpus N` (see §4).

### How strongly the frame conditions are followed

Two levers, and only one of them can be changed after the fact.

**`conditioning.guidance_scale`** (sampling time, free to change). The usual CFG
extrapolation `v_u + w · (v_c − v_u)`. Costs nothing to retry on a trained
checkpoint; above ~5 it buys adherence with artefacts.

**`model.frame_reinject_every`** (architecture, must be chosen BEFORE the run).
With the default `0`, a frame condition is concatenated at the input and never
presented again: it touches exactly **one** matrix (`input_proj`) and then has to
survive the whole depth inside the residual stream. A *global* condition, by
contrast, modulates **every** block and the final layer through AdaLN — 6 points
of contact on `S`, 28 on `XL`. That asymmetry is JASCO's design; Music ControlNet
(Wu et al., 2024) makes the opposite choice and keeps feeding the temporal
control back into the trunk.

Setting it to `N > 0` adds that path: the projected conditions are re-added to
the hidden state before every `N`-th block, through a per-block **zero-initialised,
bias-free** `Linear(sum(out_dim) → hidden)`, scaled by a **per-condition gate**
read off the AdaLN vector `c`:

```
h ← h + W_i · ( frame_proj ⊙ g_i )        g_i = W_g,i · SiLU(c)
```

Without the gate the term re-added at a given block is a function of the
conditions alone — the *same tensor at every denoising step* — while every other
path in the network is free to weigh itself against `t`: each block's own
`shift`/`scale`/`gate` come out of `adaLN_modulation(c)`, and the global
conditions ride inside that same `c`. The re-injection was the only contribution
in the model without that freedom.

```yaml
model:
  frame_reinject_every: 0   # 0 = off (JASCO) | 1 = every block | 2 = every other | ...
```

- **Each frame condition gets its own scalar** (f0, chroma, rhythm, energy), so
  they no longer necessarily rise and fall together along `t`. The projection
  still acts on their concatenation — what is per-condition is the *weight*, not
  the mixing. Before the gate the add could only be weighted as one block. It is
  inert, and says so at build time, when `conditioning.enabled_frame` is empty.
- **The gate reads `c`, not `t` alone.** With `conditioning.enabled_global`
  non-empty, `c = t_emb + g`, so how much frame condition is injected also
  depends on text/image. Deliberate — it is the DiT's own
  single-conditioning-vector design — and inert on a model with no global
  conditions, where `c` **is** the timestep embedding.
- **Block 0 is never re-injected**: `input_proj` has just delivered the
  conditions to it.
- **Zero-init** means step 0 is *exactly* the input-only model, so the extra path
  is learned only if it earns its weight — the same principle as adaLN-Zero and
  ControlNet's zero-convs. `python test_functions/test_network_cond.py` asserts this, together with
  the fact that the path then really changes the output and receives gradient.
- **The gate starts at the identity** (weight 0, **bias 1**), not at zero.
  Zeroing it as well would leave the path dead on arrival: with the gate at 0 the
  projection receives no gradient, and with the projection at 0 the gate receives
  none either, so neither could ever leave the origin. The step-0 guarantee is
  unaffected — it comes from the zero-init *projection*. The order in which the
  path wakes up is: step 0 the final layer, step 1 the projection, step 2 the
  gate.
- **Cost:** one `Linear` per selected block, plus a `Linear(hidden → n_conditions)`
  for its gate. With f0 alone (`out_dim` 16) on `XL` that is 27 × 16 × 1152 =
  0.5M params for the projections and 31k for the gates (~0.05% of the model);
  with all four frame conditions (sum 128) the projections are 4.0M. On `S` with
  f0 alone: 40 960 + 2 565 = 0.04M, 0.14% of the model.
- **It changes the state_dict**, so it is checked on `--resume` exactly like
  `model.kind`, and a checkpoint trained with one value cannot be resumed with
  another. Trying it means a **new run**, not a resume. `sampling_cond.py` and
  `test_cond.py` read the value back from the checkpoint automatically;
  checkpoints written before this option existed read back as `0`, which is what
  they were trained as. A checkpoint trained with re-injection *before the gate
  existed* is refused with an explanation rather than a wall of missing keys
  (`network_cond.check_ckpt_reinject_gate`): the stride matches, the weights do
  not.
- **`frame_reinject_every: 0` is bit-identical to the pre-re-injection code.**
  Neither the projections nor the gates are built, the parameter count and the
  initialisation from a given seed are unchanged, and an old checkpoint loads
  with `strict=True`. That is the value to use for a run meant to be compared
  with results produced before any of this existed.

### Text cross-attention — `model.text_cross_every`

**The problem it solves.** With `text_cross_every: 0` the text reaches the blocks
as one pooled vector, summed into the AdaLN conditioning vector `c`. From `c`
come `shift`/`scale`/`gate`, which `modulate` broadcasts over time — so the text
can apply exactly one affine transform, the same at every one of the ~431
frames. No word can act on frame 200 differently from frame 10, because the text
never enters an attention at all. That channel was designed in the official DiT
to carry a class label, about ten bits.

**What it adds.** A cross-attention sub-layer in every selected block, between
the self-attention and the FFN — the order of PixArt-alpha and of the
LDM/Stable-Diffusion block:

```
x = x + gate_msa * attn(modulate(norm1(x)))
x = x + cross_attn(norm_cross(x), text_tokens)     <- new
x = x + gate_mlp * mlp (modulate(norm2(x)))
```

Q comes from `x`, K and V from the caption's **token states** (768-d, CLAP's
text tower before the projection — not the pooled 512-d vector, which is a
single key and therefore a learned bias, not an attention). The conditioning
becomes time-varying and content-dependent, and its capacity grows with the
length of the text.

No AdaLN modulation on this sub-layer and no RoPE on the context, both as in
PixArt: RoPE rotates by position in the *audio* sequence, and the text keys are
not at any audio position.

**The null.** CFG needs an unconditional branch, and here it cannot be "no
keys": a softmax over an empty key set is NaN. It is a **learned null token**,
trained by the same CFG dropout that trains the zero vector of the other
conditions, and the dropout drops it on exactly the same coin as the pooled
text — one condition, two views.

**Zero-init and warm start.** The output projection starts at zero, so a model
built with the cross-attention computes *the same function* as one without it
(verified bit-identical). That is what `paths.init_from` is for: it seeds a NEW
run from an old checkpoint's weights, allowing only the keys the new
architecture adds to be missing, and refusing anything else with the list. Use
it to turn the cross-attention on for a model that is already trained —
`--resume` cannot, and should not, load one architecture into another.

At step 0 the gradient reaches only `W_o` (the trunk is fully zero-gated by
adaLN-Zero and the zero final layer); `K`/`V`/`Q` start learning once `W_o` has
left the origin, and the null token once some sample is dropped.

**What it needs from the dataset.** `global_conditions/text_labels_tok.npy`,
written by `preprocess_stream.py --global_conds text` alongside the pooled caption
embeddings. A run with `text_cross_every > 0` on a dataset without it **refuses
to start** rather than attending to the null token for a week.

**Cost.** On `L` with a stride of 1: +88M parameters on 467M (+19%), plus the
optimizer state that goes with them.

### What the text slot is fed — `conditioning.text_source`

Independent of the cross-attention, and about the *pooled* vector only:

| value | the text slot receives |
|---|---|
| `audio` | the chunk's own CLAP **audio** embedding (the AudioLDM arrangement; what every run before 14 Sept 2026 did) |
| `caption` | the CLAP **text** embedding of that chunk's written description — the vector a prompt puts there at inference |
| `mix` | one or the other per sample, `text_mix_p` of the caption; random on train, deterministic on val/test |

Measured on `dataset_ready_f0_chr_en_ryt_text_instruments` (14 Sept 2026):
audio-to-audio cosines within a class sit at +0.70..+0.82, which is what the
model sees in training, while a chunk's cosine to its own caption's text vector
is +0.24..+0.52. Under `audio`, inference hands the slot a vector from a region
it was never trained on.

The **cross-attention context is not governed by this**: it is always the
caption's tokens, because CLAP's audio tower returns one vector and there is no
sequence to attend over.

What it does **not** fix: a condition that carries no information on the data.
On material where CREPE returns mostly unvoiced (musique concrète, heavily
processed sound), the f0 curve is close to noise and no amount of re-injection or
guidance recovers a control signal that is not in the `.npz` to begin with.

### Self-attention — `model.attention`

| value | every block's self-attention |
|---|---|
| `standard` (default) | the RoPE attention of `network.py`: 16 heads × 72 on `XL`. Every run before 6 Oct 2026; bit-identical to the code before the option existed |
| `differential` | the **DIFF Transformer** of Ye et al. 2024 ([arXiv 2410.05258](https://arxiv.org/abs/2410.05258)), as in its reference code (`microsoft/unilm`, `Diff-Transformer/multihead_diffattn.py`) |

**What the differential attention computes.** Half the heads; each has two
query/key pairs over the same values:

```
A1 = softmax(q1 k1ᵀ / √d)      A2 = softmax(q2 k2ᵀ / √d)
o  = (A1 − λ·A2) v             λ = exp(λq1·λk1) − exp(λq2·λk2) + λ_init
o  = RMSNorm(o) · (1 − λ_init)
```

The weight a position receives is the difference of two maps: what both spread
alike over the sequence cancels, what `A1` puts above `A2` survives, and the
rows sum to `1 − λ`, not 1. `λ` is learned, one per block, starting from
`λ_init = 0.8 − 0.6·exp(−0.3·block)` (0.20 in block 0, 0.80 in the deep blocks of
`XL`). In the paper's retrieval test the attention given to irrelevant context
drops from 0.49–0.54 (Transformer) to 0.01–0.02.

- **The per-head RMSNorm is part of the method.** λ changes with the depth and
  during training, and the two maps cancel more or less from head to head, so
  each head's output has a scale of its own; the norm brings them to one, and
  `(1 − λ_init)` puts it back where a standard head's output sits. Paper,
  Table 6 (1.4B models, validation loss): 3.062 with it, 3.122 without — worse
  than the standard Transformer's 3.087. Stable Audio 3 uses a variant without it
  (λ fixed at 1) in its medium and large models, with no published ablation.
- **Same parameters and FLOPs.** The qkv / proj matrices keep their shapes and are
  read differently (on `XL`: 16 query/key maps of width 72, two per head, and 8
  values of width 144); the extra parameters are the four λ vectors and the norm's gain,
  `6 × head_dim` per block (12k on `XL`). Same RoPE.
- **Numbers outside language:** DiT-S/2 on ImageNet 256 (400k steps, batch 256):
  FID 67.2 → 63.9 ([arXiv 2511.00833](https://arxiv.org/abs/2511.00833)). Nothing
  published on audio.
- **It changes the state_dict and what the weights compute**, so it is checked on
  `--resume` like `model.kind`, and neither `--resume` nor `paths.init_from` can
  switch it: a different attention is a new run. `sampling_cond.py`,
  `test_cond.py` and `compare_guidance.py` read it off the checkpoint's
  **weights** (`network_cond.ckpt_attention`: a block with `attn.lambda_q1` is
  differential); checkpoints written before the option read back as `standard`.
- `python test_functions/test_network_cond.py` checks the module against an explicit, step-by-step
  computation of the formula above, and that λ and the norm receive gradient in
  every block.

### QK-norm — `model.qk_norm`

| value | q and k of every attention |
|---|---|
| `true` (default) | per-head RMSNorm with a learned gain (one for q, one for k, shared by the heads, ε = 1e-6), after the projection and before RoPE, as in Stable Audio 3 |
| `false` | used as they come out of the projection; every run before 9 Oct 2026 |

Stable Audio 3 ([arXiv 2605.17991](https://arxiv.org/abs/2605.17991), Sec. 2.2,
eq. 2) normalises q and k in every self- and cross-attention "to prevent
dot-product outputs from growing unconstrained"; both released DiTs have it
(`small` with standard, `medium` with differential attention). Without it the
logit q·kᵀ/√d grows with the norms of W_q, W_k and with the AdaLN scale, and
nothing bounds it. An XL run on SHS (batch 16, lr 1e-4, no weight decay, no
clipping) trained smoothly for 380k steps, then one isolated grad-norm spike sent
the train, validation and EMA losses back to their early-training values.

- Applied in `SelfAttention`, in both maps of `DifferentialSelfAttention` (the
  same norm for q1/q2 and for k1/k2, as in SA3) and in `CrossAttention`.
  `2 × head_dim` parameters per attention layer (4k on `XL`).
- **It changes the state_dict and what the weights compute**, so it is checked on
  `--resume` and `paths.init_from` like `model.attention`: a different value is a
  new run. `sampling_cond.py` and `test_cond.py` read it off the checkpoint's
  weights (`network_cond.ckpt_qk_norm`: an `attn.q_norm.weight` means on);
  checkpoints written before the option read back as `false` and load as before.
- `python test_functions/test_network_cond.py` checks every attention against an
  explicit computation, that q/k weights scaled ×50 leave the output unchanged
  with the norm (and not without), and that the gains receive gradient.

### Learning rate, optimizer, clipping — `training.lr_schedule`, `training.adam_*`, `training.weight_decay*`, `training.grad_clip`

| key | default | Stable Audio 3 | what it does |
|---|---|---|---|
| `lr_schedule` | `inverse_power` | `inverse_power` | after the linear warmup, `lr · (1 + step / lr_inv_gamma)^−lr_power`; `cosine`: constant until `decay_start_frac`, then cosine to 0 (every run before 9 Oct 2026) |
| `lr_inv_gamma`, `lr_power` | 1e6, 0.5 | 1e6, 0.5 | the lr is ×0.95 at 100k steps, ×0.85 at 380k, ×0.71 at 1M |
| `adam_betas` | [0.9, 0.999] | [0.9, 0.95] | AdamW betas |
| `adam_eps` | 1e-8 | — | AdamW eps |
| `weight_decay` | 0.0 | 0.01 | decoupled weight decay |
| `weight_decay_matrices_only` | true | — | with `weight_decay > 0`: two AdamW groups, decay on the weight matrices only; biases, norm gains (q/k norms included), the λ vectors and the text null token are not decayed |
| `grad_clip` | 0.0 (off) | not stated | global-norm clipping after gradient accumulation; with grad norms of 0.1–0.3, 1.0 cuts only spikes |

The paper's learning rates (Muon 1e-5, AdamW 1e-6, Sec. 3.5) belong to its
Muon+AdamW hybrid and the paragraph speaks of generator and discriminator; it
does not say they hold for pre-training, so `training.lr` keeps its value.

- The startup log prints what is in effect: `LR schedule: ...` and
  `Optimizer: AdamW | betas=... | eps=... | weight_decay=... | grad_clip=...`.
- **Resume:** keys a checkpoint predates are set to what it was trained with
  (`cosine`, [0.9, 0.999], 1e-8, one group), not to the YAML defaults.
  `grad_clip` and the schedule shape passed on the command line take effect;
  lr, betas, eps and weight_decay come back with the optimizer state, and the
  startup log says so when they differ from the config.

### Where the training puts its timesteps — `training.t_sampler`

`t = 0` is pure noise, `t = 1` the data.

| value | how the training draws `t` | below 0.1 | above 0.8 | median |
|---|---|---|---|---|
| `logit_normal` (default) | logit-normal (0, 1), SD3's `lognorm(0.00, 1.00)`; every run before 6 Oct 2026 | 1.4% | 8.3% | 0.50 |
| `truncated_logit` | Stable Audio 3's pre-training recipe ([arXiv 2605.17991](https://arxiv.org/abs/2605.17991)), for 431 frames | 9.9% | 2.3% | 0.34 |

`truncated_logit` is the two steps of their code, mirrored into this repo's convention (SA3
has `t = 1` = noise): the logit-normal (0, 1) **truncated at 0.075 and rescaled**
(`truncated_logistic_normal_rescaled`), then **shifted towards the noise** by an
amount that grows with the sequence length (`DistributionShift`, the one their
released models ship with): `logit(t') = logit(t) − μ`, `μ` linear from 0.5 at
256 tokens to 1.15 at 4096 — 0.53 for our 431 frames, `α = e^μ = 1.70`. Checked
against their code run as is: 4M draws each, largest gap between the two CDFs
0.0005.

**Why.** Measured on 6 Oct 2026 on XL SHS 200k and XL chord Lakh 50k, from the
model's own estimate of the final latent along the Euler trajectory,
`x̂₁ = x_t + (1 − t)·v`: the content of the whole clip — each channel's mean over
the 431 frames — is 90% fixed by `t = 0.12`, where `logit_normal` puts 2.4% of
the examples and `truncated_logit` 13%; above `t = 0.7` every time scale is already ≥ 99%
fixed, and `logit_normal` puts 20% of the examples there, `truncated_logit` 7%. Below
`t = 0.01` a 200k-step run with batch 16 has seen about 7 examples in all, and
there the model treats the noise as if it were already content.

- **Training only.** The network does not change and every checkpoint samples
  the same way. It may differ on a resume: pass it on the command line, since the
  checkpoint's stored config is merged over the YAML.
- **The validation loss is untouched:** it keeps its own fixed `t`, drawn once
  from the logit-normal (0, 1), so it stays comparable across samplers and with
  every earlier run.
- The startup log says where the draws go: `Training t sampler: truncated_logit (...) | draws
  with t < 0.1 (noise end): 9.9%, t > 0.8 (data end): 2.3%`.
- Above `t = 0.7` the latent changes by less than 1%, but small latent errors can
  still be heard after decoding (with DAC they can change the codes the latents
  are re-quantized to; EnCodec decodes them directly): fewer examples there is a
  trade-off to listen for.

### Where the Euler steps go — `sampling.t_schedule`

`t = 0` is pure noise, `t = 1` the data. The option places the
`sampling.euler_steps` of **every** generation of the run: the metrics step, the
panels, the uncond cards.

| value | where the steps go | steps | evaluated below t = 0.1 | below t = 0.12 | last evaluation |
|---|---|---|---|---|---|
| `uniform` (default) | equal steps from `t_min` to `t_max`; every run before 6 Oct 2026 | 100 | 10 | 12 | t = 0.989 |
| `logsnr_uniform` | Stable Audio 3's inference grid ([arXiv 2605.17991](https://arxiv.org/abs/2605.17991), §4) | 50 | 25 | 26 | t = 0.862 |

`logsnr_uniform` is their `build_schedule` with the `LogSNRShift(rate=0)` their
`models/diffusion.py` uses when a model config sets no sampling shift (their
released `-base` configs set none): the step boundaries equally spaced in
log-SNR, `log(t / (1 − t))`, from −6.2 to 2.0, the first at `t = 0` and the last
at `t = 1`, the same grid for every length; `t_min` / `t_max` do not apply to
it. Checked against their code run as is: largest gap 9e-8 (their float32).
Their base model (flow matching only, like ours) is sampled on it with 50 Euler
steps (CFG 7 in their paper); their post-trained model uses 8 ping-pong steps,
which needs their distillation and adversarial post-training.

```bash
python training_cond.py ... sampling.t_schedule=logsnr_uniform sampling.euler_steps=50
```

- **Sampling only.** The weights do not depend on it: any checkpoint, older ones
  included, can be generated either way.
- **A YAML without the key** (the server's, if not updated) reads `uniform`, and
  the command-line override still works: the key is put in at startup. On a
  resume the checkpoint's value is kept unless the command line says otherwise.
- **Who reads it.** `test_cond.py`, `sampling_cond.py` and `compare_guidance.py`
  take the checkpoint's (`uniform` for a checkpoint older than the option);
  `sampling.t_schedule=logsnr_uniform` on the test's command line and `--t_schedule logsnr_uniform`
  for the other two override it. `compare_schedules.py --schedules uniform logsnr_uniform`
  generates the two from the same noise.
- **An edit** (`sampling_cond.py edit --strength s`) starts at `t = 1 − s`: the
  same log-SNR spacing from there to 2.0 (past `t = 0.881`, i.e. `s` below 0.12,
  equal steps). SA3's own code would put the second step at more noise than the
  first, which an Euler step cannot do.
- The metric curves of two runs are comparable only with the same
  `euler_steps` and `t_schedule`. The startup log says which:
  `Sampling t schedule: logsnr_uniform (...) | 50 Euler steps, 25 of them at t < 0.1 (noise end), the last at t=0.862`.
- **`uniform` generates what it did before, bit for bit**: the same times and
  step sizes, with the same arithmetic.

### Which metrics are computed

`metrics.enabled` selects the distributional metrics, mirroring the unconditional
project's registry. `fd_dac` and `kl_dac` keep their names on both codecs: they
are measured in the latent space of the dataset's codec, DAC or EnCodec.

```yaml
metrics:
  enabled: ["fd_dac", "kl_dac"]   # [] turns them off (and skips their reference)
                                  # add "fad_vggish" for the audio-domain FAD
  seed: 0
  fidelity_device: "cuda"         # "cuda" | "cpu" — device for the re-extraction
                                  # (CREPE / beat_this / CLAP-audio) that feeds the
                                  # Condition_influence table. FD/KL always run
                                  # on the GPU regardless. The device does not
                                  # change the values, only speed: "cpu" is an
                                  # escape hatch if the metrics step runs out of
                                  # VRAM (there the model and the generations are
                                  # resident, plus the codec decoder itself when
                                  # dac_device is "cuda").
  influence_family: "influence_metrics"   # | "mir_influence_metrics" — which
                                  # metrics are the columns of the
                                  # Condition_influence table: ours, or
                                  # mir_eval's. See below.
  mir_threshold: 0.5              # when a chroma / chord pitch class counts as
                                  # ON for the mir metrics. Fixed once.
  dac_device: "cpu"               # "cpu" (default) | "cuda" — where the shared
                                  # codec decoder (DAC or EnCodec) lives. See
                                  # below: this is the one device knob that is
                                  # not free.
```

#### `influence_family` — our influence metrics or mir_eval's

Each row of the `Condition_influence` panel compares the condition given to
the model with the same descriptor re-extracted from the generation, averaged
over the `n_metrics_samples` validation generations of the step (one panel per
metrics step — see "The Condition_influence table" below). `influence_metrics` (the
default) scores it with the project's own numbers; `mir_influence_metrics` with
[mir_eval](https://github.com/craffel/mir_eval) (Raffel et al., ISMIR 2014), the
standard MIR metrics, so the numbers are comparable with the literature.

| condition | `mir_influence_metrics` |
|---|---|
| f0 | `mir_eval.melody`: `raw_pitch_accuracy` (50 cents), `raw_chroma_accuracy`, `voicing_recall`, `voicing_false_alarm` (↓ lower is better), `overall_accuracy` |
| chroma, chord | `mir_eval.multipitch`, on the pitch classes ON in each frame: `chroma_precision`, `chroma_recall`, `chroma_accuracy`, `chroma_miss_error` (↓), `chroma_false_alarm_error` (↓, can exceed 1) |
| rhythm | `mir_eval.beat`, on the beat instants read off the curves with beat_this's own rule (maximum within ±60 ms, probability > 0.5): `beat_f_measure`, `beat_cemgil`, `beat_p_score`, `beat_cmlt`, `beat_amlt`, `downbeat_f_measure` |
| midi | `mir_eval.transcription`, onsets within ±50 ms on the same key (offsets ignored): `note_precision`, `note_recall`, `note_f1`; the same for drum hits on the same class: `drum_precision`, `drum_recall`, `drum_f1`. Ours: `cosine` (sounding keys), `onset_corr`, `drum_corr` |
| energy, text, image | always ours (mir_eval has no counterpart) |

The target is the reference and the generation the estimate, on the shared
frame grid of the codec, DAC or EnCodec (no resampling). A sample whose target has no voiced frame has no raw
pitch / raw chroma / voicing recall (one with no unvoiced frame no false alarm):
it is left out of that mean — the training log prints how many generations
entered each one — instead of entering it as mir_eval's placeholder 0 or 1.

For chroma and chord mir_eval needs the LIST of notes on in each frame, while
both store 12 continuous values: a pitch class is ON when its value is
≥ `metrics.mir_threshold` (0.5) — for chroma, at least half the energy of the
frame's loudest class; for chord, a crema probability of at least one half. On
material without notes (drums, noise) chroma turns 6–7 classes of 12 ON, so there
these metrics measure agreement on noise. The same NaN rule applies: no class ON in
the target anywhere → no recall and no errors; none in the generation → no
precision.

For rhythm mir_eval compares beat INSTANTS, so both curves are first read the
way beat_this reads its own (maximum within ±60 ms — the window beat_this's code
calls ±70 ms: 7 frames of 20 ms —, probability > 0.5, each downbeat moved onto
the nearest beat). The metric functions are called one by
one, not through `mir_eval.beat.evaluate`, which drops every beat before 5 s —
on a 5 s chunk, all of them. A target with no beat (fewer than two, for P-score
and CMLt/AMLt) has no value there; a generation with no beat where the target
has them scores 0. The startup line
`[metrics] extractor=f0 | ... | metrics=mir_influence_metrics` says which family
each condition got. The training is untouched; switching the family changes the
columns, so compare runs within one family.

#### `dac_device` — the one device knob that is not free

The key keeps its name, but it places the decoder of the dataset's codec, DAC or
EnCodec. The other two change speed only. This one trades **wall clock against
VRAM**, and slightly changes the values, so it has its own section.

Measured with the DAC decoder on an RTX 5050 laptop, decoding one 5-second clip
(the EnCodec decoder has not been measured):

| | CPU | CUDA |
|---|---|---|
| per clip | 2538 ms | 194 ms (**13×**) |
| 512 generations (one metrics step: each decoded once) | 21.7 min | 1.7 min |

On a short local run the CPU decoder dominates the wall clock — it can be more
of the run than the training itself. But on the GPU it costs, and **the weights
are the small half**:

```
weights      0.29 GB   resident for the whole run
activations  0.69 GB   peak, one 5-second clip at a time
peak         0.97 GB
```

Nearly 1 GB, landing **during the metrics step**, on top of the model, the
generations and the re-extraction above. Training an `XL` on a 24 GB card —
weights, grads, Adam states and the EMA shadow all live — that spike is what
kills a run at hour 40, and a slow metrics step is far cheaper than losing days.
Hence the default is `cpu`, the historical behaviour: **no existing run changes
unless you ask.**

- **Leave it `cpu`** on a shared or large-model machine (the IRCAM servers).
- **Turn it `cuda`** on a small local run where the wall clock is the binding
  constraint.

Two caveats. First, unlike `fidelity_device` / `fad_device`, this one **does
change the values** very slightly — CPU and GPU floating point differ — so FD
and FAD produced with a GPU decoder are not directly comparable with numbers
produced by a CPU one. Comparable *within* an experiment (all runs on the same
device), not across the switch. Second, asking for `"cuda"` without CUDA is a
**hard error at startup**, not a silent fallback.

The decoder is a load-once singleton, so the device is fixed before the first
use and never changes mid-run; the startup line reports the device it actually
landed on, read back from the model (`EnCodec` in place of `DAC` on an EnCodec
dataset):

```
DAC decoder device: cuda  (~0.2 s per 5 s clip; ~1 GB peak at the metrics step)
[DAC] Model loaded once (CUDA:0) and cached for the whole run.
```

Listing a metric is an explicit request: an unsupported name is a **hard error at
startup**, not a silent skip. `fd_dac` and `kl_dac` share the generated
mean/covariance, so asking for both costs essentially the same as asking for one.

### The Condition_influence table (TensorBoard, Text window)

ONE panel per metrics step (the step slider of the Text window walks it across
training): **a row per metric, grouped by condition** (every metric of every
active condition, ours or mir_eval's per `metrics.influence_family`;
`text/clap_sim` and `image/clip_sim` for the globals), and **four columns**.
The values are means over the `n_metrics_samples` validation samples of that
step — the same generations FD and KL are computed on (and FAD, on
`n_fad_samples` of them). Each sample is generated twice from the same
starting noise, and both generations are compared with the condition the
sample was given:

- **with-cond** — generated WITH its conditions;
- **null** — generated with NO condition: the adherence reached by chance;
- **Δ influence** = with-cond − null — the effect of the conditions (on a ↓
  metric a negative Δ is the pull);
- **valid/used** — on how many samples the three are averaged, out of how many
  were measured.

```
| Condition       | Metric             | with-cond | null   | Δ influence | valid/used |
| f0_validation   | raw_pitch_accuracy | 0.4123    | 0.0517 | +0.3606     | 416/512    |
| f0_validation   | voicing_recall     | ...       |        |             |            |
| text_validation | clap_sim           | ...       |        |             |            |
```

- The three value columns are averaged over the SAME samples: those on which
  the metric is defined on both generations (an f0 accuracy is not defined on
  a target with no voiced frame, a correlation on a flat curve), so Δ is
  exactly the difference of the two means. ⚠️ in valid/used counts the samples
  that did not make it.
- With `sampling.influence_subsets` on, the panel becomes the Δ matrix (one row
  per subset of conditions, one column per metric) with the detailed table of
  every subset underneath.
- The null generations cost no extra sampling under CFG (the unconditional
  velocity is computed anyway), but they are decoded and measured like the
  conditioned ones: the audio part of the panel takes twice as long.
- TensorBoard keeps only 10 texts per tag by default, so on a long run some
  steps disappear from the slider: start it with `--samples_per_plugin text=0`
  to keep them all.
- The test and probe panels (Audio and Images windows) are listening and looking
  material: nothing measured on them is in the panel.

The legend (`Validation/Condition_influence_legend`) is written once per board
and says what every metric is. From 2 to 3 Oct 2026 the panel was replaced by a
single table (a row per step, a column per metric); it is back to the panel
per step. `test_cond.py` writes the same panel on the test split,
`Test/Condition_influence`, at the checkpoint's step, with rows `<cond>_test`
(§4).

### Global-condition similarity (a column of the table)

When the run is conditioned on a global and that global can be scored, the
Condition_influence table has a column saying **how close the generation lands
to the condition it was given**: a cosine, over the same generations as
FD / KL / FAD.

| column | space | how the generation gets there |
|---|---|---|
| `text/clap_sim` | CLAP | CLAP's **audio** tower embeds the generation; the stored condition is already a CLAP vector |
| `image/clip_sim` | CLIP | **Wav2CLIP** embeds the generation into CLIP's space; the stored condition is the CLIP vector of the picture |

Until 2 Oct 2026 the same cosines were also two scalar curves
(`Validation/Metrics/Audio_text_similarity`, `Audio_image_similarity`) on their
own `sampling.n_similarity_samples` generations. Both are gone: the number is
read in the table only. A config that still has `n_similarity_samples` is
accepted (the key is dropped at startup, with a note); on the command line it
stops the run, like any key that does not exist. A global with no embedder
installed still conditions the model; its column reads n/a. Read the absolute
cosine of an audio-image pair knowing that it is small even when the
conditioning works (see the probes section).

### Validating on the description instead of the source audio

`sampling.validation_text_from_caption` (default `false`) changes **what the
validation generations are conditioned on**, and nothing else.

|  | text slot of a validation generation | `text/clap_sim` is then |
|---|---|---|
| `false` | the chunk's own CLAP **audio** embedding — what training uses | generation vs **source audio**, in CLAP space |
| `true` | the CLAP **text** embedding of that sample's written description | generation vs **text**: a real adherence score |

With `true`, the description is the one `preprocess_stream.py` stored for every
chunk (`--text_labels_n`: the class, plus the nearest vocabulary phrases), so
with `--text_labels_n 1` on a per-instrument dataset the validation generates
from `"Sound_Violin"`, `"Sound_Piano"`, ... — i.e. it measures instrument
controllability directly. The table's `text/clap_sim` column is then the cosine
between each generation and the description it was given.

It is the vector a prompt puts in that slot at inference, so this makes the
validation measure what the model will actually be asked to do. **Training is
untouched, and so is the validation loss** — that keeps the training's own
conditioning, otherwise the two loss curves stop being comparable and stop
saying anything about overfitting. Only the generations behind the metrics and
the panels change.

Needs `global_conditions/text_labels_emb.npy` in the dataset (written by
`--global_conds text`). Without it the run says so and falls back.

### FAD-VGGish (optional, off by default)

`fd_dac` and `kl_dac` score the **latent space of the codec** (DAC or EnCodec).
`fad_vggish` scores the
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
  n_fad_samples: 512       # generations scored; each costs a codec decode + VGGish
```

It logs `Fad_vggish_cond` / `Fad_vggish_uncond` next to `Fd_dac_cond` /
`Fd_dac_uncond` (`Fd_encodec_cond` / `Fd_encodec_uncond` on EnCodec; a single
`Fad_vggish` on an unconditional run), on the same two
axes as every other distributional metric.

**What it is compared against** (`metrics.fad_reference`):

| | reference | needs | comparable with the literature |
|---|---|---|---|
| `wav` (default) | the **real** validation wavs | preprocessing run with `--save_wav val` (or `all`) | **yes** |
| `decoded` | the validation latents decoded through the codec (DAC or EnCodec) | nothing | no |

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
the cache is guarded by the same fingerprint as the normalizer and the FD
reference.

**Cost.** The FAD shares the decode of the metrics step: every generation is
decoded once, and the `sampling.n_fad_samples` of them the FAD uses are also
embedded with VGGish. The statistics are accumulated as running sums, so raising
it costs time, never memory.

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
DiT, and `model.frame_reinject_every` becomes inert — it says so at build time
rather than passing silently), no conditions are read from disk, and CFG never
engages.

```bash
python training_cond.py --config configs/training_cond_default.yaml \
    --run_name "uncond_L" model.kind=L \
    conditioning.enabled_frame='[]' conditioning.enabled_global='[]'
```

The **TensorBoard logging follows the mode**: a conditioned run logs the two-axis
scheme (`Fd_dac_cond` vs `Fd_dac_uncond`, `Kl_cond/*` vs `Kl_uncond/*`, plus the
`Condition_influence` table and the test / probe panels); an unconditional
run logs a single axis under the unconditional project's own tags (`Fd_dac`,
`Kl_real_gen`, `Kl_gen_real`), with no table and no panels. On an EnCodec
dataset `Fd_dac` reads `Fd_encodec`. `Train/*` and
`Validation/Loss*` are identical in both.

AUDIO is organised as ONE BLOCK PER PANEL plus the COLLECTED GROUPS. The
dashboard groups cards by the text before the first `/` and lays each group out
as a grid that wraps every 2-3 cards, so the SAMPLE is the group and a numeric
prefix fixes the listening order: the recording, the conditions taken from it,
the generation they produced. The generations without conditions are NOT in
those blocks: they are collected into a group of their own, so they can be heard
as a grid of peers. `audio_panel_tags()` in `training_cond.py` builds the names
and is the single source of truth.

```
test_XX/1_real_test_XX                  the recording the conditions come from
test_XX/2_f0_test_XX                    the sonified f0 target
test_XX/3..N_<cond>_test_XX             energy, chroma, ...
test_XX/N+1_generation_test_XX          the generation they produced
probe_XX/1_f0_probe_XX , /2_chroma_probe_XX , /3_generation_probe_XX

uncond generation/uncond_NN             generations without conditions
ground truth/real_test_XX               (unconditional run only) real TEST recordings
```

The TEST panels are `sampling.n_test_panels` samples of the **test** split,
spread evenly over it and the same at every step (they replaced the validation
panels on 2 Oct 2026); the PROBE panels are `sampling.n_probe_panels` stimuli.
Both are for listening and looking: nothing measured on them is in the
Condition_influence table, which is computed on the validation generations.

The uncond cards are `sampling.n_audio_samples` generations without conditions,
`uncond_00`, `uncond_01`, ... — just a number, since a generation with no
conditions has nothing of any sample in it. Card k starts from the k-th draw of
ONE noise stream seeded with `metrics.seed`, so they are all different. (Until
2 Oct 2026 they were collected from the validation and the probe pass, each of
which restarted the generator from the seed: `uncond_00` and `uncond_04` were
the same audio.) In a PURE-UNCONDITIONAL run there are no blocks: the audio
window is the uncond cards plus `n_audio_samples` validation recordings in
`ground truth/`.

WHO WRITES WHAT, AND WHEN. The `test_XX/` and `probe_XX/` panels are written by
the metrics step alone, every `intervals.metrics`: their stimuli (and the
recording) once per board, at step 0, the generation at every metrics step.
Every `intervals.audio` steps in between, only the uncond cards are refreshed —
by the same function, from the same noise draws, so the slider of `uncond_NN`
walks one generation through training. Nothing conditioned is generated between
metrics steps.

ON DISK, `runs/<run>/audio/` holds what the Audio window shows, and nothing
else:

```
audio/
├── test/test_00/                a test panel
│   ├── real.wav                 the recording (as the codec, DAC or EnCodec, decodes its latent)
│   ├── cond_f0.wav …            each frame condition, sonified
│   ├── conditions.npz           every array / vector the model received
│   ├── text.txt                 the sentence the model received (text runs)
│   ├── image.<ext>              the picture the model received (image runs)
│   ├── info.json                which test sample, guidance, noise draw
│   └── step_0050000_EMA.wav …   its generation at every metrics step
├── probe/probe_00/              a probe panel: the same, without real.wav
├── uncond/uncond_00/            step_0025000_EMA.wav … only
└── ground_truth/                real_test_XX.wav, in a run without conditions
```

A panel's conditions are the same at every step: its folder holds one copy of
them (rewritten at each metrics step) and one generation per step, named by the
step and the weights (`EMA` / `Model`); with `sampling.influence_subsets`, also
`step_<step>_<EMA|Model>_<subset>.wav`. `conditions.npz` holds the frame
conditions by name, `text` / `image` (the vectors), and `text_tokens` /
`text_mask` when the model has the cross-attention. `text.txt` is the exact
sentence CLAP encoded: always for a probe (its prompt); for a test sample, its
caption when the text slot received the caption's vector — otherwise a line
saying the slot received the CLAP audio embedding of the chunk, which no
sentence produced; with the cross-attention it also names the caption whose
tokens the model read. Files are at their real level; only the TensorBoard
cards are peak-normalized. The validation generations the table is computed on
are not written to disk (`sampling.n_val_save` is gone since 5 Oct 2026; a
config still holding it gets one log line).

A run conditioned ONLY on globals — no frame condition, so no waveform to put
beside the recording — has blocks with just the recording and the generation:
`test_XX/2_generation_test_XX`. When `text` is active, the block header carries
the label of the sample (see the probe section), and every card of that sample —
the conditions, the generation, the subset generations, the comparison images,
the image card — sits under that one header.

### Condition subsets — extra columns of the table

`sampling.influence_subsets` adds, for every combination of conditions asked
for, its own COLUMNS to the Condition_influence table, headed by its label
(`no_chroma`, `only_f0`, ...): the same metrics, on the `n_metrics_samples`
generations conditioned on that subset only.

```yaml
influence_subsets: []                            # off (default): the table's own columns
influence_subsets: ["all", "loo"]                # the standard ablation
influence_subsets: ["all", "loo", "singletons"]
influence_subsets: [["f0"], ["f0", "energy"]]    # hand-picked
```

`all` = every active condition (free — the conditioned pass the step already
runs IS it, and its columns are the table's own). `loo` = leave-one-out, the
marginal contribution of each condition at the point where the model is
actually used. `singletons` = each condition alone. Every subset covers EVERY
active condition, not only the ones it gives: the off-subset columns (marked
`°`) are the side effects — give f0 alone and watch what happens to chroma. Each
subset other than `all` is a full extra generation pass, decoded and measured,
so **the metrics step grows linearly with the number of subsets**.

**The subsets vary the FRAME conditions only.** The globals are handed to the
model in every subset, so `text` and `image` columns are never marked `°`, and
what they show is the *side effect*: what dropping chroma does to how well the
generation still matches its prompt. A run with no frame condition has nothing
to subset.

Each subset also gets its own audio card inside every test block
(`test_XX/4_gen_no_chroma_test_XX`), so the whole combination ladder of one
test sample plays side by side.

### Probe sets — the ablation instrument

Elementary synthetic stimuli, unambiguous by construction, with targets
extracted by the run's OWN extractor. They let you hear and see "does this
conditioning work at all", which real material cannot show on its own: a real
f0 contour is ornamented, a real energy envelope jittery, a real beat grid may
not exist on this material, so a middling score there does not separate a weak
conditioning from an ambiguous target. Since 2 Oct 2026 the probes are
listening and looking material only: they have no rows in the table.

| condition | stimuli | the stimulus IS |
|---|---|---|
| f0 | scale, arpeggio, octave leap, sustained note, rests | a waveform |
| energy | crescendo, diminuendo, four stabs, swell, plateau, staircases | a waveform |
| chroma | one sustained triad; simple cadences (I-IV-V, i-iv-V-i in A minor, a tritone resolving to G); a moving pitch class, fifths, clusters, whole-tone and quartal sets; two with rests, one repeated chord | a waveform |
| rhythm | click grids at fixed tempi, downbeat every N, accelerando | a waveform |
| chord | sustained triads, I-IV-V, single pitch class, clusters (the chroma bank's earlier list, kept as its own) | a waveform |
| midi | scale, sustained chord, arpeggio, a repeated note, melody + bass, I-IV-V-I, low and high register, stabs then rest; drums alone (backbeat, rock groove, open hi-hats, tom fill); drums with bass / walking bass / a small band | **notes**: the target is their roll; the waveform is only to listen to |
| **text** | chosen by the dataset (see below): its own labels, in turn, when its captions are single labels (`drum`, `guitar`, `piano`, `violin`, `drum`, …); otherwise 16 instrument/style descriptions ("solo pipe organ in a large reverberant church", "fast electronic dance beat…") | a **string**, CLAP-encoded |
| **image** | 16 abstract figures — colour fields, stripes, checkerboard, rings, gradient, noise | a **.png**, CLIP-encoded |

The two families differ only in medium. A frame target is a `(n_frames, dim)`
curve **extracted** from a waveform (midi excepted: its target is **rolled** from
the stimulus's notes, the way the dataset's MIDI is); a global target is a `(dim,)` vector
**encoded** from a string or a picture, by the run's own CLAP/CLIP — so the
probe drives the model in exactly the space it was conditioned in. Nothing
re-extracts a picture from audio, so a global probe has no "target vs
re-extracted" plot: the image is shown as-is and the prompt names the panel.

> **The text bank follows the dataset.** The probe speaks at the level of
> detail of the text the model is trained on, which the preprocessing records
> as `n_terms` in `global_conditions/text_labels.json` (1 = the caption is the
> class alone). Single labels → the probe uses exactly those labels, the
> strings CLAP was given, repeated in turn to fill the panels (each panel
> starts from its own noise, so a repeat is one more generation, not a copy).
> Richer captions → the 16 descriptions. A dataset without text has no text
> condition and so no text probe at all. At startup the training prints which
> bank it took: `[text-probe] the dataset's captions are single labels ->
> those labels, in turn: drum, guitar, piano, violin`. The labels are read
> from the dataset, never written in the code.

> **Read the abstract IMAGE bank knowing what it is.** What reaches the model is
> not the picture but CLIP's *reading* of it, and a flat colour field has no
> musical meaning for CLIP to read: its embedding lands far from the paintings
> and album covers the training conditioned on. A weak-sounding image probe does
> **not** by itself prove the image conditioning is broken — unlike the f0 bank,
> where a rising scale is unambiguous and a failure is a failure. The table's
> `image/clip_sim` column, on in-corpus images, carries that verdict. What the bank *does* establish
> — and what the build prints as "bank spread" — is whether 16 distinct stimuli
> produce 16 distinct embeddings, i.e. whether the slot can carry information at
> all. The same report is printed for the text bank.

A bank is built for **every condition active in the run** — there is nothing to
turn on per condition. The listening panels have two sizes:

```yaml
sampling:
  n_test_panels: 8    # samples of the TEST split: 0 (off) .. as many as it holds
  n_probe_panels: 8   # probe stimuli per bank: 0 (off) .. 16
```

Each probe panel drives all the active conditions at once with the i-th
stimulus of their own bank. Cost is one generation (plus one re-extraction, for
the images) per panel per metrics step, whatever the number of conditions.
Neither size has anything to do with the Condition_influence table, which is
computed on all the `n_metrics_samples` validation generations.

**Fewer probe stimuli than a bank holds** (16): that many are drawn **at random
but fixed** — seeded by `probe_conditions.PROBE_SUBSET_SEED`, so the subset is the
same at every step and in every run, and a probe panel changes only because the
model did. Random rather than the first N, because the banks are ordered by kind
(the first 4 f0 stimuli are all scales, arpeggios and leaps). The subset is cached
in its own folder (`<probe dir>/subset_NN`), so runs of different sizes sharing a
`cache_dir` do not rebuild each other's bank. A text bank made of the dataset's
single labels is cycled to the size asked for instead, so every label stays in.
**More than 16**: all 16, and the startup log says so.

The keys these two replaced, `n_influence_samples_valid` / `_probe` (and
`n_influence_samples` before them), are gone. Left in a config they are removed
at startup with a log line that names the new ones; given on the command line
they stop the run with the same pointer.

> Removed knobs, no longer read at all: `n_probes` (→ `n_probe_panels`),
> `n_cond_plot` and its alias `n_f0_plot` (every panel is plotted), and the
> per-condition `n_f0_probe` / `n_energy_probe` / `n_chroma_probe` /
> `n_rhythm_probe` (already inert before). A leftover in your config does nothing.

All the banks live in ONE module, `probe_conditions.py`: one bank, one
synthesizer and one plot branch per condition (chord reuses chroma's
synthesizer), so every condition is set up, built, drawn and scored the same
way. Adding one more means adding a bank and a synthesizer, nothing else.

Each FRAME condition gets the SAME two images, N of each, grouped **per sample**
so the Images tab collapses into the same sections as the Audio tab instead of
one flat wall:

```
test_XX/<cond>_target_vs_gen         target vs generated, on a real test sample
probe_XX/<cond>_target_vs_gen        the same, on the unambiguous probe stimulus
test_XX/image_condition              the picture that sample was conditioned on
probe_XX/image_condition             the probe figure, in the same block
```

The two global conditions occupy the same blocks by other means: the **image**
is shown as a card (there is nothing to compare it against), and the **text**
prompt names the block itself — `probe_03 [string quartet playing a slow
sustained chord]`, or `probe_03 [violin]` on a dataset of single labels — and
for a test sample its caption from the dataset (or, on a dataset written before
the captions sidecar, the category plus the nearest phrase to its stored CLAP
vector, `test_00 [Baroque sacred · "solo pipe organ" (+0.58)]`: a retrieval
over `text_vocab`, which is why the cosine is beside it).

Each image is drawn in the form that suits the shape — f0 on a log-Hz axis with
a voicing ribbon, energy and rhythm as overlaid curves, chroma and chord as
paired heatmaps — from the condition re-extracted from that panel's own
generation; its title carries that sample's first metric. Plus the audio block
(`probe_XX/`, `test_XX/`).

Build any probe set standalone to look at it before training:

```bash
python probe_conditions.py f0     ./cache/f0_probe     --n_frames 431
python probe_conditions.py energy ./cache/probe_energy --n_frames 431
python probe_conditions.py chroma ./cache/probe_chroma --n_frames 431
python probe_conditions.py rhythm ./cache/probe_rhythm --n_frames 431
python probe_conditions.py chord  ./cache/probe_chord  --n_frames 431
# an EnCodec dataset: its codec and its frame count (250 for 5 s)
python probe_conditions.py f0     ./cache/f0_probe_enc --n_frames 250 --codec encodec
# the global banks: --n_frames does not apply (one embedding, no chunk geometry)
python probe_conditions.py text   ./cache/probe_text    # always the descriptions
python probe_conditions.py image  ./cache/probe_image
```

Each prints its bank and, for the global ones, the **spread** — every stimulus's
cosine to its nearest neighbour and the bank's mean pairwise cosine, with a
warning on any pair above 0.95 ("these two stimuli drive the model with the same
condition"). A label repeated on purpose is listed as `repeat of [NN]` and left
out of the spread. Read it before trusting a global probe.

> The rhythm bank is ordered by how reliably `beat_this` recovers the intended
> tempo: 12 of the 16 grids come back at the right metrical level, the last four
> (60/160/180 bpm and the ritardando) fall into the tempo-octave ambiguity of
> beat tracking. They are still self-consistent — the target is whatever the
> run's extractor produced — but they are confusing to look at, so they sit at
> the end of the bank.

IMAGES carry the matching overlays for every active condition, in the SAME
per-sample blocks as the audio: `test_XX/<cond>_target_vs_gen` and
`probe_XX/<cond>_target_vs_gen`. See the probe section above for the forms each
takes.

The COLLECTED audio group `uncond generation/` (the same model with no
conditions) holds `sampling.n_audio_samples` cards in all, each from its own
noise draw; `ground truth/` exists only in an unconditioned run.

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
> and let the normalizer refit (a few minutes; the FD reference rebuilds with
> it). Copying the cache across is never worth it.

Monitor: `tensorboard --logdir <runs_dir> --samples_per_plugin text=0` — the
option makes the Text window keep EVERY step of the Condition_influence panel
(TensorBoard keeps 10 per tag by default, so on a long run some steps vanish
from the slider). It is an option of TensorBoard, not of the training: it has
to be given every time TensorBoard is started.

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

"Same numbers" includes the Condition_influence table, which is computed on the
conditioned generations alone. The fused sampler also gets the generations
without conditions for free out of the CFG math (they feed `metrics_uncond` and
the null column of the table); the serial one generates them only when the
table or `metrics_uncond` needs them.

The value does **not** change which samples are generated (the noise is drawn per
sample), so the metric values are the same for any setting. Batching does change
CUDA op ordering, so the fused and the serial path agree to floating-point
tolerance, not bit for bit.

---

## 4) Evaluate & sample

**Test a checkpoint — `test_cond.py`.** The test is the training's metrics step
for one checkpoint, on the TEST split: the same functions write the same things
to TensorBoard, minus the loss.

```bash
python test_cond.py --ckpt runs/<run>/checkpoints/best_model_step<N>.pt
# on an IRCAM GPU server, through the test's own GPU-lock wrapper
# (the twin of launch_training_cond.py: same arguments):
python launch_test_cond.py --ckpt <ckpt>
# the same on N GPUs (see "Several GPUs" below):
python launch_test_cond.py --num-gpus 2 --ckpt <ckpt> --metrics_samples all
```

(Checkpoints are named `best_model_step<N>.pt`, `checkpoint_step<N>.pt`, and
`checkpoint_last_step<N>.pt` — pick the one you want.)

**The config.** The checkpoint carries its training config: model, dataset and
split (`splits.json`), conditions, guidance, Euler steps, `metrics.*`, seed —
everything that decides HOW the numbers are computed, so a test number is the
twin of the run's validation curve. `configs/test_cond_default.yaml` (or `--config`) is
layered on top and decides only what the test does:

```yaml
sampling:
  n_metrics_samples: 512   # test samples measured: a number, all, or 0 (panels only)
  n_test_panels: 8         # test samples shown as panels
  n_probe_panels: 8        # probe stimuli shown as panels
  n_audio_samples: 4       # uncond generations
```

The command line wins over both: dotlist overrides as for the training (e.g.
`sampling.n_metrics_samples=all metrics.dac_device=cuda`), plus the shortcuts
`--metrics_samples N|all`, `--n_samples` (= `n_test_panels`), `--steps`,
`--guidance`, `--seed`, `--duration_s`. A key that does not exist stops the test.
The Euler grid is the checkpoint's `sampling.t_schedule` (`uniform` for a
checkpoint older than the option): e.g. `--steps 50 sampling.t_schedule=logsnr_uniform`
tests it on Stable Audio 3's grid; the JSON records `euler_steps` and
`t_schedule`.

**What it writes**, at the checkpoint's training step, to `runs/<run>/test_logs/`:

| window | content |
|---|---|
| Time series | `Test/Metrics/*`: FD (`Fd_dac_*` or `Fd_encodec_*`, by the checkpoint's codec), KL (both directions), FAD-VGGish if in `metrics.enabled`, on `n_metrics_samples` test samples spread over the split, against the real test data |
| Text | `Test/Condition_influence`: the training's panel on those same generations, at the checkpoint's step -- a row per metric of every condition (`<cond>_test`, text / image included), columns with-cond / null / Δ / valid/used; its legend; the config |
| Images | `test_XX/` and `probe_XX/`: target vs re-extracted condition |
| Audio | `test_XX/` (recording, sonified conditions, generation), `probe_XX/` (stimuli, generation), `uncond generation/uncond_NN` |

The panels are the training's: the same test samples (for the same
`n_test_panels`), the same probe stimuli (for the same `n_probe_panels`), the
same noise, so for the same checkpoint the test shows the generations the
training showed at that step. The text / image a generation is conditioned on is
the training's too: a test sample carries the one stored by the preprocessing
(the CLAP text vector of its caption when `sampling.validation_text_from_caption`
is on), a probe the stimulus of `probe_conditions.py`. An unconditional
checkpoint gets no table and no panels: the metrics on a single axis, the uncond
cards, the real test recordings (`ground truth/real_test_XX`) and the config.

What the Audio window shows goes to `runs/<run>/test_outputs/`, laid out as the
training's `audio/` folder (`test/`, `probe/`, `uncond/`, `ground_truth/` for an
unconditional checkpoint; one generation per tested checkpoint), the numbers to
`runs/<run>/test_outputs/metrics_<checkpoint>_<N|all>.json`. To keep the panel
of every checkpoint tested on the slider: `tensorboard --logdir runs/<run>/test_logs --samples_per_plugin text=0`.
The cost is the generation: hours for a full test split with 100 Euler steps;
`n_metrics_samples: 0` writes only the panels and the uncond cards. The codec
decoder (DAC or EnCodec) runs on `metrics.dac_device` from the checkpoint's config unless
overridden on the command line.

FAD needs the real TEST audio. With `metrics.fad_reference: wav` the test wavs
must exist — a dataset preprocessed with `--save_wav val` has none, and the test
stops at startup saying so. Either add them (re-run `preprocess_stream.py` on the
same output dir with `--save_wav test`: it does not re-encode the latents), or run
the test with `metrics.fad_reference=decoded` (the real test latents decoded
through the codec, DAC or EnCodec; not comparable with published FAD values).

**Several GPUs — `launch_test_cond.py --num-gpus N`**, the twin of the
training's ("Several GPUs" in §3: the launcher locks the N GPUs, starts one
process per GPU, stays alive as their parent and stops them all if one dies).

- **One GPU does not change.** `python test_cond.py ...` and `--num-gpus 1`
  run the test in a single process, as before: none of the multi-GPU code runs
  (checked on 5 Oct 2026: the old and the new code give the same JSON numbers,
  panels, cards and TensorBoard values, byte for byte).
- **Split over the GPUs:** the metrics step, as in the training — test sample
  j of the `n_metrics_samples` is generated, decoded and measured on GPU
  j mod N, from the SAME starting noise it has on one GPU — and the references
  of the real test data (FD, FAD). GPU 0 adds up what the others measured
  and computes FD, KL, FAD and the table: the numbers of one GPU up to
  floating-point rounding (the same sums, added in another order). The JSON
  records how many GPUs (`gpus`). Checked on 5 Oct 2026 with 2 processes on
  the CPU against 1: the 73 numbers of the JSON within 3e-15 (relative), the
  panels and the cards byte for byte. NCCL on real GPUs cannot run on the
  laptop: the first multi-GPU test on a server is the first run of that path.
- **GPU 0 alone:** TensorBoard, `test_outputs/`, the console; the probe banks,
  the panels and the uncond cards, while the others wait (time-out 180 min,
  `DIST_TIMEOUT_MIN` in `training_cond.py`). The other processes write their
  console to `runs/<run>/test_logs/rank<R>_<checkpoint>.log`; their errors still
  reach the launcher's console.
- **Panels only** (`n_metrics_samples: 0`): nothing to split — GPU 0 makes the
  panels, the other processes exit at once.
- **Stopping:** Ctrl+C or a kill stops every process at once. A test writes its
  numbers at the end, so there is nothing to save half-way.
- The checkpoint is opened memory-mapped (on one GPU too): only the weights go
  to the GPU, not the optimizer state, and the N processes share one copy of
  the file in memory.

**Generate / edit:** `sampling_cond.py` takes the checkpoint and the mode as
**positional** arguments (`checkpoint` then `generate`|`edit`). The codec (DAC
or EnCodec) is read from the checkpoint: an edit encodes the source with it, and
every generation is decoded with it.

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
`--steps` (default 50) and `--t_schedule uniform|logsnr_uniform` (default: the checkpoint's
`sampling.t_schedule`) set the Euler steps and where they go.

---

## Running on IRCAM servers

- Launch training through the GPU-lock wrapper (needs the internal `manage_gpus`):
  ```bash
  python launch_training_cond.py --num-gpus 1 --config configs/training_cond_default.yaml [overrides]
  ```
  Elsewhere (e.g. a local Windows box) run `python training_cond.py` directly.
- Launch a test through its own wrapper, the twin of the training's
  (`--num-gpus N` for N GPUs, see §4):
  ```bash
  python launch_test_cond.py --ckpt <ckpt> [--config configs/test_cond_default.yaml] [overrides]
  ```
  Elsewhere run `python test_cond.py` directly.
- Model caches (DAC / EnCodec / CREPE / beat_this / HuggingFace) are auto-redirected to the
  machine-local disk when `/data/anasynth_nonbp/baione` exists, to avoid the NFS HOME quota.
- The default `paths.runs_dir` / `paths.cache_dir` are **relative** (`./runs`,
  `./cache`). On IRCAM, override them with the shared absolute paths:
  ```bash
  python training_cond.py --config training_cond_default.yaml \
      paths.runs_dir=/data2/anasynth_nonbp/baione/runs \
      paths.cache_dir=/data2/anasynth_nonbp/baione/cache
  ```
- Several GPUs: `python launch_training_cond.py --num-gpus N ...` (see
  "Several GPUs" in §3) and `python launch_test_cond.py --num-gpus N ...`
  (§4). With `--num-gpus 1` nothing changes.

---

## End-to-end quickstart (small dataset)

```bash
# 0) sanity check on a handful of files (Windows: keep --num_workers 0)
python preprocess_stream.py <SRC_mini> out_mini --device cuda \
    --acoustic_rules --conditions f0 --num_workers 0 --batch_size 4

# 1) full preprocessing (add --codec encodec for EnCodec latents)
python preprocess_stream.py <SRC> dataset_ready_cond --device cuda \
    --acoustic_rules --conditions f0 --num_workers 0 --batch_size 8

# 2) short training run (reads splits.json, writes the normalizer into a FRESH cache)
python training_cond.py --config training_cond_default.yaml --run_name smoke \
    conditioning.enabled_frame='[f0]' training.num_steps=200 \
    paths.dataset_root=./dataset_ready_cond/latents \
    paths.condition_root=./dataset_ready_cond/conditions \
    paths.runs_dir=./runs paths.cache_dir=./cache_smoke

# 3) evaluate on the recorded test set (use the actual best_model_step<N>.pt written)
python test_cond.py --ckpt runs/smoke/checkpoints/best_model_step<N>.pt
```
