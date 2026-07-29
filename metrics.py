# metrics.py
#
# Evaluation metrics for unconditional audio generation, computed entirely in
# the (normalized) DAC latent space:
#
#   - FD-DAC (Frechet DAC Distance): Frechet/Wasserstein-2 distance between two
#     multivariate Gaussians fitted on the DAC latents, with FULL covariance.
#   - KL divergence: Kullback-Leibler divergence between the same two
#     multivariate Gaussians (full covariance), in BOTH directions
#     (real||gen and gen||real), since KL is asymmetric.
#
# DESIGN / ASSUMPTIONS (discussed and chosen deliberately):
#   - Both metrics model the real and generated latent distributions as
#     multivariate Gaussians N(mu, Sigma) with a FULL covariance over the DAC
#     pre-quantizer latent dimension (72x72; the number comes from the tensor at
#     runtime, this is just what it is for the 44 kHz DAC).
#     This is the SAME assumption for both, so FD-DAC and KL are directly
#     comparable and live in the SAME representation space.
#   - Space: the NORMALIZED latent space. Real latents come normalized from the
#     dataset; generated latents are taken PRE-denormalization (i.e. straight
#     out of the model, already in normalized space). So both distributions are
#     in the same space and the metrics measure a genuine distribution gap, not
#     a normalization offset.
#   - A non-parametric / non-Gaussian KL (e.g. kNN Kozachenko-Leonenko) was
#     considered but rejected: in 1024-D it is dominated by the curse of
#     dimensionality and is LESS reliable than the closed-form Gaussian KL, not
#     more "truthful". The Gaussian assumption is identical to the one already
#     accepted for FD-DAC.
#
# This module exposes FOUR metrics, all single-Gaussian, FULL-covariance,
# selectable from config (metrics.enabled):
#   - fd_dac      : Frechet on DAC latents       (latent-only, frame-level)
#   - kl_dac      : KL both directions on DAC     (latent-only, frame-level)
#   - fad_encodec : Frechet on Encodec embeddings (supervisor/Roebel; decode+Encodec,
#                   frame-level ~13ms; reference = REAL val wavs, no DAC)
#   - fad_vggish  : Frechet on VGGish embeddings  (decode+VGGish, ~0.96s window;
#                   reference = REAL val wavs, no DAC)
# fd_dac/kl_dac are latent-only (no decode). The two FADs decode the GENERATED
# latents to audio (unavoidable: the model only outputs DAC latents) and embed;
# their reference is the real val audio, embedded directly (no DAC on the real
# side). NONE of these use Gaussian mixtures: all are single multivariate
# Gaussians, exactly like the official FAD/FD/KL.

import os
import torch
import numpy as np
from typing import Optional
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils')


# ============================================================
# COMMON: mean / covariance from cumulative sums
# ============================================================

def compute_mu_sigma(sum_x: torch.Tensor, sum_xx: torch.Tensor, n) -> tuple:
    """Unbiased mean and full covariance from cumulative sums."""
    if isinstance(n, torch.Tensor):
        n = n.item()
    mu = sum_x / n
    sigma = (sum_xx - torch.outer(sum_x, mu)) / (n - 1)
    return mu, sigma, n


# ============================================================
# FRECHET DISTANCE — numerically stable (eigendecomposition)
# ============================================================

def symmetric_psd_matrix_sqrt(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Principal square root of a positive semi-definite matrix.
    Uses torch eigendecomposition (more stable than scipy.linalg.sqrtm).
    """
    m = 0.5 * (m + m.T)
    eigvals, eigvecs = torch.linalg.eigh(m)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T


def compute_frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Frechet Distance between two multivariate Gaussians.

    FD(P1, P2) = ||mu1 - mu2||^2 + Tr(Sigma1) + Tr(Sigma2)
                 - 2 * Tr(sqrt(Sigma1^{1/2} Sigma2 Sigma1^{1/2}))
    """
    mu1 = mu1.reshape(-1).double()
    mu2 = mu2.reshape(-1).double()
    sigma1 = sigma1.double()
    sigma2 = sigma2.double()

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"

    diff = mu1 - mu2

    eye = torch.eye(sigma1.shape[0], dtype=sigma1.dtype, device=sigma1.device)
    sigma1 = sigma1 + eps * eye
    sigma2 = sigma2 + eps * eye

    sqrt_sigma1 = symmetric_psd_matrix_sqrt(sigma1, eps)
    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    middle = 0.5 * (middle + middle.T)
    covmean = symmetric_psd_matrix_sqrt(middle, eps)

    fd = (
        diff @ diff
        + torch.trace(sigma1)
        + torch.trace(sigma2)
        - 2.0 * torch.trace(covmean)
    )
    return fd.to(torch.float32)


# ============================================================
# KL DIVERGENCE — full-covariance multivariate Gaussian
# ============================================================

def gaussian_kl_fullcov(
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    KL( N(mu_p, Sigma_p) || N(mu_q, Sigma_q) ) for FULL-covariance Gaussians.

        KL = 0.5 * [ tr(Sq^-1 Sp)
                     + (mu_q - mu_p)^T Sq^-1 (mu_q - mu_p)
                     - d
                     + ln(det Sq / det Sp) ]

    NUMERICAL STABILITY (essential in 1024-D):
      Computed via Cholesky factorization, never via explicit inverse or naive
      det:
        - Sp, Sq are symmetrized and regularized (S + eps*I) so they are SPD.
        - log det S = 2 * sum(log(diag(L)))  where S = L L^T.
        - tr(Sq^-1 Sp) and the Mahalanobis term use cholesky_solve (triangular
          solves), avoiding the cost and instability of forming Sq^-1.

    NOTE on direction (KL is asymmetric):
      KL(real || gen): penalizes the model for NOT covering regions where the
                       real data lives (mode-covering view).
      KL(gen || real): penalizes the model for generating where real data is
                       unlikely (mode-seeking view).
      evaluate_generation logs BOTH so nothing is lost.
    """
    mu_p = mu_p.reshape(-1).double()
    mu_q = mu_q.reshape(-1).double()
    Sp = sigma_p.double()
    Sq = sigma_q.double()
    d = mu_p.shape[0]

    eye = torch.eye(d, dtype=torch.float64, device=Sp.device)
    Sp = 0.5 * (Sp + Sp.T) + eps * eye
    Sq = 0.5 * (Sq + Sq.T) + eps * eye

    Lp = torch.linalg.cholesky(Sp)
    Lq = torch.linalg.cholesky(Sq)

    logdet_p = 2.0 * torch.log(torch.diag(Lp)).sum()
    logdet_q = 2.0 * torch.log(torch.diag(Lq)).sum()

    # tr(Sq^-1 Sp) via X = Sq^-1 Sp (cholesky_solve), then trace.
    X = torch.cholesky_solve(Sp, Lq)
    tr_term = torch.trace(X)

    # Mahalanobis: (mu_q - mu_p)^T Sq^-1 (mu_q - mu_p)
    diff = (mu_q - mu_p).unsqueeze(1)
    sol = torch.cholesky_solve(diff, Lq)
    maha = (diff * sol).sum()

    kl = 0.5 * (tr_term + maha - d + (logdet_q - logdet_p))
    return float(kl.item())


# ============================================================
# REFERENCE STATS ON THE VALIDATION SET (shared by FD-DAC and KL)
# ============================================================

@torch.no_grad()
def precompute_latent_reference(
    val_dataset,
    cache_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_accum: int = 50,
) -> dict:
    """
    Pre-compute mu and full covariance (Sigma) of the REAL latents over the
    whole validation set, in the normalized space. These same stats feed BOTH
    FD-DAC and the KL divergence (same Gaussian, same space).

    Online accumulation with sum_x and sum_xx for numerical stability.

    Cache: if cache_path exists -> load; if given but missing -> compute & save;
    if None -> compute without saving.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if cache_path is not None and Path(cache_path).exists():
        print(f"[Latent Reference] Loading cache: {cache_path}")
        stats = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        print(f"[Latent Reference] {stats['n_total']} frame, "
              f"dim={stats['mu'].shape[0]}, "
              f"mu range [{stats['mu'].min():.3f}, {stats['mu'].max():.3f}]")
        return stats

    print(f"[Latent Reference] Accumulation on {len(val_dataset)} samples "
          f"(device={device}, batch_accum={batch_accum})...")

    sum_x = None
    sum_xx = None
    count = 0
    buffer = []

    for idx in tqdm(range(len(val_dataset)), desc="Latent reference"):
        frames, _ = val_dataset[idx]   # (n_frames, dim) normalized
        buffer.append(frames)

        if len(buffer) >= batch_accum or idx == len(val_dataset) - 1:
            batch = torch.cat(buffer, dim=0).to(device=device, dtype=torch.float64)
            buffer = []

            if sum_x is None:
                dim = batch.shape[-1]
                sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
                sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=device)

            sum_x = sum_x + batch.sum(dim=0)
            sum_xx = sum_xx + batch.T @ batch
            count = count + batch.shape[0]

    mu, sigma, _ = compute_mu_sigma(sum_x, sum_xx, count)

    mu_cpu = mu.cpu()
    sigma_cpu = sigma.cpu()

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[Latent Reference] Done: {count} frame, "
          f"dim={mu_cpu.shape[0]}, "
          f"mu range [{mu_cpu.min():.3f}, {mu_cpu.max():.3f}]")

    stats = {"mu": mu_cpu, "sigma": sigma_cpu, "n_total": count}

    if cache_path is not None:
        cache_path_obj = Path(cache_path)
        cache_path_obj.parent.mkdir(parents=True, exist_ok=True)
        # Publish atomically (temp + os.replace): an interruption during a plain
        # torch.save leaves a TRUNCATED cache file, which the next run finds,
        # tries to load, and fails on -- every time, until it is deleted by hand.
        _tmp = str(cache_path_obj) + ".tmp"
        torch.save(stats, _tmp)
        os.replace(_tmp, str(cache_path_obj))
        print(f"[Latent Reference] Cache saved in: {cache_path}")

    return stats


# Backwards-compatible alias: training.py historically imported
# precompute_fd_dac_reference. It is the same thing now (reference stats are
# shared between FD-DAC and KL), so we keep the old name working.
def precompute_fd_dac_reference(val_dataset, cache_path=None, device=None,
                                batch_accum: int = 50) -> dict:
    return precompute_latent_reference(
        val_dataset, cache_path=cache_path, device=device, batch_accum=batch_accum)


# ============================================================
# GENERATED-SIDE STATS (streamed in blocks to bound GPU memory)
# ============================================================

def _generated_mu_sigma(generated_latents: torch.Tensor, device: str,
                        block_size: int = 16):
    """
    mu and full covariance of the GENERATED latents, accumulating sum_x/sum_xx
    one block of `block_size` samples at a time. Peak GPU memory is independent
    of N (only one block on the GPU at once), mirroring the reference path.

    generated_latents: (N, n_frames, dim) on CPU.
    Returns (mu, sigma) as float64 GPU tensors.
    """
    dim = generated_latents.shape[-1]
    sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    count = 0

    n = generated_latents.shape[0]
    for start in range(0, n, block_size):
        block = generated_latents[start:start + block_size]
        block = block.reshape(-1, dim).to(device=device, dtype=torch.float64)
        sum_x = sum_x + block.sum(dim=0)
        sum_xx = sum_xx + block.T @ block
        count = count + block.shape[0]
        del block

    mu_gen, sigma_gen, _ = compute_mu_sigma(sum_x, sum_xx, count)
    return mu_gen, sigma_gen


# ============================================================
# FD-DAC
# ============================================================

def compute_fd_dac(
    generated_latents: torch.Tensor,
    ref_stats: dict,
    device: str = "cuda",
    block_size: int = 16,
) -> float:
    """
    Frechet DAC Distance between generated latents and the reference.

    generated_latents: (N, n_frames, dim) on CPU. Streamed to GPU in blocks.
    ref_stats: dict with 'mu' and 'sigma' (full covariance) of the reference.
    """
    mu_gen, sigma_gen = _generated_mu_sigma(generated_latents, device, block_size)

    mu_ref = ref_stats["mu"].to(device)
    sigma_ref = ref_stats["sigma"].to(device)

    fd = compute_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

    del mu_gen, sigma_gen, mu_ref, sigma_ref
    if device == "cuda":
        torch.cuda.empty_cache()

    return float(fd.item())


# ============================================================
# KL (both directions), sharing the generated stats
# ============================================================

def compute_kl_both(
    generated_latents: torch.Tensor,
    ref_stats: dict,
    device: str = "cuda",
    block_size: int = 16,
) -> dict:
    """
    KL divergence (full-covariance Gaussian) between the REAL reference and the
    GENERATED latents, in BOTH directions.

    Returns {"kl_real_gen": KL(real||gen), "kl_gen_real": KL(gen||real)}.
    Same Gaussian assumption and same (normalized) latent space as FD-DAC.
    """
    mu_gen, sigma_gen = _generated_mu_sigma(generated_latents, device, block_size)

    mu_ref = ref_stats["mu"].to(device)
    sigma_ref = ref_stats["sigma"].to(device)

    # real = reference (P = real), gen = generated (Q = gen)
    kl_real_gen = gaussian_kl_fullcov(mu_ref, sigma_ref, mu_gen, sigma_gen)
    kl_gen_real = gaussian_kl_fullcov(mu_gen, sigma_gen, mu_ref, sigma_ref)

    del mu_gen, sigma_gen, mu_ref, sigma_ref
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"kl_real_gen": kl_real_gen, "kl_gen_real": kl_gen_real}


# ============================================================
# DAC-latent metrics, selection-driven, sharing ONE mu/sigma
# ============================================================

# The DAC-latent metrics available to the CONDITIONED pipeline. (The audio-FAD
# metrics of ALL_METRICS -- fad_encodec / fad_vggish -- belong to the
# unconditional evaluate_generation path and are not wired here.)
DAC_METRICS = ("fd_dac", "kl_dac")


def compute_dac_metrics(
    generated_latents: torch.Tensor,
    ref_stats: dict,
    enabled=DAC_METRICS,
    device: str = "cuda",
    block_size: int = 16,
) -> dict:
    """
    Compute the REQUESTED DAC-latent metrics over `generated_latents`, fitting the
    generated mu/full-covariance ONCE and reusing it for every requested metric.

    `enabled`: any subset of DAC_METRICS ("fd_dac", "kl_dac"). Only the requested
    metrics are computed; the others come back as None (and cost nothing).

    Sharing the stats is what makes the selection cheap AND fast at the same time:
    the expensive part is the covariance accumulation over all frames, which is
    done once regardless of how many metrics are requested; each metric on top of
    it is a small (dim x dim) linear-algebra step. Results are NUMERICALLY
    IDENTICAL to calling compute_fd_dac() / compute_kl_both() separately.

    Returns {"fd_dac": float|None, "kl_real_gen": tensor|None,
             "kl_gen_real": tensor|None}.
    """
    out = {"fd_dac": None, "kl_real_gen": None, "kl_gen_real": None}
    want_fd = "fd_dac" in enabled
    want_kl = "kl_dac" in enabled
    if not (want_fd or want_kl):
        return out

    # the one expensive pass, shared by every requested metric
    mu_gen, sigma_gen = _generated_mu_sigma(generated_latents, device, block_size)
    mu_ref = ref_stats["mu"].to(device)
    sigma_ref = ref_stats["sigma"].to(device)

    if want_fd:
        out["fd_dac"] = float(
            compute_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen).item())
    if want_kl:
        out["kl_real_gen"] = gaussian_kl_fullcov(mu_ref, sigma_ref, mu_gen, sigma_gen)
        out["kl_gen_real"] = gaussian_kl_fullcov(mu_gen, sigma_gen, mu_ref, sigma_ref)

    del mu_gen, sigma_gen, mu_ref, sigma_ref
    if device == "cuda":
        torch.cuda.empty_cache()

    return out


# ============================================================
# EVALUATION FUNCTION (for the training loop)
# ============================================================

# ============================================================
# AUDIO EMBEDDERS for the FAD metrics (Encodec = Roebel, VGGish = 1s window).
# Lazy-loaded, so this module imports fine without encodec / vggish installed.
# ============================================================

class EncodecEmbedder:
    """Encodec encoder embeddings (pre-quantization), frame-level (~13 ms @ 24kHz).
    Faithful to the supervisor's fad.py.get_embeddings."""

    def __init__(self, audio_sr_model: int = 24000, device: str = "cpu"):
        self.audio_sr_model = audio_sr_model
        self.device = device
        self._model = None
        self.embedding_dim = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from encodec import EncodecModel
        except ImportError:
            raise ImportError("encodec not found. Install with: pip install encodec")
        model = (EncodecModel.encodec_model_48khz() if self.audio_sr_model == 48000
                 else EncodecModel.encodec_model_24khz())
        model.set_target_bandwidth(24.0)
        model.eval().to(self.device)
        self._model = model
        self.embedding_dim = model.encoder.dimension
        print(f"[Encodec] {self.audio_sr_model // 1000}kHz on {self.device} "
              f"(dim={self.embedding_dim})")

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
        """audio (b, c, n) -> (b*n_enc, d), encoder output pooled over frames."""
        self._load()
        x = audio.to(self.device).float()
        if self._model.sample_rate != audio_sr:
            import librosa
            x_np = librosa.resample(
                x.detach().cpu().numpy(),
                orig_sr=audio_sr, target_sr=self._model.sample_rate, axis=-1)
            x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32)).to(self.device)
        if self._model.sample_rate == 48000:
            if x.shape[1] != 2:
                x = torch.cat((x, x), dim=1)            # 48kHz model wants stereo
        elif x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)             # 24kHz model wants mono
        e = self._model.encoder(x)                      # (b, d, n)
        return e.permute(0, 2, 1).reshape(-1, e.shape[1])


class VGGishEmbedder:
    """VGGish embeddings, ~0.96 s temporal window per vector (128-D). The ~1s
    window is intrinsic to VGGish (log-mel patches), which is why this is the
    "1-second-window" FAD. Loaded via torch.hub (harritaylor/torchvggish).
    NOTE: first load downloads weights via torch.hub (needs network once). This
    loader is the single piece to verify in your environment."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self.embedding_dim = 128

    def _load(self):
        if self._model is not None:
            return
        model = torch.hub.load('harritaylor/torchvggish', 'vggish', postprocess=False)
        model.eval().to(self.device)
        model.device = self.device
        self._model = model
        print(f"[VGGish] loaded on {self.device} (dim={self.embedding_dim}, ~0.96s window)")

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
        """audio (b, c, n) -> (n_windows, 128). Processes the (mono) clip; VGGish
        resamples to 16 kHz internally."""
        self._load()
        wav = audio[0].mean(dim=0).cpu().numpy()        # 1-D mono
        emb = self._model(wav, fs=audio_sr)             # (n_windows, 128)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb.to(self.device)


# ============================================================
# AUDIO-SIDE mu/sigma (shared by reference wavs and decoded generated latents)
# ============================================================

@torch.no_grad()
def _audio_clips_to_mu_sigma(clip_iter, n_items, embedder, device, desc):
    """clip_iter yields (wav (b,c,T), sr). Accumulate mu/sigma of embeddings."""
    sum_x = None
    sum_xx = None
    count = 0
    for wav, sr in tqdm(clip_iter, total=n_items, desc=desc):
        emb = embedder.embed(wav, audio_sr=sr).to(device=device, dtype=torch.float64)
        if sum_x is None:
            d = emb.shape[-1]
            sum_x = torch.zeros(d, dtype=torch.float64, device=device)
            sum_xx = torch.zeros(d, d, dtype=torch.float64, device=device)
        sum_x = sum_x + emb.sum(dim=0)
        sum_xx = sum_xx + emb.T @ emb
        count = count + emb.shape[0]
        del emb
    mu, sigma, _ = compute_mu_sigma(sum_x, sum_xx, count)
    return mu, sigma, count


@torch.no_grad()
def precompute_audio_reference(val_wav_root, embedder, cache_path=None,
                               device: str = "cuda") -> dict:
    """Reference mu/sigma = embeddings of the REAL val wavs (NO DAC). Cached.
    val_wav_root is searched recursively for *.wav."""
    if cache_path is not None and Path(cache_path).exists():
        print(f"[Audio ref] loading cache: {cache_path}")
        return torch.load(str(cache_path), map_location="cpu", weights_only=False)
    import soundfile as sf
    wavs = sorted(Path(val_wav_root).rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav under {val_wav_root}")
    print(f"[Audio ref] embedding {len(wavs)} real val wavs (NO DAC)...")

    def _iter():
        for p in wavs:
            # soundfile (libsndfile) reads WAV with no system FFmpeg, unlike
            # torchaudio.load (torchcodec backend) which needs FFmpeg DLLs and
            # fails on bare Windows envs.
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)  # (T, c)
            w = torch.from_numpy(data.T.copy())                           # (c, T)
            yield w.unsqueeze(0), sr                                      # (1, c, T)

    mu, sigma, count = _audio_clips_to_mu_sigma(_iter(), len(wavs), embedder, device, "Audio ref")
    stats = {"mu": mu.cpu(), "sigma": sigma.cpu(), "n_total": count}
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        _tmp = str(cache_path) + ".tmp"          # atomic publish, see above
        torch.save(stats, _tmp)
        os.replace(_tmp, str(cache_path))
        print(f"[Audio ref] cache saved: {cache_path}")
    return stats


@torch.no_grad()
def _decode_embed_mu_sigma(generated_latents, normalizer, embedder, device, desc):
    """Decode generated latents to audio (DAC) and embed -> mu/sigma."""
    from audio_dataset_npy import decode_latents, DAC_SAMPLE_RATE

    def _iter():
        for i in range(generated_latents.shape[0]):
            z = generated_latents[i].T                  # (dim, n_frames) normalized
            z = normalizer.denormalize(z)               # raw DAC latents
            wav = decode_latents(z, device=device)      # (1, T) @ 44.1kHz
            yield wav.unsqueeze(0), DAC_SAMPLE_RATE      # (1, 1, T)

    return _audio_clips_to_mu_sigma(_iter(), generated_latents.shape[0], embedder, device, desc)


# ============================================================
# CONFIG-DRIVEN SETUP + EVALUATION
#
# Metric registry. All single-Gaussian, FULL covariance:
#   fd_dac      : Frechet on DAC latents          (latent-only, frame-level)
#   kl_dac      : KL both directions on DAC        (latent-only, frame-level)
#   fad_encodec : Frechet on Encodec embeddings    (Roebel; decode+Encodec, frame)
#   fad_vggish  : Frechet on VGGish embeddings     (decode+VGGish, ~0.96s window)
# ============================================================

ALL_METRICS = ("fd_dac", "kl_dac", "fad_encodec", "fad_vggish")


def make_embedders(enabled, device: str = "cuda", encodec_sr: int = 24000) -> dict:
    """Instantiate only the audio embedders needed by `enabled`."""
    emb = {}
    if "fad_encodec" in enabled:
        emb["encodec"] = EncodecEmbedder(audio_sr_model=encodec_sr, device=device)
    if "fad_vggish" in enabled:
        emb["vggish"] = VGGishEmbedder(device=device)
    return emb


def build_references(enabled, val_dataset, val_wav_root, embedders, cache_dir,
                     device: str = "cuda", strict: bool = True) -> dict:
    """Pre-compute (and cache) only the references needed by `enabled`.
    DAC reference is the real val latents; the FAD references are the real val
    wavs embedded directly (no DAC).

    A metric in `enabled` is an EXPLICIT request, so by default
    (strict=True) a FAD reference that cannot be built — missing val wavs, an
    unloadable audio backend (FFmpeg / torchcodec), or VGGish weights that
    cannot be fetched — is a HARD ERROR that STOPS the run at startup, BEFORE
    any training, with a clear message. This prevents silently training for
    hours believing a metric is on when it is not. Set strict=False
    (cfg.metrics.strict) only when you deliberately want the run to proceed and
    skip any FAD that cannot be built (e.g. a sweep across offline nodes)."""
    cache_dir = Path(cache_dir)
    refs = {}
    if "fd_dac" in enabled or "kl_dac" in enabled:
        refs["dac"] = precompute_latent_reference(
            val_dataset, cache_path=str(cache_dir / "latent_ref_stats.pt"), device=device)
    for name, key, fname in (("fad_encodec", "encodec", "fad_encodec_ref_stats.pt"),
                             ("fad_vggish",  "vggish",  "fad_vggish_ref_stats.pt")):
        if name not in enabled:
            continue
        try:
            refs[key] = precompute_audio_reference(
                val_wav_root, embedders[key],
                cache_path=str(cache_dir / fname), device=device)
        except Exception as e:
            need = (f"need real val wavs under '{val_wav_root}', a working audio "
                    f"backend (FFmpeg/torchcodec)"
                    + (" and network access for VGGish weights" if key == "vggish" else "")
                    + ".")
            if strict:
                raise RuntimeError(
                    f"[Metrics] FATAL: metric '{name}' is enabled but its reference "
                    f"could not be built ({type(e).__name__}: {e}). {need} "
                    f"Fix the environment, remove '{name}' from metrics.enabled, "
                    f"or set metrics.strict=false to skip it and continue.") from e
            print(f"[Metrics] WARNING (strict=false): could not build the {name} "
                  f"reference ({type(e).__name__}: {e}). Skipping {name} this run — {need}")
    return refs


@torch.no_grad()
def evaluate_generation(model, normalizer, val_dataset, *, enabled, references,
                        embedders=None, n_samples: int = 64, euler_steps: int = 50,
                        t_min: float = 0.001, t_max: float = 0.999,
                        seed: Optional[int] = None,
                        device: str = "cuda", use_amp: bool = False) -> dict:
    """
    Generate N samples ONCE, then compute every metric in `enabled`, reusing the
    same generated latents (DAC metrics use them directly; audio metrics decode
    them once per embedder). Returns a flat dict of scalars plus the latents:
      {"fd_dac":.., "kl_real_gen":.., "kl_gen_real":.., "fad_encodec":..,
       "fad_vggish":.., "generated_latents": <tensor>}.
    Only the enabled keys are present. token_dim is inferred (72 or 1024).
    (t_min, t_max, euler_steps come from cfg.sampling so the metric sampler
    matches the one used to generate the logged audio — see euler_integrate.)
    `seed` (cfg.metrics.seed) fixes the generation noise via a DEDICATED Generator,
    so the same checkpoint yields the same FD/KL every eval (comparable across
    checkpoints) WITHOUT touching the global training RNG. null = free-running.
    """
    # NOTE: evaluate_generation() is the UNCONDITIONAL end-to-end audio-metrics
    # helper and is NOT used by the conditioned pipeline (which uses
    # precompute_latent_reference + condition_metrics). Its Euler sampler lives in
    # the uncond `sampling.py`. Guard the import so a stray call fails clearly
    # instead of raising a bare ModuleNotFoundError (report #13).
    try:
        from sampling import euler_integrate      # shared sampler (avoids divergence)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "evaluate_generation() is the unconditional metrics path (needs the "
            "uncond sampling.py / euler_integrate); the conditioned pipeline does "
            "not use it -- see precompute_latent_reference + condition_metrics.") from e
    embedders = embedders or {}
    model.eval()
    n_frames = val_dataset.n_frames
    token_dim = val_dataset[0][0].shape[-1]             # 72 or 1024, no hardcode

    gen_rng = None
    if seed is not None:
        gen_rng = torch.Generator(device=device)        # dedicated, isolated RNG
        gen_rng.manual_seed(int(seed))

    gen_list = []
    n_skipped = 0
    for _ in tqdm(range(n_samples), desc="Metrics: generating samples"):
        x = torch.randn(1, n_frames, token_dim, device=device, generator=gen_rng)
        x = euler_integrate(model, x, steps=euler_steps,
                            t_min=t_min, t_max=t_max, use_amp=use_amp)
        gf = x[0].cpu()
        if torch.isfinite(gf).all():            # drop NaN/inf samples so they do
            gen_list.append(gf)                 # not poison mu/sigma (-> NaN metrics)
        else:
            n_skipped += 1
        del x
    if device == "cuda":
        torch.cuda.empty_cache()

    if n_skipped:
        print(f"[Metrics] WARNING: {n_skipped}/{n_samples} generated samples "
              f"were non-finite and skipped.")
    if not gen_list:
        print("[Metrics] WARNING: all generated samples were non-finite; "
              "skipping metrics this eval.")
        return {}

    generated_latents = torch.stack(gen_list)           # CPU, normalized, finite

    out = {}
    if "fd_dac" in enabled:
        out["fd_dac"] = compute_fd_dac(generated_latents, references["dac"], device=device)
    if "kl_dac" in enabled:
        kl = compute_kl_both(generated_latents, references["dac"], device=device)
        out["kl_real_gen"] = kl["kl_real_gen"]
        out["kl_gen_real"] = kl["kl_gen_real"]
    for name, key in (("fad_encodec", "encodec"), ("fad_vggish", "vggish")):
        if (name in enabled and key in references
                and embedders is not None and embedders.get(key) is not None):
            mu_g, sigma_g, _ = _decode_embed_mu_sigma(
                generated_latents, normalizer, embedders[key], device,
                desc=f"Metrics: {name} (decode+embed)")
            ref = references[key]
            out[name] = float(compute_frechet_distance(
                ref["mu"].to(device), ref["sigma"].to(device), mu_g, sigma_g).item())
    if device == "cuda":
        torch.cuda.empty_cache()

    out["generated_latents"] = generated_latents
    return out


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    print("Test compute_frechet_distance...")
    d = 128
    torch.manual_seed(0)
    mu1 = torch.randn(d).double()
    A = torch.randn(d, d).double()
    sigma1 = (A @ A.T) / d
    mu2 = torch.randn(d).double()
    B = torch.randn(d, d).double()
    sigma2 = (B @ B.T) / d
    fd = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"  FD: {fd.item():.4f} (> 0)")
    assert fd.item() > 0
    print("  OK\n")

    print("Test gaussian_kl_fullcov...")
    kl_same = gaussian_kl_fullcov(mu1, sigma1 + torch.eye(d), mu1, sigma1 + torch.eye(d))
    print(f"  KL(P||P) ~ 0: {kl_same:.2e}")
    assert abs(kl_same) < 1e-5
    kl_pq = gaussian_kl_fullcov(mu1, sigma1 + torch.eye(d), mu2, sigma2 + torch.eye(d))
    kl_qp = gaussian_kl_fullcov(mu2, sigma2 + torch.eye(d), mu1, sigma1 + torch.eye(d))
    print(f"  KL(P||Q)={kl_pq:.4f}  KL(Q||P)={kl_qp:.4f}  (asymmetric, > 0)")
    assert kl_pq > 0 and kl_qp > 0
    print("  OK\n")

    print("Test compute_fd_dac + compute_kl_both (block accumulation)...")
    n_samples, n_frames, dim = 10, 430, 1024
    gen_latents = torch.randn(n_samples, n_frames, dim)
    ref_stats = {
        "mu": torch.zeros(dim, dtype=torch.float64),
        "sigma": torch.eye(dim, dtype=torch.float64),
        "n_total": 1000,
    }
    fd_dac = compute_fd_dac(gen_latents, ref_stats, device="cpu", block_size=4)
    kl = compute_kl_both(gen_latents, ref_stats, device="cpu", block_size=4)
    print(f"  FD-DAC: {fd_dac:.4f}")
    print(f"  KL(real||gen): {kl['kl_real_gen']:.4f} | KL(gen||real): {kl['kl_gen_real']:.4f}")
    # block-size invariance
    fd_a = compute_fd_dac(gen_latents, ref_stats, device="cpu", block_size=4)
    fd_b = compute_fd_dac(gen_latents, ref_stats, device="cpu", block_size=16)
    assert abs(fd_a - fd_b) < 1e-9
    print("  OK (block-size invariant)")
