# metrics.py
#
# Evaluation metrics for generating audio:
#   - FD-DAC (Frechet DAC Distance): Frechet distance on the DAC latents
#     with full covariance (upgrade of the previous diagonal KL)
#   - FAD (Frechet Audio Distance): quality perception with Encodec
#
# Key differences with the previous diagonal KL:
#   - Full covariance (1024x1024) invece di varianza per-canale
#   - It captures the correlations among dimensions of the DAC latents
#   - it uses the same Frechet formula of the FAD but in the latent space
#
# Requirements:
#   pip install encodec einops soundfile

import os
import torch
import numpy as np
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils')


# ============================================================
# FRECHET DISTANCE — torch computation numerically stable
# ============================================================

def symmetric_psd_matrix_sqrt(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Principal square root of a positive semi-definite matrix.
    It uses eigendecomposition torch (more stable than scipy.linalg.sqrtm).
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


def compute_mu_sigma(sum_x: torch.Tensor, sum_xx: torch.Tensor, n) -> tuple:
    """Compute unbiased mean and covariance from cumulative sums."""
    if isinstance(n, torch.Tensor):
        n = n.item()
    mu = sum_x / n
    sigma = (sum_xx - torch.outer(sum_x, mu)) / (n - 1)
    return mu, sigma, n


# ============================================================
# FD-DAC: FRECHET DISTANCE ON DAC LATENTS (substitute KL)
# ============================================================

@torch.no_grad()
def precompute_fd_dac_reference(
    val_dataset,
    cache_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_accum: int = 50,
) -> dict:
    """
    Pre-computes mu e sigma (full covariance) in the DAC latent space
    for the computation of the Frechet DAC Distance.

    Online accumulation with sum_x e sum_xx for numeric stability.
    The latents used are those normalized from the dataset (coherently with what the model generates during the training).

    Cache: if cache_path is given and exists, it loads the stats.
    If cache_path is given but does not exists, computes and saves.
    If cache_path is None, computes without saving.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cache hit: load the already computed stats
    if cache_path is not None and Path(cache_path).exists():
        print(f"[FD-DAC Reference] Loading cache: {cache_path}")
        stats = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        print(f"[FD-DAC Reference] {stats['n_total']} frame, "
              f"dim={stats['mu'].shape[0]}, "
              f"mu range [{stats['mu'].min():.3f}, {stats['mu'].max():.3f}]")
        return stats

    print(f"[FD-DAC Reference] Accumulation on {len(val_dataset)} samples "
          f"(device={device}, batch_accum={batch_accum})...")

    sum_x = None
    sum_xx = None
    count = 0
    buffer = []

    for idx in tqdm(range(len(val_dataset)), desc="FD-DAC reference"):
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

    # Move on CPU for saving (8MB per sigma 1024x1024)
    mu_cpu = mu.cpu()
    sigma_cpu = sigma.cpu()

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[FD-DAC Reference] Done: {count} frame, "
          f"dim={mu_cpu.shape[0]}, "
          f"mu range [{mu_cpu.min():.3f}, {mu_cpu.max():.3f}]")

    stats = {"mu": mu_cpu, "sigma": sigma_cpu, "n_total": count}

    # Save cache on disk if path is given
    if cache_path is not None:
        cache_path_obj = Path(cache_path)
        cache_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, str(cache_path_obj))
        print(f"[FD-DAC Reference] Cache saved in: {cache_path}")

    return stats


def compute_fd_dac(
    generated_latents: torch.Tensor,
    fd_dac_ref_stats: dict,
    device: str = "cuda",
    block_size: int = 16,
) -> float:
    """
    Computes Frechet DAC Distance among generated latents and reference.

    generated_latents: (N, n_frames, dim) tensor of generated latents
                       (already normalized, as output from the model).
                       EXPECTED ON CPU: this function streams it to the GPU in
                       small blocks, it never moves the whole tensor at once.
    fd_dac_ref_stats: dict with 'mu' e 'sigma' of the reference

    MEMORY NOTE
    -----------
    The previous version did:
        gen_flat = generated_latents.reshape(-1, dim).to(device, float64)
        sum_xx   = gen_flat.T @ gen_flat
    i.e. it pushed the FULL (N*n_frames, dim) matrix onto the GPU in float64.
    With N=256, n_frames=431, dim=1024 that single tensor is
    256*431*1024*8 bytes ~= 0.9 GB, plus the intermediate products. On top of
    a ~8.5 GB resident G model that extra spike was enough to OOM exactly at
    the metrics step (and it scaled with n_metrics_samples, which is why 256
    OOM'd while 128 did not).

    This version accumulates sum_x / sum_xx online, one block of `block_size`
    samples at a time, mirroring precompute_fd_dac_reference. Peak GPU memory
    is now independent of N: only one block lives on the GPU at any moment,
    plus the two fixed (dim,) and (dim,dim) accumulators. The mathematical
    result is identical (same sums, just summed in chunks).
    """
    dim = generated_latents.shape[-1]

    sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    count = 0

    n = generated_latents.shape[0]
    for start in range(0, n, block_size):
        block = generated_latents[start:start + block_size]      # (b, n_frames, dim) on CPU
        block = block.reshape(-1, dim).to(device=device, dtype=torch.float64)
        sum_x = sum_x + block.sum(dim=0)
        sum_xx = sum_xx + block.T @ block
        count = count + block.shape[0]
        del block

    mu_gen, sigma_gen, _ = compute_mu_sigma(sum_x, sum_xx, count)

    mu_ref = fd_dac_ref_stats["mu"].to(device)
    sigma_ref = fd_dac_ref_stats["sigma"].to(device)

    fd = compute_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

    # Drop the GPU accumulators before returning so they don't linger into the
    # next phase (FAD / decode) of the metrics step.
    del sum_x, sum_xx, mu_gen, sigma_gen, mu_ref, sigma_ref
    if device == "cuda":
        torch.cuda.empty_cache()

    return float(fd.item())


# ============================================================
# FAD CALCULATOR with Encodec
# ============================================================

class FADCalculator:
    """
    FAD with encoder Encodec (Facebook neural audio codec).

    Workflow:
      1. precompute_reference_stats(...): computes the Encodec embedding stats
         (mu_ref, sigma_ref) over the validation WAVs and caches them (.pt)
      2. compute_fad_against_reference(...): compute embedding of the generated
         and returns FAD against reference
    """

    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        self.model = None
        self.embedding_dim = None
        self.audio_sr = None
        self.encodec_sr = None

        self.ref_dir = None
        self.ref_stats_path = None
        self.ref_n_samples = 0
        self.mu_ref = None
        self.sigma_ref = None

    def _load_model(self, audio_sr: int = 44100):
        """Load Encodec, lazy."""
        if self.model is not None:
            return

        from encodec import EncodecModel

        if audio_sr == 48000:
            self.model = EncodecModel.encodec_model_48khz()
            self.encodec_sr = 48000
        else:
            self.model = EncodecModel.encodec_model_24khz()
            self.encodec_sr = 24000

        self.model.set_target_bandwidth(24.0)
        self.embedding_dim = self.model.encoder.dimension
        self.audio_sr = audio_sr

        self.model.to(self.device)
        self.model.eval()

        print(f"[FAD/Encodec] Model loaded: {self.encodec_sr/1000:.0f}kHz, "
              f"embedding_dim={self.embedding_dim}, device={self.device}")

    @torch.no_grad()
    def _get_embeddings(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Computes the Encodec embeddings.

        wav: (T,) o (1,T) o (B,T) o (B,1,T)
        return: (N, embedding_dim) dove N = batch * n_frames
        """
        import torchaudio

        # Normalizes shape to (B, 1, T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)
        elif wav.dim() == 3 and wav.shape[1] != 1:
            wav = wav.mean(dim=1, keepdim=True)

        wav = wav.to(self.device).float()

        # Resampling
        if self.audio_sr != self.encodec_sr:
            wav = torchaudio.functional.resample(
                wav, orig_freq=self.audio_sr, new_freq=self.encodec_sr,
            )

        # Encodec 48kHz requires stereo
        if self.encodec_sr == 48000 and wav.shape[1] != 2:
            wav = torch.cat([wav, wav], dim=1)

        # Embedding pre-quantization
        emb = self.model.encoder(wav)  # (B, D, N_frames)

        from einops import rearrange
        emb = rearrange(emb, "b d n -> (b n) d")
        return emb

    @torch.no_grad()
    def precompute_reference_stats(
        self,
        val_dataset,
        normalizer,
        wav_root: str,
        latent_root: str,
        sr: int = 44100,
        cache_dir: Optional[str] = None,
    ):
        """
        Computes and caches the reference Encodec stats (mu_ref, sigma_ref)
        over the FULL validation set.

        STRADA B — no WAV duplication
        ------------------------------
        The previous version did this in two passes over all val samples:
          1. read each original WAV from wav_root and REWRITE a copy into
             cache_dir/reference_wavs/ref_XXXXX.wav
          2. re-read those copies and compute Encodec embeddings
        Step 1 duplicated the entire validation audio on disk
        (65k files x ~0.9 MB ~= 57 GB), which filled /data and got the process
        killed mid-way ("Terminated") on the modern dataset.

        This version keeps ALL validation samples (no subsampling — same as
        before / as on gottan) but computes the embeddings in a SINGLE pass,
        reading each WAV directly from wav_root and never writing a copy. For
        the rare samples whose WAV is missing, it falls back to decoding the
        latent with DAC in-memory (still no file written). Disk usage for the
        reference is therefore essentially zero; only the small stats .pt is
        saved.
        """
        import soundfile as sf

        if cache_dir is None:
            raise ValueError(
                "cache_dir is mandatory. Set it in the training script."
            )

        self._load_model(audio_sr=sr)

        n_samples = len(val_dataset)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.ref_stats_path = str(cache_dir / "ref_stats_encodec.pt")

        # Cache hit: load stats already computed
        if Path(self.ref_stats_path).exists():
            print(f"[FAD Reference] Load cache stats: {self.ref_stats_path}")
            stats = torch.load(self.ref_stats_path, map_location="cpu", weights_only=False)
            self.mu_ref = stats["mu"].to(self.device)
            self.sigma_ref = stats["sigma"].to(self.device)
            self.ref_n_samples = stats.get("n_samples", n_samples)
            print(f"[FAD Reference] {self.ref_n_samples} sample, "
                  f"dim={self.mu_ref.shape[0]}")
            return

        print(f"[FAD Reference] Computing Encodec embeddings on {n_samples} "
              f"validation WAVs (read in place, no copy on disk)...")

        # Pre-resolve the original WAV path for every sample, and check whether
        # any are missing (those few will be reconstructed with DAC in-memory).
        wav_paths = []
        need_dac = False
        for idx in range(n_samples):
            npy_path, start, label_idx = val_dataset.samples[idx]
            rel_path = npy_path.relative_to(Path(latent_root))
            wav_path = Path(wav_root) / rel_path.with_suffix(".wav")
            wav_paths.append(wav_path)
            if not wav_path.exists():
                need_dac = True

        dac_model = None
        if need_dac:
            import dac
            dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
            dac_model.to("cpu")
            dac_model.eval()
            print("[FAD Reference] Some validation WAVs are missing - "
                  "those will be reconstructed in-memory with DAC (no file written).")

        dim = self.embedding_dim
        sum_x = torch.zeros(dim, dtype=torch.float64, device=self.device)
        sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        count = 0

        # SINGLE streaming pass: read WAV (or rebuild) -> embedding -> accumulate.
        # No WAV copy is ever written to disk.
        for idx in tqdm(range(n_samples), desc="FAD ref: embedding"):
            wav_path = wav_paths[idx]
            if wav_path.exists():
                audio, _ = sf.read(str(wav_path), dtype='float32')
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                wav_t = torch.from_numpy(audio)
            else:
                # Fallback: reconstruct from the latent, in memory only.
                frames, _ = val_dataset[idx]
                z = frames.T
                z = normalizer.denormalize(z)
                waveform = dac_model.decode(z.unsqueeze(0).float()).squeeze()
                wav_t = waveform.cpu()
                if wav_t.dim() > 1:
                    wav_t = wav_t.squeeze()

            emb = self._get_embeddings(wav_t.float()).double()
            sum_x = sum_x + emb.sum(dim=0)
            sum_xx = sum_xx + emb.T @ emb
            count = count + emb.shape[0]

        if dac_model is not None:
            del dac_model

        self.ref_n_samples = n_samples

        # Compute and save
        mu_ref, sigma_ref, _ = compute_mu_sigma(sum_x, sum_xx, count)
        self.mu_ref = mu_ref
        self.sigma_ref = sigma_ref

        torch.save({
            "mu": mu_ref.cpu(),
            "sigma": sigma_ref.cpu(),
            "n_samples": n_samples,
            "n_embeddings": count,
            "embedding_dim": dim,
        }, self.ref_stats_path)
        print(f"[FAD Reference] Cache saved in: {self.ref_stats_path}")

    @torch.no_grad()
    def compute_fad_against_reference(
        self,
        generated_wavs: List[torch.Tensor],
        sr: int = 44100,
    ) -> float:
        """
        Compute FAD of the generated against the reference cached stats.
        """
        assert self.mu_ref is not None and self.sigma_ref is not None, \
            "Calls precompute_reference_stats() before"

        if self.model is None:
            self._load_model(audio_sr=sr)

        dim = self.embedding_dim
        sum_x = torch.zeros(dim, dtype=torch.float64, device=self.device)
        sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        count = 0

        for wav in generated_wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() > 1:
                wav = wav.squeeze()
            emb = self._get_embeddings(wav.float()).double()
            sum_x = sum_x + emb.sum(dim=0)
            sum_xx = sum_xx + emb.T @ emb
            count = count + emb.shape[0]

        if count < 2:
            print("[FAD] Too few embeddings for computing the covariance")
            return float("nan")

        mu_gen, sigma_gen, _ = compute_mu_sigma(sum_x, sum_xx, count)

        fad = compute_frechet_distance(
            self.mu_ref, self.sigma_ref, mu_gen, sigma_gen,
        )
        return float(fad.item())

    @torch.no_grad()
    def compute_fad(
        self,
        real_wavs: List[torch.Tensor],
        generated_wavs: List[torch.Tensor],
        sr: int = 44100,
    ) -> float:
        """FAD standalone without pre-computed reference."""
        if self.model is None:
            self._load_model(audio_sr=sr)

        dim = self.embedding_dim

        # Real
        sum_x_r = torch.zeros(dim, dtype=torch.float64, device=self.device)
        sum_xx_r = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        count_r = 0
        for wav in real_wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() > 1:
                wav = wav.squeeze()
            emb = self._get_embeddings(wav.float()).double()
            sum_x_r = sum_x_r + emb.sum(dim=0)
            sum_xx_r = sum_xx_r + emb.T @ emb
            count_r = count_r + emb.shape[0]

        # Generated
        sum_x_g = torch.zeros(dim, dtype=torch.float64, device=self.device)
        sum_xx_g = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        count_g = 0
        for wav in generated_wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() > 1:
                wav = wav.squeeze()
            emb = self._get_embeddings(wav.float()).double()
            sum_x_g = sum_x_g + emb.sum(dim=0)
            sum_xx_g = sum_xx_g + emb.T @ emb
            count_g = count_g + emb.shape[0]

        mu_r, sigma_r, _ = compute_mu_sigma(sum_x_r, sum_xx_r, count_r)
        mu_g, sigma_g, _ = compute_mu_sigma(sum_x_g, sum_xx_g, count_g)

        fad = compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
        return float(fad.item())


# ============================================================
# EVALUATION FUNCTION (for the training loop)
# ============================================================

@torch.no_grad()
def evaluate_generation(
    model,
    normalizer,
    val_dataset,
    n_samples: int = 64,
    euler_steps: int = 50,
    device: str = "cuda",
    use_amp: bool = True,
    fad_calculator: FADCalculator = None,
    fd_dac_ref_stats: dict = None,
) -> dict:
    """
    Generate N samples and compute FD-DAC + FAD against pre-computed reference.
    """
    from audio_dataset_npy import DAC_SAMPLE_RATE

    model.eval()
    n_frames = val_dataset.n_frames
    token_dim = 1024
    T_MIN, T_MAX = 0.001, 0.999

    # Generate N samples. Each generated latent is moved to CPU right away, so
    # the GPU only ever holds the single (1, n_frames, token_dim) tensor being
    # integrated, not the whole batch of N.
    generated_latents_list = []
    for i in tqdm(range(n_samples), desc="Metrics: generating samples"):
        x = torch.randn(1, n_frames, token_dim, device=device)
        dt = (T_MAX - T_MIN) / euler_steps
        for s in range(euler_steps):
            t_val = T_MIN + s * dt
            t = torch.ones(1, device=device) * t_val
            with torch.amp.autocast('cuda', enabled=use_amp):
                v = model(x, t)
            x = x + v.float() * dt
        gen_frames = x[0].cpu()
        generated_latents_list.append(gen_frames)
        del x

    # Free the GPU activations left by the generation loop before the FD-DAC /
    # decode / FAD phases run.
    if device == "cuda":
        torch.cuda.empty_cache()

    generated_latents = torch.stack(generated_latents_list)   # on CPU

    # FD-DAC: Frechet distance on the DAC latents (full covariance).
    # compute_fd_dac streams generated_latents to the GPU in small blocks, so
    # passing the full CPU tensor here does NOT cause a big GPU allocation.
    fd_dac = None
    if fd_dac_ref_stats is not None:
        fd_dac = compute_fd_dac(generated_latents, fd_dac_ref_stats, device=device)

    # Decode with DAC to compute FAD.
    #
    # SPEED NOTE: decoding the N generated latents with DAC on CPU is extremely
    # slow — measured as the phase that pinned one core at 100% for many
    # minutes and made the metrics step look "frozen" (and, when a watchdog or
    # the user gave up, killed). DAC decode is a GPU job: here we run it on
    # `device` (the same GPU used for training). At this point the generation
    # loop is done and its activations have been freed (empty_cache above), and
    # the N latents already live on CPU, so the GPU has room. We still decode
    # ONE sample at a time and move each waveform to CPU immediately, so peak
    # VRAM stays tiny (one 5 s clip at a time), and we show a progress bar so
    # the phase is visibly advancing instead of appearing stuck.
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to(device)
    dac_model.eval()

    generated_wavs = []
    for gen_frames in tqdm(generated_latents_list, desc="Metrics: DAC decode"):
        z = gen_frames.T
        z = normalizer.denormalize(z).to(device)
        wav = dac_model.decode(z.unsqueeze(0).float()).squeeze()
        generated_wavs.append(wav.cpu())
        del z, wav

    del dac_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # FAD
    fad = None
    if fad_calculator is not None and fad_calculator.mu_ref is not None:
        fad = fad_calculator.compute_fad_against_reference(
            generated_wavs, sr=DAC_SAMPLE_RATE,
        )

    return {
        "fd_dac": fd_dac,
        "fad": fad,
        "generated_wavs": generated_wavs,
        "generated_latents": generated_latents,
    }


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
    print(f"  FD: {fd.item():.4f} (deve essere > 0)")
    assert fd.item() > 0
    print("  OK!\n")

    print("Test compute_fd_dac (blocked accumulation)...")
    n_samples, n_frames, dim = 10, 430, 1024
    gen_latents = torch.randn(n_samples, n_frames, dim)
    ref_stats = {
        "mu": torch.zeros(dim, dtype=torch.float64),
        "sigma": torch.eye(dim, dtype=torch.float64),
        "n_total": 1000,
    }
    fd_dac = compute_fd_dac(gen_latents, ref_stats, device="cpu", block_size=4)
    print(f"  FD-DAC: {fd_dac:.4f}")
    print("  OK!\n")

    print("Test FAD con Encodec...")
    try:
        fad_calc = FADCalculator(device="cpu")
        wav1 = [torch.randn(44100) for _ in range(5)]
        wav2 = [torch.randn(44100) * 0.5 for _ in range(5)]
        fad = fad_calc.compute_fad(wav1, wav2, sr=44100)
        print(f"  FAD (random vs mitigated noise): {fad:.4f}")
        print("  OK!")
    except ImportError as e:
        print(f" Missing dependency: {e}")
