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
#     multivariate Gaussians N(mu, Sigma) with FULL 1024x1024 covariance.
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
# The FAD (Encodec-based) metric was REMOVED on purpose; everything here is
# latent-only, so there is no audio decoding, no Encodec, and no wav_root
# needed for the metrics.

import os
import torch
import numpy as np
from typing import List, Optional
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
        torch.save(stats, str(cache_path_obj))
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
    ref_stats: dict = None,
) -> dict:
    """
    Generate N samples in the normalized latent space and compute, against the
    pre-computed real reference (ref_stats):
        - FD-DAC
        - KL(real||gen) and KL(gen||real)

    All metrics are latent-only and use the generated latents PRE-denormalization
    (i.e. exactly the model output, in normalized space), matching the space of
    the reference. No audio decoding is involved.

    Returns a dict with the scalar metrics and the generated latents.
    """
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

    if device == "cuda":
        torch.cuda.empty_cache()

    generated_latents = torch.stack(generated_latents_list)   # on CPU, normalized

    fd_dac = None
    kl_real_gen = None
    kl_gen_real = None

    if ref_stats is not None:
        fd_dac = compute_fd_dac(generated_latents, ref_stats, device=device)
        kl = compute_kl_both(generated_latents, ref_stats, device=device)
        kl_real_gen = kl["kl_real_gen"]
        kl_gen_real = kl["kl_gen_real"]

    return {
        "fd_dac": fd_dac,
        "kl_real_gen": kl_real_gen,
        "kl_gen_real": kl_gen_real,
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
