# Tests for metrics.py: Frechet distance, Gaussian KL, block accumulation.
# Run with `pytest test_functions` or `python test_functions/test_metrics.py`.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from metrics import (compute_frechet_distance, gaussian_kl_fullcov,
                     compute_fd_dac, compute_kl_both)


def _two_gaussians(d=128):
    torch.manual_seed(0)
    mu1 = torch.randn(d).double()
    A = torch.randn(d, d).double()
    sigma1 = (A @ A.T) / d
    mu2 = torch.randn(d).double()
    B = torch.randn(d, d).double()
    sigma2 = (B @ B.T) / d
    return mu1, sigma1, mu2, sigma2


def test_frechet_distance():
    mu1, sigma1, mu2, sigma2 = _two_gaussians()
    fd = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"  FD: {fd.item():.4f} (> 0)")
    assert fd.item() > 0


def test_gaussian_kl_fullcov():
    mu1, sigma1, mu2, sigma2 = _two_gaussians()
    d = mu1.shape[0]
    kl_same = gaussian_kl_fullcov(mu1, sigma1 + torch.eye(d), mu1, sigma1 + torch.eye(d))
    print(f"  KL(P||P) ~ 0: {kl_same:.2e}")
    assert abs(kl_same) < 1e-5
    kl_pq = gaussian_kl_fullcov(mu1, sigma1 + torch.eye(d), mu2, sigma2 + torch.eye(d))
    kl_qp = gaussian_kl_fullcov(mu2, sigma2 + torch.eye(d), mu1, sigma1 + torch.eye(d))
    print(f"  KL(P||Q)={kl_pq:.4f}  KL(Q||P)={kl_qp:.4f}  (asymmetric, > 0)")
    assert kl_pq > 0 and kl_qp > 0


def test_block_accumulation():
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
    fd_a = compute_fd_dac(gen_latents, ref_stats, device="cpu", block_size=4)
    fd_b = compute_fd_dac(gen_latents, ref_stats, device="cpu", block_size=16)
    assert abs(fd_a - fd_b) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  OK\n")