# Distributional metrics: FD on the codec latents, KL, FAD-VGGish.

import os
import torch
import numpy as np
from typing import Optional
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils')


def compute_mu_sigma(sum_x: torch.Tensor, sum_xx: torch.Tensor, n) -> tuple:
    if isinstance(n, torch.Tensor):
        n = n.item()
    mu = sum_x / n
    sigma = (sum_xx - torch.outer(sum_x, mu)) / (n - 1)
    return mu, sigma, n


def symmetric_psd_matrix_sqrt(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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


def gaussian_kl_fullcov(
    mu_p: torch.Tensor,
    sigma_p: torch.Tensor,
    mu_q: torch.Tensor,
    sigma_q: torch.Tensor,
    eps: float = 1e-6,
) -> float:
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

    X = torch.cholesky_solve(Sp, Lq)
    tr_term = torch.trace(X)

    diff = (mu_q - mu_p).unsqueeze(1)
    sol = torch.cholesky_solve(diff, Lq)
    maha = (diff * sol).sum()

    kl = 0.5 * (tr_term + maha - d + (logdet_q - logdet_p))
    return float(kl.item())


@torch.no_grad()
def precompute_latent_reference(
    val_dataset,
    cache_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_accum: int = 50,
) -> dict:
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
        frames, _ = val_dataset[idx]
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
        _tmp = str(cache_path_obj) + ".tmp"
        torch.save(stats, _tmp)
        os.replace(_tmp, str(cache_path_obj))
        print(f"[Latent Reference] Cache saved in: {cache_path}")

    return stats


def precompute_fd_dac_reference(val_dataset, cache_path=None, device=None,
                                batch_accum: int = 50) -> dict:
    return precompute_latent_reference(
        val_dataset, cache_path=cache_path, device=device, batch_accum=batch_accum)


def _generated_mu_sigma(generated_latents: torch.Tensor, device: str,
                        block_size: int = 16):
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


def compute_fd_dac(
    generated_latents: torch.Tensor,
    ref_stats: dict,
    device: str = "cuda",
    block_size: int = 16,
) -> float:
    mu_gen, sigma_gen = _generated_mu_sigma(generated_latents, device, block_size)

    mu_ref = ref_stats["mu"].to(device)
    sigma_ref = ref_stats["sigma"].to(device)

    fd = compute_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

    del mu_gen, sigma_gen, mu_ref, sigma_ref
    if device == "cuda":
        torch.cuda.empty_cache()

    return float(fd.item())


def compute_kl_both(
    generated_latents: torch.Tensor,
    ref_stats: dict,
    device: str = "cuda",
    block_size: int = 16,
) -> dict:
    mu_gen, sigma_gen = _generated_mu_sigma(generated_latents, device, block_size)

    mu_ref = ref_stats["mu"].to(device)
    sigma_ref = ref_stats["sigma"].to(device)

    kl_real_gen = gaussian_kl_fullcov(mu_ref, sigma_ref, mu_gen, sigma_gen)
    kl_gen_real = gaussian_kl_fullcov(mu_gen, sigma_gen, mu_ref, sigma_ref)

    del mu_gen, sigma_gen, mu_ref, sigma_ref
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"kl_real_gen": kl_real_gen, "kl_gen_real": kl_gen_real}


DAC_METRICS = ("fd_dac", "kl_dac")


def compute_dac_metrics(
    generated_latents: torch.Tensor,
    ref_stats: dict,
    enabled=DAC_METRICS,
    device: str = "cuda",
    block_size: int = 16,
) -> dict:
    out = {"fd_dac": None, "kl_real_gen": None, "kl_gen_real": None}
    want_fd = "fd_dac" in enabled
    want_kl = "kl_dac" in enabled
    if not (want_fd or want_kl):
        return out

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


class EncodecEmbedder:
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
                x = torch.cat((x, x), dim=1)
        elif x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        e = self._model.encoder(x)
        return e.permute(0, 2, 1).reshape(-1, e.shape[1])


class VGGishEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self.embedding_dim = 128

    def _load(self):
        if self._model is not None:
            return
        model = torch.hub.load('harritaylor/torchvggish', 'vggish',
                               postprocess=False, trust_repo=True)
        model.eval().to(self.device)
        model.device = self.device
        self._model = model
        print(f"[VGGish] loaded on {self.device} (dim={self.embedding_dim}, ~0.96s window)")

    @torch.no_grad()
    def embed(self, audio: torch.Tensor, audio_sr: int) -> torch.Tensor:
        self._load()
        wav = audio[0].mean(dim=0).cpu().numpy()
        emb = self._model(wav, fs=audio_sr)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return emb.to(self.device)


@torch.no_grad()
def _audio_clips_to_mu_sigma(clip_iter, n_items, embedder, device, desc):
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
def precompute_audio_reference(val_wav_source, embedder, cache_path=None,
                               device: str = "cuda") -> dict:
    if cache_path is not None and Path(cache_path).exists():
        print(f"[Audio ref] loading cache: {cache_path}")
        return torch.load(str(cache_path), map_location="cpu", weights_only=False)
    import soundfile as sf
    if isinstance(val_wav_source, (str, Path)):
        wavs = sorted(Path(val_wav_source).rglob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav under {val_wav_source}")
    else:
        wavs = [Path(w) for w in val_wav_source]
        missing = [w for w in wavs if not w.exists()]
        if not wavs:
            raise FileNotFoundError(
                "precompute_audio_reference received an EMPTY file list.")
        if missing:
            raise FileNotFoundError(
                f"{len(missing)}/{len(wavs)} reference wavs do not exist "
                f"(first: {missing[0]}). Re-run preprocess_stream.py with "
                f"--save_wav, or switch metrics.fad_reference to 'decoded'.")
    print(f"[Audio ref] embedding {len(wavs)} real val wavs (NO DAC)...")

    def _iter():
        for p in wavs:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
            w = torch.from_numpy(data.T.copy())
            yield w.unsqueeze(0), sr

    mu, sigma, count = _audio_clips_to_mu_sigma(_iter(), len(wavs), embedder, device, "Audio ref")
    stats = {"mu": mu.cpu(), "sigma": sigma.cpu(), "n_total": count}
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        _tmp = str(cache_path) + ".tmp"
        torch.save(stats, _tmp)
        os.replace(_tmp, str(cache_path))
        print(f"[Audio ref] cache saved: {cache_path}")
    return stats


@torch.no_grad()
def _decode_embed_mu_sigma(generated_latents, normalizer, embedder, device, desc):
    from audio_dataset_npy import decode_latents
    import latent_codec as lc
    sr = lc.active().sample_rate

    def _iter():
        for i in range(generated_latents.shape[0]):
            z = generated_latents[i].T
            z = normalizer.denormalize(z)
            wav = decode_latents(z, device=device)
            yield wav.unsqueeze(0), sr

    return _audio_clips_to_mu_sigma(_iter(), generated_latents.shape[0], embedder, device, desc)


ALL_METRICS = ("fd_dac", "kl_dac", "fad_encodec", "fad_vggish")

COND_METRICS = ("fd_dac", "kl_dac", "fad_vggish")


@torch.no_grad()
def compute_audio_mu_sigma(clip_iter, n_items, embedder, device: str = "cuda",
                           desc: str = "Audio stats"):
    return _audio_clips_to_mu_sigma(clip_iter, n_items, embedder, device, desc)


def compute_fad(mu_gen, sigma_gen, ref_stats: dict, device: str = "cuda") -> float:
    mu_ref = ref_stats["mu"].to(device)
    sigma_ref = ref_stats["sigma"].to(device)
    fad = compute_frechet_distance(mu_ref, sigma_ref,
                                   mu_gen.to(device), sigma_gen.to(device))
    del mu_ref, sigma_ref
    if device == "cuda":
        torch.cuda.empty_cache()
    return float(fad.item())


def make_embedders(enabled, device: str = "cuda", encodec_sr: int = 24000) -> dict:
    emb = {}
    if "fad_encodec" in enabled:
        emb["encodec"] = EncodecEmbedder(audio_sr_model=encodec_sr, device=device)
    if "fad_vggish" in enabled:
        emb["vggish"] = VGGishEmbedder(device=device)
    return emb


def build_references(enabled, val_dataset, val_wav_root, embedders, cache_dir,
                     device: str = "cuda", strict: bool = True) -> dict:
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
                    f"or call build_references(strict=False) to skip it and "
                    f"continue.") from e
            print(f"[Metrics] WARNING (strict=false): could not build the {name} "
                  f"reference ({type(e).__name__}: {e}). Skipping {name} this run — {need}")
    return refs


@torch.no_grad()
def evaluate_generation(model, normalizer, val_dataset, *, enabled, references,
                        embedders=None, n_samples: int = 64, euler_steps: int = 50,
                        t_min: float = 0.001, t_max: float = 0.999,
                        seed: Optional[int] = None,
                        device: str = "cuda", use_amp: bool = False) -> dict:
    try:
        from sampling import euler_integrate
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "evaluate_generation() is the unconditional metrics path (needs the "
            "uncond sampling.py / euler_integrate); the conditioned pipeline does "
            "not use it -- see precompute_latent_reference + condition_metrics.") from e
    embedders = embedders or {}
    model.eval()
    n_frames = val_dataset.n_frames
    token_dim = val_dataset[0][0].shape[-1]

    gen_rng = None
    if seed is not None:
        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(int(seed))

    gen_list = []
    n_skipped = 0
    for _ in tqdm(range(n_samples), desc="Metrics: generating samples"):
        x = torch.randn(1, n_frames, token_dim, device=device, generator=gen_rng)
        x = euler_integrate(model, x, steps=euler_steps,
                            t_min=t_min, t_max=t_max, use_amp=use_amp)
        gf = x[0].cpu()
        if torch.isfinite(gf).all():
            gen_list.append(gf)
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

    generated_latents = torch.stack(gen_list)

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


