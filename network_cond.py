# network_cond.py
#
# Conditioned Diffusion Transformer (DiT) for DAC audio latents.
#
# This file EXTENDS network.py without changing it. Every shared component
# below (modulate, RoPE helpers, TimestepEmbedder, SelfAttention, FFN,
# DiTBlock, FinalLayer) is copied VERBATIM from network.py, which uses the
# two intentional deviations from the official DiT:
#   - RoPE inside self-attention instead of an additive sin/cos pos embedding
#     (Su et al., RoFormer, 2021)
#   - SwiGLU FFN instead of the GELU MLP (Shazeer, 2020)
# Position is therefore encoded inside the attention (RoPE); there is NO
# additive positional embedding, exactly as in network.py.
#
# Conditioning is added ON TOP, without touching the block:
#
#   * FRAME-LEVEL conditions (f0, chroma, rhythm, energy) are injected by
#     CONCATENATION ON THE FEATURE DIMENSION at the input, exactly as in
#     JASCO (audiocraft/models/flow_matching.py, forward):
#         for each temporal condition c: x = torch.concat((x, c), dim=-1)
#         input_ = self.emb(x)
#     Each condition is first projected by a single Linear (raw_dim -> out_dim)
#     in conditions.FrameConditionEncoder (== JASCO MelodyConditioner's
#     output_proj). The concatenation widens input_proj from token_dim to
#     token_dim + sum(out_dim); everything after input_proj is unchanged.
#
#   * GLOBAL conditions (text-CLAP, image-CLIP) are injected via AdaLN, exactly
#     as the class label in the official DiT (c = t + y): encoded to
#     hidden_size and ADDED to the timestep embedding to form the conditioning
#     vector c, which modulates every block and the final layer.
#
# CFG: passing null (zero) conditions yields the unconditional output.
#
# No patching: every DAC frame is directly a token (72-dim, DAC pre-quantizer
# latents; TOKEN_DIM = DAC_LATENT_DIM, inherited from audio_dataset_npy).

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_dataset_npy import DAC_LATENT_DIM, MAX_FRAMES
from conditions import FrameConditionEncoder, GlobalConditionEncoder


# ============================================================
# TOKEN DIM = DAC_LATENT_DIM directly
# ============================================================
TOKEN_DIM = DAC_LATENT_DIM   # 72 (DAC pre-quantizer latents) - no patching


# ============================================================
# MODULATE (identical to network.py / facebookresearch/DiT)
# ============================================================
def modulate(x, shift, scale):
    """
    AdaLN modulation.
    x:     (B, T, D)
    shift: (B, D)
    scale: (B, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================
# RoPE  (copied 1:1 from network.py; rotate-half convention)
# ============================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input. Identical to
    transformers.models.llama.modeling_llama.rotate_half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Identical to transformers.models.llama.modeling_llama.apply_rotary_pos_emb.
    q, k: (B, n_heads, S, head_dim)   cos, sin: (B|1, S, head_dim)
    unsqueeze_dim=1 broadcasts cos/sin over the head axis."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_default_rope_parameters(head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """inv_freq = 1 / theta^(2i/head_dim), i=0..head_dim/2-1.
    Same as transformers _compute_default_rope_parameters (rope_type='default')."""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


# ============================================================
# TIMESTEP EMBEDDING (identical to network.py)
# ============================================================
class TimestepEmbedder(nn.Module):
    """
    Sinusoidal embedding of the (continuous) timestep followed by a
    2-layer MLP. Same structure as in the official DiT.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ============================================================
# SELF ATTENTION  (RoPE applied exactly as in HF Llama) - identical to network.py
# ============================================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, max_seq_len: int = 4096,
                 theta: float = 10000.0):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_size // n_heads

        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # inv_freq buffer, non-persistent like HF (deterministic -> not saved).
        inv_freq = compute_default_rope_parameters(self.head_dim, theta=theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, S: int, device, dtype):
        # Mirrors LlamaRotaryEmbedding.forward (rope_type='default'):
        # freqs = positions (outer) inv_freq ; emb = cat(freqs, freqs).
        t = torch.arange(S, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)              # (S, head_dim)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]  # (1, S, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        # -> (B, n_heads, S, head_dim), the same layout HF rotates in
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        cos, sin = self._cos_sin(S, x.device, x.dtype)       # (1, S, head_dim)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)          # identical to HF

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, S, -1)
        return self.proj(x)


# ============================================================
# FFN  (SwiGLU) - identical to network.py
# ============================================================
class FFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0,
                 multiple_of: int = 256, dropout: float = 0.0):
        super().__init__()
        # SwiGLU has 3 matrices (w1 gate, w3 up, w2 down) vs 2 of a classic FFN.
        # To match params/FLOPs to a standard FFN at mlp_ratio*hidden, the width
        # is scaled by 2/3 (Shazeer 2020), then rounded to a multiple for tensor
        # core efficiency (LLaMA convention). Old behaviour: inner = hidden*4
        # (3 matrices at 4x) = +50% FFN params vs the matched sizing. Kept 1:1
        # with network.py so the conditioned backbone matches the unconditional.
        inner = int(2 * (mlp_ratio * hidden_size) / 3)
        inner = multiple_of * ((inner + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(hidden_size, inner, bias=False)   # gate
        self.w3 = nn.Linear(hidden_size, inner, bias=False)   # up
        self.w2 = nn.Linear(inner, hidden_size, bias=False)   # down
        # Two dropouts with the same p, mirroring timm Mlp (drop1 after the
        # activation/gating on the hidden tensor, drop2 after the output proj).
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w3(x)   # gated hidden  (B, T, inner)
        h = self.drop1(h)
        h = self.w2(h)                         # output proj   (B, T, hidden)
        h = self.drop2(h)
        return h


# ============================================================
# DIT BLOCK (identical to network.py: RoPE attn + SwiGLU FFN)
# ============================================================
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Structurally identical to the official DiTBlock:
        x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * mlp (modulate(norm2(x), shift_mlp, scale_mlp))
    norm2 acts on the NEW x (post-attention), not on the input.

    Deviations from official DiT (see network.py docstring):
      - self.attn uses RoPE (SelfAttention) instead of timm Attention
      - self.mlp  is SwiGLU (FFN) instead of timm GELU Mlp
    `drop` is wired ONLY into the FFN (not the attention). Default 0.0 -> inert.

    Frame-level conditions are NOT handled here (they are concatenated at the
    input); global conditions reach the block only through `c` (AdaLN), exactly
    like the class label in DiT.
    """

    def __init__(self, hidden_size, num_heads, max_seq_len=4096,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = SelfAttention(hidden_size, num_heads, max_seq_len=max_seq_len)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = FFN(hidden_size, mlp_ratio=mlp_ratio, dropout=drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================
# FINAL LAYER (identical to network.py)
# ============================================================
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================
# CONDITIONED AUDIO DIT
# ============================================================
class ConditionedAudioDiT(nn.Module):
    """
    Conditioned Diffusion Transformer for DAC audio latents.

    Block structure identical to the unconditional AudioDiT (network.py):
    RoPE self-attention + SwiGLU FFN + AdaLN-Zero, no additive pos embedding.
    Conditioning added on top WITHOUT touching the block:
      - frame-level conditions -> concatenated on the feature dim at the input
        (JASCO), then projected to hidden by a single input projection;
      - global conditions       -> added to the AdaLN conditioning vector c
        (official DiT class-label mechanism).

    Configurations (same as AudioDiT):
        'S':  6 layers,  512 hidden,  8 heads   head_dim=64
        'B': 12 layers,  768 hidden, 12 heads   head_dim=64
        'G': 18 layers, 1024 hidden, 16 heads   head_dim=64  <- between B and L
        'L': 24 layers, 1024 hidden, 16 heads   head_dim=64
        'XL':28 layers, 1152 hidden, 16 heads   head_dim=72  <- official DiT-XL

    Args:
        frame_cond_dims:     {name: raw_dim}, e.g.
                             {"f0": 2, "chroma": 12, "rhythm": 2}.
                             Raw per-frame dimensionality produced by the
                             extractors (conditions.py). Empty/None -> no frame
                             conditioning (input_proj is token_dim -> hidden,
                             exactly the unconditional case).
        frame_cond_out_dims: {name: out_dim}, the per-condition projection
                             width (JASCO bottleneck). Must have the same keys
                             as frame_cond_dims. Empty/None when no frame conds.
        global_cond_configs: {name: {"dim": d}}, e.g.
                             {"text": {"dim": 512}, "image": {"dim": 512}}.
                             Empty/None -> timestep-only AdaLN (no global).

    CFG: passing null (zero) conditions yields the unconditional output.
    """

    CONFIGS = {
        'S':  dict(n_layers=6,  hidden_size=512,  n_heads=8),
        'B':  dict(n_layers=12, hidden_size=768,  n_heads=12),
        'G':  dict(n_layers=18, hidden_size=1024, n_heads=16),
        'L':  dict(n_layers=24, hidden_size=1024, n_heads=16),
        'XL': dict(n_layers=28, hidden_size=1152, n_heads=16),
    }

    def __init__(
        self,
        token_dim:           int   = TOKEN_DIM,
        max_seq_len:         int   = MAX_FRAMES + 16,
        kind:                str   = 'L',
        mlp_ratio:           float = 4.0,
        drop:                float = 0.0,
        frame_cond_dims:     Optional[Dict[str, int]]  = None,
        frame_cond_out_dims: Optional[Dict[str, int]]  = None,
        global_cond_configs: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()
        cfg = self.CONFIGS[kind]
        self.kind        = kind
        self.token_dim   = token_dim
        self.max_seq_len = max_seq_len
        hidden_size      = cfg['hidden_size']
        n_layers         = cfg['n_layers']
        n_heads          = cfg['n_heads']

        self.frame_cond_dims     = dict(frame_cond_dims)     if frame_cond_dims     else {}
        self.frame_cond_out_dims = dict(frame_cond_out_dims) if frame_cond_out_dims else {}
        self.global_cond_configs = dict(global_cond_configs) if global_cond_configs else {}
        self.has_frame  = len(self.frame_cond_dims)     > 0
        self.has_global = len(self.global_cond_configs) > 0

        if self.has_frame:
            missing = set(self.frame_cond_dims) - set(self.frame_cond_out_dims)
            if missing:
                raise ValueError(
                    f"frame_cond_out_dims is missing the out_dim for {sorted(missing)}. "
                    f"Every frame condition must declare both raw_dim and out_dim."
                )

        # ----- Frame-level conditioning (JASCO concat) -----
        # FrameConditionEncoder holds one Linear(raw_dim -> out_dim) per
        # condition and returns the concatenation of the projected conditions
        # in a fixed canonical order. The network then concatenates that with
        # the noisy latent on the feature dim.
        frame_extra = 0
        if self.has_frame:
            self.frame_encoder = FrameConditionEncoder(
                self.frame_cond_dims, self.frame_cond_out_dims,
            )
            frame_extra = self.frame_encoder.total_out_dim

        # Token projection (no patching). Width grows by the concatenated
        # frame-condition channels (= 0 in the unconditional / no-frame case,
        # which makes this identical to network.py's input_proj).
        self.input_proj = nn.Linear(token_dim + frame_extra, hidden_size, bias=True)

        # Timestep embedder (sinusoidal + MLP)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # ----- Global conditioning (AdaLN, added to c) -----
        if self.has_global:
            self.global_encoder = GlobalConditionEncoder(
                self.global_cond_configs, hidden_size,
            )

        # DiT blocks (RoPE handles position inside the attention; max_seq_len is
        # plumbed through so each block can build its rope inv_freq buffer).
        # IDENTICAL to network.py's blocks.
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, max_seq_len=max_seq_len,
                     mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(n_layers)
        ])

        # Final layer (AdaLN-Zero modulated by c)
        self.final_layer = FinalLayer(hidden_size, token_dim)

        # Initialise weights as in network.py / the official DiT
        self.initialize_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[ConditionedAudioDiT-{kind}] {n_params/1e6:.1f}M params | "
              f"hidden={hidden_size} | layers={n_layers} | heads={n_heads}")
        print(f"  Frame conditions (concat): "
              f"{self.frame_cond_dims if self.has_frame else 'NONE'}"
              + (f" -> out {self.frame_cond_out_dims} "
                 f"(+{frame_extra} input channels)" if self.has_frame else ""))
        print(f"  Global conditions (AdaLN): "
              f"{list(self.global_cond_configs.keys()) if self.has_global else 'NONE'}")

    def initialize_weights(self):
        """
        Identical to network.py / facebookresearch/DiT:
          - xavier_uniform_ on every nn.Linear, bias to 0 (this also covers the
            new frame projections and global encoder; no special init is needed
            because adaLN-Zero + zero final layer already make the model start
            as the identity, exactly as in JASCO which does not gate the
            concatenated conditions)
          - normal_(std=0.02) on the two layers of the timestep MLP
          - zero-out adaLN_modulation[-1] of every block (adaLN-Zero)
          - zero-out adaLN_modulation[-1] and linear of the final layer
        (No pos_embed init: position is handled by RoPE.)
        """
        # Basic init: xavier_uniform on every Linear, bias to 0
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Timestep MLP: normal init (std=0.02), as in the official DiT
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in every DiT block (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out the final layer modulation + projection (zero output)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _gather_frame_conditions(
        self,
        frame_conditions: Optional[Dict[str, torch.Tensor]],
        B: int, T: int, device, dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        Return a dict with EVERY expected frame condition present. Missing or
        None conditions are filled with zeros (the null condition for CFG), so
        the concatenation width is always constant. Zeros are the same null
        used by make_null_frame_conditions and by the training CFG dropout.
        """
        out = {}
        fc = frame_conditions or {}
        for name, raw_dim in self.frame_cond_dims.items():
            c = fc.get(name, None)
            if c is None:
                c = torch.zeros(B, T, raw_dim, device=device, dtype=dtype)
            else:
                c = c.to(device=device, dtype=dtype)
            out[name] = c
        return out

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        frame_conditions:  Optional[Dict[str, torch.Tensor]] = None,
        global_conditions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        x: (B, n_frames, token_dim)   one token per DAC frame
        t: (B,)                        timestep in [0, 1]
        frame_conditions:  {"f0": (B,T,2), "chroma": (B,T,12), "rhythm": (B,T,2)} or None
        global_conditions: {"text":  (B,d),  "image":  (B,d), ...}    or None

        Returns:
            velocity field of shape (B, n_frames, token_dim)
        """
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()
        B, T, _ = x.shape

        # ----- FRAME conditions: concat on the feature dim (JASCO) -----
        if self.has_frame:
            fc = self._gather_frame_conditions(frame_conditions, B, T, x.device, x.dtype)
            frame_proj = self.frame_encoder(fc)        # (B, T, sum(out_dim))
            x = torch.cat([x, frame_proj], dim=-1)     # (B, T, token_dim + sum(out_dim))

        # Token projection (position is injected by RoPE inside attention; no
        # additive positional embedding, exactly as in network.py).
        x = self.input_proj(x)

        # ----- GLOBAL conditions: AdaLN vector c = t_emb + g_global -----
        c = self.t_embedder(t)                          # (B, hidden)
        if self.has_global and global_conditions:
            g = self.global_encoder(global_conditions)  # (B, hidden) or None
            if g is not None:
                c = c + g

        # DiT blocks (block is identical to the unconditional network.py)
        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    B, N = 2, 430   # ~5 seconds

    x = torch.randn(B, N, TOKEN_DIM)
    t = torch.rand(B)

    # --- f0 only ---
    print("=== Frame: f0 only ===")
    model = ConditionedAudioDiT(
        kind='S',
        frame_cond_dims={"f0": 2},
        frame_cond_out_dims={"f0": 16},
        global_cond_configs={},
    )
    out = model(x, t, frame_conditions={"f0": torch.randn(B, N, 2)})
    print(f"  input {x.shape} -> output {out.shape}")
    assert out.shape == x.shape
    assert out.shape == model(x, t).shape  # frame conds omitted -> zeros

    # --- All three frame conditions + global text/image ---
    print("\n=== Frame: f0+chroma+rhythm | Global: text+image ===")
    model2 = ConditionedAudioDiT(
        kind='S',
        frame_cond_dims={"f0": 2, "chroma": 12, "rhythm": 2},
        frame_cond_out_dims={"f0": 16, "chroma": 64, "rhythm": 32},
        global_cond_configs={"text": {"dim": 512}, "image": {"dim": 512}},
    )
    out2 = model2(
        x, t,
        frame_conditions={
            "f0": torch.randn(B, N, 2),
            "chroma": torch.randn(B, N, 12),
            "rhythm": torch.randn(B, N, 2),
        },
        global_conditions={"text": torch.randn(B, 512), "image": torch.randn(B, 512)},
    )
    print(f"  full conditioned -> {out2.shape}")
    assert out2.shape == x.shape

    # --- Global only ---
    print("\n=== Global only (text) ===")
    model3 = ConditionedAudioDiT(
        kind='S',
        frame_cond_dims={},
        frame_cond_out_dims={},
        global_cond_configs={"text": {"dim": 512}},
    )
    out3 = model3(x, t, global_conditions={"text": torch.randn(B, 512)})
    print(f"  text only -> {out3.shape}")
    assert out3.shape == x.shape

    print("\nTest passed!")
