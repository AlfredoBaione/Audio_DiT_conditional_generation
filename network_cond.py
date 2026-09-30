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
#     OPTIONAL PER-BLOCK RE-INJECTION (`frame_reinject_every`, default 0 = off,
#     i.e. plain JASCO). With JASCO's input-only concatenation the frame
#     conditions touch exactly ONE matrix -- input_proj -- and then have to
#     survive the whole depth inside the residual stream, while a GLOBAL
#     condition modulates every block through AdaLN. Music ControlNet (Wu et
#     al., 2024) makes the opposite choice and keeps feeding the temporal
#     control back into the trunk. This flag adds that path: the SAME
#     `frame_proj` produced once by the encoder is re-added to the hidden state
#     before selected blocks through a per-block, ZERO-INITIALISED projection
#     (ControlNet's zero-conv / DiT's adaLN-Zero principle), so at init it
#     contributes exactly 0: given the same shared weights the network computes
#     the same function as the input-only one, and learns the extra path only if
#     it pays. (The two are the same FUNCTION at init, not the same WEIGHTS: the
#     extra modules also consume the RNG stream, so two separately seeded builds
#     do not draw identical shared weights.)
#
#     The re-injected term is scaled by a PER-CONDITION GATE driven by the
#     AdaLN vector c (identity-init). The projection alone is a function of the
#     conditions only, so without the gate the term re-added at a given block
#     is the same tensor at every denoising step, while every other path in the
#     network -- the block's own shift/scale/gate, and the global conditions
#     that ride inside c -- is free to weigh itself against t. The gate closes
#     that gap, one scalar per frame condition, so f0 and chroma can follow
#     different schedules in t instead of rising and falling together.
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
# CROSS-ATTENTION ON A TEXT SEQUENCE  (PixArt-alpha)
# ============================================================
class CrossAttention(nn.Module):
    """
    The audio tokens QUERY a text sequence: Q from x, K and V from the context.

    WHY IT EXISTS AT ALL. The pooled text vector reaches the network through
    AdaLN, where it becomes shift/scale/gate -- vectors of shape (B, hidden)
    that `modulate` broadcasts over time. So it can only apply ONE affine
    transform, the same at every one of the ~431 frames: there is no path by
    which a word could act on frame 200 differently from frame 10. Here every
    audio token computes its own query and reads the words it needs. Three
    things follow that AdaLN cannot give: the conditioning becomes time-varying,
    it becomes content-dependent (Q is a function of x, so the same word weighs
    differently depending on what is already being generated, and differently
    again at each depth), and its capacity grows with the length of the text
    instead of being compressed into one vector that also has to share a channel
    with the timestep embedding.

    NO POSITIONAL ENCODING ON THE CONTEXT. RoPE rotates queries and keys by
    their position in the AUDIO sequence; the text keys are not at any audio
    position, and rotating them by one would assert an alignment between word i
    and frame i that nothing in the data supports. PixArt does the same: the
    cross-attention sees the text as a set the model orders through the token
    states themselves, which already carry the sentence's own positions.

    NO AdaLN MODULATION on this sub-layer, again as in PixArt: the branch reads
    the text directly, and the block's six adaLN parameters stay what they are
    in the official DiT, driving the self-attention and the FFN.

    `ctx_mask` is a BOOLEAN key mask, True where a token is real. It is not
    optional in practice: captions have different lengths, they are padded into
    one tensor, and padding that reaches K and V is padding the model reads as
    words. The caller must never hand a row whose mask is all-False -- softmax
    over an empty set of keys is NaN, not zero -- which is why
    ConditionedAudioDiT substitutes a learned null token instead of masking a
    row out entirely.

    The output projection is ZERO-INITIALISED (see initialize_weights), the
    adaLN-Zero / ControlNet principle and PixArt's own choice: at step 0 this
    sub-layer contributes exactly 0, so a model built with cross-attention
    computes the same function as one without it and can be warm-started from a
    checkpoint that never had it.
    """

    def __init__(self, hidden_size: int, n_heads: int, ctx_dim: int):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=True)
        # ONE Linear for K and V: they read the same context, so a single
        # matmul over the (short) text sequence is cheaper than two and the
        # split is free.
        self.kv = nn.Linear(ctx_dim, 2 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        L = ctx.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(ctx.to(x.dtype)).view(B, L, 2, self.n_heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)      # (B, heads, L, head_dim)
        v = kv[:, :, 1].transpose(1, 2)
        attn_mask = None
        if ctx_mask is not None:
            # (B, 1, 1, L): broadcast over heads and over every audio token.
            # Boolean: True = this key takes part.
            attn_mask = ctx_mask.view(B, 1, 1, L)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


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
    input); the pooled global conditions reach the block only through `c`
    (AdaLN), exactly like the class label in DiT.

    OPTIONAL CROSS-ATTENTION on a text sequence (`cross_attn_ctx_dim` > 0),
    inserted BETWEEN the self-attention and the FFN -- the order of PixArt-alpha
    and of the LDM/Stable-Diffusion BasicTransformerBlock:

        x = x + gate_msa * attn(modulate(norm1(x)))
        x = x + cross_attn(norm_cross(x), ctx)        <- only when built
        x = x + gate_mlp * mlp (modulate(norm2(x)))

    When it is NOT built, this block is network.py's DiT block down to the last
    operation and the last parameter -- which is what keeps a run without text
    comparable to every run that came before it, and keeps its checkpoints
    loadable by this code.
    """

    def __init__(self, hidden_size, num_heads, max_seq_len=4096,
                 mlp_ratio=4.0, drop=0.0, cross_attn_ctx_dim=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = SelfAttention(hidden_size, num_heads, max_seq_len=max_seq_len)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = FFN(hidden_size, mlp_ratio=mlp_ratio, dropout=drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Built only when asked, so a block without text carries no extra
        # tensor in the state_dict at all.
        self.norm_cross = None
        self.cross_attn = None
        if int(cross_attn_ctx_dim or 0) > 0:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False,
                                           eps=1e-6)
            self.cross_attn = CrossAttention(hidden_size, num_heads,
                                             int(cross_attn_ctx_dim))

    def forward(self, x, c, ctx=None, ctx_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        if self.cross_attn is not None:
            # NOT guarded on `ctx is not None`. A block that has the sub-layer
            # and silently skips it when the caller forgot the context would be
            # a THIRD state -- neither "conditioned" nor the learned null the
            # CFG dropout trained -- and it would never raise. The model always
            # supplies a context (its null token when there is no text), so the
            # only way to arrive here without one is a bug, and it should say so.
            if ctx is None:
                raise ValueError(
                    "this DiTBlock has a cross-attention but no text context "
                    "reached it; ConditionedAudioDiT.forward must pass one "
                    "(its learned null token when there is no text)")
            x = x + self.cross_attn(self.norm_cross(x), ctx, ctx_mask)
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
        (JASCO), then projected to hidden by a single input projection, and
        OPTIONALLY re-added before selected blocks (frame_reinject_every);
      - global conditions       -> added to the AdaLN conditioning vector c
        (official DiT class-label mechanism).

    The DiTBlock itself is still untouched in both cases: the re-injection is a
    residual add performed by this module BETWEEN blocks, so a block's forward
    signature stays (x, c) exactly as in network.py.

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
        frame_reinject_every: stride of the per-block re-injection of the frame
                             conditions. 0 (default) = OFF, plain JASCO
                             input-only concatenation. 1 = re-inject before
                             EVERY block, 2 = every other block, and so on.
                             Block 0 is never re-injected: input_proj has just
                             delivered the conditions to it, so an add there
                             would be redundant. Ignored (with a printed note)
                             when there are no frame conditions.
                             The re-injected term is scaled per condition by a
                             gate read off c (see frame_reinject_gate), so each
                             condition can be weighted differently at different
                             denoising steps.

    CFG: passing null (zero) conditions yields the unconditional output. This
    holds for the re-injection too -- it is a function of the same frame_proj,
    so a null (zero) condition drives the re-injection path as well and the
    unconditional branch stays a single, coherent input.
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
        frame_reinject_every: int = 0,
        text_cross_every:     int = 0,
        text_ctx_dim:         int = 0,
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
        # ----- Cross-attention on the text SEQUENCE (optional) -----
        # WHICH BLOCKS: strided from 0, so every block when the stride is 1 --
        # unlike the frame re-injection, which skips block 0 because input_proj
        # has just handed it the conditions. Nothing hands block 0 the text, so
        # it gets the sub-layer like any other.
        #
        # It is OFF unless the run actually has a text condition: a cross-
        # attention over a context that is always the null token is a per-block
        # learned bias with a few million parameters, which is worse than
        # nothing and would sit in the state_dict forever.
        self.text_cross_every = int(text_cross_every or 0)
        if self.text_cross_every < 0:
            raise ValueError(
                f"text_cross_every must be >= 0 (0 = off), got {text_cross_every}.")
        self.text_ctx_dim = int(text_ctx_dim or 0)
        self.text_cross_layers = []
        if self.text_cross_every > 0 and "text" in self.global_cond_configs:
            if self.text_ctx_dim <= 0:
                raise ValueError(
                    "text_cross_every > 0 needs text_ctx_dim: the width of ONE "
                    "token state of the text encoder (768 for CLAP "
                    "clap-htsat-unfused). It is not the pooled dim (512) -- see "
                    "CLAPTextCondition.encode_tokens.")
            self.text_cross_layers = [
                i for i in range(n_layers) if i % self.text_cross_every == 0
            ]

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, max_seq_len=max_seq_len,
                     mlp_ratio=mlp_ratio, drop=drop,
                     cross_attn_ctx_dim=(self.text_ctx_dim
                                         if i in self.text_cross_layers else 0))
            for i in range(n_layers)
        ])

        # THE NULL CONTEXT, learned, ONE token.
        # Classifier-free guidance needs an unconditional branch, and for this
        # path "unconditional" cannot be "no keys": softmax over an empty set is
        # NaN, so a row can never be masked out entirely. It cannot be a zero
        # SEQUENCE either -- the zero vector is a perfectly ordinary point of a
        # hidden-state space that is not normalised, and the model would have to
        # learn "no text" as a coincidence. PixArt learns the null the same way.
        # Trained by the CFG dropout exactly like the zero vector of the other
        # conditions, so guidance extrapolates from something the model saw.
        self.text_null = None
        if self.text_cross_layers:
            self.text_null = nn.Parameter(torch.randn(1, 1, self.text_ctx_dim) * 0.02)

        # ----- Per-block re-injection of the frame conditions (optional) -----
        # WHICH BLOCKS: strided over 1..n_layers-1. Index 0 is excluded on
        # purpose -- input_proj feeds block 0 directly, so re-adding there would
        # only duplicate what the concatenation already delivered.
        #
        # WHAT: one Linear(total_out_dim -> hidden) per selected block, applied
        # to the SAME frame_proj the input concatenation uses (the encoder runs
        # once per forward; only these projections are per-block). Each block
        # therefore gets its own learned view of the conditions instead of
        # sharing one.
        #
        # bias=False ON PURPOSE: a bias would add a constant vector to the
        # residual stream at that depth REGARDLESS of the condition -- a plain
        # learned offset that says nothing about f0/chroma/rhythm/energy, and
        # that the block's own AdaLN shift already provides. Without it the
        # re-injected term is strictly a function of the conditions.
        #
        # ZERO-INIT (see initialize_weights): at step 0 every re-injection
        # contributes exactly 0, so a run with frame_reinject_every>0 starts
        # from the same function as the input-only model and the extra path has
        # to earn its weight. Same principle as adaLN-Zero and ControlNet's
        # zero-convs; without it, N_layers random projections would inject noise
        # into the trunk from the first step.
        self.frame_reinject_every = int(frame_reinject_every or 0)
        if self.frame_reinject_every < 0:
            raise ValueError(
                f"frame_reinject_every must be >= 0 (0 = off), got "
                f"{frame_reinject_every}."
            )
        self.reinject_layers = []
        if self.frame_reinject_every > 0 and self.has_frame:
            self.reinject_layers = [
                i for i in range(1, n_layers)
                if i % self.frame_reinject_every == 0
            ]
        if self.reinject_layers:
            self.frame_reinject = nn.ModuleDict({
                str(i): nn.Linear(frame_extra, hidden_size, bias=False)
                for i in self.reinject_layers
            })

            # PER-CONDITION GATE on the re-injected term, a function of c.
            #
            # WHY: frame_proj and the projection above are both functions of
            # the CONDITIONS ALONE, so the term re-added at block i would be
            # the same tensor at every denoising step -- identical where x is
            # still almost pure noise and where it is almost data. Everything
            # else in this network is free to weigh itself against t: each
            # block's shift/scale/gate come out of adaLN_modulation(c), and the
            # global conditions ride inside that same c. The re-injection was
            # the only contribution in the model without that freedom.
            #
            # PER CONDITION, not one gate for the whole term: it emits one
            # scalar per frame condition, and each scales its OWN slice of
            # frame_proj -- the positional slots FrameConditionEncoder
            # concatenated, in the same canonical order -- so f0 and chroma can
            # follow different schedules in t instead of rising and falling
            # together. With a single frame condition the two are identical.
            #
            # It reads c, NOT t alone: with global conditions active
            # c = t_emb + g, so how much frame condition is injected also
            # depends on text/image. That is deliberate (it is the DiT's own
            # single-conditioning-vector design) and it is inert on a model
            # with no global conditions, where c IS the timestep embedding.
            #
            # SiLU + Linear is the same shape as adaLN_modulation, so the gate
            # is the block's own idiom applied to the one path that lacked it.
            self.frame_reinject_gate = nn.ModuleDict({
                str(i): nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, len(self.frame_encoder.names)),
                )
                for i in self.reinject_layers
            })

            # Slot widths in canonical order, used to expand the per-condition
            # scalars back to the frame_proj width. persistent=False: it is
            # derived from frame_cond_out_dims, which the checkpoint already
            # carries, so it must not become a state_dict entry of its own.
            self.register_buffer(
                "frame_slot_repeats",
                torch.tensor(
                    [self.frame_cond_out_dims[n]
                     for n in self.frame_encoder.names],
                    dtype=torch.long,
                ),
                persistent=False,
            )

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

        # Re-injection report. A setting that ends up doing NOTHING is printed
        # as such rather than passing silently: asking for it and getting plain
        # JASCO without being told is exactly the kind of run that gets
        # misattributed later.
        if self.reinject_layers:
            n_re = len(self.reinject_layers)
            n_fc = len(self.frame_encoder.names)
            re_params = n_re * frame_extra * hidden_size
            gate_params = n_re * (hidden_size + 1) * n_fc
            print(f"  Frame re-injection: every {self.frame_reinject_every} "
                  f"block(s) -> {n_re} of {n_layers} blocks "
                  f"{self.reinject_layers} | zero-init, bias-free | "
                  f"gate on c, {n_fc} per block, identity-init | "
                  f"+{(re_params + gate_params)/1e6:.2f}M params")
        elif self.frame_reinject_every > 0 and not self.has_frame:
            print(f"  Frame re-injection: REQUESTED (every "
                  f"{self.frame_reinject_every}) but INACTIVE -- this model has "
                  f"no frame conditions to re-inject.")
        elif self.frame_reinject_every > n_layers - 1 and self.has_frame:
            print(f"  Frame re-injection: REQUESTED (every "
                  f"{self.frame_reinject_every}) but INACTIVE -- the stride "
                  f"exceeds the {n_layers - 1} eligible blocks (1..{n_layers-1}).")
        else:
            print(f"  Frame re-injection: OFF (input concat only, plain JASCO)")

        # Cross-attention report, same rule: a setting that does nothing says so.
        if self.text_cross_layers:
            n_x = len(self.text_cross_layers)
            # per block: LayerNorm (affine-free, 0) + Q + K,V + W_o
            x_params = n_x * ((hidden_size * hidden_size + hidden_size)
                              + (self.text_ctx_dim * 2 * hidden_size + 2 * hidden_size)
                              + (hidden_size * hidden_size + hidden_size))
            print(f"  Text cross-attention: every {self.text_cross_every} "
                  f"block(s) -> {n_x} of {n_layers} blocks | "
                  f"ctx_dim={self.text_ctx_dim} | W_o zero-init | "
                  f"learned null token | +{x_params/1e6:.2f}M params")
        elif self.text_cross_every > 0 and "text" not in self.global_cond_configs:
            print(f"  Text cross-attention: REQUESTED (every "
                  f"{self.text_cross_every}) but INACTIVE -- this model has no "
                  f"'text' global condition to attend to.")
        else:
            print(f"  Text cross-attention: OFF (the text reaches the blocks "
                  f"only as a pooled vector through AdaLN)")

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

        # Zero-out every per-block frame re-injection. MUST come after
        # self.apply(_basic_init) above, which has just given these Linears a
        # xavier_uniform_ init like every other one; this overrides it. The
        # effect is the adaLN-Zero / ControlNet zero-conv one: at step 0 the
        # re-injection adds exactly 0 to the residual stream, so the model is
        # the input-only (JASCO) model, and training turns the extra path on
        # gradually instead of starting from N random projections shouting into
        # the trunk.
        if self.reinject_layers:
            for lin in self.frame_reinject.values():
                nn.init.constant_(lin.weight, 0)

            # The gate starts at the IDENTITY (weight 0, bias 1), NOT at zero.
            # Zeroing it as well would leave the whole path dead on arrival:
            # with the gate at 0 the projection above receives no gradient, and
            # with the projection at 0 the gate receives none either, so
            # neither could ever leave the origin. A gate of exactly 1 keeps
            # the step-0 guarantee intact -- the zero-init projection still
            # contributes exactly 0, so the model is still the input-only
            # (JASCO) one -- while keeping the projection's gradient alive. The
            # gate itself starts moving from step 1, once the projection is no
            # longer zero.
            for gate in self.frame_reinject_gate.values():
                nn.init.constant_(gate[-1].weight, 0)
                nn.init.constant_(gate[-1].bias, 1.0)

        # Zero-out the OUTPUT projection of every cross-attention. Same reason
        # and same place in the order as above: _basic_init has just given it a
        # xavier init like every other Linear, and this overrides it. The effect
        # is PixArt's: at step 0 the whole sub-layer adds exactly 0, so the
        # model computes the same function as one built without cross-attention
        # -- which is what makes warm-starting from such a checkpoint exact
        # rather than approximately right.
        #
        # ONLY the output projection, and it is worth being precise about what
        # that buys, because the obvious guess is wrong. At step 0, with W_o=0,
        # Q, K and V receive NO gradient at all: the gradient reaching the
        # attention output is grad_out @ W_o.T, which is zero. What IS non-zero
        # is the gradient of W_o ITSELF -- it is grad_out against the attention
        # output, and that output is non-zero because Q/K/V are xavier-
        # initialised. So W_o leaves the origin on the first step, and from the
        # second one the whole sub-layer trains normally. (Measured, 14 Sept
        # 2026: after one backward, grad on W_o non-zero, grad on Q/K/V and on
        # the null token exactly zero.)
        #
        # That is why the zero goes HERE and nowhere else. Zero Q or K,V as
        # well and the attention output is zero too, so W_o's own gradient
        # vanishes with everything else and the path never starts.
        for block in self.blocks:
            if block.cross_attn is not None:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

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

    def _gather_global_conditions(
        self,
        global_conditions: Optional[Dict[str, torch.Tensor]],
        B: int, device, dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        The global counterpart of _gather_frame_conditions: return a dict with
        EVERY expected global condition present, missing ones zero-filled.

        Same reason as the frame version, and the same null: the CFG dropout at
        training time replaces a dropped global with a ZERO VECTOR, so zeros --
        pushed through the projection, its bias and the final LayerNorm -- are
        what the model learned "no condition" to be. Leaving a name out of the
        dict instead removed its projection from the sum altogether, a third
        state that was never trained. That is what a partial dict looked like:
        `sampling_cond.py --prompt ... --allow_null_global_conditions` (a text
        vector, no image) built exactly one, as did any call that passed None or
        {} while the model had global conditions.
        """
        out = {}
        gc = global_conditions or {}
        for name, cfg in self.global_cond_configs.items():
            c = gc.get(name, None)
            if c is None:
                c = torch.zeros(B, int(cfg["dim"]), device=device, dtype=dtype)
            else:
                c = c.to(device=device, dtype=dtype)
            out[name] = c
        return out

    def _gather_text_context(self, ctx, mask, B, device):
        """
        -> ((B, L, ctx_dim) float32, (B, L) bool) for the cross-attention, with
        the LEARNED NULL TOKEN standing in wherever there is no text.

        The third counterpart of _gather_frame_conditions / _gather_global_
        conditions, and it exists for the same reason: there must be exactly TWO
        states, "a text" and "the null the dropout trained", never a third that
        nobody trained and nothing reports.

        `ctx=None` means the WHOLE batch is unconditional -- that is the call
        the CFG null branch makes, and the one a caller that has no text at all
        makes. Per-SAMPLE dropout cannot say it that way, so it says it with the
        mask instead: a row whose mask is entirely False is a dropped sample,
        and it comes back here as the null token. That is also the only reason
        this is not a plain passthrough -- a row with no valid key at all would
        make scaled_dot_product_attention take a softmax over an empty set and
        return NaN, which would poison the loss of the whole batch.
        """
        null = self.text_null.to(device=device, dtype=torch.float32)
        if ctx is None:
            return (null.expand(B, 1, -1),
                    torch.ones(B, 1, dtype=torch.bool, device=device))
        ctx = ctx.to(device=device, dtype=torch.float32)
        if ctx.dim() != 3 or ctx.shape[0] != B:
            raise ValueError(
                f"text_context must be (B, L, {self.text_ctx_dim}) with B={B}, "
                f"got {tuple(ctx.shape)}")
        if ctx.shape[2] != self.text_ctx_dim:
            raise ValueError(
                f"text_context has width {ctx.shape[2]}, this model was built "
                f"for {self.text_ctx_dim}. The context is the TOKEN-level state "
                f"of the text encoder (768 for CLAP), not the pooled embedding "
                f"(512) that goes into the AdaLN slot.")
        if mask is None:
            mask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device).bool()
            if mask.shape != ctx.shape[:2]:
                raise ValueError(
                    f"text_context_mask {tuple(mask.shape)} does not match the "
                    f"context {tuple(ctx.shape[:2])}")
        empty = ~mask.any(dim=1)
        if bool(empty.any()):
            ctx, mask = ctx.clone(), mask.clone()
            ctx[empty] = 0.0
            ctx[empty, 0] = null[0, 0]
            mask[empty] = False
            mask[empty, 0] = True
        return ctx, mask

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        frame_conditions:  Optional[Dict[str, torch.Tensor]] = None,
        global_conditions: Optional[Dict[str, torch.Tensor]] = None,
        text_context:      Optional[torch.Tensor] = None,
        text_context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, n_frames, token_dim)   one token per DAC frame
        t: (B,)                        timestep in [0, 1]
        frame_conditions:  {"f0": (B,T,2), "chroma": (B,T,12), "rhythm": (B,T,2)} or None
        global_conditions: {"text":  (B,d),  "image":  (B,d), ...}    or None
        text_context:      (B, L, text_ctx_dim) token states of the text, or
                           None for "no text" (the learned null token). Ignored
                           by a model built without cross-attention.
        text_context_mask: (B, L) bool, True where a token is real. A row that
                           is all-False is a CFG-dropped sample and becomes the
                           null token.

        Returns:
            velocity field of shape (B, n_frames, token_dim)
        """
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()
        B, T, _ = x.shape

        # ----- FRAME conditions: concat on the feature dim (JASCO) -----
        frame_proj = None
        if self.has_frame:
            fc = self._gather_frame_conditions(frame_conditions, B, T, x.device, x.dtype)
            frame_proj = self.frame_encoder(fc)        # (B, T, sum(out_dim))
            x = torch.cat([x, frame_proj], dim=-1)     # (B, T, token_dim + sum(out_dim))

        # Token projection (position is injected by RoPE inside attention; no
        # additive positional embedding, exactly as in network.py).
        x = self.input_proj(x)

        # ----- GLOBAL conditions: AdaLN vector c = t_emb + g_global -----
        # Gathered, NOT tested for truthiness: a model WITH global conditions
        # handed an empty or partial dict must still see the null it was trained
        # against (zeros through the projection), exactly as the frame branch
        # above zero-fills its missing slots. Skipping the encoder for an empty
        # dict, as this used to, made "no global given" a different input from
        # "global dropped by CFG" -- two nulls where the training had one.
        # NB: float32, not x.dtype -- by this point x has been through
        # input_proj, which under autocast returns fp16. The conditions arrive
        # in fp32 and autocast casts them itself inside the projection; forcing
        # fp16 here would only lose precision before it.
        c = self.t_embedder(t)                          # (B, hidden)
        if self.has_global:
            gc = self._gather_global_conditions(global_conditions, B,
                                                x.device, torch.float32)
            g = self.global_encoder(gc)                 # (B, hidden) or None
            if g is not None:
                c = c + g

        # DiT blocks (block is identical to the unconditional network.py).
        # When enabled, the frame conditions are re-added to the residual stream
        # BEFORE the selected blocks, so that block reads them from its own
        # input instead of relying on what survived from input_proj. The add is
        # done here, outside the block: DiTBlock.forward is still (x, c).
        #
        # frame_proj is the tensor computed once above -- the same one the input
        # concatenation used, nulls included -- so the conditioned and
        # unconditional CFG branches stay exactly the two inputs the training
        # saw. dtype: under autocast frame_proj is fp32 and x is fp16; the
        # Linear returns fp16 and the add matches x. Outside autocast both are
        # fp32.
        # The text SEQUENCE, resolved once for the whole depth: every block that
        # has a cross-attention reads the same context, exactly as every block
        # reads the same c.
        ctx = ctx_mask = None
        if self.text_cross_layers:
            ctx, ctx_mask = self._gather_text_context(
                text_context, text_context_mask, B, x.device)

        reinject = self.frame_reinject if (self.reinject_layers and
                                           frame_proj is not None) else None
        for i, block in enumerate(self.blocks):
            if reinject is not None and str(i) in reinject:
                # One scalar per frame condition, expanded to the width of its
                # own slice of frame_proj and broadcast over time: the gate
                # schedules each condition along t, it does not weigh one frame
                # against another. Under autocast the gate returns fp16 and
                # frame_proj is fp32; the product promotes to fp32 and the
                # re-injection Linear casts it back, exactly as before.
                gate = self.frame_reinject_gate[str(i)](c)
                gate = torch.repeat_interleave(
                    gate, self.frame_slot_repeats, dim=-1).unsqueeze(1)
                x = x + reinject[str(i)](frame_proj * gate)
            x = block(x, c, ctx, ctx_mask)

        x = self.final_layer(x, c)
        return x


# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def ckpt_frame_reinject_every(ckpt: dict) -> int:
    """
    Recover `frame_reinject_every` from a checkpoint, for the scripts that
    rebuild the model from the checkpoint's own fields (sampling_cond.py,
    test_cond.py). It is an ARCHITECTURE parameter -- it adds one tensor per
    selected block to the state_dict -- so getting it wrong is a load failure,
    not a subtly different sample.

    Three sources, in order:
      1. the top-level "frame_reinject_every" field written by
         training_cond.build_ckpt_data;
      2. the stored full config, config["model"]["frame_reinject_every"], as a
         safety net for a checkpoint whose top-level field is missing;
      3. 0 -- the correct answer for every checkpoint written before this
         option existed, which is exactly what those runs were trained as.
    """
    v = ckpt.get("frame_reinject_every", None)
    if v is None:
        cfg = ckpt.get("config", None)
        if isinstance(cfg, dict):
            v = (cfg.get("model", None) or {}).get("frame_reinject_every", None)
    return int(v or 0)


def ckpt_text_cross_every(ckpt: dict) -> int:
    """
    Recover `text_cross_every` from a checkpoint, the twin of
    ckpt_frame_reinject_every and for the same reason: it adds four tensors per
    selected block plus the null token, so a script that rebuilds the model from
    a checkpoint and guesses it wrong gets a load failure, not a slightly
    different sample.

    Three sources, in order: the top-level field, the stored config, then 0 --
    which is the correct answer for every checkpoint written before the
    cross-attention existed, and is exactly what those runs were trained as.
    """
    v = ckpt.get("text_cross_every", None)
    if v is None:
        cfg = ckpt.get("config", None)
        if isinstance(cfg, dict):
            v = (cfg.get("model", None) or {}).get("text_cross_every", None)
    return int(v or 0)


def ckpt_text_ctx_dim(ckpt: dict) -> int:
    """
    The width of ONE text token as the checkpoint was trained, read off the
    state_dict itself rather than off a config field.

    Off the WEIGHTS on purpose: this number is not a preference, it is a shape,
    and the only thing that can state it without being able to lie is the tensor
    whose shape it is. `blocks.<i>.cross_attn.kv.weight` is (2*hidden, ctx_dim).
    Returns 0 when the checkpoint has no cross-attention.
    """
    sd = ckpt.get("model_state_dict", None)
    if not isinstance(sd, dict) or not sd:
        sd = ckpt.get("ema_state_dict", None)
    if not isinstance(sd, dict):
        return 0
    for k, v in sd.items():
        if k.endswith("cross_attn.kv.weight") and hasattr(v, "shape"):
            return int(v.shape[1])
    return 0


def check_ckpt_reinject_gate(ckpt: dict, where: str = "this checkpoint") -> None:
    """
    Refuse, with an explanation, a checkpoint whose re-injection PREDATES the
    per-condition gate.

    `frame_reinject_every` alone no longer pins the state_dict down. A run
    trained with re-injection before the gate existed has the per-block
    projections and none of the `frame_reinject_gate.*` tensors, so every
    architecture check passes -- same kind, same conditions, same stride -- and
    the failure only surfaces two frames deeper, as a raw wall of
    "Missing key(s) in state_dict: frame_reinject_gate.1.1.weight, ...", which
    says nothing about what to do.

    Called by every script that rebuilds a model from a checkpoint. Silent (the
    normal case) for `frame_reinject_every == 0`, where neither the projections
    nor the gates exist, and for any checkpoint written since the gate.
    """
    if ckpt_frame_reinject_every(ckpt) <= 0:
        return
    sd = ckpt.get("model_state_dict", None)
    if not isinstance(sd, dict):
        sd = ckpt.get("ema_state_dict", None)
    if not isinstance(sd, dict) or not sd:
        return                      # nothing to inspect; let the load speak
    if any(k.startswith("frame_reinject_gate.") for k in sd):
        return                      # written after the gate: nothing to say
    raise RuntimeError(
        f"{where} was trained with model.frame_reinject_every="
        f"{ckpt_frame_reinject_every(ckpt)} BEFORE the per-condition gate on "
        f"the re-injection existed, so it has the per-block projections but "
        f"none of the `frame_reinject_gate.*` weights this code builds. It "
        f"cannot be loaded or resumed as-is: the two are different "
        f"architectures that happen to agree on every other field.\n"
        f"  - to keep using that checkpoint, check out the code from before "
        f"the gate (network_cond.py.pre-reinject-gate.bak);\n"
        f"  - to use this code, start a NEW run -- a gate initialised to 1 "
        f"makes step 0 identical to the ungated model, so nothing is lost "
        f"except the steps already spent."
    )


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

    # --- Per-block frame re-injection ---
    print("\n=== Frame re-injection (every block) ===")
    cond_kw = dict(kind='S',
                   frame_cond_dims={"f0": 2, "chroma": 12},
                   frame_cond_out_dims={"f0": 16, "chroma": 64},
                   global_cond_configs={})
    model4_off = ConditionedAudioDiT(**cond_kw, frame_reinject_every=0)
    model4     = ConditionedAudioDiT(**cond_kw, frame_reinject_every=1)

    # 6 blocks -> blocks 1..5 are re-injected, block 0 is not (input_proj).
    assert model4_off.reinject_layers == []
    assert model4.reinject_layers == [1, 2, 3, 4, 5], model4.reinject_layers

    f0c = {"f0": torch.randn(B, N, 2), "chroma": torch.randn(B, N, 12)}
    out4 = model4(x, t, frame_conditions=f0c)
    print(f"  re-injected -> {out4.shape}")
    assert out4.shape == x.shape

    # A FRESHLY INITIALISED DiT IS THE CONSTANT-ZERO FUNCTION: adaLN-Zero makes
    # every block the identity (gate=0) and the final layer is zeroed, so
    # comparing two models at init compares 0 with 0 and would pass whatever the
    # re-injection did. So first make the baseline a non-degenerate network
    # (small random gates + a real final projection), THEN copy its weights into
    # the re-injecting model so the two differ ONLY by the extra path.
    with torch.no_grad():
        for blk in model4_off.blocks:
            nn.init.normal_(blk.adaLN_modulation[-1].weight, std=0.02)
            nn.init.normal_(blk.adaLN_modulation[-1].bias,   std=0.02)
        nn.init.normal_(model4_off.final_layer.linear.weight, std=0.02)
        nn.init.normal_(model4_off.final_layer.adaLN_modulation[-1].weight, std=0.02)

    missing, unexpected = model4.load_state_dict(model4_off.state_dict(),
                                                 strict=False)
    assert unexpected == [], unexpected
    assert missing and all(k.startswith("frame_reinject.") or
                           k.startswith("frame_reinject_gate.")
                           for k in missing), missing
    print(f"  shared weights copied; only {len(missing)} re-injection tensors "
          f"are new (projections + per-condition gates)")

    with torch.no_grad():
        base = model4_off(x, t, frame_conditions=f0c)
        assert base.abs().max().item() > 0, "baseline is still degenerate"
        d = (model4(x, t, frame_conditions=f0c) - base).abs().max().item()
    print(f"  zero-init check: max |reinject - baseline| = {d:.3e}")
    assert d == 0.0, "re-injection is not zero at init"

    # ...and once the projections are non-zero, it must actually change the
    # output (i.e. the path is really wired into the trunk).
    with torch.no_grad():
        for lin in model4.frame_reinject.values():
            nn.init.normal_(lin.weight, std=0.02)
        d2 = (model4(x, t, frame_conditions=f0c) - base).abs().max().item()
    print(f"  wired check:     max |reinject - baseline| = {d2:.3e}")
    assert d2 > 0.0, "re-injection has no effect on the output"

    # The re-injected term must depend on the CONDITIONS, not just on depth:
    # two different f0 curves must give two different outputs through that path.
    with torch.no_grad():
        f0c_b = {"f0": torch.randn(B, N, 2), "chroma": torch.randn(B, N, 12)}
        d3 = (model4(x, t, frame_conditions=f0c)
              - model4(x, t, frame_conditions=f0c_b)).abs().max().item()
    print(f"  sensitivity:     max |cond_A - cond_B| = {d3:.3e}")
    assert d3 > 0.0

    # Every re-injection projection must receive gradient from the loss.
    model4.zero_grad(set_to_none=True)
    model4(x, t, frame_conditions=f0c).pow(2).mean().backward()
    for k, lin in model4.frame_reinject.items():
        g = lin.weight.grad
        assert g is not None and g.abs().sum().item() > 0, f"no grad on block {k}"
    print(f"  grad check:      all {len(model4.frame_reinject)} projections "
          f"receive gradient")

    # --- Stride > 1, and the two inert configurations ---
    print("\n=== Re-injection: stride 2 / inert cases ===")
    model6 = ConditionedAudioDiT(
        kind='S', frame_cond_dims={"f0": 2}, frame_cond_out_dims={"f0": 16},
        global_cond_configs={}, frame_reinject_every=2,
    )
    assert model6.reinject_layers == [2, 4], model6.reinject_layers
    assert model6(x, t, frame_conditions={"f0": torch.randn(B, N, 2)}).shape == x.shape

    # Requested but no frame conditions -> inert, and says so.
    model7 = ConditionedAudioDiT(
        kind='S', frame_cond_dims={}, frame_cond_out_dims={},
        global_cond_configs={"text": {"dim": 512}}, frame_reinject_every=1,
    )
    assert model7.reinject_layers == []
    assert not hasattr(model7, "frame_reinject")

    # Stride larger than the depth -> inert, and says so.
    model8 = ConditionedAudioDiT(
        kind='S', frame_cond_dims={"f0": 2}, frame_cond_out_dims={"f0": 16},
        global_cond_configs={}, frame_reinject_every=99,
    )
    assert model8.reinject_layers == []

    try:
        ConditionedAudioDiT(kind='S', frame_cond_dims={"f0": 2},
                            frame_cond_out_dims={"f0": 16},
                            global_cond_configs={}, frame_reinject_every=-1)
        raise AssertionError("negative stride should have raised")
    except ValueError as e:
        print(f"  negative stride rejected: {e}")

    print("\nTest passed!")
