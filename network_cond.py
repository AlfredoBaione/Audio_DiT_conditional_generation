# The conditioned DiT.

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from conditions import FrameConditionEncoder, GlobalConditionEncoder



ATTENTION_KINDS = ("standard", "differential")
TEXT_CONDITIONING_KINDS = ("ada", "cross_att", "all")
ADALN_KINDS = ("shared", "per_block")
T_FEATURE_CYCLES = {
    "high_freq": (0.5, 10000.0),
    "low_freq": (1e-4 / (2 * math.pi), 1.0 / (2 * math.pi)),
}
QK_NORM_EPS = 1e-6


def left_pad(x: torch.Tensor, length: int) -> torch.Tensor:
    return F.pad(x, (0, 0, length - x.shape[1], 0))


def fp32_norm(norm: Optional[nn.Module], x: torch.Tensor) -> torch.Tensor:
    if norm is None:
        return x
    with torch.autocast(device_type=x.device.type, enabled=False):
        return norm(x.float()).to(x.dtype)


def make_qk_norms(head_dim: int, enabled: bool):
    if not enabled:
        return None, None
    return (nn.RMSNorm(head_dim, eps=QK_NORM_EPS, elementwise_affine=True),
            nn.RMSNorm(head_dim, eps=QK_NORM_EPS, elementwise_affine=True))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    r = cos.shape[-1]
    rotated, rest = x[..., :r], x[..., r:]
    return torch.cat((rotated * cos + rotate_half(rotated) * sin, rest), dim=-1)


def rope_dim(head_dim: int) -> int:
    return max(head_dim // 2, 32)


def rope_inv_freq(rot_dim: int, theta: float = 10000.0) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.int64).float() / rot_dim))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, t_features="high_freq", n_features=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        ramp = torch.linspace(0, 1, n_features // 2)
        log_min, log_max = map(math.log, T_FEATURE_CYCLES[t_features])
        self.register_buffer("freqs", torch.exp(ramp * (log_max - log_min) + log_min),
                             persistent=False)

    def forward(self, t):
        args = t[:, None] * self.freqs * 2 * math.pi
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, differential: bool = False,
                 theta: float = 10000.0, qk_norm: bool = False):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_size // n_heads
        self.n_proj   = 5 if differential else 3

        self.qkv  = nn.Linear(hidden_size, self.n_proj * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm, self.k_norm = make_qk_norms(self.head_dim, qk_norm)

        inv_freq = rope_inv_freq(rope_dim(self.head_dim), theta=theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, S: int, device, dtype):
        t = torch.arange(S, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _attend(self, q, k, v, cos, sin):
        q = apply_rotary_pos_emb(fp32_norm(self.q_norm, q), cos, sin)
        k = apply_rotary_pos_emb(fp32_norm(self.k_norm, k), cos, sin)
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        parts = (self.qkv(x).reshape(B, S, self.n_proj, self.n_heads, self.head_dim)
                 .permute(2, 0, 3, 1, 4).unbind(0))
        v = parts[2]
        cos, sin = self._cos_sin(S, x.device, x.dtype)
        out = self._attend(parts[0], parts[1], v, cos, sin)
        if self.n_proj == 5:
            out = out - self._attend(parts[3], parts[4], v, cos, sin)
        return self.proj(out.transpose(1, 2).reshape(B, S, -1))


class FFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.proj = nn.Linear(hidden_size, 2 * inner)
        self.out = nn.Linear(inner, hidden_size)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h, gate = self.proj(x).chunk(2, dim=-1)
        h = self.drop1(h * F.silu(gate))
        return self.drop2(self.out(h))


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, differential: bool = False,
                 qk_norm: bool = False):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.differential = differential
        self.q = nn.Linear(hidden_size, (2 if differential else 1) * hidden_size, bias=False)
        self.kv = nn.Linear(hidden_size, (3 if differential else 2) * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm, self.k_norm = make_qk_norms(self.head_dim, qk_norm)

    def _split(self, p: torch.Tensor, n: int):
        B, S, _ = p.shape
        return p.reshape(B, S, n, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

    def _attend(self, q, k, v):
        return F.scaled_dot_product_attention(
            fp32_norm(self.q_norm, q), fp32_norm(self.k_norm, k), v)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if self.differential:
            q1, q2 = self._split(self.q(x), 2)
            k1, k2, v = self._split(self.kv(ctx), 3)
            out = self._attend(q1, k1, v) - self._attend(q2, k2, v)
        else:
            q, = self._split(self.q(x), 1)
            k, v = self._split(self.kv(ctx), 2)
            out = self._attend(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, token_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, token_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        scale, shift = self.adaLN(c).unsqueeze(1).chunk(2, dim=-1)
        return self.linear(fp32_norm(self.norm, x) * (1 + scale) + shift)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 mlp_ratio=4.0, drop=0.0, cross_attention=False,
                 attention="standard", qk_norm=False,
                 inpaint_ctx_dim=0, adaln="shared"):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-5)
        self.attn  = SelfAttention(hidden_size, num_heads,
                                   differential=attention == "differential",
                                   qk_norm=qk_norm)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-5)
        self.mlp   = FFN(hidden_size, mlp_ratio=mlp_ratio, dropout=drop)
        self.adaLN = None
        self.adaLN_bias = None
        if adaln == "per_block":
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        else:
            self.adaLN_bias = nn.Parameter(torch.randn(6 * hidden_size) / hidden_size ** 0.5)
        self.norm_cross = None
        self.cross_attn = None
        if cross_attention:
            self.norm_cross = nn.RMSNorm(hidden_size, eps=1e-5)
            self.cross_attn = CrossAttention(hidden_size, num_heads,
                                             differential=attention == "differential",
                                             qk_norm=qk_norm)
        self.inpaint_embed = None
        if int(inpaint_ctx_dim or 0) > 0:
            self.inpaint_embed = nn.Sequential(
                nn.Linear(int(inpaint_ctx_dim), hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

    def forward(self, x, mod, ctx=None, inpaint=None):
        if self.adaLN is None:
            scale_a, shift_a, gate_a, scale_f, shift_f, gate_f = (
                self.adaLN_bias + mod).unsqueeze(1).chunk(6, dim=-1)
            gate_a, gate_f = torch.sigmoid(1 - gate_a), torch.sigmoid(1 - gate_f)
        else:
            scale_a, shift_a, gate_a, scale_f, shift_f, gate_f = (
                self.adaLN(mod)).unsqueeze(1).chunk(6, dim=-1)
        x = x + gate_a * self.attn(fp32_norm(self.norm1, x) * (1 + scale_a) + shift_a)
        if self.cross_attn is not None:
            x = x + self.cross_attn(fp32_norm(self.norm_cross, x), ctx)
        if self.inpaint_embed is not None:
            x = x + left_pad(self.inpaint_embed(inpaint), x.shape[1])
        x = x + gate_f * self.mlp(fp32_norm(self.norm2, x) * (1 + scale_f) + shift_f)
        return x


class ConditionedAudioDiT(nn.Module):
    CONFIGS = {
        'S':  dict(n_layers=6,  hidden_size=512,  n_heads=8),
        'B':  dict(n_layers=12, hidden_size=768,  n_heads=12),
        'G':  dict(n_layers=18, hidden_size=1024, n_heads=16),
        'L':  dict(n_layers=24, hidden_size=1024, n_heads=16),
        'XL': dict(n_layers=28, hidden_size=1152, n_heads=18),
    }

    def __init__(
        self,
        token_dim:           int,
        kind:                str   = 'L',
        mlp_ratio:           float = 4.0,
        drop:                float = 0.0,
        frame_cond_dims:     Optional[Dict[str, int]]  = None,
        frame_cond_out_dims: Optional[Dict[str, int]]  = None,
        global_cond_configs: Optional[Dict[str, dict]] = None,
        frame_reinject_every: int = 0,
        text_conditioning:    str = "ada",
        text_ctx_dim:         int = 0,
        attention:            str = "standard",
        qk_norm:              bool = False,
        inpainting:           bool = False,
        memory_tokens:        int  = 0,
        t_features:           str  = "high_freq",
        adaln:                str  = "shared",
    ):
        super().__init__()
        cfg = self.CONFIGS[kind]
        self.kind        = kind
        self.token_dim   = token_dim
        hidden_size      = cfg['hidden_size']
        n_layers         = cfg['n_layers']
        n_heads          = cfg['n_heads']
        if attention not in ATTENTION_KINDS:
            raise ValueError(
                f"attention must be one of {ATTENTION_KINDS}, got {attention!r}")
        if t_features not in T_FEATURE_CYCLES:
            raise ValueError(
                f"t_features must be one of {tuple(T_FEATURE_CYCLES)}, got {t_features!r}")
        if adaln not in ADALN_KINDS:
            raise ValueError(f"adaln must be one of {ADALN_KINDS}, got {adaln!r}")
        self.attention   = attention
        self.qk_norm     = bool(qk_norm)
        self.inpainting  = bool(inpainting)
        self.inpaint_ctx_dim = token_dim + 1 if self.inpainting else 0

        self.frame_cond_dims     = dict(frame_cond_dims)     if frame_cond_dims     else {}
        self.frame_cond_out_dims = dict(frame_cond_out_dims) if frame_cond_out_dims else {}
        if text_conditioning not in TEXT_CONDITIONING_KINDS:
            raise ValueError(f"text_conditioning must be one of "
                             f"{TEXT_CONDITIONING_KINDS}, got {text_conditioning!r}")
        configs = dict(global_cond_configs) if global_cond_configs else {}
        self.text_conditioning = text_conditioning
        self.text_in_cross = "text" in configs and text_conditioning != "ada"
        self.global_cond_configs = {k: v for k, v in configs.items()
                                    if k != "text" or text_conditioning != "cross_att"}
        self.has_frame  = len(self.frame_cond_dims)     > 0
        self.has_global = len(self.global_cond_configs) > 0

        if self.has_frame:
            missing = set(self.frame_cond_dims) - set(self.frame_cond_out_dims)
            if missing:
                raise ValueError(
                    f"frame_cond_out_dims is missing the out_dim for {sorted(missing)}. "
                    f"Every frame condition must declare both raw_dim and out_dim."
                )

        frame_extra = 0
        if self.has_frame:
            self.frame_encoder = FrameConditionEncoder(
                self.frame_cond_dims, self.frame_cond_out_dims,
            )
            frame_extra = self.frame_encoder.total_out_dim

        self.pre_conv = nn.Conv1d(token_dim + frame_extra, token_dim + frame_extra, 1, bias=False)
        self.input_proj = nn.Linear(token_dim + frame_extra, hidden_size, bias=False)

        self.n_memory = int(memory_tokens)
        self.memory_tokens = (nn.Parameter(torch.randn(self.n_memory, hidden_size))
                              if self.n_memory > 0 else None)

        self.t_embedder = TimestepEmbedder(hidden_size, t_features)
        self.adaLN = None
        if adaln == "shared":
            self.adaLN = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size),
            )

        if self.has_global:
            self.global_encoder = GlobalConditionEncoder(
                self.global_cond_configs, hidden_size,
            )

        self.text_ctx_dim = int(text_ctx_dim or 0)
        if self.text_in_cross:
            if self.text_ctx_dim <= 0:
                raise ValueError(f"text_conditioning={text_conditioning!r} needs "
                                 f"text_ctx_dim, the width of one text token")
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_ctx_dim, hidden_size, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=False),
            )
            self.text_pad = nn.Parameter(torch.randn(self.text_ctx_dim) * 0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads,
                     mlp_ratio=mlp_ratio, drop=drop,
                     cross_attention=self.text_in_cross,
                     attention=attention, qk_norm=self.qk_norm,
                     inpaint_ctx_dim=self.inpaint_ctx_dim,
                     adaln=adaln)
            for i in range(n_layers)
        ])

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

            self.frame_reinject_gate = nn.ModuleDict({
                str(i): nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, len(self.frame_encoder.names)),
                )
                for i in self.reinject_layers
            })

            self.register_buffer(
                "frame_slot_repeats",
                torch.tensor(
                    [self.frame_cond_out_dims[n]
                     for n in self.frame_encoder.names],
                    dtype=torch.long,
                ),
                persistent=False,
            )

        self.final_layer = None
        if adaln == "per_block":
            self.final_layer = FinalLayer(hidden_size, token_dim)
        else:
            self.output_proj = nn.Linear(hidden_size, token_dim, bias=False)
            self.post_conv = nn.Conv1d(token_dim, token_dim, 1, bias=False)

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
            print("  Frame re-injection: OFF (input concat only)")

        if "text" in configs:
            print(f"  Text conditioning: {text_conditioning}"
                  + (f" | tokens of {self.text_ctx_dim} values -> Linear-SiLU-Linear, "
                     f"cross-attention in every block" if self.text_in_cross else ""))

        _d = hidden_size // n_heads
        print(f"  Self-attention: {attention} ({n_heads} heads x {_d})"
              + (f" | out = A(q1,k1) v - A(q2,k2) v | +"
                 f"{n_layers * 2 * hidden_size ** 2 / 1e6:.2f}M params"
                 if attention == "differential" else "")
              + f" | RoPE on {rope_dim(_d)} of {_d} values per head")
        print(f"  Memory tokens: {self.n_memory}" if self.n_memory else "  Memory tokens: OFF")
        print(f"  Time features: {t_features} | cos and sin of 2 pi f t, f from "
              f"{T_FEATURE_CYCLES[t_features][0]:.3g} to {T_FEATURE_CYCLES[t_features][1]:.3g}")
        print(f"  AdaLN: {adaln} | "
              + ("SiLU-Linear on c in every block, gate g; final layer with LayerNorm, "
                 "scale, shift and a Linear with bias" if adaln == "per_block" else
                 "one Linear-SiLU-Linear on c + a bias per block, gate sigmoid(1 - g); "
                 "no modulation at the output"))

        if self.qk_norm:
            n_att = n_layers * (2 if self.text_in_cross else 1)
            print(f"  QK-norm: per-head RMSNorm on q and k before RoPE "
                  f"(eps {QK_NORM_EPS:g}) in {n_att} attention "
                  f"layer(s) | +{n_att * 2 * _d} params")
        else:
            print("  QK-norm: OFF")

        if self.inpainting:
            ip_params = n_layers * ((self.inpaint_ctx_dim + 1) * hidden_size
                                    + (hidden_size + 1) * hidden_size)
            print(f"  Inpainting context: mask + masked latent "
                  f"({self.inpaint_ctx_dim} values per frame) -> Linear-SiLU-"
                  f"Linear added before the FFN of every block | "
                  f"last layer zero-init | +{ip_params/1e6:.2f}M params")
        else:
            print("  Inpainting context: OFF")

    def initialize_weights(self):
        for block in self.blocks:
            if block.adaLN is None:
                nn.init.zeros_(block.attn.proj.weight)
                nn.init.zeros_(block.mlp.out.weight)
                nn.init.zeros_(block.mlp.out.bias)
            else:
                nn.init.zeros_(block.adaLN[-1].weight)
                nn.init.zeros_(block.adaLN[-1].bias)

        nn.init.zeros_(self.pre_conv.weight)
        if self.final_layer is None:
            nn.init.zeros_(self.post_conv.weight)
        else:
            nn.init.zeros_(self.final_layer.adaLN[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN[-1].bias)
            nn.init.zeros_(self.final_layer.linear.weight)
            nn.init.zeros_(self.final_layer.linear.bias)

        if self.reinject_layers:
            for lin in self.frame_reinject.values():
                nn.init.constant_(lin.weight, 0)

            for gate in self.frame_reinject_gate.values():
                nn.init.constant_(gate[-1].weight, 0)
                nn.init.constant_(gate[-1].bias, 1.0)

        for block in self.blocks:
            if block.cross_attn is not None:
                nn.init.zeros_(block.cross_attn.proj.weight)
            if block.inpaint_embed is not None:
                nn.init.constant_(block.inpaint_embed[-1].weight, 0)
                nn.init.constant_(block.inpaint_embed[-1].bias, 0)

    def _gather_frame_conditions(
        self,
        frame_conditions: Optional[Dict[str, torch.Tensor]],
        B: int, T: int, device, dtype,
    ) -> Dict[str, torch.Tensor]:
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
        if ctx is None:
            return torch.zeros(B, 1, self.text_ctx_dim, device=device, dtype=torch.float32)
        ctx = ctx.to(device=device, dtype=torch.float32)
        if ctx.dim() != 3 or ctx.shape[0] != B or ctx.shape[2] != self.text_ctx_dim:
            raise ValueError(
                f"text_context must be (B, L, {self.text_ctx_dim}) with B={B}, "
                f"got {tuple(ctx.shape)}")
        mask = mask.to(device=device).bool()
        if mask.shape != ctx.shape[:2]:
            raise ValueError(
                f"text_context_mask {tuple(mask.shape)} does not match the "
                f"context {tuple(ctx.shape[:2])}")
        ctx = torch.where(mask.unsqueeze(-1), ctx, self.text_pad)
        return ctx.masked_fill(~mask.any(dim=1).view(B, 1, 1), 0.0)

    def _gather_inpaint_context(self, inpaint_context, B, T, device):
        if inpaint_context is None:
            return torch.zeros(B, T, self.inpaint_ctx_dim, device=device,
                               dtype=torch.float32)
        if tuple(inpaint_context.shape) != (B, T, self.inpaint_ctx_dim):
            raise ValueError(
                f"inpaint_context must be (B, T, {self.inpaint_ctx_dim}) = "
                f"({B}, {T}, mask + {self.token_dim} masked latent values), "
                f"got {tuple(inpaint_context.shape)}")
        return inpaint_context.to(device=device, dtype=torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        frame_conditions:  Optional[Dict[str, torch.Tensor]] = None,
        global_conditions: Optional[Dict[str, torch.Tensor]] = None,
        text_context:      Optional[torch.Tensor] = None,
        text_context_mask: Optional[torch.Tensor] = None,
        inpaint_context:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()
        B, T, _ = x.shape

        inpaint = None
        if self.inpainting:
            inpaint = self._gather_inpaint_context(inpaint_context, B, T, x.device)
        elif inpaint_context is not None:
            raise ValueError(
                "an inpainting context was passed to a model built without "
                "inpainting (training.inpainting: false); it would be ignored")

        frame_proj = None
        if self.has_frame:
            fc = self._gather_frame_conditions(frame_conditions, B, T, x.device, x.dtype)
            frame_proj = self.frame_encoder(fc)
            x = torch.cat([x, frame_proj], dim=-1)

        x = x + self.pre_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_proj(x)
        if self.n_memory:
            x = torch.cat((self.memory_tokens.expand(B, -1, -1), x), dim=1)

        c = self.t_embedder(t)
        if self.has_global:
            gc = self._gather_global_conditions(global_conditions, B,
                                                x.device, torch.float32)
            c = c + self.global_encoder(gc)
        mod = c if self.adaLN is None else self.adaLN(c)

        ctx = None
        if self.text_in_cross:
            ctx = self.text_proj(self._gather_text_context(
                text_context, text_context_mask, B, x.device))

        reinject = self.frame_reinject if (self.reinject_layers and
                                           frame_proj is not None) else None
        for i, block in enumerate(self.blocks):
            if reinject is not None and str(i) in reinject:
                gate = self.frame_reinject_gate[str(i)](c)
                gate = torch.repeat_interleave(
                    gate, self.frame_slot_repeats, dim=-1).unsqueeze(1)
                x = x + left_pad(reinject[str(i)](frame_proj * gate), x.shape[1])
            x = block(x, mod, ctx, inpaint)

        x = x[:, self.n_memory:]
        if self.final_layer is not None:
            return self.final_layer(x, c)
        x = self.output_proj(x)
        return x + self.post_conv(x.transpose(1, 2)).transpose(1, 2)


def ckpt_text_ctx_dim(ckpt: dict) -> int:
    for k, v in ckpt["model_state_dict"].items():
        if k.endswith("text_proj.0.weight"):
            return int(v.shape[1])
    return 0
